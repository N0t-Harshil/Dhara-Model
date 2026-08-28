from __future__ import annotations

"""Cooperative SIGINT/SIGTERM shutdown coordinator.

Signal policy:
  * First signal  — flag the shutdown, log ``[SHUTDOWN]``, and unwind with
                    ``SystemExit(128 + signum)`` so ordinary ``finally``
                    cleanup (pool drain, checkpoint flush, telemetry stop)
                    still runs. Worker modules must never install signal
                    handlers; they only poll ``requested()`` at safe points.
  * Second signal — the teardown is hung (e.g. a blocked pool join); force an
                    immediate hard exit with ``os._exit`` so an operator kill
                    can never be ignored.
  * Grace window  — a daemon watchdog forces ``os._exit`` ``grace_sec`` after
                    the first signal as a backstop for teardown paths that
                    neither return nor re-signal.

Only the orchestrator (``main.py``) calls ``install()``. The coordinator is
idempotent and safe to install/query from any thread.
"""

import logging
import os
import signal
import sys
import threading

logger = logging.getLogger(__name__)

_SIGNAL_NAMES = {
    signal.SIGINT: "SIGINT",
    getattr(signal, "SIGTERM", None): "SIGTERM",
}


class ShutdownCoordinator:
    def __init__(self, grace_sec: float = 30.0) -> None:
        self._grace_sec = float(grace_sec)
        self._event = threading.Event()
        self._signum: int | None = None
        self._count = 0
        self._lock = threading.Lock()
        self._watchdog: threading.Timer | None = None
        self._prev_handlers: dict = {}

    # -- queries (safe from any thread) ---------------------------

    def requested(self) -> bool:
        return self._event.is_set()

    def reason(self) -> str | None:
        with self._lock:
            return _SIGNAL_NAMES.get(self._signum) if self._signum is not None else None

    # -- signal path ----------------------------------------------

    def request(self, signum: int) -> None:
        """Handle one shutdown signal: first = graceful, further = force."""
        with self._lock:
            self._count += 1
            self._signum = signum
            self._event.set()
            if self._count > 1:
                logger.error(
                    "[SHUTDOWN] signal %s (%d) received again while draining — "
                    "forcing immediate exit",
                    _SIGNAL_NAMES.get(signum, "?"), signum)
                os._exit(128 + signum)
            if self._watchdog is None:
                self._watchdog = threading.Timer(
                    self._grace_sec, self._force, args=(signum,))
                self._watchdog.daemon = True
                self._watchdog.start()
        logger.warning(
            "[SHUTDOWN] signal %s (%d) received — graceful shutdown, draining "
            "(send again, or wait up to %.0fs, to force)",
            _SIGNAL_NAMES.get(signum, "?"), signum, self._grace_sec)
        raise SystemExit(128 + signum)

    def _force(self, signum: int) -> None:
        logger.error(
            "[SHUTDOWN] did not exit within grace window after signal %s (%d) — "
            "forcing immediate exit",
            _SIGNAL_NAMES.get(signum, "?"), signum)
        os._exit(128 + signum)

    def _handle(self, signum, frame) -> None:
        self.request(signum)

    # -- installation ---------------------------------------------

    def install(self, grace_sec: float | None = None) -> None:
        """Register the cooperative handlers for SIGINT and SIGTERM.

        Idempotent: repeated calls do not re-wire handlers already owned by
        this coordinator. Saves the previous handlers so ``reset()`` can
        restore them.
        """
        if grace_sec is not None:
            self._grace_sec = float(grace_sec)
        for sig in (signal.SIGINT, signal.SIGTERM):
            if self._owns(sig):
                continue
            self._prev_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, self._handle)

    def _owns(self, sig: int) -> bool:
        try:
            return signal.getsignal(sig) == self._handle
        except (ValueError, TypeError, OSError):
            return False

    def reset(self) -> None:
        """Restore previously installed handlers and disarm the watchdog
        (used by tests and by a coordinator reused across runs). A stale
        watchdog must never hard-exit a process that already recovered."""
        with self._lock:
            if self._watchdog is not None:
                self._watchdog.cancel()
                self._watchdog = None
            self._event.clear()
            self._signum = None
            self._count = 0
        for sig, handler in self._prev_handlers.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, TypeError, OSError):
                pass
        self._prev_handlers.clear()