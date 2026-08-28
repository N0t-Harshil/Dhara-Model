from __future__ import annotations

"""Cooperative shutdown coordinator tests (Phase 2).

Hermetic: no subprocesses, no real teardown. Monkeypatches ``os._exit`` so
the force path is asserted without killing the test runner. Signal
registration/restoration is exercised carefully and wrapped in try/finally.
"""

import os
import signal
import threading

import pytest

from src.utils.shutdown import ShutdownCoordinator


def _monkeypatched_exit(monkeypatch):
    """Stub os._exit with one that actually terminates control flow (raises),
    so the force path is asserted exactly like a real hard exit."""
    class _HardExit(Exception):
        def __init__(self, code):
            super().__init__(code)
            self.code = code

    calls = {}
    monkeypatch.setattr(
        "os._exit",
        lambda code: (_ for _ in ()).throw(_HardExit(code)),
    )
    return calls, _HardExit


class TestShutdownCoordinator:
    def test_not_requested_by_default(self):
        c = ShutdownCoordinator()
        assert not c.requested()
        assert c.reason() is None

    def test_request_raises_system_exit_with_128_plus_signum(self):
        c = ShutdownCoordinator(grace_sec=3600)
        with pytest.raises(SystemExit) as ei:
            c.request(signal.SIGTERM)
        assert ei.value.code == 128 + signal.SIGTERM

    def test_request_flags_shutdown_and_records_reason(self):
        c = ShutdownCoordinator(grace_sec=3600)
        with pytest.raises(SystemExit):
            c.request(signal.SIGINT)
        assert c.requested()
        assert c.reason() == "SIGINT"

    def test_handle_matches_request(self):
        c = ShutdownCoordinator(grace_sec=3600)
        with pytest.raises(SystemExit) as ei:
            c._handle(signal.SIGTERM, None)
        assert ei.value.code == 128 + signal.SIGTERM
        assert c.requested()

    def test_second_signal_forces_hard_exit(self, monkeypatch):
        calls, _HardExit = _monkeypatched_exit(monkeypatch)
        c = ShutdownCoordinator(grace_sec=3600)
        with pytest.raises(SystemExit):
            c._handle(signal.SIGTERM, None)
        with pytest.raises(_HardExit) as ei:
            c._handle(signal.SIGTERM, None)
        assert ei.value.code == 128 + signal.SIGTERM
        assert not calls  # stub raised; nothing recorded

    def test_force_exits_hard_after_grace(self, monkeypatch):
        calls, _HardExit = _monkeypatched_exit(monkeypatch)
        c = ShutdownCoordinator(grace_sec=3600)
        with pytest.raises(SystemExit):
            c._handle(signal.SIGTERM, None)
        with pytest.raises(_HardExit) as ei:
            c._force(signal.SIGTERM)
        assert ei.value.code == 128 + signal.SIGTERM
        assert not calls

    def test_install_is_idempotent_and_removeable(self):
        c = ShutdownCoordinator(grace_sec=3600)
        try:
            c.install()
            c.install()  # idempotent — no error, no re-wire
            assert c._owns(signal.SIGINT)
            assert c._owns(signal.SIGTERM)
        finally:
            c.reset()
        assert not c._owns(signal.SIGINT)
        assert not c._owns(signal.SIGTERM)

    def test_different_instances_do_not_clobber_lock(self):
        a = ShutdownCoordinator(grace_sec=3600)
        b = ShutdownCoordinator(grace_sec=3600)
        with pytest.raises(SystemExit):
            a.request(signal.SIGINT)
        try:
            assert not b.requested()  # unrelated instance stays clean
            assert a.requested()
        finally:
            a.reset()

    def test_reset_disarms_watchdog_and_clears_state(self):
        c = ShutdownCoordinator(grace_sec=3600)
        with pytest.raises(SystemExit):
            c._handle(signal.SIGTERM, None)
        assert c._watchdog is not None and c._watchdog.is_alive()
        c.reset()
        assert c._watchdog is None, "reset() must cancel the armed watchdog"
        assert not c.requested()
        assert c.reason() is None
        # Reuse after reset stays clean (no stale hard-exit scheduled).
        with pytest.raises(SystemExit):
            c._handle(signal.SIGINT, None)
        assert c.requested()