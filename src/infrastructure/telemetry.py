from __future__ import annotations

"""Pipeline runtime telemetry.

A small, lock-guarded metrics accumulator plus optional monitor thread that
periodically samples GPU/CPU state and renders a compact status line. Design:

  * every counter write/read is under a single RLock — safe from any thread
    (producers, workers, the Trainer callbacks, and the orchestrator).
  * nothing here can raise into training: all sampling is best-effort and
    guarded; failures degrade the metric to 0/None rather than propagating.
  * no heavy per-step work — only aggregate counters, computed lazily in
    snapshot().
"""

import logging
import os
import subprocess
import threading
import time
from typing import Any, Dict, Optional

try:
    import psutil
except ImportError:  # pragma: no cover — optional
    psutil = None

logger = logging.getLogger(__name__)


class PipelineTelemetry:
    def __init__(self, enabled: bool = True, interval_sec: float = 30.0,
                 gpu_util_sampling: bool = True) -> None:
        self.enabled = enabled
        self.interval = max(1.0, float(interval_sec))
        self._gpu_sample = gpu_util_sampling
        self._lock = threading.RLock()
        self._t0 = time.monotonic()
        self._started_at = self._t0
        self._counters: Dict[str, float] = {}
        self._last = {}  # last-value event metrics (gpu util, mem, ...)
        self._monitor: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._smoke_gpu_avail = _gpu_command_available()

    # -- reporting -------------------------------------------------

    def record(self, name: str, value: float = 1.0) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._counters[name] = self._counters.get(name, 0.0) + value

    def set_event(self, name: str, value: float) -> None:
        """Overwrite a momentary metric (gpu util %, mem, active dataset)."""
        if not self.enabled:
            return
        with self._lock:
            self._last[name] = value

    def elapsed(self) -> float:
        return time.monotonic() - self._t0

    # -- lifecycle -------------------------------------------------

    def start(self) -> None:
        if not self.enabled:
            return
        if self._monitor is None:
            self._stop.clear()
            self._monitor = threading.Thread(
                target=self._sample_loop, daemon=True, name="telemetry")
            self._monitor.start()

    def stop(self) -> None:
        self._stop.set()
        if self._monitor is not None:
            self._monitor.join(timeout=2.0)
            if self._monitor.is_alive():
                logger.warning("Telemetry sampler did not exit within 2s; abandoning thread.")
            self._monitor = None

    def _sample_loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self._sample_once()
            except Exception:  # noqa: BLE001 — telemetry never breaks training
                pass

    def _sample_once(self) -> None:
        if self._gpu_sample and self._smoke_gpu_avail:
            util = _gpu_utilization()
            if util is not None:
                self.set_event("gpu_util_pct", util)
            mem_free, mem_total = _gpu_memory()
            if mem_free is not None:
                self.set_event("gpu_mem_free_gb", mem_free)
                self.set_event("gpu_mem_total_gb", mem_total)
        if psutil is not None:
            self.set_event("cpu_util_pct", psutil.cpu_percent(interval=None))
            self.set_event("ram_used_gb", psutil.virtual_memory().used / (1024 ** 3))

    # -- snapshot --------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            c = dict(self._counters)
            ev = dict(self._last)
        out: Dict[str, Any] = {"uptime_s": round(time.monotonic() - self._t0, 1)}
        out.update(c)
        out.update(ev)
        samples = c.get("samples", 0)
        tokens = c.get("tokens", 0)
        out["samples_per_sec"] = samples / max(1e-9, out["uptime_s"])
        out["tokens_per_sec"] = tokens / max(1e-9, out["uptime_s"])
        out["network_wait_s"] = round(c.get("network_wait_s", 0.0), 1)
        out["cache_hit_pct"] = _pct(
            c.get("cache_hits", 0.0), c.get("cache_attempts", 0.0))
        out["metadata_hit_pct"] = _pct(
            c.get("metadata_hits", 0.0), c.get("metadata_attempts", 0.0))
        return out

    def summary_line(self, extra: Optional[Dict[str, Any]] = None) -> str:
        s = self.snapshot()
        parts = [
            f"up={_fmt_dur(s['uptime_s'])}",
            f"samp/s={s['samples_per_sec']:.0f}",
            f"tok/s={s['tokens_per_sec']:.0f}",
        ]
        gpu_util = s.get("gpu_util_pct")
        ctx = {"gpu": f"{gpu_util:.0f}%" if gpu_util is not None else "n/a"}
        if s.get("gpu_mem_free_gb") is not None:
            ctx["gpu_mem"] = f"{s['gpu_mem_free_gb']:.1f}/{s['gpu_mem_total_gb']:.1f}G"
        if s.get("cache_hit_pct"):
            ctx["cache_hit"] = f"{s['cache_hit_pct']:.0f}%"
        if s.get("network_wait_s"):
            ctx["net_wait"] = f"{s['network_wait_s']}s"
        if extra:
            ctx.update(extra)
        for k, v in ctx.items():
            parts.append(f"{k}={v}")
        return " | ".join(parts)

    def finish(self) -> Dict[str, Any]:
        """Stop the monitor and return a final snapshot (for the report)."""
        self.stop()
        return self.snapshot()


def _fmt_dur(sec: float) -> str:
    sec = int(sec)
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _pct(n: float, d: float) -> float:
    return (100.0 * n / d) if d > 0 else 0.0


def _gpu_command_available() -> bool:
    """True if nvidia-smi is on PATH (sampling would not be futile)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--version"], capture_output=True, timeout=3)
        return out.returncode == 0
    except Exception:  # noqa: BLE001
        return False


def _gpu_memory() -> Optional[tuple]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, timeout=3, text=True)
        if out.returncode == 0 and out.stdout.strip():
            # One "free,total" pair per GPU line; sum across GPUs.
            free_total = 0.0
            total_total = 0.0
            for line in out.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",") if p.strip()]
                if len(parts) < 2:
                    continue
                free_total += float(parts[0])
                total_total += float(parts[1])
            if total_total > 0:
                return free_total / 1024.0, total_total / 1024.0
    except Exception:  # noqa: BLE001
        pass
    return None


def _gpu_utilization() -> Optional[float]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, timeout=3, text=True)
        if out.returncode == 0 and out.stdout.strip():
            return float(out.stdout.strip().split(",")[0])
    except Exception:  # noqa: BLE001
        pass
    return None