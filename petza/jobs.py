"""One background job at a time (import, teach, mark, review) with progress."""
from __future__ import annotations

import threading
import time
import traceback


class Busy(Exception):
    pass


class JobRunner:
    def __init__(self):
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.state = {"kind": None, "status": "idle", "progress": 0.0, "message": "", "result": None, "error": None}

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self.state)

    def running(self) -> bool:
        return self.snapshot()["status"] == "running"

    def start(self, kind: str, fn) -> None:
        """fn(progress, should_stop) -> result (JSON-able)."""
        with self._lock:
            if self.state["status"] == "running":
                raise Busy(f"Already busy with: {self.state['kind']}")
            self._stop.clear()
            self.state = {"kind": kind, "status": "running", "progress": 0.0, "message": "Starting",
                          "result": None, "error": None, "started": time.time()}

        def progress(frac: float, message: str):
            with self._lock:
                self.state["progress"] = round(max(0.0, min(1.0, frac)), 4)
                self.state["message"] = message

        def body():
            try:
                result = fn(progress, self._stop.is_set)
                with self._lock:
                    self.state.update(status="stopped" if self._stop.is_set() else "done", progress=1.0, result=result)
            except Exception as e:
                traceback.print_exc()
                with self._lock:
                    self.state.update(status="stopped" if self._stop.is_set() else "error", error=str(e))

        threading.Thread(target=body, daemon=True, name=f"job-{kind}").start()

    def cancel(self) -> None:
        self._stop.set()
