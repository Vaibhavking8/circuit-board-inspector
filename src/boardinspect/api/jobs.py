"""
api/jobs.py — in-memory job registry for background calibration/inspection.

Jobs run in daemon threads. The registry is a module-level singleton
(JOBS) so routes.py can import and use it without passing it around.
"""
from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class JobState:
    """State of one background job."""

    job_id: str
    status: str = "queued"      # queued | running | complete | failed
    progress: float = 0.0
    result: dict[str, Any] | None = None
    error: str | None = None


class JobManager:
    """Thread-safe in-memory job registry."""

    def __init__(self) -> None:
        self._jobs: dict[str, JobState] = {}
        self._lock = threading.Lock()

    def create(self) -> str:
        """Create a new job entry and return its job_id."""
        job_id = uuid.uuid4().hex[:8]
        with self._lock:
            self._jobs[job_id] = JobState(job_id=job_id)
        return job_id

    def get(self, job_id: str) -> JobState | None:
        """Return the JobState for *job_id*, or None if not found."""
        with self._lock:
            return self._jobs.get(job_id)

    def set_running(self, job_id: str) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].status = "running"

    def set_progress(self, job_id: str, progress: float) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].progress = max(0.0, min(1.0, progress))

    def set_complete(self, job_id: str, result: dict[str, Any]) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].status = "complete"
                self._jobs[job_id].progress = 1.0
                self._jobs[job_id].result = result

    def set_failed(self, job_id: str, error: str) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].status = "failed"
                self._jobs[job_id].error = error

    def run_in_background(self, job_id: str, fn: Callable[[], None]) -> None:
        """Start *fn* in a daemon thread, managing status transitions."""

        def _wrapper() -> None:
            self.set_running(job_id)
            try:
                fn()
            except Exception as exc:
                logger.exception("Job %s raised an exception", job_id)
                self.set_failed(job_id, str(exc))

        t = threading.Thread(target=_wrapper, daemon=True)
        t.start()


# Module-level singleton used by routes.py
JOBS = JobManager()
