"""In-memory ingest job tracking.

No queue -- appropriate at the "dozens-hundreds of datasets" scale this app
targets. A job in progress during a server restart is lost; revisit with a
real job store if ingest volume or durability needs grow.
"""

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from thermal_dataset_manager.models.metadata import DatasetMetadata

JobStatus = Literal["pending", "running", "succeeded", "failed"]


@dataclass
class IngestJob:
    job_id: str
    status: JobStatus = "pending"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    dataset_id: str | None = None
    errors: list[str] = field(default_factory=list)


class JobTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, IngestJob] = {}

    def create(self) -> IngestJob:
        job = IngestJob(job_id=str(uuid4()))
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> IngestJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def mark_running(self, job_id: str) -> None:
        with self._lock:
            self._jobs[job_id].status = "running"

    def mark_succeeded(self, job_id: str, metadata: DatasetMetadata) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = "succeeded"
            job.dataset_id = metadata.auto.dataset_id

    def mark_failed(self, job_id: str, errors: list[str]) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = "failed"
            job.errors = errors


_tracker = JobTracker()


def get_job_tracker() -> JobTracker:
    return _tracker
