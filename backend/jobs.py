import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class JobStatus(str, Enum):
    QUEUED = "queued"
    EXTRACTING = "extracting"
    FEATURES = "features"
    MATCHING = "matching"
    SPARSE = "sparse"
    UNDISTORTING = "undistorting"
    DENSE_DEPTH = "dense_depth"
    DENSE_FUSION = "dense_fusion"
    MESHING = "meshing"
    CONVERTING = "converting"
    DONE = "done"
    FAILED = "failed"


STATUS_LABELS = {
    JobStatus.QUEUED: "Queued",
    JobStatus.EXTRACTING: "Extracting frames from video",
    JobStatus.FEATURES: "Detecting features in images",
    JobStatus.MATCHING: "Matching features across images",
    JobStatus.SPARSE: "Building sparse 3D model",
    JobStatus.UNDISTORTING: "Undistorting images",
    JobStatus.DENSE_DEPTH: "Computing dense depth maps",
    JobStatus.DENSE_FUSION: "Fusing depth maps into point cloud",
    JobStatus.MESHING: "Generating mesh",
    JobStatus.CONVERTING: "Exporting OBJ model",
    JobStatus.DONE: "Done",
    JobStatus.FAILED: "Failed",
}


@dataclass
class Job:
    job_id: str
    work_dir: Path
    status: JobStatus = JobStatus.QUEUED
    progress: int = 0
    message: str = "Queued"
    error: str = ""
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    output_obj: Optional[Path] = None
    output_mtl: Optional[Path] = None
    downloaded: bool = False
    is_video: bool = False


_jobs: dict[str, Job] = {}
_lock = threading.Lock()


def create_job(work_dir: Path, is_video: bool = False) -> Job:
    job_id = str(uuid.uuid4())
    job = Job(job_id=job_id, work_dir=work_dir, is_video=is_video)
    with _lock:
        _jobs[job_id] = job
    return job


def get_job(job_id: str) -> Optional[Job]:
    with _lock:
        return _jobs.get(job_id)


def update_job(job_id: str, **kwargs) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        for key, value in kwargs.items():
            setattr(job, key, value)
        if "status" in kwargs and "message" not in kwargs:
            job.message = STATUS_LABELS.get(kwargs["status"], kwargs["status"])


def all_jobs() -> list[Job]:
    with _lock:
        return list(_jobs.values())


def remove_job(job_id: str) -> None:
    with _lock:
        _jobs.pop(job_id, None)
