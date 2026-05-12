import asyncio
import shutil
import time

from .config import CLEANUP_INTERVAL, JOB_TTL_SECS
from .jobs import JobStatus, all_jobs, remove_job


async def cleanup_loop() -> None:
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL)
        _cleanup_expired()


def _cleanup_expired() -> None:
    now = time.time()
    terminal = {JobStatus.DONE, JobStatus.FAILED}

    for job in all_jobs():
        expired = (now - job.created_at) > JOB_TTL_SECS
        done_and_downloaded = job.status in terminal and job.downloaded

        if expired or done_and_downloaded:
            shutil.rmtree(job.work_dir, ignore_errors=True)
            remove_job(job.job_id)
