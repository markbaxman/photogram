import os
import pathlib

WORK_DIR = pathlib.Path(os.getenv("WORK_DIR", "/tmp/photogram_jobs"))
COLMAP_BIN = os.getenv("COLMAP_BIN", "colmap")
FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")

FRAME_FPS = int(os.getenv("FRAME_FPS", "1"))
JOB_TTL_SECS = int(os.getenv("JOB_TTL_SECS", "3600"))
CLEANUP_INTERVAL = int(os.getenv("CLEANUP_INTERVAL", "300"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "4096"))

import shutil
HAS_NVIDIA = shutil.which("nvidia-smi") is not None
