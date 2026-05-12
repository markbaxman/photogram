import asyncio
import io
import mimetypes
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path

import aiofiles
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .cleanup import cleanup_loop
from .config import APP_VERSION, MAX_UPLOAD_MB, WORK_DIR
from .jobs import JobStatus, create_job, get_job, update_job
from .pipeline import run_pipeline

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp", "image/tiff"}
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/quicktime", "video/x-msvideo", "video/mpeg", "video/webm"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    task = asyncio.create_task(cleanup_loop())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="Photogrammetry App", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/version")
async def get_version():
    return JSONResponse({"version": APP_VERSION})


@app.post("/upload")
async def upload(
    files: list[UploadFile] = File(default=[]),
    video: UploadFile = File(default=None),
):
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024

    if video and video.filename:
        mime = video.content_type or mimetypes.guess_type(video.filename)[0] or ""
        if mime not in ALLOWED_VIDEO_TYPES:
            raise HTTPException(400, f"Unsupported video type: {mime}")

        job = create_job(WORK_DIR / "placeholder", is_video=True)
        job_dir = WORK_DIR / job.job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        update_job(job.job_id, work_dir=job_dir)
        job.work_dir = job_dir

        suffix = Path(video.filename).suffix or ".mp4"
        dest = job_dir / f"raw{suffix}"
        async with aiofiles.open(dest, "wb") as f:
            total = 0
            while chunk := await video.read(1024 * 1024):
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(413, "Video file too large")
                await f.write(chunk)

    elif files:
        valid_files = [f for f in files if f.filename]
        if not valid_files:
            raise HTTPException(400, "No files provided")

        for f in valid_files:
            mime = f.content_type or mimetypes.guess_type(f.filename)[0] or ""
            if mime not in ALLOWED_IMAGE_TYPES:
                raise HTTPException(400, f"Unsupported file type: {mime} ({f.filename})")

        job = create_job(WORK_DIR / "placeholder", is_video=False)
        job_dir = WORK_DIR / job.job_id
        images_dir = job_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        update_job(job.job_id, work_dir=job_dir)
        job.work_dir = job_dir

        for f in valid_files:
            dest = images_dir / Path(f.filename).name
            async with aiofiles.open(dest, "wb") as out:
                total = 0
                while chunk := await f.read(1024 * 1024):
                    total += len(chunk)
                    if total > max_bytes:
                        raise HTTPException(413, f"File {f.filename} too large")
                    await out.write(chunk)
    else:
        raise HTTPException(400, "Provide either 'video' or 'files'")

    asyncio.create_task(run_pipeline(job.job_id))
    return JSONResponse({"job_id": job.job_id})


@app.get("/status/{job_id}")
async def status(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return JSONResponse({
        "job_id": job.job_id,
        "status": job.status,
        "progress": job.progress,
        "message": job.message,
        "error": job.error,
    })


@app.get("/download/{job_id}")
async def download(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job.status != JobStatus.DONE:
        raise HTTPException(400, f"Job is not ready (status: {job.status})")
    if not job.output_obj or not job.output_obj.exists():
        raise HTTPException(500, "Output file missing")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(job.output_obj, "model.obj")
        if job.output_mtl and job.output_mtl.exists():
            zf.write(job.output_mtl, "model.mtl")
    buf.seek(0)

    update_job(job_id, downloaded=True)

    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=room_model.zip"},
    )


@app.get("/jobs/{job_id}/files/{filename}")
async def serve_output_file(job_id: str, filename: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job.status != JobStatus.DONE:
        raise HTTPException(400, "Model not ready yet")

    output_dir = job.work_dir / "output"
    target = (output_dir / filename).resolve()

    if not str(target).startswith(str(output_dir.resolve())):
        raise HTTPException(403, "Access denied")
    if not target.exists():
        raise HTTPException(404, "File not found")

    return FileResponse(target)


@app.get("/viewer/{job_id}", response_class=HTMLResponse)
async def viewer_page(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return FileResponse(FRONTEND_DIR / "viewer.html")
