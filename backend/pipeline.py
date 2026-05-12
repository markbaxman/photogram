import asyncio
import logging
import re
import subprocess
import time
from pathlib import Path
from typing import Callable, Optional

import trimesh

from .config import COLMAP_BIN, FFMPEG_BIN, FRAME_FPS, HAS_NVIDIA
from .jobs import Job, JobStatus, update_job

logger = logging.getLogger(__name__)


def _count_images(images_dir: Path) -> int:
    return len(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg")) + list(images_dir.glob("*.png")))


def _run_colmap_step(
    job: Job,
    step_name: str,
    args: list[str],
    progress_range: tuple[int, int],
    log_parser: Optional[Callable[[str, Job, int, int], None]] = None,
) -> None:
    start_pct, end_pct = progress_range
    update_job(job.job_id, progress=start_pct, message=f"{job.message}")

    logger.info(f"Running {step_name}: {' '.join(args)}")

    process = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    line_count = 0
    output_lines = []
    try:
        for line in process.stdout:
            line = line.rstrip()
            output_lines.append(line)
            if log_parser:
                log_parser(line, job, start_pct, end_pct)
            else:
                line_count += 1
                mid = (start_pct + end_pct) // 2
                if line_count == 1:
                    update_job(job.job_id, progress=mid)
    except Exception as e:
        logger.error(f"Error reading stdout: {e}")
        process.terminate()

    # Read any remaining stderr before wait
    stderr_lines = []
    try:
        if process.stderr:
            stderr_lines = process.stderr.readlines()
    except Exception as e:
        logger.error(f"Error reading stderr: {e}")

    returncode = process.wait()
    if returncode != 0:
        stderr_text = "".join(stderr_lines[-20:]) if stderr_lines else ""
        last_output = "\n".join(output_lines[-10:]) if output_lines else "(no output)"
        error_msg = f"{step_name} failed with exit code {returncode}"
        if stderr_text:
            error_msg += f"\nStderr:\n{stderr_text}"
        if last_output != "(no output)":
            error_msg += f"\nLast stdout:\n{last_output}"
        raise RuntimeError(error_msg)

    update_job(job.job_id, progress=end_pct)


def _parse_feature_progress(line: str, job: Job, start: int, end: int) -> None:
    m = re.search(r"\[\s*(\d+)/\s*(\d+)\]", line)
    if m:
        current, total = int(m.group(1)), int(m.group(2))
        if total > 0:
            pct = start + int((current / total) * (end - start))
            update_job(job.job_id, progress=pct)


def _parse_mapper_progress(line: str, job: Job, start: int, end: int) -> None:
    if "Registering image" in line:
        m = re.search(r"Registering image #\d+ \((\d+)\)", line)
        if m:
            registered = int(m.group(1))
            images_dir = job.work_dir / "images"
            total = _count_images(images_dir)
            if total > 0:
                pct = start + int((registered / total) * (end - start))
                update_job(job.job_id, progress=min(pct, end - 1))


def _parse_stereo_progress(line: str, job: Job, start: int, end: int) -> None:
    m = re.search(r"Processing image (\d+) / (\d+)", line)
    if m:
        current, total = int(m.group(1)), int(m.group(2))
        if total > 0:
            pct = start + int((current / total) * (end - start))
            update_job(job.job_id, progress=pct)


def _extract_frames(job: Job, source_path: Path) -> Path:
    images_dir = job.work_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    update_job(job.job_id, status=JobStatus.EXTRACTING, progress=5)

    cmd = [
        FFMPEG_BIN, "-i", str(source_path),
        "-vf", f"fps={FRAME_FPS}",
        "-q:v", "2",
        str(images_dir / "frame_%06d.jpg"),
        "-hide_banner", "-loglevel", "info",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr[-500:]}")

    count = _count_images(images_dir)
    if count < 5:
        raise RuntimeError(
            f"Only {count} frames extracted — video may be too short or corrupt. "
            "Try uploading a longer video or use image mode."
        )

    # Validate that at least one extracted image is readable
    try:
        from PIL import Image
        img_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg"))
        if img_files:
            img = Image.open(str(img_files[0]))
            img.verify()
    except Exception as e:
        raise RuntimeError(f"Extracted images may be corrupted: {e}")

    update_job(job.job_id, progress=10, message=f"Extracted {count} frames")
    return images_dir


def _ply_to_obj(job: Job, ply_path: Path, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    update_job(job.job_id, status=JobStatus.CONVERTING, progress=95)

    mesh = trimesh.load(str(ply_path), force="mesh")

    obj_path = out_dir / "model.obj"
    mtl_path = out_dir / "model.mtl"

    result = trimesh.exchange.obj.export_obj(mesh, include_texture=False)
    obj_path.write_bytes(result.encode() if isinstance(result, str) else result)

    mtl_path.write_text(
        "newmtl material0\nKa 0.2 0.2 0.2\nKd 0.8 0.8 0.8\nKs 0.0 0.0 0.0\n"
    )

    obj_content = obj_path.read_text()
    if "mtllib" not in obj_content:
        obj_path.write_text(f"mtllib model.mtl\n{obj_content}")

    return obj_path, mtl_path


async def run_pipeline(job_id: str) -> None:
    from .jobs import get_job

    job = get_job(job_id)
    if job is None:
        return

    loop = asyncio.get_event_loop()

    try:
        await loop.run_in_executor(None, _run_pipeline_sync, job)
    except Exception as exc:
        update_job(
            job_id,
            status=JobStatus.FAILED,
            error=str(exc),
            message=f"Failed: {exc}",
            finished_at=time.time(),
        )


def _run_pipeline_sync(job: Job) -> None:
    import shutil

    work_dir = job.work_dir
    images_dir = work_dir / "images"
    db_path = work_dir / "database.db"
    sparse_dir = work_dir / "sparse"
    dense_dir = work_dir / "dense"
    output_dir = work_dir / "output"

    # Verify COLMAP binary exists
    colmap_path = shutil.which(COLMAP_BIN)
    if not colmap_path:
        raise RuntimeError(f"COLMAP binary not found: {COLMAP_BIN}")
    logger.info(f"Using COLMAP binary: {colmap_path}")

    sparse_dir.mkdir(parents=True, exist_ok=True)
    dense_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 0: Extract frames if video input
    if job.is_video:
        video_path = next(work_dir.glob("raw.*"), None)
        if video_path is None:
            raise RuntimeError("No video file found in job directory")
        _extract_frames(job, video_path)
    else:
        image_count = _count_images(images_dir)
        if image_count < 5:
            raise RuntimeError(
                f"Only {image_count} images uploaded. Please upload at least 5 overlapping images."
            )
        update_job(job.job_id, progress=10, message=f"Processing {image_count} images")

    # Remove existing database to start fresh
    if db_path.exists():
        db_path.unlink()

    # Step 1: Feature extraction (10→25)
    update_job(job.job_id, status=JobStatus.FEATURES)

    # Log image directory info
    image_count = _count_images(images_dir)
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg")) + list(images_dir.glob("*.png"))
    logger.info(f"Feature extraction: {image_count} images found in {images_dir}")
    if image_files:
        logger.info(f"First image: {image_files[0].name} ({image_files[0].stat().st_size} bytes)")

    _run_colmap_step(
        job,
        "feature_extractor",
        [
            COLMAP_BIN, "feature_extractor",
            "--database_path", str(db_path),
            "--image_path", str(images_dir),
            "--SiftExtraction.use_gpu", "0",
        ],
        (10, 25),
        log_parser=_parse_feature_progress,
    )

    # Step 2: Feature matching (25→40)
    update_job(job.job_id, status=JobStatus.MATCHING)
    _run_colmap_step(
        job,
        "sequential_matcher",
        [
            COLMAP_BIN, "sequential_matcher",
            "--database_path", str(db_path),
        ],
        (25, 40),
    )

    # Step 3: Sparse reconstruction (40→55)
    update_job(job.job_id, status=JobStatus.SPARSE)
    _run_colmap_step(
        job,
        "mapper",
        [
            COLMAP_BIN, "mapper",
            "--database_path", str(db_path),
            "--image_path", str(images_dir),
            "--output_path", str(sparse_dir),
        ],
        (40, 55),
        log_parser=_parse_mapper_progress,
    )

    sparse_model = sparse_dir / "0"
    if not sparse_model.exists():
        raise RuntimeError(
            "Sparse reconstruction produced no model. "
            "Not enough overlapping images were matched. "
            "Ensure images have 60–80% overlap and good lighting."
        )

    # Step 4: Image undistortion (55→62)
    update_job(job.job_id, status=JobStatus.UNDISTORTING)
    _run_colmap_step(
        job,
        "image_undistorter",
        [
            COLMAP_BIN, "image_undistorter",
            "--image_path", str(images_dir),
            "--input_path", str(sparse_model),
            "--output_path", str(dense_dir),
            "--output_type", "COLMAP",
        ],
        (55, 62),
    )

    # Step 5: Dense depth estimation (62→78)
    update_job(job.job_id, status=JobStatus.DENSE_DEPTH)
    patch_match_args = [
        COLMAP_BIN, "patch_match_stereo",
        "--workspace_path", str(dense_dir),
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.geom_consistency", "true",
        "--PatchMatchStereo.depth_min", "0.1",
        "--PatchMatchStereo.depth_max", "10.0",
    ]
    if not HAS_NVIDIA:
        patch_match_args += ["--PatchMatchStereo.gpu_index", "-1"]
    _run_colmap_step(job, "patch_match_stereo", patch_match_args, (62, 78), log_parser=_parse_stereo_progress)

    # Step 6: Stereo fusion (78→88)
    fused_ply = dense_dir / "fused.ply"
    update_job(job.job_id, status=JobStatus.DENSE_FUSION)
    _run_colmap_step(
        job,
        "stereo_fusion",
        [
            COLMAP_BIN, "stereo_fusion",
            "--workspace_path", str(dense_dir),
            "--workspace_format", "COLMAP",
            "--input_type", "geometric",
            "--output_path", str(fused_ply),
            "--StereoFusion.min_num_pixels", "3",
        ],
        (78, 88),
    )

    # Step 7: Poisson meshing (88→95)
    mesh_ply = dense_dir / "mesh.ply"
    update_job(job.job_id, status=JobStatus.MESHING)
    _run_colmap_step(
        job,
        "poisson_mesher",
        [
            COLMAP_BIN, "poisson_mesher",
            "--input_path", str(fused_ply),
            "--output_path", str(mesh_ply),
            "--PoissonMeshing.trim", "7",
        ],
        (88, 95),
    )

    # Step 8: Convert PLY → OBJ+MTL (95→100)
    obj_path, mtl_path = _ply_to_obj(job, mesh_ply, output_dir)

    update_job(
        job.job_id,
        status=JobStatus.DONE,
        progress=100,
        message="3D model ready",
        output_obj=obj_path,
        output_mtl=mtl_path,
        finished_at=time.time(),
    )
