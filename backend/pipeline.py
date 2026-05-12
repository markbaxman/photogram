import asyncio
import logging
import os
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

    work_dir = job.work_dir
    log_file = work_dir / f"{step_name}.log"

    try:
        with open(log_file, 'w') as logf:
            env = os.environ.copy()
            env['QT_QPA_PLATFORM'] = 'offscreen'
            env['LIBGL_ALWAYS_INDIRECT'] = '1'
            env['MESA_GL_VERSION_OVERRIDE'] = '4.3'
            result = subprocess.run(
                args,
                stdout=logf,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
                timeout=600,
                env=env,
            )

        log_content = log_file.read_text()
        output_lines = log_content.split('\n')

        if log_parser:
            for line in output_lines:
                if line.strip():
                    log_parser(line, job, start_pct, end_pct)
        elif output_lines:
            update_job(job.job_id, progress=(start_pct + end_pct) // 2)

        if result.returncode != 0:
            last_lines = "\n".join(output_lines[-30:])
            error_msg = f"{step_name} failed with exit code {result.returncode}\n\nLast output:\n{last_lines}" if last_lines.strip() else f"{step_name} failed with exit code {result.returncode} (no output)"
            raise RuntimeError(error_msg)

        update_job(job.job_id, progress=end_pct)

    except subprocess.TimeoutExpired:
        raise RuntimeError(f"{step_name} timed out after 10 minutes")
    except Exception as e:
        raise


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


def _read_points3d_binary(binary_file: Path) -> list[tuple[float, float, float, int, int, int]]:
    """Read points from COLMAP points3D.bin format."""
    import struct
    points = []
    try:
        with open(binary_file, 'rb') as f:
            num_points = struct.unpack('<Q', f.read(8))[0]
            for _ in range(num_points):
                point_id = struct.unpack('<Q', f.read(8))[0]
                x, y, z = struct.unpack('<ddd', f.read(24))
                r, g, b = struct.unpack('<BBB', f.read(3))
                error = struct.unpack('<d', f.read(8))[0]
                track_length = struct.unpack('<Q', f.read(8))[0]
                f.read(track_length * 8)  # Skip track data
                points.append((x, y, z, r, g, b))
    except Exception as e:
        raise RuntimeError(f"Failed to read points3D.bin: {e}")
    return points


def _create_mesh_from_points(points_ply: Path, output_dir: Path) -> Path:
    """Create a simple mesh from sparse point cloud using trimesh."""
    try:
        mesh = trimesh.load(str(points_ply))
        # Convert point cloud to mesh using convex hull or simple triangulation
        if hasattr(mesh, 'vertices') and len(mesh.vertices) > 3:
            # Use convex hull for a simple closed mesh
            mesh = mesh.convex_hull
            mesh_path = output_dir / "mesh.ply"
            mesh.export(str(mesh_path))
            return mesh_path
    except Exception as e:
        raise RuntimeError(f"Failed to create mesh from points: {e}")


def _create_mesh_from_colmap_points(sparse_model: Path, output_dir: Path) -> Path:
    """Create OBJ mesh from COLMAP sparse point cloud."""
    points_bin = sparse_model / "points3D.bin"
    if not points_bin.exists():
        raise RuntimeError(f"Points file not found: {points_bin}")

    points = _read_points3d_binary(points_bin)
    if not points:
        raise RuntimeError("No points found in sparse reconstruction")

    # Create mesh using convex hull
    import numpy as np
    vertices = np.array([[p[0], p[1], p[2]] for p in points], dtype=np.float64)
    colors = np.array([[p[3], p[4], p[5]] for p in points], dtype=np.uint8)

    # Try to create convex hull
    try:
        mesh = trimesh.Trimesh(vertices=vertices)
        mesh = mesh.convex_hull
        mesh_path = output_dir / "mesh.ply"
        mesh.export(str(mesh_path))
        return mesh_path
    except Exception as e:
        # If convex hull fails, export as simple point cloud OBJ
        obj_path = output_dir / "mesh.obj"
        mtl_path = output_dir / "mesh.mtl"

        obj_content = "# Point cloud from sparse reconstruction\n"
        for x, y, z, r, g, b in points:
            obj_content += f"v {x} {y} {z}\n"

        obj_path.write_text(obj_content)
        mtl_path.write_text("newmtl material0\nKa 0.2 0.2 0.2\nKd 0.8 0.8 0.8\nKs 0.0 0.0 0.0\n")

        return obj_path


async def run_pipeline(job_id: str) -> None:
    from .jobs import get_job
    import tempfile

    job = get_job(job_id)
    if job is None:
        print(f"Job {job_id} not found", flush=True)
        return

    log_file = Path(tempfile.gettempdir()) / f"pipeline_{job_id}.log"
    msg = f"[PIPELINE] Starting job {job_id}"
    print(msg, flush=True)
    with open(log_file, "a") as f:
        f.write(msg + "\n")

    loop = asyncio.get_event_loop()

    try:
        job.log_file = log_file
        await loop.run_in_executor(None, _run_pipeline_sync, job)
        msg = f"[PIPELINE] Job {job_id} completed successfully"
        print(msg, flush=True)
        with open(log_file, "a") as f:
            f.write(msg + "\n")
    except Exception as exc:
        msg = f"[PIPELINE] Job {job_id} failed: {exc}"
        print(msg, flush=True)
        with open(log_file, "a") as f:
            f.write(msg + "\n")
        update_job(
            job_id,
            status=JobStatus.FAILED,
            error=str(exc),
            message=f"Failed: {exc}",
            finished_at=time.time(),
        )


def _run_pipeline_sync(job: Job) -> None:
    import shutil
    import tempfile

    log_file = getattr(job, 'log_file', Path(tempfile.gettempdir()) / f"pipeline_{job.job_id}.log")

    def log(msg):
        print(msg, flush=True)
        with open(log_file, "a") as f:
            f.write(msg + "\n")

    log(f"[_run_pipeline_sync] Starting for job {job.job_id}")

    work_dir = job.work_dir
    images_dir = work_dir / "images"
    db_path = work_dir / "database.db"
    sparse_dir = work_dir / "sparse"
    dense_dir = work_dir / "dense"
    output_dir = work_dir / "output"

    # Verify COLMAP binary exists and works
    colmap_path = shutil.which(COLMAP_BIN)
    if not colmap_path:
        raise RuntimeError(f"COLMAP binary not found: {COLMAP_BIN}")
    log(f"[COLMAP] Using binary: {colmap_path}")

    # Test that COLMAP runs
    try:
        env = os.environ.copy()
        env['QT_QPA_PLATFORM'] = 'offscreen'
        env['LIBGL_ALWAYS_INDIRECT'] = '1'
        env['MESA_GL_VERSION_OVERRIDE'] = '4.3'
        result = subprocess.run([COLMAP_BIN, "--version"], capture_output=True, text=True, timeout=5, env=env)
        logger.info(f"COLMAP version check: {result.stdout.strip()}")
    except Exception as e:
        logger.warning(f"Could not get COLMAP version: {e}")

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

    # Validate before feature extraction
    if not images_dir.exists():
        raise RuntimeError(f"Images directory does not exist: {images_dir}")

    image_count = _count_images(images_dir)
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg")) + list(images_dir.glob("*.png"))
    log(f"[FEATURE_EXTRACT] Found {image_count} images in {images_dir}")

    if image_count == 0:
        raise RuntimeError(f"No images found in {images_dir}")

    if image_files:
        logger.info(f"First image: {image_files[0].name} ({image_files[0].stat().st_size} bytes)")
        try:
            from PIL import Image
            img = Image.open(str(image_files[0]))
            img.verify()
            logger.info(f"First image validation: OK (size {img.width}x{img.height})")
        except Exception as e:
            logger.warning(f"Could not verify image: {e}")

    if not db_path.parent.exists():
        raise RuntimeError(f"Work directory does not exist: {db_path.parent}")

    _run_colmap_step(
        job,
        "feature_extractor",
        [
            COLMAP_BIN, "feature_extractor",
            "--database_path", str(db_path),
            "--image_path", str(images_dir),
            "--SiftExtraction.use_gpu", "0",
            "--SiftExtraction.max_image_size", "3200",
            "--SiftExtraction.peak_threshold", "0.03",
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
            "--SiftMatching.use_gpu", "0",
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

    # Step 4: Point cloud to mesh conversion (55→95)
    # Dense reconstruction requires CUDA which is not available, so create mesh from sparse points.
    update_job(job.job_id, status=JobStatus.MESHING)

    log(f"[SPARSE_MESH] Converting sparse point cloud to mesh...")
    mesh_path = _create_mesh_from_colmap_points(sparse_model, output_dir)
    log(f"[SPARSE_MESH] Created mesh at {mesh_path}")

    update_job(job.job_id, progress=95)

    # Step 5: Convert to OBJ+MTL format if needed (95→100)
    update_job(job.job_id, progress=95)

    # If mesh_path is already OBJ, use it directly; otherwise convert from PLY
    if isinstance(mesh_path, Path) and mesh_path.suffix.lower() == '.obj':
        obj_path = mesh_path
        mtl_path = output_dir / "mesh.mtl"
        if not mtl_path.exists():
            mtl_path.write_text("newmtl material0\nKa 0.2 0.2 0.2\nKd 0.8 0.8 0.8\nKs 0.0 0.0 0.0\n")
    else:
        # Convert PLY to OBJ
        if isinstance(mesh_path, Path):
            obj_path, mtl_path = _ply_to_obj(job, mesh_path, output_dir)
        else:
            raise RuntimeError("No valid mesh produced")

    update_job(
        job.job_id,
        status=JobStatus.DONE,
        progress=100,
        message="3D model ready",
        output_obj=obj_path,
        output_mtl=mtl_path,
        finished_at=time.time(),
    )
