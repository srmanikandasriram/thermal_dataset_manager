import logging
import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile
from pydantic import BaseModel
from sqlmodel import Session

from thermal_dataset_manager.api.deps import get_app_settings, get_engine, get_storage
from thermal_dataset_manager.api.jobs import IngestJob, JobTracker, get_job_tracker
from thermal_dataset_manager.catalog.db import get_session
from thermal_dataset_manager.config import Settings
from thermal_dataset_manager.ingest.inspect_file import inspect_source_file
from thermal_dataset_manager.ingest.key_registry import CameraKeyPair, KeyRegistryError
from thermal_dataset_manager.ingest.pipeline import (
    DuplicateDatasetError,
    UnsupportedSourceFormatError,
    detect_source_format,
    write_raw_dataset,
)
from thermal_dataset_manager.ingest.recompute import recompute_metadata
from thermal_dataset_manager.ingest.validation import IngestValidationError
from thermal_dataset_manager.storage import layout
from thermal_dataset_manager.storage.filesystem import FilesystemDatasetStorage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ingest", tags=["ingest"])


class ImportPathRequest(BaseModel):
    path: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    dataset_id: str | None
    errors: list[str]


class KeyInfoResponse(BaseModel):
    key: str
    shape: list[int]
    dtype: str


class CameraMapping(BaseModel):
    """One camera stream's key mapping -- either auto-detected via the key
    registry, or built manually by the user in the ingest UI.
    """

    camera_label: str | None = None
    frames_key: str
    timestamps_key: str


class UploadInspectionResponse(BaseModel):
    upload_token: str
    filename: str
    keys: list[KeyInfoResponse]
    auto_mapping: list[CameraMapping] | None
    auto_mapping_error: str | None


class ConfirmIngestRequest(BaseModel):
    upload_token: str
    filename: str
    mapping: list[CameraMapping]


def _run_ingest_job(
    job_id: str,
    source_path: Path,
    storage: FilesystemDatasetStorage,
    session: Session,
    settings: Settings,
    tracker: JobTracker,
    cleanup_dir: Path | None,
    camera_mapping: list[CameraKeyPair] | None = None,
) -> None:
    """Tracked phase: validate + write the raw zarr store. Once this
    succeeds the job is done as far as the user is concerned -- the dataset
    is already locked, browsable, and listed (with "pending" metadata_status
    and cheap-only fields). Stats/thumbnail computation then runs as an
    untracked continuation, decoupled from the ingest job: it can take
    longer, and a failure there doesn't retroactively fail an ingest that
    already succeeded (it just leaves metadata_status stuck "pending",
    fixable later via a metadata recompute).
    """
    tracker.mark_running(job_id)
    dataset_id: str | None = None
    try:
        metadata = write_raw_dataset(
            source_path,
            storage,
            session,
            default_telemetry_row_count=settings.default_telemetry_row_count,
            zarr_target_chunk_bytes=settings.zarr_target_chunk_bytes,
            camera_mapping=camera_mapping,
        )
        dataset_id = metadata.auto.dataset_id
        tracker.mark_succeeded(job_id, metadata)
    except (
        IngestValidationError,
        UnsupportedSourceFormatError,
        KeyRegistryError,
        DuplicateDatasetError,
    ) as e:
        errors = e.errors if isinstance(e, (IngestValidationError, DuplicateDatasetError)) else [str(e)]
        tracker.mark_failed(job_id, errors)
    except Exception as e:  # noqa: BLE001 - last resort so a job never gets stuck "running"
        logger.exception("Unexpected error writing raw dataset for %s", source_path)
        tracker.mark_failed(job_id, [f"Unexpected error: {e}"])
    finally:
        if cleanup_dir is not None and cleanup_dir.exists():
            shutil.rmtree(cleanup_dir, ignore_errors=True)

    if dataset_id is not None:
        try:
            recompute_metadata(storage, dataset_id, session, stats_sample_frames=settings.stats_sample_frames)
        except Exception:  # noqa: BLE001 - the ingest job already succeeded; don't undo that
            logger.exception("Metadata computation failed for dataset %s", dataset_id)

    session.close()


@router.post("/upload", response_model=UploadInspectionResponse)
async def upload_capture(
    file: UploadFile,
    storage: FilesystemDatasetStorage = Depends(get_storage),
    settings: Settings = Depends(get_app_settings),
) -> UploadInspectionResponse:
    """Stage an uploaded file and report what's in it: its raw array keys,
    plus an auto-detected camera mapping if the key registry recognizes the
    naming convention. Does not ingest -- call /confirm with a (possibly
    user-edited) mapping to actually convert and index the dataset.
    """
    if file.filename is None:
        raise HTTPException(status_code=422, detail="Missing filename")

    try:
        source_format = detect_source_format(Path(file.filename))
    except UnsupportedSourceFormatError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    # Opportunistic cleanup of orphaned uploads from prior sessions -- cheap
    # (a handful of directory stats at this scale), and means a long-running
    # server doesn't need a separate scheduled job to stay tidy.
    storage.sweep_stale_staging(settings.staging_max_age_seconds)

    upload_token = str(uuid4())
    staging_dir = storage.new_staging_dir(upload_token)
    dest_path = staging_dir / file.filename
    try:
        with open(dest_path, "wb") as out:
            while chunk := await file.read(1024 * 1024):
                out.write(chunk)
    except Exception as e:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"Upload failed or was interrupted: {e}") from e

    try:
        inspection = inspect_source_file(dest_path, source_format)
    except Exception as e:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise HTTPException(status_code=422, detail=f"Failed to read file: {e}") from e

    auto_mapping = None
    if inspection.auto_pairs is not None:
        auto_mapping = [
            CameraMapping(camera_label=p.camera_label, frames_key=p.frames_key, timestamps_key=p.timestamps_key)
            for p in inspection.auto_pairs
        ]

    return UploadInspectionResponse(
        upload_token=upload_token,
        filename=file.filename,
        keys=[KeyInfoResponse(key=k.key, shape=list(k.shape), dtype=k.dtype) for k in inspection.keys],
        auto_mapping=auto_mapping,
        auto_mapping_error=inspection.auto_mapping_error,
    )


@router.delete("/upload/{upload_token}")
def cancel_upload(
    upload_token: str,
    storage: FilesystemDatasetStorage = Depends(get_storage),
) -> dict:
    """Discard a staged upload that was never (or won't be) confirmed --
    e.g. the user backed out of the mapping review step."""
    storage.remove_staging_dir(upload_token)
    return {"cancelled": True}


@router.post("/confirm", response_model=JobStatusResponse)
def confirm_ingest(
    request: ConfirmIngestRequest,
    background_tasks: BackgroundTasks,
    storage: FilesystemDatasetStorage = Depends(get_storage),
    settings: Settings = Depends(get_app_settings),
    tracker: JobTracker = Depends(get_job_tracker),
    engine=Depends(get_engine),
) -> JobStatusResponse:
    """Ingest a previously-staged upload using the given camera key mapping."""
    staging_dir = layout.staging_dir(storage.data_root, request.upload_token)
    source_path = staging_dir / request.filename
    if not source_path.exists():
        raise HTTPException(
            status_code=404, detail="Staged upload not found (it may have expired or already been ingested)"
        )
    if not request.mapping:
        raise HTTPException(status_code=422, detail="At least one camera mapping is required")

    pairs = [
        CameraKeyPair(
            camera_index=i,
            camera_label=m.camera_label,
            frames_key=m.frames_key,
            timestamps_key=m.timestamps_key,
        )
        for i, m in enumerate(request.mapping)
    ]

    job = tracker.create()
    session = get_session(engine)

    background_tasks.add_task(
        _run_ingest_job, job.job_id, source_path, storage, session, settings, tracker, staging_dir, pairs
    )

    return JobStatusResponse(job_id=job.job_id, status=job.status, dataset_id=None, errors=[])


@router.post("/import-path", response_model=JobStatusResponse)
def import_path(
    request: ImportPathRequest,
    background_tasks: BackgroundTasks,
    storage: FilesystemDatasetStorage = Depends(get_storage),
    settings: Settings = Depends(get_app_settings),
    tracker: JobTracker = Depends(get_job_tracker),
    engine=Depends(get_engine),
) -> JobStatusResponse:
    source_path = Path(request.path)
    if not source_path.exists():
        raise HTTPException(status_code=404, detail=f"Path does not exist: {source_path}")

    job = tracker.create()
    session = get_session(engine)

    background_tasks.add_task(
        _run_ingest_job, job.job_id, source_path, storage, session, settings, tracker, None
    )

    return JobStatusResponse(job_id=job.job_id, status=job.status, dataset_id=None, errors=[])


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(
    job_id: str,
    tracker: JobTracker = Depends(get_job_tracker),
) -> JobStatusResponse:
    job: IngestJob | None = tracker.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(job_id=job.job_id, status=job.status, dataset_id=job.dataset_id, errors=job.errors)
