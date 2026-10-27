"""Recompute a dataset's auto-metadata from its already-ingested zarr store.

For datasets whose folder structure and key naming are already settled,
there's no need to re-upload/re-ingest just to pick up new or corrected
computed fields (stats, derived values, etc.) -- this reads directly from
the immutable zarr store (never re-reading the original source file, which
may no longer even be around) and rewrites metadata.json's `auto` section
in place. `user` metadata (name, tags, description, ...) is left untouched.

Also clears and regenerates preview caches, since a metadata recompute is
a natural point to pick up rendering changes too (e.g. a new colormap).
"""

import logging
import shutil

from sqlmodel import Session

from thermal_dataset_manager import __version__ as INGEST_TOOL_VERSION
from thermal_dataset_manager.catalog import repository
from thermal_dataset_manager.ingest import preview_gen
from thermal_dataset_manager.ingest.metadata_compute import compute_camera_metadata
from thermal_dataset_manager.models.metadata import AutoMetadata, DatasetMetadata
from thermal_dataset_manager.storage import layout, zarr_store
from thermal_dataset_manager.storage.filesystem import FilesystemDatasetStorage

logger = logging.getLogger(__name__)


class DatasetNotFoundError(ValueError):
    pass


def recompute_metadata(
    storage: FilesystemDatasetStorage,
    dataset_id: str,
    db_session: Session,
    *,
    stats_sample_frames: int,
) -> DatasetMetadata:
    if not storage.dataset_exists(dataset_id):
        raise DatasetNotFoundError(dataset_id)

    existing = storage.read_metadata(dataset_id)
    store_path = storage.zarr_store_path(dataset_id)
    root = zarr_store.open_group(store_path)

    previous_hashes = {c.camera_index: c.content_sha256 for c in existing.auto.cameras}
    cameras = []
    for camera_index in zarr_store.camera_indices(store_path):
        cam_group = root[f"camera_{camera_index}"]
        cameras.append(
            compute_camera_metadata(
                camera_index=camera_index,
                camera_label=cam_group.attrs.get("camera_label"),
                frames=cam_group["counts"],
                telemetry=cam_group["telemetry"] if "telemetry" in cam_group.array_keys() else None,
                timestamps=cam_group["timestamps"][:],
                telemetry_row_count=int(cam_group.attrs["telemetry_row_count"]),
                timestamp_kind=cam_group.attrs["timestamp_kind"],
                stats_sample_frames=stats_sample_frames,
            ).model_copy(update={"content_sha256": previous_hashes.get(camera_index)})
        )

    auto = AutoMetadata(
        dataset_id=dataset_id,
        source_format=existing.auto.source_format,
        source_filename=existing.auto.source_filename,
        source_sha256=existing.auto.source_sha256,
        ingested_at=existing.auto.ingested_at,
        ingest_tool_version=INGEST_TOOL_VERSION,
        camera_count=len(cameras),
        cameras=cameras,
        other_variables=existing.auto.other_variables,
    )
    metadata = DatasetMetadata(auto=auto, user=existing.user)

    storage.write_metadata(dataset_id, metadata, locked=True)

    for camera in cameras:
        camera_previews_dir = layout.previews_dir(storage.data_root, dataset_id, camera.camera_index)
        shutil.rmtree(camera_previews_dir / "frames", ignore_errors=True)
        preview_gen.generate_thumbnail(storage.data_root, dataset_id, store_path, camera)

    repository.upsert_from_metadata(db_session, metadata)
    return metadata


def recompute_all(
    storage: FilesystemDatasetStorage,
    db_session: Session,
    *,
    stats_sample_frames: int,
) -> tuple[int, dict[str, str]]:
    """Recompute every dataset's metadata. Returns (succeeded_count, {dataset_id: error})."""
    errors: dict[str, str] = {}
    succeeded = 0
    for dataset_id in storage.list_dataset_ids():
        try:
            recompute_metadata(storage, dataset_id, db_session, stats_sample_frames=stats_sample_frames)
            succeeded += 1
        except Exception as e:  # noqa: BLE001 - one bad dataset shouldn't abort the batch
            logger.exception("Failed to recompute metadata for dataset %s", dataset_id)
            errors[dataset_id] = str(e)
    return succeeded, errors
