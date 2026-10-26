import logging
from pathlib import Path

import numpy as np
from sqlmodel import Session
from ulid import ULID

from thermal_dataset_manager import __version__ as INGEST_TOOL_VERSION
from thermal_dataset_manager.catalog import repository
from thermal_dataset_manager.ingest import metadata_compute
from thermal_dataset_manager.ingest.key_registry import CameraKeyPair, discover_camera_keys
from thermal_dataset_manager.ingest.readers.common import (
    build_normalized_capture_from_pairs,
    describe_other_variables,
    open_source,
)
from thermal_dataset_manager.ingest.recompute import recompute_metadata
from thermal_dataset_manager.ingest.validation import IngestValidationError, validate
from thermal_dataset_manager.models.metadata import DatasetMetadata, SourceFormat, UserMetadata
from thermal_dataset_manager.storage import zarr_store
from thermal_dataset_manager.storage.filesystem import FilesystemDatasetStorage

logger = logging.getLogger(__name__)

_NPZ_ZST_SUFFIX = ".npz.zst"


class UnsupportedSourceFormatError(ValueError):
    pass


class DuplicateDatasetError(ValueError):
    """Raised when a source file byte-identical to an already-ingested one is
    submitted again. Carries the existing dataset's identifying details."""

    def __init__(self, source_sha256: str, existing: "repository.DatasetRecord") -> None:
        self.source_sha256 = source_sha256
        self.existing_dataset_id = existing.dataset_id
        self.existing_name = existing.name
        self.existing_source_filename = existing.source_filename
        self.existing_ingested_at = existing.ingested_at
        super().__init__(
            f"Source file is a duplicate (sha256 {source_sha256}) of already-ingested dataset "
            f"{existing.dataset_id!r} (name {existing.name!r}, from file "
            f"{existing.source_filename!r}, ingested {existing.ingested_at.isoformat()})"
        )

    @property
    def errors(self) -> list[str]:
        return [str(self)]


def detect_source_format(path: Path) -> SourceFormat:
    name = path.name.lower()
    if name.endswith(_NPZ_ZST_SUFFIX):
        return "npz.zst"
    if name.endswith(".npz"):
        return "npz"
    if name.endswith((".h5", ".hdf5")):
        return "h5"
    raise UnsupportedSourceFormatError(f"Unsupported source file extension: {path.name}")


def _dataset_name(source_path: Path, cam, multi_camera: bool) -> str:
    if not multi_camera:
        return source_path.stem
    suffix = cam.camera_label or f"cam{cam.camera_index}"
    return f"{source_path.stem}_{suffix}"


def write_raw_dataset(
    source_path: Path,
    storage: FilesystemDatasetStorage,
    db_session: Session,
    *,
    default_telemetry_row_count: int,
    zarr_target_chunk_bytes: int,
    camera_mapping: list[CameraKeyPair] | None = None,
    capture_id: str | None = None,
) -> list[DatasetMetadata]:
    """Validate a source capture file and commit it into the canonical Zarr
    layout. Deliberately does *not* scan pixel data for stats or render
    thumbnails; `metadata.auto.metadata_status` is left "pending" with
    cheap-only fields (frame counts/shapes/fps/duration) populated, and each
    dataset is immediately locked, browsable, and listed. Call
    `ingest.recompute.recompute_metadata` next to fill in the rest -- see
    `ingest_dataset` below for the combined, synchronous convenience path
    (used by the CLI).

    One dataset is written per camera stream in the source file, so a dataset
    is always a single thermal video. Each camera's telemetry rows are
    stripped from the frames and stored in a separate `telemetry` array in
    the same store. The datasets from a multi-camera file share a
    `capture_id` (generated, unless one is passed in -- e.g. to group
    several single-camera files). Only camera frames and timestamps are
    ingested; the source file's other arrays are summarized as free text in
    the metadata (`other_variables`) and otherwise dropped. Cameras are
    loaded, validated and written one at a time to keep memory to a single
    camera. Cameras already ingested from a byte-identical file are skipped;
    if that is every camera, DuplicateDatasetError is raised.

    By default, camera streams are auto-detected from the file's array keys
    via the key registry. Pass `camera_mapping` to bypass auto-detection.
    """
    source_format = detect_source_format(source_path)

    # Duplicate check first: hashing the file is far cheaper than decoding it,
    # and re-ingesting a byte-identical capture would only create a redundant
    # immutable dataset.
    source_sha256 = metadata_compute.sha256_of_file(source_path)

    with open_source(source_path, source_format) as source:
        # Everything up to loading the frames works from array headers alone.
        infos = source.infos
        pairs = camera_mapping if camera_mapping is not None else discover_camera_keys(list(infos))

        # Only uint16 frame arrays are ingested; anything else is never loaded.
        usable = [p for p in pairs if infos[p.frames_key].dtype == np.uint16]
        if not usable:
            skipped = ", ".join(f"{p.frames_key} ({infos[p.frames_key].dtype})" for p in pairs)
            raise IngestValidationError([f"No uint16 frame arrays found; skipped: {skipped}"])
        for p in pairs:
            if p not in usable:
                logger.warning(
                    "%s: skipping %s, dtype %s is not uint16", source_path, p.frames_key, infos[p.frames_key].dtype
                )

        new_pairs = []
        first_duplicate = None
        for pair in usable:
            existing = repository.find_by_source(db_session, source_sha256, pair.camera_index)
            if existing is None:
                new_pairs.append(pair)
            else:
                first_duplicate = first_duplicate or existing
                logger.warning(
                    "%s camera %d already ingested as dataset %s; skipping",
                    source_path, pair.camera_index, existing.dataset_id,
                )
        if not new_pairs:
            raise DuplicateDatasetError(source_sha256, first_duplicate)

        other_variables = describe_other_variables(
            source, {k for p in usable for k in (p.frames_key, p.timestamps_key)}
        )
        multi_camera = len(usable) > 1
        if capture_id is None and multi_camera:
            capture_id = str(ULID())

        # One camera at a time: load, validate, write, then let it go, so peak
        # memory is a single camera's frames. A failure partway leaves the
        # earlier cameras ingested; re-running skips them (duplicate check).
        results: list[DatasetMetadata] = []
        for pair in new_pairs:
            single = build_normalized_capture_from_pairs(
                source, [pair], default_telemetry_row_count, other_variables
            )
            validate(single)
            cam = single.cameras[0]
            dataset_id = str(ULID())

            auto_metadata = metadata_compute.compute_auto_metadata(
                dataset_id=dataset_id,
                source_path=source_path,
                source_format=source_format,
                capture=single,
                ingest_tool_version=INGEST_TOOL_VERSION,
                stats_sample_frames=0,
                compute_stats=False,
                source_sha256=source_sha256,
            )
            metadata = DatasetMetadata(
                auto=auto_metadata,
                user=UserMetadata(name=_dataset_name(source_path, cam, multi_camera), capture_id=capture_id),
            )

            zarr_store.write(
                storage.zarr_store_path(dataset_id),
                single.cameras,
                source_format,
                zarr_target_chunk_bytes,
            )
            storage.write_metadata(dataset_id, metadata, locked=False)
            storage.write_manifest(
                dataset_id,
                {
                    "dataset_id": dataset_id,
                    "source_filename": source_path.name,
                    "source_sha256": auto_metadata.source_sha256,
                    "camera_index": cam.camera_index,
                    "ingest_tool_version": INGEST_TOOL_VERSION,
                    "ingested_at": auto_metadata.ingested_at.isoformat(),
                },
            )
            storage.lock_dataset(dataset_id)
            repository.upsert_from_metadata(db_session, metadata)
            results.append(metadata)
            del single, cam  # release this camera's frames before loading the next

    return results


def ingest_dataset(
    source_path: Path,
    storage: FilesystemDatasetStorage,
    db_session: Session,
    *,
    default_telemetry_row_count: int,
    zarr_target_chunk_bytes: int,
    stats_sample_frames: int,
    camera_mapping: list[CameraKeyPair] | None = None,
    capture_id: str | None = None,
) -> list[DatasetMetadata]:
    """Convenience wrapper composing write_raw_dataset + recompute_metadata
    into one synchronous call -- one fully-computed dataset per camera in the
    source file.
    """
    bare = write_raw_dataset(
        source_path,
        storage,
        db_session,
        default_telemetry_row_count=default_telemetry_row_count,
        zarr_target_chunk_bytes=zarr_target_chunk_bytes,
        camera_mapping=camera_mapping,
        capture_id=capture_id,
    )
    return [
        recompute_metadata(storage, m.auto.dataset_id, db_session, stats_sample_frames=stats_sample_frames)
        for m in bare
    ]
