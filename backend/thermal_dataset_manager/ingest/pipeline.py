from pathlib import Path

from sqlmodel import Session
from ulid import ULID

from thermal_dataset_manager import __version__ as INGEST_TOOL_VERSION
from thermal_dataset_manager.catalog import repository
from thermal_dataset_manager.ingest import metadata_compute
from thermal_dataset_manager.ingest.key_registry import CameraKeyPair, discover_camera_keys
from thermal_dataset_manager.ingest.readers.common import build_normalized_capture_from_pairs, load_array_map
from thermal_dataset_manager.ingest.recompute import recompute_metadata
from thermal_dataset_manager.ingest.validation import validate
from thermal_dataset_manager.models.metadata import DatasetMetadata, SourceFormat, UserMetadata
from thermal_dataset_manager.storage import zarr_store
from thermal_dataset_manager.storage.filesystem import FilesystemDatasetStorage

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


def write_raw_dataset(
    source_path: Path,
    storage: FilesystemDatasetStorage,
    db_session: Session,
    *,
    default_telemetry_row_count: int,
    zarr_target_chunk_bytes: int,
    camera_mapping: list[CameraKeyPair] | None = None,
) -> DatasetMetadata:
    """Validate a source capture file and commit it into the canonical Zarr
    layout -- the part that genuinely needs the user's input (the file
    itself, and the camera key mapping). Deliberately does *not* scan pixel
    data for stats or render thumbnails; `metadata.auto.metadata_status` is
    left "pending" with cheap-only fields (frame counts/shapes/fps/duration)
    populated, and the dataset is immediately locked, browsable, and
    listed. Call `ingest.recompute.recompute_metadata` next (e.g. from a
    background task) to fill in the rest -- see `ingest_dataset` below for
    the combined, synchronous convenience path (used by the CLI).

    By default, camera streams are auto-detected from the file's array keys
    via the key registry. Pass `camera_mapping` (e.g. built from a user's
    manual key mapping in the UI) to bypass auto-detection entirely.
    """
    source_format = detect_source_format(source_path)

    # Duplicate check first: hashing the file is far cheaper than decoding it,
    # and re-ingesting a byte-identical capture would only create a redundant
    # immutable dataset.
    source_sha256 = metadata_compute.sha256_of_file(source_path)
    existing = repository.find_by_source_sha256(db_session, source_sha256)
    if existing is not None:
        raise DuplicateDatasetError(source_sha256, existing)

    array_map = load_array_map(source_path, source_format)
    pairs = camera_mapping if camera_mapping is not None else discover_camera_keys(list(array_map.keys()))
    capture = build_normalized_capture_from_pairs(array_map, pairs, default_telemetry_row_count)
    validate(capture)

    dataset_id = str(ULID())

    auto_metadata = metadata_compute.compute_auto_metadata(
        dataset_id=dataset_id,
        source_path=source_path,
        source_format=source_format,
        capture=capture,
        ingest_tool_version=INGEST_TOOL_VERSION,
        stats_sample_frames=0,
        compute_stats=False,
        source_sha256=source_sha256,
    )
    metadata = DatasetMetadata(
        auto=auto_metadata,
        user=UserMetadata(name=source_path.stem),
    )

    store_path = storage.zarr_store_path(dataset_id)
    zarr_store.write(
        store_path, capture.cameras, source_format, zarr_target_chunk_bytes, capture.extra_arrays
    )

    storage.write_metadata(dataset_id, metadata, locked=False)

    storage.write_manifest(
        dataset_id,
        {
            "dataset_id": dataset_id,
            "source_filename": source_path.name,
            "source_sha256": auto_metadata.source_sha256,
            "ingest_tool_version": INGEST_TOOL_VERSION,
            "ingested_at": auto_metadata.ingested_at.isoformat(),
        },
    )

    storage.lock_dataset(dataset_id)
    repository.upsert_from_metadata(db_session, metadata)

    return metadata


def ingest_dataset(
    source_path: Path,
    storage: FilesystemDatasetStorage,
    db_session: Session,
    *,
    default_telemetry_row_count: int,
    zarr_target_chunk_bytes: int,
    stats_sample_frames: int,
    camera_mapping: list[CameraKeyPair] | None = None,
) -> DatasetMetadata:
    """Convenience wrapper composing write_raw_dataset + recompute_metadata
    into one synchronous call -- for the CLI and anywhere else that wants a
    single "fully ingested, stats and all" result rather than the two-phase
    flow the API uses.
    """
    bare = write_raw_dataset(
        source_path,
        storage,
        db_session,
        default_telemetry_row_count=default_telemetry_row_count,
        zarr_target_chunk_bytes=zarr_target_chunk_bytes,
        camera_mapping=camera_mapping,
    )
    return recompute_metadata(
        storage, bare.auto.dataset_id, db_session, stats_sample_frames=stats_sample_frames
    )
