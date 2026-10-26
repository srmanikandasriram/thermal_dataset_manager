import os
import stat

from tests.conftest import TELEMETRY_ROWS, write_single_camera_npz
from thermal_dataset_manager.ingest.pipeline import ingest_dataset

INGEST_KWARGS = dict(
    default_telemetry_row_count=TELEMETRY_ROWS,
    zarr_target_chunk_bytes=1024 * 1024,
    stats_sample_frames=8,
)


def test_metadata_writable_through_lock(tmp_path, storage, db_session):
    src = tmp_path / "capture.npz"
    write_single_camera_npz(src, n_frames=5)
    [metadata] = ingest_dataset(src, storage, db_session, **INGEST_KWARGS)
    dataset_id = metadata.auto.dataset_id

    metadata_path = storage.dataset_dir(dataset_id) / "metadata.json"
    mode_before = stat.S_IMODE(os.stat(metadata_path).st_mode)
    assert not (mode_before & stat.S_IWUSR)

    updated = metadata.model_copy(deep=True)
    updated.user.name = "renamed"
    storage.write_metadata(dataset_id, updated, locked=True)

    # File is readable and content updated, and ends up read-only again.
    mode_after = stat.S_IMODE(os.stat(metadata_path).st_mode)
    assert not (mode_after & stat.S_IWUSR)

    reloaded = storage.read_metadata(dataset_id)
    assert reloaded.user.name == "renamed"


def test_zarr_store_is_not_writable_after_ingest(tmp_path, storage, db_session):
    src = tmp_path / "capture.npz"
    write_single_camera_npz(src, n_frames=5)
    [metadata] = ingest_dataset(src, storage, db_session, **INGEST_KWARGS)
    store_path = storage.zarr_store_path(metadata.auto.dataset_id)

    for root, dirs, files in os.walk(store_path):
        for f in files:
            mode = stat.S_IMODE(os.stat(os.path.join(root, f)).st_mode)
            assert not (mode & stat.S_IWUSR), f"{f} should be read-only"
