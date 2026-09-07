import numpy as np
import pytest

from tests.conftest import (
    TELEMETRY_ROWS,
    write_multi_camera_npz,
    write_single_camera_h5,
    write_single_camera_npz,
    write_single_camera_npz_zst,
)
from thermal_dataset_manager.catalog import repository
from thermal_dataset_manager.ingest.pipeline import (
    DuplicateDatasetError,
    UnsupportedSourceFormatError,
    ingest_dataset,
)
from thermal_dataset_manager.ingest.validation import IngestValidationError
from thermal_dataset_manager.storage import zarr_store

INGEST_KWARGS = dict(
    default_telemetry_row_count=TELEMETRY_ROWS,
    zarr_target_chunk_bytes=1024 * 1024,
    stats_sample_frames=8,
)


def test_ingest_squeezes_trailing_singleton_channel_dim(tmp_path, storage, db_session):
    """Some sources emit frames as (n_frames, H, W, 1) rather than
    (n_frames, H, W) -- the trailing singleton dim should be squeezed out
    rather than tripping the "frames must be 3-D" validation check.
    """
    src = tmp_path / "channel_dim.npz"
    frames = np.random.default_rng(0).integers(0, 60000, size=(10, 12, 16, 1), dtype=np.uint16)
    ts = 1_700_000_000.0 + np.arange(10) / 30.0
    np.savez(src, raw_thr_frames=frames, raw_thr_tstamps=ts)

    metadata = ingest_dataset(src, storage, db_session, **INGEST_KWARGS)

    cam = metadata.auto.cameras[0]
    assert cam.frame_height == 12 - TELEMETRY_ROWS
    assert cam.frame_width == 16

    store_path = storage.zarr_store_path(metadata.auto.dataset_id)
    frame = zarr_store.read_frame(store_path, 0, 0)
    assert frame.shape == (12, 16)


def test_ingest_rejects_byte_identical_duplicate(tmp_path, storage, db_session):
    src = tmp_path / "capture.npz"
    write_single_camera_npz(src, n_frames=8)
    first = ingest_dataset(src, storage, db_session, **INGEST_KWARGS)

    # Same file again (and a renamed copy) -- both are byte-identical.
    dup = tmp_path / "capture_copy.npz"
    dup.write_bytes(src.read_bytes())

    for path in (src, dup):
        with pytest.raises(DuplicateDatasetError) as excinfo:
            ingest_dataset(path, storage, db_session, **INGEST_KWARGS)
        e = excinfo.value
        assert e.existing_dataset_id == first.auto.dataset_id
        assert e.source_sha256 == first.auto.source_sha256
        assert first.auto.dataset_id in str(e)

    # Only the first dataset was ever written.
    assert len(list(storage.list_dataset_ids())) == 1


def test_ingest_allows_different_content_same_shape(tmp_path, storage, db_session):
    a = tmp_path / "a.npz"
    b = tmp_path / "b.npz"
    write_single_camera_npz(a, n_frames=8, seed=1)
    write_single_camera_npz(b, n_frames=8, seed=2)

    ingest_dataset(a, storage, db_session, **INGEST_KWARGS)
    ingest_dataset(b, storage, db_session, **INGEST_KWARGS)

    assert len(list(storage.list_dataset_ids())) == 2


def test_ingest_single_camera_npz(tmp_path, storage, db_session):
    src = tmp_path / "capture.npz"
    write_single_camera_npz(src, n_frames=20)

    metadata = ingest_dataset(src, storage, db_session, **INGEST_KWARGS)

    assert metadata.auto.source_format == "npz"
    assert metadata.auto.camera_count == 1
    cam = metadata.auto.cameras[0]
    assert cam.frame_count == 20
    assert cam.frame_height == 10  # 12 total - 2 telemetry rows
    assert cam.telemetry_row_count == TELEMETRY_ROWS
    assert cam.fps_estimate == pytest.approx(30.0, rel=0.05)
    assert metadata.user.name == "capture"

    # Catalog was updated.
    record = repository.get(db_session, metadata.auto.dataset_id)
    assert record is not None
    assert record.total_frame_count == 20

    # Dataset is locked read-only.
    store_path = storage.zarr_store_path(metadata.auto.dataset_id)
    assert store_path.exists()
    import os
    import stat

    mode = stat.S_IMODE(os.stat(store_path).st_mode)
    assert not (mode & stat.S_IWUSR)


def test_ingest_npz_zst(tmp_path, storage, db_session):
    src = tmp_path / "capture.npz.zst"
    write_single_camera_npz_zst(src, n_frames=10)
    metadata = ingest_dataset(src, storage, db_session, **INGEST_KWARGS)
    assert metadata.auto.source_format == "npz.zst"
    assert metadata.auto.cameras[0].frame_count == 10


def test_ingest_h5(tmp_path, storage, db_session):
    src = tmp_path / "capture.h5"
    write_single_camera_h5(src, n_frames=12)
    metadata = ingest_dataset(src, storage, db_session, **INGEST_KWARGS)
    assert metadata.auto.source_format == "h5"
    assert metadata.auto.cameras[0].frame_count == 12


def test_ingest_multi_camera(tmp_path, storage, db_session):
    src = tmp_path / "multi.npz"
    write_multi_camera_npz(src, n_frames=10, n_cameras=2)
    metadata = ingest_dataset(src, storage, db_session, **INGEST_KWARGS)

    assert metadata.auto.camera_count == 2
    indices = {cam.camera_index for cam in metadata.auto.cameras}
    assert indices == {0, 1}

    store_path = storage.zarr_store_path(metadata.auto.dataset_id)
    assert zarr_store.camera_indices(store_path) == [0, 1]

    # Each camera has its own thumbnail.
    for cam_idx in (0, 1):
        thumb = storage.data_root / "datasets" / metadata.auto.dataset_id / "previews" / f"camera_{cam_idx}" / "thumbnail.png"
        assert thumb.exists()


def test_ingest_unsupported_extension(tmp_path, storage, db_session):
    src = tmp_path / "capture.txt"
    src.write_text("nope")
    with pytest.raises(UnsupportedSourceFormatError):
        ingest_dataset(src, storage, db_session, **INGEST_KWARGS)


def test_ingest_rejects_mismatched_frame_and_timestamp_counts(tmp_path, storage, db_session):
    src = tmp_path / "bad.npz"
    frames = np.zeros((10, 12, 16), dtype=np.uint16)
    timestamps = np.arange(5, dtype=np.float64)  # wrong length
    np.savez(src, raw_thr_frames=frames, raw_thr_tstamps=timestamps)

    with pytest.raises(IngestValidationError) as exc_info:
        ingest_dataset(src, storage, db_session, **INGEST_KWARGS)
    assert any("frame_count" in e for e in exc_info.value.errors)


def test_ingested_dataset_readable_from_disk(tmp_path, storage, db_session):
    src = tmp_path / "capture.npz"
    write_single_camera_npz(src, n_frames=6)
    metadata = ingest_dataset(src, storage, db_session, **INGEST_KWARGS)

    reloaded = storage.read_metadata(metadata.auto.dataset_id)
    assert reloaded == metadata
