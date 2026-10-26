import pytest

from thermal_dataset_manager.catalog import repository
from thermal_dataset_manager.ingest.pipeline import ingest_dataset
from thermal_dataset_manager.ingest.recompute import DatasetNotFoundError, recompute_all, recompute_metadata
from tests.conftest import TELEMETRY_ROWS, write_multi_camera_npz, write_single_camera_npz

INGEST_KWARGS = dict(
    default_telemetry_row_count=TELEMETRY_ROWS,
    zarr_target_chunk_bytes=1024 * 1024,
    stats_sample_frames=8,
)


def test_recompute_reproduces_equivalent_metadata(tmp_path, storage, db_session):
    src = tmp_path / "capture.npz"
    write_single_camera_npz(src, n_frames=12)
    [original] = ingest_dataset(src, storage, db_session, **INGEST_KWARGS)

    recomputed = recompute_metadata(storage, original.auto.dataset_id, db_session, stats_sample_frames=8)

    orig_cam = original.auto.cameras[0]
    new_cam = recomputed.auto.cameras[0]
    assert new_cam.frame_count == orig_cam.frame_count
    assert new_cam.frame_height == orig_cam.frame_height
    assert new_cam.frame_width == orig_cam.frame_width
    assert new_cam.fps_estimate == pytest.approx(orig_cam.fps_estimate)
    assert new_cam.counts_min == orig_cam.counts_min
    assert new_cam.counts_max == orig_cam.counts_max


def test_recompute_preserves_user_metadata(tmp_path, storage, db_session):
    src = tmp_path / "capture.npz"
    write_single_camera_npz(src, n_frames=6)
    [original] = ingest_dataset(src, storage, db_session, **INGEST_KWARGS)
    dataset_id = original.auto.dataset_id

    edited = original.model_copy(deep=True)
    edited.user.name = "My renamed dataset"
    edited.user.tags = ["outdoor"]
    storage.write_metadata(dataset_id, edited, locked=True)
    repository.upsert_from_metadata(db_session, edited)

    recomputed = recompute_metadata(storage, dataset_id, db_session, stats_sample_frames=8)

    assert recomputed.user.name == "My renamed dataset"
    assert recomputed.user.tags == ["outdoor"]


def test_recompute_preserves_split_dataset_identity(tmp_path, storage, db_session):
    src = tmp_path / "two_cam.npz"
    write_multi_camera_npz(src, n_frames=8, n_cameras=2)
    for original in ingest_dataset(src, storage, db_session, **INGEST_KWARGS):
        recomputed = recompute_metadata(storage, original.auto.dataset_id, db_session, stats_sample_frames=8)

        assert recomputed.auto.camera_count == 1
        assert recomputed.auto.cameras[0].camera_index == original.auto.cameras[0].camera_index
        assert recomputed.user.capture_id == original.user.capture_id


def test_recompute_updates_catalog(tmp_path, storage, db_session):
    src = tmp_path / "capture.npz"
    write_single_camera_npz(src, n_frames=6)
    [original] = ingest_dataset(src, storage, db_session, **INGEST_KWARGS)
    dataset_id = original.auto.dataset_id

    # Directly corrupt the catalog row to prove recompute repairs it.
    record = repository.get(db_session, dataset_id)
    record.total_frame_count = 999
    db_session.add(record)
    db_session.commit()

    recompute_metadata(storage, dataset_id, db_session, stats_sample_frames=8)

    record = repository.get(db_session, dataset_id)
    assert record.total_frame_count == 6


def test_recompute_regenerates_thumbnail(tmp_path, storage, db_session):
    src = tmp_path / "capture.npz"
    write_single_camera_npz(src, n_frames=6)
    [original] = ingest_dataset(src, storage, db_session, **INGEST_KWARGS)
    dataset_id = original.auto.dataset_id

    thumb_path = storage.data_root / "datasets" / dataset_id / "previews" / "camera_0" / "thumbnail.png"
    assert thumb_path.exists()
    thumb_path.unlink()

    recompute_metadata(storage, dataset_id, db_session, stats_sample_frames=8)

    assert thumb_path.exists()


def test_recompute_unknown_dataset_raises(tmp_path, storage, db_session):
    with pytest.raises(DatasetNotFoundError):
        recompute_metadata(storage, "does-not-exist", db_session, stats_sample_frames=8)


def test_recompute_all(tmp_path, storage, db_session):
    for i in range(3):
        src = tmp_path / f"capture_{i}.npz"
        write_single_camera_npz(src, n_frames=5, seed=i)
        ingest_dataset(src, storage, db_session, **INGEST_KWARGS)

    succeeded, errors = recompute_all(storage, db_session, stats_sample_frames=8)

    assert succeeded == 3
    assert errors == {}


def test_recompute_preserves_other_variables_summary(tmp_path, storage, db_session):
    import numpy as np

    src = tmp_path / "with_aux.npz"
    np.savez(
        src,
        raw_thr_frames=np.zeros((4, 12, 16), dtype=np.uint16),
        raw_thr_tstamps=1_700_000_000.0 + np.arange(4) / 30.0,
        imu=np.arange(8, dtype=np.float32),
    )
    [original] = ingest_dataset(src, storage, db_session, **INGEST_KWARGS)
    recomputed = recompute_metadata(storage, original.auto.dataset_id, db_session, stats_sample_frames=8)
    assert recomputed.auto.other_variables == "imu: shape=(8,), dtype=float32"
