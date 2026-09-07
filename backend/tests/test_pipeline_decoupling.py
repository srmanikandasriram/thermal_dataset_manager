from thermal_dataset_manager.ingest.pipeline import ingest_dataset, write_raw_dataset
from thermal_dataset_manager.ingest.recompute import recompute_metadata
from tests.conftest import TELEMETRY_ROWS, write_single_camera_npz


def test_write_raw_dataset_is_pending_with_cheap_fields_only(tmp_path, storage, db_session):
    src = tmp_path / "capture.npz"
    write_single_camera_npz(src, n_frames=10)

    bare = write_raw_dataset(
        src,
        storage,
        db_session,
        default_telemetry_row_count=TELEMETRY_ROWS,
        zarr_target_chunk_bytes=1024 * 1024,
    )

    assert bare.auto.metadata_status == "pending"
    cam = bare.auto.cameras[0]
    # Cheap fields (shape/fps/duration) are populated immediately.
    assert cam.frame_count == 10
    assert cam.fps_estimate is not None
    assert cam.duration_seconds is not None
    # Stats requiring a pixel-data scan are deferred.
    assert cam.counts_min is None
    assert cam.counts_max is None
    assert cam.counts_mean is None
    assert cam.counts_std is None


def test_raw_dataset_is_immediately_locked_and_catalogued(tmp_path, storage, db_session):
    from thermal_dataset_manager.catalog import repository

    src = tmp_path / "capture.npz"
    write_single_camera_npz(src, n_frames=5)

    bare = write_raw_dataset(
        src,
        storage,
        db_session,
        default_telemetry_row_count=TELEMETRY_ROWS,
        zarr_target_chunk_bytes=1024 * 1024,
    )

    # Immutability lock already applied -- not deferred to metadata computation.
    reloaded = storage.read_metadata(bare.auto.dataset_id)
    assert reloaded.auto.metadata_status == "pending"

    record = repository.get(db_session, bare.auto.dataset_id)
    assert record is not None
    assert record.total_frame_count == 5


def test_recompute_after_write_raw_dataset_flips_to_ready(tmp_path, storage, db_session):
    src = tmp_path / "capture.npz"
    write_single_camera_npz(src, n_frames=10)

    bare = write_raw_dataset(
        src,
        storage,
        db_session,
        default_telemetry_row_count=TELEMETRY_ROWS,
        zarr_target_chunk_bytes=1024 * 1024,
    )
    ready = recompute_metadata(storage, bare.auto.dataset_id, db_session, stats_sample_frames=8)

    assert ready.auto.metadata_status == "ready"
    cam = ready.auto.cameras[0]
    assert cam.counts_min is not None
    assert cam.counts_max is not None


def test_ingest_dataset_convenience_wrapper_is_fully_ready(tmp_path, storage, db_session):
    """The CLI/simple-call path should still produce a fully-computed
    result in one call, composed from the two decoupled phases."""
    src = tmp_path / "capture.npz"
    write_single_camera_npz(src, n_frames=10)

    metadata = ingest_dataset(
        src,
        storage,
        db_session,
        default_telemetry_row_count=TELEMETRY_ROWS,
        zarr_target_chunk_bytes=1024 * 1024,
        stats_sample_frames=8,
    )

    assert metadata.auto.metadata_status == "ready"
    assert metadata.auto.cameras[0].counts_min is not None
