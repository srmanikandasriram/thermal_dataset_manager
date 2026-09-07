import numpy as np

from thermal_dataset_manager.api.ingest import _run_ingest_job
from thermal_dataset_manager.api.jobs import JobTracker
from thermal_dataset_manager.config import Settings


def test_unrecognized_keys_mark_job_failed_not_stuck_running(tmp_path, storage, db_session):
    """Regression test: an error type the pipeline raises that isn't one of
    the two originally-anticipated exception types must still mark the job
    failed, not leave it "running" forever (see KeyRegistryError bug).
    """
    src = tmp_path / "unrecognized.npz"
    frames = np.zeros((5, 12, 16), dtype=np.uint16)
    timestamps = np.arange(5, dtype=np.float64)
    np.savez(src, some_other_key=frames, another_key=timestamps)

    tracker = JobTracker()
    job = tracker.create()
    settings = Settings(data_root=storage.data_root)

    _run_ingest_job(job.job_id, src, storage, db_session, settings, tracker, cleanup_dir=None)

    result = tracker.get(job.job_id)
    assert result.status == "failed"
    assert result.errors
    assert "No recognized frame/timestamp keys" in result.errors[0]


def test_letter_suffixed_multi_camera_keys_ingest_successfully(tmp_path, storage, db_session):
    """Matches a real-world capture layout: raw_thr_frames_A/_B rather than _0/_1."""
    from thermal_dataset_manager.ingest.pipeline import ingest_dataset

    src = tmp_path / "two_cam.npz"
    frames_a = np.random.default_rng(0).integers(0, 60000, size=(10, 12, 16), dtype=np.uint16)
    frames_b = np.random.default_rng(1).integers(0, 60000, size=(10, 12, 16), dtype=np.uint16)
    ts = 1_700_000_000.0 + np.arange(10) / 30.0
    np.savez(
        src,
        raw_thr_frames_A=frames_a,
        raw_thr_tstamps_A=ts,
        raw_thr_frames_B=frames_b,
        raw_thr_tstamps_B=ts,
        thr_cam_timestamp_offset_A=np.zeros(10),
        thr_cam_timestamp_offset_B=np.zeros(10),
    )

    metadata = ingest_dataset(
        src,
        storage,
        db_session,
        default_telemetry_row_count=2,
        zarr_target_chunk_bytes=1024 * 1024,
        stats_sample_frames=8,
    )

    assert metadata.auto.camera_count == 2
    # _A/_B are normalized to the canonical _0/_1 camera labels.
    assert {c.camera_label for c in metadata.auto.cameras} == {"0", "1"}

    # thr_cam_timestamp_offset_A/_B aren't mapped to any camera's
    # frames/timestamps -- they should be preserved as extra arrays, not
    # silently dropped.
    assert {e.key for e in metadata.auto.extra_arrays} == {
        "thr_cam_timestamp_offset_A",
        "thr_cam_timestamp_offset_B",
    }
