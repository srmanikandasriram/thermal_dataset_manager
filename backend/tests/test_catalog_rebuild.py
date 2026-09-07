from tests.conftest import TELEMETRY_ROWS, write_single_camera_npz
from thermal_dataset_manager.catalog import repository
from thermal_dataset_manager.catalog.rebuild import rebuild_catalog
from thermal_dataset_manager.ingest.pipeline import ingest_dataset

INGEST_KWARGS = dict(
    default_telemetry_row_count=TELEMETRY_ROWS,
    zarr_target_chunk_bytes=1024 * 1024,
    stats_sample_frames=8,
)


def test_rebuild_restores_catalog_from_disk(tmp_path, storage, db_session):
    src = tmp_path / "capture.npz"
    write_single_camera_npz(src, n_frames=10)
    metadata = ingest_dataset(src, storage, db_session, **INGEST_KWARGS)

    # Simulate catalog loss.
    repository.delete(db_session, metadata.auto.dataset_id)
    assert repository.get(db_session, metadata.auto.dataset_id) is None

    count = rebuild_catalog(storage, db_session)

    assert count == 1
    record = repository.get(db_session, metadata.auto.dataset_id)
    assert record is not None
    assert record.total_frame_count == 10


def test_search_by_tag_and_text(tmp_path, storage, db_session):
    src = tmp_path / "capture.npz"
    write_single_camera_npz(src, n_frames=5)
    metadata = ingest_dataset(src, storage, db_session, **INGEST_KWARGS)

    updated = metadata.model_copy(deep=True)
    updated.user.tags = ["outdoor", "day"]
    updated.user.description = "a rooftop scene"
    storage.write_metadata(metadata.auto.dataset_id, updated, locked=True)
    repository.upsert_from_metadata(db_session, updated)

    results = repository.search(db_session, tags=["outdoor"])
    assert len(results) == 1
    assert results[0].dataset_id == metadata.auto.dataset_id

    results = repository.search(db_session, tags=["nonexistent"])
    assert results == []

    results = repository.search(db_session, q="rooftop")
    assert len(results) == 1
