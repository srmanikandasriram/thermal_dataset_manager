import numpy as np
import pytest
from fastapi.testclient import TestClient

from tests.conftest import TELEMETRY_ROWS
from thermal_dataset_manager.api import deps
from thermal_dataset_manager.app import create_app
from thermal_dataset_manager.catalog.db import get_session, init_db, make_engine
from thermal_dataset_manager.config import Settings
from thermal_dataset_manager.ingest.pipeline import ingest_dataset
from thermal_dataset_manager.storage import zarr_store
from thermal_dataset_manager.storage.filesystem import FilesystemDatasetStorage

INGEST_KWARGS = dict(
    default_telemetry_row_count=TELEMETRY_ROWS,
    zarr_target_chunk_bytes=1024 * 1024,
    stats_sample_frames=8,
)


def _write_capture_with_extra_key(path, n_frames=10):
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 60000, size=(n_frames, 12, 16), dtype=np.uint16)
    ts = 1_700_000_000.0 + np.arange(n_frames) / 30.0
    offsets = np.arange(n_frames, dtype=np.float64) * 0.001
    np.savez(path, raw_thr_frames=frames, raw_thr_tstamps=ts, thr_cam_timestamp_offset=offsets)
    return offsets


def test_extra_array_written_to_zarr_and_readable(tmp_path, storage, db_session):
    src = tmp_path / "capture.npz"
    offsets = _write_capture_with_extra_key(src)

    metadata = ingest_dataset(src, storage, db_session, **INGEST_KWARGS)

    assert [e.key for e in metadata.auto.extra_arrays] == ["thr_cam_timestamp_offset"]
    extra_info = metadata.auto.extra_arrays[0]
    assert extra_info.shape == [10]
    assert extra_info.dtype == "float64"

    store_path = storage.zarr_store_path(metadata.auto.dataset_id)
    roundtripped = zarr_store.read_extra_array(store_path, "thr_cam_timestamp_offset")
    np.testing.assert_allclose(roundtripped, offsets)


def _write_capture_with_scalar_extra_key(path, n_frames=10):
    """Regression fixture: a source file where the unmapped extra key is a
    0-d/scalar array (e.g. a single per-capture offset value), not an
    array with a leading dimension -- this previously crashed on read.
    """
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 60000, size=(n_frames, 12, 16), dtype=np.uint16)
    ts = 1_700_000_000.0 + np.arange(n_frames) / 30.0
    offset = np.array(3.14)
    np.savez(path, raw_thr_frames=frames, raw_thr_tstamps=ts, thr_cam_timestamp_offset=offset)
    return offset


def test_scalar_extra_array_written_to_zarr_and_readable(tmp_path, storage, db_session):
    src = tmp_path / "capture.npz"
    _write_capture_with_scalar_extra_key(src)

    metadata = ingest_dataset(src, storage, db_session, **INGEST_KWARGS)

    extra_info = metadata.auto.extra_arrays[0]
    assert extra_info.shape == []

    store_path = storage.zarr_store_path(metadata.auto.dataset_id)
    roundtripped = zarr_store.read_extra_array(store_path, "thr_cam_timestamp_offset")
    assert float(roundtripped) == pytest.approx(3.14)


def test_read_extra_array_missing_key_raises(tmp_path, storage, db_session):
    src = tmp_path / "capture.npz"
    _write_capture_with_extra_key(src)
    metadata = ingest_dataset(src, storage, db_session, **INGEST_KWARGS)
    store_path = storage.zarr_store_path(metadata.auto.dataset_id)

    with pytest.raises(KeyError):
        zarr_store.read_extra_array(store_path, "does_not_exist")


def test_dataset_with_no_unmapped_keys_has_empty_extra_arrays(tmp_path, storage, db_session):
    from tests.conftest import write_single_camera_npz

    src = tmp_path / "clean.npz"
    write_single_camera_npz(src, n_frames=5)
    metadata = ingest_dataset(src, storage, db_session, **INGEST_KWARGS)
    assert metadata.auto.extra_arrays == []


@pytest.fixture
def client(tmp_path):
    data_root = tmp_path / "data_root"
    data_root.mkdir()
    storage = FilesystemDatasetStorage(data_root)
    engine = make_engine(f"sqlite:///{tmp_path / 'catalog.sqlite'}")
    init_db(engine)
    settings = Settings(data_root=data_root, catalog_db=f"sqlite:///{tmp_path / 'catalog.sqlite'}")

    app = create_app()
    app.dependency_overrides[deps.get_storage] = lambda: storage
    app.dependency_overrides[deps.get_db_session] = lambda: get_session(engine)
    app.dependency_overrides[deps.get_app_settings] = lambda: settings

    with TestClient(app) as c:
        c.storage = storage
        c.engine = engine
        yield c


def test_api_get_extra_array(client, tmp_path):
    src = tmp_path / "capture.npz"
    offsets = _write_capture_with_extra_key(src)
    with get_session(client.engine) as session:
        metadata = ingest_dataset(src, client.storage, session, **INGEST_KWARGS)

    resp = client.get(f"/api/datasets/{metadata.auto.dataset_id}/extra/thr_cam_timestamp_offset")
    assert resp.status_code == 200
    body = resp.json()
    assert body["key"] == "thr_cam_timestamp_offset"
    assert body["shape"] == [10]
    np.testing.assert_allclose(body["values"], offsets)


def test_api_get_extra_array_unknown_key_404(client, tmp_path):
    src = tmp_path / "capture.npz"
    _write_capture_with_extra_key(src)
    with get_session(client.engine) as session:
        metadata = ingest_dataset(src, client.storage, session, **INGEST_KWARGS)

    resp = client.get(f"/api/datasets/{metadata.auto.dataset_id}/extra/nope")
    assert resp.status_code == 404


def test_api_get_scalar_extra_array(client, tmp_path):
    """Regression test: a scalar (0-d) extra array previously crashed the
    endpoint with a 500 ("too many indices for array") on read.
    """
    src = tmp_path / "capture.npz"
    _write_capture_with_scalar_extra_key(src)
    with get_session(client.engine) as session:
        metadata = ingest_dataset(src, client.storage, session, **INGEST_KWARGS)

    resp = client.get(f"/api/datasets/{metadata.auto.dataset_id}/extra/thr_cam_timestamp_offset")
    assert resp.status_code == 200
    body = resp.json()
    assert body["shape"] == []
    assert body["values"] == pytest.approx(3.14)
