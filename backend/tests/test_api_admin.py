import pytest
from fastapi.testclient import TestClient

from tests.conftest import write_single_camera_npz
from thermal_dataset_manager.api import deps
from thermal_dataset_manager.app import create_app
from thermal_dataset_manager.catalog.db import get_session, init_db, make_engine
from thermal_dataset_manager.config import Settings
from thermal_dataset_manager.ingest.pipeline import ingest_dataset
from thermal_dataset_manager.storage.filesystem import FilesystemDatasetStorage

INGEST_KWARGS = dict(default_telemetry_row_count=2, zarr_target_chunk_bytes=1024 * 1024, stats_sample_frames=8)


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


def test_rebuild_catalog_endpoint(client, tmp_path):
    src = tmp_path / "capture.npz"
    write_single_camera_npz(src, n_frames=5)
    with get_session(client.engine) as session:
        ingest_dataset(src, client.storage, session, **INGEST_KWARGS)

    resp = client.post("/api/admin/rebuild-catalog")
    assert resp.status_code == 200
    assert resp.json()["datasets_indexed"] == 1


def test_recompute_all_metadata_endpoint(client, tmp_path):
    for i in range(3):
        src = tmp_path / f"capture_{i}.npz"
        write_single_camera_npz(src, n_frames=5, seed=i)
        with get_session(client.engine) as session:
            ingest_dataset(src, client.storage, session, **INGEST_KWARGS)

    resp = client.post("/api/admin/recompute-all-metadata")
    assert resp.status_code == 200
    body = resp.json()
    assert body["succeeded"] == 3
    assert body["errors"] == {}
