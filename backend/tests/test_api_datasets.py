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
        c.data_root = data_root
        yield c


def _ingest_sample(client, tmp_path, n_frames=10):
    src = tmp_path / "capture.npz"
    write_single_camera_npz(src, n_frames=n_frames)
    with get_session(client.engine) as session:
        [meta] = ingest_dataset(src, client.storage, session, **INGEST_KWARGS)
        return meta


def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200


def test_list_and_get_dataset(client, tmp_path):
    metadata = _ingest_sample(client, tmp_path)

    resp = client.get("/api/datasets")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["items"]) == 1
    summary = body["items"][0]
    assert summary["dataset_id"] == metadata.auto.dataset_id
    # summary carries the optional-column fields the list view can show
    assert summary["source_filename"] == metadata.auto.source_filename
    assert "scene_category" in summary and "capture_location" in summary

    resp = client.get(f"/api/datasets/{metadata.auto.dataset_id}")
    assert resp.status_code == 200
    assert resp.json()["auto"]["dataset_id"] == metadata.auto.dataset_id


def test_get_dataset_404(client):
    resp = client.get("/api/datasets/does-not-exist")
    assert resp.status_code == 404


def test_list_sorts_by_requested_field(client, tmp_path):
    # three captures with distinct frame counts and names
    specs = [("bravo", 7), ("alpha", 3), ("charlie", 12)]
    for name, n in specs:
        src = tmp_path / f"{name}.npz"
        write_single_camera_npz(src, n_frames=n, seed=n)
        with get_session(client.engine) as session:
            [meta] = ingest_dataset(src, client.storage, session, **INGEST_KWARGS)
            client.patch(f"/api/datasets/{meta.auto.dataset_id}/metadata", json={"name": name})

    by_name = client.get("/api/datasets?sort=name&order=asc").json()
    assert [i["name"] for i in by_name["items"]] == ["alpha", "bravo", "charlie"]

    by_frames_desc = client.get("/api/datasets?sort=total_frame_count&order=desc").json()
    assert [i["total_frame_count"] for i in by_frames_desc["items"]] == [12, 7, 3]

    # unknown sort field is rejected by the Literal type
    assert client.get("/api/datasets?sort=bogus").status_code == 422


def test_list_sort_is_stable_across_pages(client, tmp_path):
    for i in range(5):
        src = tmp_path / f"cap_{i}.npz"
        write_single_camera_npz(src, n_frames=5, seed=i)  # identical frame counts -> ties
        with get_session(client.engine) as session:
            ingest_dataset(src, client.storage, session, **INGEST_KWARGS)

    q = "/api/datasets?sort=total_frame_count&order=asc&limit=2"
    ids = []
    for offset in (0, 2, 4):
        ids += [i["dataset_id"] for i in client.get(f"{q}&offset={offset}").json()["items"]]
    assert len(ids) == len(set(ids)) == 5  # no row repeated or skipped despite the ties


def test_list_paginates_with_total(client, tmp_path):
    for i in range(5):
        src = tmp_path / f"capture_{i}.npz"
        write_single_camera_npz(src, n_frames=5, seed=i)
        with get_session(client.engine) as session:
            ingest_dataset(src, client.storage, session, **INGEST_KWARGS)

    first = client.get("/api/datasets?limit=2&offset=0").json()
    assert first["total"] == 5
    assert [i["dataset_id"] for i in first["items"]] != []
    assert len(first["items"]) == 2

    second = client.get("/api/datasets?limit=2&offset=2").json()
    assert second["total"] == 5
    assert len(second["items"]) == 2

    last = client.get("/api/datasets?limit=2&offset=4").json()
    assert last["total"] == 5
    assert len(last["items"]) == 1

    seen = {i["dataset_id"] for page in (first, second, last) for i in page["items"]}
    assert len(seen) == 5

    # total reflects filters, not the whole table.
    filtered = client.get("/api/datasets?q=nonexistentterm").json()
    assert filtered["total"] == 0
    assert filtered["items"] == []


def test_patch_metadata(client, tmp_path):
    metadata = _ingest_sample(client, tmp_path)
    resp = client.patch(
        f"/api/datasets/{metadata.auto.dataset_id}/metadata",
        json={"name": "New Name", "tags": ["outdoor"]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["user"]["name"] == "New Name"
    assert body["user"]["tags"] == ["outdoor"]

    resp = client.get("/api/datasets?tags=outdoor")
    assert len(resp.json()["items"]) == 1


def test_get_thumbnail_and_frame(client, tmp_path):
    metadata = _ingest_sample(client, tmp_path)
    dataset_id = metadata.auto.dataset_id

    resp = client.get(f"/api/datasets/{dataset_id}/cameras/0/thumbnail.png")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"

    resp = client.get(f"/api/datasets/{dataset_id}/cameras/0/frames/0.png")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"

    resp = client.get(f"/api/datasets/{dataset_id}/cameras/0/frames/9999.png")
    assert resp.status_code == 404


def test_thumbnail_regenerates_when_cache_missing(client, tmp_path):
    """Previews are a derived cache -- deleting one (e.g. to pick up a
    colormap change) should trigger lazy regeneration, not a 404.
    """
    metadata = _ingest_sample(client, tmp_path)
    dataset_id = metadata.auto.dataset_id

    thumb_path = client.storage.data_root / "datasets" / dataset_id / "previews" / "camera_0" / "thumbnail.png"
    assert thumb_path.exists()
    thumb_path.unlink()

    resp = client.get(f"/api/datasets/{dataset_id}/cameras/0/thumbnail.png")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert thumb_path.exists()


def test_recompute_dataset_metadata_endpoint(client, tmp_path):
    from thermal_dataset_manager.catalog import repository

    metadata = _ingest_sample(client, tmp_path, n_frames=6)
    dataset_id = metadata.auto.dataset_id

    # Corrupt the catalog to prove recompute repairs it from the zarr store.
    with get_session(client.engine) as session:
        record = repository.get(session, dataset_id)
        record.total_frame_count = 999
        session.add(record)
        session.commit()

    resp = client.post(f"/api/datasets/{dataset_id}/recompute-metadata")
    assert resp.status_code == 200
    body = resp.json()
    assert body["auto"]["metadata_status"] == "ready"
    assert body["auto"]["cameras"][0]["frame_count"] == 6

    with get_session(client.engine) as session:
        record = repository.get(session, dataset_id)
        assert record.total_frame_count == 6


def test_recompute_dataset_metadata_404(client):
    resp = client.post("/api/datasets/does-not-exist/recompute-metadata")
    assert resp.status_code == 404


def test_default_sort_is_random_but_stable_per_seed(client, tmp_path):
    for i in range(8):
        src = tmp_path / f"cap_{i}.npz"
        write_single_camera_npz(src, n_frames=5, seed=i)
        with get_session(client.engine) as session:
            ingest_dataset(src, client.storage, session, **INGEST_KWARGS)

    def ids(query):
        return [i["dataset_id"] for i in client.get(f"/api/datasets?{query}").json()["items"]]

    whole = ids("seed=1&limit=100")
    assert ids("seed=1&limit=100") == whole  # same seed -> same order
    assert ids("seed=2&limit=100") != whole  # other seed -> different order
    assert whole != ids("sort=ingested_at&order=desc&limit=100")  # not just the recency order

    # paging through one seed neither repeats nor skips rows, and matches the full order
    paged = []
    for offset in (0, 3, 6):
        paged += ids(f"seed=1&limit=3&offset={offset}")
    assert paged == whole
