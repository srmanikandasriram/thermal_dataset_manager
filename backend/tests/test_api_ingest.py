import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tests.conftest import write_single_camera_npz
from thermal_dataset_manager.api import deps
from thermal_dataset_manager.app import create_app
from thermal_dataset_manager.catalog.db import get_session, init_db, make_engine
from thermal_dataset_manager.config import Settings
from thermal_dataset_manager.storage.filesystem import FilesystemDatasetStorage


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
    # Background ingest jobs open their own session (they outlive the request),
    # so they resolve the engine via this dependency rather than get_db_session.
    app.dependency_overrides[deps.get_engine] = lambda: engine

    with TestClient(app) as c:
        c.storage = storage
        c.engine = engine
        c.data_root = data_root
        yield c


def _poll_job(client, job_id, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = client.get(f"/api/ingest/jobs/{job_id}")
        body = resp.json()
        if body["status"] in ("succeeded", "failed"):
            return body
        time.sleep(0.05)
    raise TimeoutError(f"job {job_id} did not finish in time")


def test_upload_recognized_keys_returns_auto_mapping(client, tmp_path):
    src = tmp_path / "capture.npz"
    write_single_camera_npz(src, n_frames=5)

    with open(src, "rb") as f:
        resp = client.post("/api/ingest/upload", files={"file": ("capture.npz", f, "application/octet-stream")})

    assert resp.status_code == 200
    body = resp.json()
    assert body["auto_mapping"] is not None
    assert body["auto_mapping_error"] is None
    assert body["auto_mapping"][0]["frames_key"] == "raw_thr_frames"
    assert body["auto_mapping"][0]["timestamps_key"] == "raw_thr_tstamps"
    assert {k["key"] for k in body["keys"]} == {"raw_thr_frames", "raw_thr_tstamps"}


def test_upload_unrecognized_keys_returns_no_auto_mapping(client, tmp_path):
    src = tmp_path / "weird.npz"
    frames = np.zeros((5, 12, 16), dtype=np.uint16)
    timestamps = np.arange(5, dtype=np.float64)
    np.savez(src, thermal_data=frames, frame_times=timestamps)

    with open(src, "rb") as f:
        resp = client.post("/api/ingest/upload", files={"file": ("weird.npz", f, "application/octet-stream")})

    assert resp.status_code == 200
    body = resp.json()
    assert body["auto_mapping"] is None
    assert "No recognized" in body["auto_mapping_error"]
    assert {k["key"] for k in body["keys"]} == {"thermal_data", "frame_times"}
    for key_info in body["keys"]:
        if key_info["key"] == "thermal_data":
            assert key_info["shape"] == [5, 12, 16]
            assert key_info["dtype"] == "uint16"


def test_confirm_with_auto_mapping_ingests_successfully(client, tmp_path):
    src = tmp_path / "capture.npz"
    write_single_camera_npz(src, n_frames=6)

    with open(src, "rb") as f:
        upload_resp = client.post("/api/ingest/upload", files={"file": ("capture.npz", f, "application/octet-stream")})
    inspection = upload_resp.json()

    confirm_resp = client.post(
        "/api/ingest/confirm",
        json={
            "upload_token": inspection["upload_token"],
            "filename": inspection["filename"],
            "mapping": inspection["auto_mapping"],
        },
    )
    assert confirm_resp.status_code == 200
    job = _poll_job(client, confirm_resp.json()["job_id"])
    assert job["status"] == "succeeded"
    assert job["dataset_id"]

    resp = client.get(f"/api/datasets/{job['dataset_id']}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["auto"]["cameras"][0]["frame_count"] == 6
    # TestClient runs BackgroundTasks synchronously to completion before
    # the confirm call returns, so by the time we poll the job as
    # "succeeded" the untracked metadata-computation continuation has also
    # already run -- in a real deployment this happens moments after the
    # job is reported done, not necessarily before the client observes it.
    assert body["auto"]["metadata_status"] == "ready"
    assert body["auto"]["cameras"][0]["counts_min"] is not None


def test_confirm_respects_per_camera_telemetry_row_count(client, tmp_path):
    src = tmp_path / "capture.npz"
    write_single_camera_npz(src, n_frames=6)

    with open(src, "rb") as f:
        upload_resp = client.post("/api/ingest/upload", files={"file": ("capture.npz", f, "application/octet-stream")})
    inspection = upload_resp.json()
    # Auto-mapping carries the server default so the UI can show it.
    assert inspection["auto_mapping"][0]["telemetry_row_count"] == 2

    mapping = [{**inspection["auto_mapping"][0], "telemetry_row_count": 0}]
    confirm_resp = client.post(
        "/api/ingest/confirm",
        json={
            "upload_token": inspection["upload_token"],
            "filename": inspection["filename"],
            "mapping": mapping,
        },
    )
    job = _poll_job(client, confirm_resp.json()["job_id"])
    assert job["status"] == "succeeded"

    body = client.get(f"/api/datasets/{job['dataset_id']}").json()
    cam = body["auto"]["cameras"][0]
    assert cam["telemetry_row_count"] == 0
    assert cam["frame_height"] == 12  # conftest frames are 12 rows total


def test_confirm_with_manual_mapping_for_unrecognized_keys(client, tmp_path):
    src = tmp_path / "weird.npz"
    frames = np.random.default_rng(0).integers(0, 60000, size=(7, 12, 16), dtype=np.uint16)
    timestamps = 1_700_000_000.0 + np.arange(7) / 30.0
    np.savez(src, thermal_data=frames, frame_times=timestamps)

    with open(src, "rb") as f:
        upload_resp = client.post("/api/ingest/upload", files={"file": ("weird.npz", f, "application/octet-stream")})
    inspection = upload_resp.json()
    assert inspection["auto_mapping"] is None

    confirm_resp = client.post(
        "/api/ingest/confirm",
        json={
            "upload_token": inspection["upload_token"],
            "filename": inspection["filename"],
            "mapping": [{"camera_label": None, "frames_key": "thermal_data", "timestamps_key": "frame_times"}],
        },
    )
    assert confirm_resp.status_code == 200
    job = _poll_job(client, confirm_resp.json()["job_id"])
    assert job["status"] == "succeeded"

    resp = client.get(f"/api/datasets/{job['dataset_id']}")
    assert resp.json()["auto"]["cameras"][0]["frame_count"] == 7


def test_confirm_without_mapping_rejected(client, tmp_path):
    src = tmp_path / "capture.npz"
    write_single_camera_npz(src, n_frames=3)
    with open(src, "rb") as f:
        upload_resp = client.post("/api/ingest/upload", files={"file": ("capture.npz", f, "application/octet-stream")})
    inspection = upload_resp.json()

    resp = client.post(
        "/api/ingest/confirm",
        json={"upload_token": inspection["upload_token"], "filename": inspection["filename"], "mapping": []},
    )
    assert resp.status_code == 422


def test_cancel_upload_removes_staging_dir(client, tmp_path):
    src = tmp_path / "capture.npz"
    write_single_camera_npz(src, n_frames=4)
    with open(src, "rb") as f:
        upload_resp = client.post("/api/ingest/upload", files={"file": ("capture.npz", f, "application/octet-stream")})
    inspection = upload_resp.json()

    staging_dir = client.data_root / "staging" / inspection["upload_token"]
    assert staging_dir.exists()

    resp = client.delete(f"/api/ingest/upload/{inspection['upload_token']}")
    assert resp.status_code == 200
    assert not staging_dir.exists()

    # Confirming after cancel should now 404 -- the staged file is gone.
    confirm_resp = client.post(
        "/api/ingest/confirm",
        json={
            "upload_token": inspection["upload_token"],
            "filename": inspection["filename"],
            "mapping": inspection["auto_mapping"],
        },
    )
    assert confirm_resp.status_code == 404


def test_cancel_unknown_upload_token_is_a_noop(client):
    resp = client.delete("/api/ingest/upload/does-not-exist")
    assert resp.status_code == 200


def test_confirm_unknown_upload_token_404(client):
    resp = client.post(
        "/api/ingest/confirm",
        json={
            "upload_token": "does-not-exist",
            "filename": "capture.npz",
            "mapping": [{"frames_key": "a", "timestamps_key": "b"}],
        },
    )
    assert resp.status_code == 404
