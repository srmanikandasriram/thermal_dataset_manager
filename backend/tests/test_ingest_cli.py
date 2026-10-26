"""The ingest CLI is a thin wrapper; these tests lock in its argument
handling (particularly --telemetry-rows) rather than re-testing the
pipeline, which has its own coverage."""

import importlib.util
from pathlib import Path

import numpy as np

from tests.conftest import write_single_camera_npz
from thermal_dataset_manager.config import Settings

_spec = importlib.util.spec_from_file_location(
    "ingest_cli", Path(__file__).resolve().parent.parent / "scripts" / "ingest_cli.py"
)
cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cli)


def _run(monkeypatch, tmp_path, argv, capture_kwargs):
    data_root = tmp_path / "root"
    data_root.mkdir(exist_ok=True)
    settings = Settings(data_root=data_root, catalog_db=f"sqlite:///{tmp_path / 'catalog.sqlite'}")
    monkeypatch.setattr(cli, "get_settings", lambda: settings)

    def fake_ingest(path, storage, session, **kwargs):
        capture_kwargs.append(kwargs)
        auto = type("A", (), {"dataset_id": "X"})()
        user = type("U", (), {"name": "n", "capture_id": None})()
        return [type("M", (), {"auto": auto, "user": user})()]

    monkeypatch.setattr(cli, "ingest_dataset", fake_ingest)
    monkeypatch.setattr("sys.argv", ["ingest_cli.py", *argv])
    return cli.main()


def test_defaults_to_configured_telemetry_row_count(monkeypatch, tmp_path):
    src = tmp_path / "capture.npz"
    write_single_camera_npz(src, n_frames=4)
    seen: list[dict] = []

    assert _run(monkeypatch, tmp_path, [str(src)], seen) == 0
    assert seen[0]["default_telemetry_row_count"] == 2


def test_telemetry_rows_flag_overrides(monkeypatch, tmp_path):
    src = tmp_path / "capture.npz"
    write_single_camera_npz(src, n_frames=4)
    seen: list[dict] = []

    for value in (0, -2):
        seen.clear()
        assert _run(monkeypatch, tmp_path, ["--telemetry-rows", str(value), str(src)], seen) == 0
        assert seen[0]["default_telemetry_row_count"] == value


def test_telemetry_rows_flag_applies_to_every_file(monkeypatch, tmp_path):
    paths = []
    for i in range(3):
        p = tmp_path / f"c{i}.npz"
        frames = np.random.default_rng(i).integers(0, 60000, size=(4, 12, 16), dtype=np.uint16)
        ts = 1_700_000_000.0 + np.arange(4) / 30.0
        np.savez(p, raw_thr_frames=frames, raw_thr_tstamps=ts)
        paths.append(str(p))
    seen: list[dict] = []

    assert _run(monkeypatch, tmp_path, ["--telemetry-rows", "0", *paths], seen) == 0
    assert [k["default_telemetry_row_count"] for k in seen] == [0, 0, 0]
