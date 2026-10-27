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


def test_directories_are_expanded_and_one_bad_file_does_not_stop_the_run(monkeypatch, tmp_path, capsys):
    folder = tmp_path / "captures" / "nested"
    folder.mkdir(parents=True)
    for name in ("a.npz", "b.npz"):
        write_single_camera_npz(folder / name, n_frames=4)
    (folder / "notes.txt").write_text("ignore me")
    (folder / "movie.yuv").write_bytes(b"\0")

    ingested: list[str] = []
    data_root = tmp_path / "root"
    data_root.mkdir()
    settings = Settings(data_root=data_root, catalog_db=f"sqlite:///{tmp_path / 'catalog.sqlite'}")
    monkeypatch.setattr(cli, "get_settings", lambda: settings)

    def fake_ingest(path, storage, session, **kwargs):
        ingested.append(path.name)
        if path.name == "a.npz":
            raise RuntimeError("boom")
        auto = type("A", (), {"dataset_id": "X"})()
        user = type("U", (), {"name": "n", "capture_id": None})()
        return [type("M", (), {"auto": auto, "user": user})()]

    monkeypatch.setattr(cli, "ingest_dataset", fake_ingest)
    monkeypatch.setattr("sys.argv", ["ingest_cli.py", str(tmp_path / "captures")])

    assert cli.main() == 1  # a.npz failed
    assert ingested == ["a.npz", "b.npz"]  # but b.npz was still attempted; txt/yuv ignored
    assert "FAILED" in capsys.readouterr().err


def _setup_move_run(monkeypatch, tmp_path, outcomes):
    """Run main() with a fake ingest whose per-filename outcome is given by `outcomes`
    ('ok' | 'dup' | 'boom'). Returns a function to call main with extra argv."""
    from thermal_dataset_manager.ingest.pipeline import DuplicateDatasetError

    data_root = tmp_path / "root"
    data_root.mkdir(exist_ok=True)
    settings = Settings(data_root=data_root, catalog_db=f"sqlite:///{tmp_path / 'catalog.sqlite'}")
    monkeypatch.setattr(cli, "get_settings", lambda: settings)

    def fake_ingest(path, storage, session, **kwargs):
        outcome = outcomes[path.name]
        if outcome == "boom":
            raise RuntimeError("boom")
        if outcome == "dup":
            existing = type(
                "R", (), {"dataset_id": "D", "name": "n", "source_filename": "f", "ingested_at": __import__("datetime").datetime.now()}
            )()
            raise DuplicateDatasetError("sha", existing)
        auto = type("A", (), {"dataset_id": "X"})()
        user = type("U", (), {"name": "n", "capture_id": None})()
        return [type("M", (), {"auto": auto, "user": user})()]

    monkeypatch.setattr(cli, "ingest_dataset", fake_ingest)

    def run(*argv):
        monkeypatch.setattr("sys.argv", ["ingest_cli.py", *argv])
        return cli.main()

    return run


def _tree(tmp_path):
    todo = tmp_path / "to_be_ingested"
    (todo / "sceneA").mkdir(parents=True)
    for rel in ("sceneA/ok.npz", "sceneA/dup.npz", "bad.npz"):
        (todo / rel).write_bytes(b"x")
    return todo


def test_move_ingested_moves_only_successes_keeping_structure(monkeypatch, tmp_path):
    todo, done = _tree(tmp_path), tmp_path / "ingested"
    run = _setup_move_run(monkeypatch, tmp_path, {"ok.npz": "ok", "dup.npz": "dup", "bad.npz": "boom"})

    assert run("--move-ingested-to", str(done), str(todo)) == 1  # bad.npz failed

    assert (done / "sceneA" / "ok.npz").exists() and not (todo / "sceneA" / "ok.npz").exists()
    assert (todo / "sceneA" / "dup.npz").exists()  # duplicates stay unless asked
    assert (todo / "bad.npz").exists()  # failures stay


def test_move_duplicates_flag_moves_duplicates_too(monkeypatch, tmp_path):
    todo, done = _tree(tmp_path), tmp_path / "ingested"
    run = _setup_move_run(monkeypatch, tmp_path, {"ok.npz": "ok", "dup.npz": "dup", "bad.npz": "ok"})

    assert run("--move-ingested-to", str(done), "--move-duplicates", str(todo)) == 0
    assert sorted(p.name for p in done.rglob("*.npz")) == ["bad.npz", "dup.npz", "ok.npz"]
    assert list(todo.rglob("*.npz")) == []


def test_move_never_overwrites_existing_file(monkeypatch, tmp_path):
    todo, done = _tree(tmp_path), tmp_path / "ingested"
    (done / "sceneA").mkdir(parents=True)
    (done / "sceneA" / "ok.npz").write_bytes(b"different")
    run = _setup_move_run(monkeypatch, tmp_path, {"ok.npz": "ok", "dup.npz": "ok", "bad.npz": "ok"})

    assert run("--move-ingested-to", str(done), str(todo)) == 1
    assert (done / "sceneA" / "ok.npz").read_bytes() == b"different"
    assert (todo / "sceneA" / "ok.npz").read_bytes() == b"x"  # source untouched


def test_destination_inside_scanned_folder_is_not_rescanned(monkeypatch, tmp_path):
    root = tmp_path / "raw"
    (root / "ingested").mkdir(parents=True)
    (root / "ingested" / "old.npz").write_bytes(b"x")
    (root / "new.npz").write_bytes(b"x")
    seen = []
    run = _setup_move_run(monkeypatch, tmp_path, {"new.npz": "ok", "old.npz": "boom"})
    monkeypatch.setattr(cli, "ingest_dataset", lambda path, *a, **k: seen.append(path.name) or _ok())

    def _ok():
        auto = type("A", (), {"dataset_id": "X"})()
        user = type("U", (), {"name": "n", "capture_id": None})()
        return [type("M", (), {"auto": auto, "user": user})()]

    assert run("--move-ingested-to", str(root / "ingested"), str(root)) == 0
    assert seen == ["new.npz"]


def test_move_duplicates_requires_move_ingested_to(monkeypatch, tmp_path):
    run = _setup_move_run(monkeypatch, tmp_path, {})
    import pytest

    with pytest.raises(SystemExit):
        run("--move-duplicates", str(tmp_path))
