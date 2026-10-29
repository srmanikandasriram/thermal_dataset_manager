import numpy as np
import pytest

from tests.conftest import TELEMETRY_ROWS, write_multi_camera_npz, write_single_camera_npz
from thermal_dataset_manager.ingest.pipeline import ingest_dataset
from thermal_dataset_manager.loader import Library

KW = dict(default_telemetry_row_count=TELEMETRY_ROWS, zarr_target_chunk_bytes=1024 * 1024, stats_sample_frames=8)


@pytest.fixture
def lib(tmp_path, storage, db_session):
    """A Library over two single-camera files and one two-camera file."""
    a, b, c = tmp_path / "a.npz", tmp_path / "b.npz", tmp_path / "c.npz"
    write_single_camera_npz(a, n_frames=10, seed=1)
    write_single_camera_npz(b, n_frames=20, seed=2)
    write_multi_camera_npz(c, n_frames=6)
    for p in (a, b, c):
        ingest_dataset(p, storage, db_session, **KW)
    db_session.commit()
    library = Library.__new__(Library)
    library.storage = storage
    library._engine = db_session.get_bind()
    return library


def test_find_all_and_filters(lib):
    assert len(lib.find()) == 4
    assert {v.n_frames for v in lib.find(min_frames=10)} == {10, 20}
    assert [v.name for v in lib.find(max_frames=6)] == ["c_0", "c_1"]
    assert len(lib.find(has_telemetry=False)) == 4  # synthetic frames carry no valid telemetry line
    assert lib.find(pipeline_enabled=["gain"]) == []
    assert len(lib.find(limit=2)) == 2
    assert [v.name for v in lib.find(where=lambda md: md.user.name == "b")] == ["b"]


def test_capture_groups_cameras(lib):
    [first, _] = lib.find(max_frames=6)
    assert first.capture_id
    assert sorted(v.name for v in lib.capture(first.capture_id)) == ["c_0", "c_1"]


def test_frames_selection_modes(lib):
    [v] = lib.find(where=lambda md: md.user.name == "b")
    full = v.frames()
    assert full.shape == (20, v.shape[0], v.shape[1]) and full.dtype == np.uint16
    assert np.array_equal(v.frames(slice(5, 8)), full[5:8])
    assert np.array_equal(v.frames(3), full[3])
    assert np.array_equal(v.frames([7, 2, 2, 15]), full[[7, 2, 2, 15]])
    assert np.array_equal(np.concatenate(list(v.iter_batches(6))), full)
    assert np.array_equal(np.concatenate(list(v.iter_batches(4, slice(0, 20, 2)))), full[::2])


def test_time_selection(lib):
    [v] = lib.find(where=lambda md: md.user.name == "b")
    sl = v.time_slice(0.1, 0.2)  # synthetic timestamps are 30 fps
    assert np.array_equal(v.frames(time=(0.1, 0.2)), v.frames(sl))
    assert 2 <= sl.stop - sl.start <= 4
    with pytest.raises(ValueError):
        v.frames(slice(0, 2), time=(0, 1))


@pytest.mark.parametrize("trc", [2, -2])
def test_raw_reattaches_telemetry_rows(tmp_path, storage, db_session, trc):
    src = tmp_path / "x.npz"
    write_single_camera_npz(src, n_frames=5)
    original = np.load(src)["raw_thr_frames"]
    ingest_dataset(src, storage, db_session, **{**KW, "default_telemetry_row_count": trc})
    db_session.commit()
    library = Library.__new__(Library)
    library.storage, library._engine = storage, db_session.get_bind()
    [v] = library.find()
    assert np.array_equal(v.frames(raw=True), original)
    assert v.telemetry_rows().shape == (5, 2, original.shape[2])


def test_library_refuses_a_root_without_a_catalog(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        Library(tmp_path / "typo")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="no catalog"):
        Library(empty)
    assert not (empty / "catalog.sqlite").exists()  # and it must not create one
