import numpy as np
import pytest

from tests.conftest import (
    TELEMETRY_ROWS,
    write_multi_camera_npz,
    write_single_camera_h5,
    write_single_camera_npz,
    write_single_camera_npz_zst,
)
from thermal_dataset_manager.catalog import repository
from thermal_dataset_manager.ingest.pipeline import (
    DuplicateDatasetError,
    UnsupportedSourceFormatError,
    ingest_dataset,
)
from thermal_dataset_manager.ingest.validation import IngestValidationError
from thermal_dataset_manager.storage import zarr_store

INGEST_KWARGS = dict(
    default_telemetry_row_count=TELEMETRY_ROWS,
    zarr_target_chunk_bytes=1024 * 1024,
    stats_sample_frames=8,
)


def test_ingest_squeezes_trailing_singleton_channel_dim(tmp_path, storage, db_session):
    """Some sources emit frames as (n_frames, H, W, 1) rather than
    (n_frames, H, W) -- the trailing singleton dim should be squeezed out
    rather than tripping the "frames must be 3-D" validation check.
    """
    src = tmp_path / "channel_dim.npz"
    frames = np.random.default_rng(0).integers(0, 60000, size=(10, 12, 16, 1), dtype=np.uint16)
    ts = 1_700_000_000.0 + np.arange(10) / 30.0
    np.savez(src, raw_thr_frames=frames, raw_thr_tstamps=ts)

    [metadata] = ingest_dataset(src, storage, db_session, **INGEST_KWARGS)

    cam = metadata.auto.cameras[0]
    assert cam.frame_height == 12 - TELEMETRY_ROWS
    assert cam.frame_width == 16

    store_path = storage.zarr_store_path(metadata.auto.dataset_id)
    assert zarr_store.read_frame_image(store_path, 0, 0).shape == (12 - TELEMETRY_ROWS, 16)
    assert zarr_store.read_frame_telemetry(store_path, 0, 0).shape == (TELEMETRY_ROWS, 16)


def test_ingest_rejects_byte_identical_duplicate(tmp_path, storage, db_session):
    src = tmp_path / "capture.npz"
    write_single_camera_npz(src, n_frames=8)
    [first] = ingest_dataset(src, storage, db_session, **INGEST_KWARGS)

    # Same file again (and a renamed copy) -- both are byte-identical.
    dup = tmp_path / "capture_copy.npz"
    dup.write_bytes(src.read_bytes())

    for path in (src, dup):
        with pytest.raises(DuplicateDatasetError) as excinfo:
            ingest_dataset(path, storage, db_session, **INGEST_KWARGS)
        e = excinfo.value
        assert e.existing_dataset_id == first.auto.dataset_id
        assert e.source_sha256 == first.auto.source_sha256
        assert first.auto.dataset_id in str(e)

    # Only the first dataset was ever written.
    assert len(list(storage.list_dataset_ids())) == 1


def test_ingest_allows_different_content_same_shape(tmp_path, storage, db_session):
    a = tmp_path / "a.npz"
    b = tmp_path / "b.npz"
    write_single_camera_npz(a, n_frames=8, seed=1)
    write_single_camera_npz(b, n_frames=8, seed=2)

    ingest_dataset(a, storage, db_session, **INGEST_KWARGS)
    ingest_dataset(b, storage, db_session, **INGEST_KWARGS)

    assert len(list(storage.list_dataset_ids())) == 2


def test_ingest_with_no_telemetry_rows(tmp_path, storage, db_session):
    """A capture with no telemetry rows: default_telemetry_row_count=0 keeps
    the full frame height and computes stats over every row."""
    src = tmp_path / "no_telemetry.npz"
    write_single_camera_npz(src, n_frames=8)

    kwargs = {**INGEST_KWARGS, "default_telemetry_row_count": 0}
    [metadata] = ingest_dataset(src, storage, db_session, **kwargs)

    cam = metadata.auto.cameras[0]
    assert cam.telemetry_row_count == 0
    assert cam.frame_height == 12  # full height, nothing sliced off
    assert cam.counts_min is not None

    store_path = storage.zarr_store_path(metadata.auto.dataset_id)
    assert zarr_store.read_frame_image(store_path, 0, 0).shape == (12, 16)
    assert zarr_store.read_frame_telemetry(store_path, 0, 0).shape == (0, 16)
    assert "telemetry" not in zarr_store.open_group(store_path)["camera_0"].array_keys()


def test_ingest_with_bottom_telemetry_rows(tmp_path, storage, db_session):
    src = tmp_path / "bottom_telemetry.npz"
    write_single_camera_npz(src, n_frames=8)

    kwargs = {**INGEST_KWARGS, "default_telemetry_row_count": -2}
    [metadata] = ingest_dataset(src, storage, db_session, **kwargs)

    cam = metadata.auto.cameras[0]
    assert cam.telemetry_row_count == -2
    assert cam.frame_height == 10  # 12 total - bottom 2

    store_path = storage.zarr_store_path(metadata.auto.dataset_id)
    source = np.load(src)["raw_thr_frames"]
    np.testing.assert_array_equal(zarr_store.read_frame_image(store_path, 0, 0), source[0, :-2, :])
    np.testing.assert_array_equal(zarr_store.read_frame_telemetry(store_path, 0, 0), source[0, -2:, :])


def test_ingest_single_camera_npz(tmp_path, storage, db_session):
    src = tmp_path / "capture.npz"
    write_single_camera_npz(src, n_frames=20)

    [metadata] = ingest_dataset(src, storage, db_session, **INGEST_KWARGS)

    assert metadata.auto.source_format == "npz"
    assert metadata.auto.camera_count == 1
    cam = metadata.auto.cameras[0]
    assert cam.frame_count == 20
    assert cam.frame_height == 10  # 12 total - 2 telemetry rows
    assert cam.telemetry_row_count == TELEMETRY_ROWS
    assert cam.fps_estimate == pytest.approx(30.0, rel=0.05)
    assert metadata.user.name == "capture"

    # Catalog was updated.
    record = repository.get(db_session, metadata.auto.dataset_id)
    assert record is not None
    assert record.total_frame_count == 20

    # Dataset is locked read-only.
    store_path = storage.zarr_store_path(metadata.auto.dataset_id)
    assert store_path.exists()
    import os
    import stat

    mode = stat.S_IMODE(os.stat(store_path).st_mode)
    assert not (mode & stat.S_IWUSR)


def test_ingest_npz_zst(tmp_path, storage, db_session):
    src = tmp_path / "capture.npz.zst"
    write_single_camera_npz_zst(src, n_frames=10)
    [metadata] = ingest_dataset(src, storage, db_session, **INGEST_KWARGS)
    assert metadata.auto.source_format == "npz.zst"
    assert metadata.auto.cameras[0].frame_count == 10


def test_ingest_h5(tmp_path, storage, db_session):
    src = tmp_path / "capture.h5"
    write_single_camera_h5(src, n_frames=12)
    [metadata] = ingest_dataset(src, storage, db_session, **INGEST_KWARGS)
    assert metadata.auto.source_format == "h5"
    assert metadata.auto.cameras[0].frame_count == 12


def test_ingest_multi_camera_file_splits_into_one_dataset_per_camera(tmp_path, storage, db_session):
    src = tmp_path / "multi.npz"
    write_multi_camera_npz(src, n_frames=10, n_cameras=2)
    datasets = ingest_dataset(src, storage, db_session, **INGEST_KWARGS)

    assert len(datasets) == 2
    assert len(set(m.auto.dataset_id for m in datasets)) == 2
    assert sorted(m.auto.cameras[0].camera_index for m in datasets) == [0, 1]
    capture_ids = {m.user.capture_id for m in datasets}
    assert len(capture_ids) == 1 and None not in capture_ids

    source = np.load(src)
    for m in datasets:
        assert m.auto.camera_count == 1
        cam = m.auto.cameras[0]
        idx = cam.camera_index
        store_path = storage.zarr_store_path(m.auto.dataset_id)
        # The zarr group keeps the source numbering, so there is no 0-vs-1 ambiguity.
        assert zarr_store.camera_indices(store_path) == [idx]
        expected = source[f"raw_thr_frames_{idx}"]
        np.testing.assert_array_equal(zarr_store.read_frame_image(store_path, idx, 3), expected[3, TELEMETRY_ROWS:, :])
        np.testing.assert_array_equal(
            zarr_store.read_frame_telemetry(store_path, idx, 3), expected[3, :TELEMETRY_ROWS, :]
        )
        thumb = storage.data_root / "datasets" / m.auto.dataset_id / "previews" / f"camera_{idx}" / "thumbnail.png"
        assert thumb.exists()

    # Siblings are findable through the catalog by their shared capture id.
    siblings = repository.search(db_session, capture_id=capture_ids.pop())
    assert {r.dataset_id for r in siblings} == {m.auto.dataset_id for m in datasets}


def test_explicit_capture_id_groups_separate_files(tmp_path, storage, db_session):
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    write_single_camera_npz(a, n_frames=6, seed=1)
    write_single_camera_npz(b, n_frames=6, seed=2)
    [ma] = ingest_dataset(a, storage, db_session, capture_id="run42", **INGEST_KWARGS)
    [mb] = ingest_dataset(b, storage, db_session, capture_id="run42", **INGEST_KWARGS)
    assert ma.user.capture_id == mb.user.capture_id == "run42"


def test_single_camera_file_has_no_capture_id_by_default(tmp_path, storage, db_session):
    src = tmp_path / "capture.npz"
    write_single_camera_npz(src, n_frames=6)
    [m] = ingest_dataset(src, storage, db_session, **INGEST_KWARGS)
    assert m.user.capture_id is None


def test_multi_camera_duplicate_skips_already_ingested_cameras(tmp_path, storage, db_session):
    src = tmp_path / "multi.npz"
    write_multi_camera_npz(src, n_frames=6, n_cameras=2)
    first = ingest_dataset(src, storage, db_session, **INGEST_KWARGS)
    assert len(first) == 2

    # Forget camera 1's dataset: re-ingest should add just that camera back.
    gone = next(m for m in first if m.auto.cameras[0].camera_index == 1)
    repository.delete(db_session, gone.auto.dataset_id)
    again = ingest_dataset(src, storage, db_session, **INGEST_KWARGS)
    assert [m.auto.cameras[0].camera_index for m in again] == [1]

    with pytest.raises(DuplicateDatasetError):
        ingest_dataset(src, storage, db_session, **INGEST_KWARGS)


def test_other_arrays_are_summarized_not_stored(tmp_path, storage, db_session):
    src = tmp_path / "multi_extra.npz"
    arrays = {}
    for i in range(2):
        arrays[f"raw_thr_frames_{i}"] = np.zeros((4, 12, 16), dtype=np.uint16)
        arrays[f"raw_thr_tstamps_{i}"] = 1_700_000_000.0 + np.arange(4) / 30.0
    arrays["imu"] = np.arange(8, dtype=np.float32)
    arrays["gps"] = np.zeros((3, 2), dtype=np.float64)
    np.savez(src, **arrays)
    for m in ingest_dataset(src, storage, db_session, **INGEST_KWARGS):
        assert m.auto.other_variables.splitlines() == [
            "imu: shape=(8,), dtype=float32",
            "gps: shape=(3, 2), dtype=float64",
        ]
        root = zarr_store.open_group(storage.zarr_store_path(m.auto.dataset_id))
        assert "extra" not in list(root.group_keys())
        assert list(root.group_keys()) == [f"camera_{m.auto.cameras[0].camera_index}"]


def test_no_other_variables_gives_empty_summary(tmp_path, storage, db_session):
    src = tmp_path / "capture.npz"
    write_single_camera_npz(src, n_frames=6)
    [m] = ingest_dataset(src, storage, db_session, **INGEST_KWARGS)
    assert m.auto.other_variables == ""


def test_ingest_unsupported_extension(tmp_path, storage, db_session):
    src = tmp_path / "capture.txt"
    src.write_text("nope")
    with pytest.raises(UnsupportedSourceFormatError):
        ingest_dataset(src, storage, db_session, **INGEST_KWARGS)


def test_ingest_rejects_mismatched_frame_and_timestamp_counts(tmp_path, storage, db_session):
    src = tmp_path / "bad.npz"
    frames = np.zeros((10, 12, 16), dtype=np.uint16)
    timestamps = np.arange(5, dtype=np.float64)  # wrong length
    np.savez(src, raw_thr_frames=frames, raw_thr_tstamps=timestamps)

    with pytest.raises(IngestValidationError) as exc_info:
        ingest_dataset(src, storage, db_session, **INGEST_KWARGS)
    assert any("frame_count" in e for e in exc_info.value.errors)


def test_ingested_dataset_readable_from_disk(tmp_path, storage, db_session):
    src = tmp_path / "capture.npz"
    write_single_camera_npz(src, n_frames=6)
    [metadata] = ingest_dataset(src, storage, db_session, **INGEST_KWARGS)

    reloaded = storage.read_metadata(metadata.auto.dataset_id)
    assert reloaded == metadata


def _multi_camera_arrays(second_dtype=np.uint16, n_frames=4):
    arrays = {}
    for i, dtype in enumerate([np.uint16, second_dtype]):
        arrays[f"raw_thr_frames_{i}"] = np.zeros((n_frames, 12, 16), dtype=dtype)
        arrays[f"raw_thr_tstamps_{i}"] = 1_700_000_000.0 + np.arange(n_frames) / 30.0
    return arrays


def test_non_uint16_frames_are_skipped_and_summarized(tmp_path, storage, db_session):
    src = tmp_path / "mixed.npz"
    np.savez(src, **_multi_camera_arrays(second_dtype=np.float32))
    [m] = ingest_dataset(src, storage, db_session, **INGEST_KWARGS)

    assert m.auto.cameras[0].camera_index == 0
    assert "raw_thr_frames_1: shape=(4, 12, 16), dtype=float32" in m.auto.other_variables
    assert m.user.capture_id is None  # only one camera was ingested


def test_no_uint16_frames_is_an_error(tmp_path, storage, db_session):
    src = tmp_path / "float_frames.npz"
    np.savez(
        src,
        raw_thr_frames=np.zeros((4, 12, 16), dtype=np.float32),
        raw_thr_tstamps=1_700_000_000.0 + np.arange(4) / 30.0,
    )
    with pytest.raises(IngestValidationError, match="No uint16 frame arrays"):
        ingest_dataset(src, storage, db_session, **INGEST_KWARGS)
    assert list(storage.list_dataset_ids()) == []


@pytest.mark.parametrize("fmt", ["npz", "npz.zst", "h5"])
def test_sources_read_headers_without_loading_data(tmp_path, fmt, monkeypatch):
    from tests.conftest import write_single_camera_h5, write_single_camera_npz_zst
    from thermal_dataset_manager.ingest.readers.common import open_source

    src = tmp_path / f"capture.{fmt}"
    {"npz": write_single_camera_npz, "npz.zst": write_single_camera_npz_zst, "h5": write_single_camera_h5}[fmt](
        src, n_frames=5
    )
    with open_source(src, fmt) as source:
        assert source.infos["raw_thr_frames"].shape == (5, 12, 16)
        assert source.infos["raw_thr_frames"].dtype == np.uint16
        assert source.infos["raw_thr_tstamps"].shape == (5,)
        assert source.read("raw_thr_tstamps").shape == (5,)


@pytest.mark.parametrize("fmt", ["npz", "h5"])
def test_ingest_only_reads_frames_and_timestamps(tmp_path, storage, db_session, monkeypatch, fmt):
    import h5py

    from thermal_dataset_manager.ingest.readers import h5_reader, npz_reader

    frames = np.zeros((4, 12, 16), dtype=np.uint16)
    ts = 1_700_000_000.0 + np.arange(4) / 30.0
    src = tmp_path / f"aux.{fmt}"
    if fmt == "npz":
        np.savez(src, raw_thr_frames=frames, raw_thr_tstamps=ts, imu=np.arange(8.0), big=np.zeros(1000))
        cls = npz_reader.NpzSource
    else:
        with h5py.File(src, "w") as f:
            f["raw_thr_frames"], f["raw_thr_tstamps"] = frames, ts
            f["imu"], f["big"] = np.arange(8.0), np.zeros(1000)
        cls = h5_reader.H5Source

    read_keys: list[str] = []
    original = cls.read
    monkeypatch.setattr(cls, "read", lambda self, key: (read_keys.append(key), original(self, key))[1])

    [m] = ingest_dataset(src, storage, db_session, **INGEST_KWARGS)

    assert sorted(read_keys) == ["raw_thr_frames", "raw_thr_tstamps"]
    assert sorted(m.auto.other_variables.splitlines()) == [
        "big: shape=(1000,), dtype=float64",
        "imu: shape=(8,), dtype=float64",
    ]


def test_cameras_are_loaded_and_written_one_at_a_time(tmp_path, storage, db_session, monkeypatch):
    from thermal_dataset_manager.ingest.readers import npz_reader

    src = tmp_path / "two.npz"
    np.savez(src, **_multi_camera_arrays())

    datasets_on_disk_at_read: dict[str, int] = {}
    original = npz_reader.NpzSource.read

    def spy(self, key):
        datasets_on_disk_at_read[key] = len(list(storage.list_dataset_ids()))
        return original(self, key)

    monkeypatch.setattr(npz_reader.NpzSource, "read", spy)
    ingest_dataset(src, storage, db_session, **INGEST_KWARGS)

    # Camera 1's arrays are only read once camera 0's dataset is already written.
    assert datasets_on_disk_at_read["raw_thr_frames_0"] == 0
    assert datasets_on_disk_at_read["raw_thr_frames_1"] == 1


def test_failure_on_later_camera_keeps_earlier_one_and_rerun_resumes(tmp_path, storage, db_session):
    src = tmp_path / "bad_second.npz"
    arrays = _multi_camera_arrays()
    arrays["raw_thr_tstamps_1"] = arrays["raw_thr_tstamps_1"][::-1].copy()  # non-monotonic
    np.savez(src, **arrays)

    with pytest.raises(IngestValidationError):
        ingest_dataset(src, storage, db_session, **INGEST_KWARGS)
    assert len(list(storage.list_dataset_ids())) == 1

    # Re-running skips the camera that made it and fails again only on the bad one.
    with pytest.raises(IngestValidationError):
        ingest_dataset(src, storage, db_session, **INGEST_KWARGS)
    assert len(list(storage.list_dataset_ids())) == 1
