import numpy as np
import pytest
import zarr

from thermal_dataset_manager.ingest.types import NormalizedCameraCapture, image_row_slice, telemetry_row_slice
from thermal_dataset_manager.storage import zarr_store


def _capture(
    camera_index: int, n_frames: int, h: int, w: int, telemetry_row_count: int = 2
) -> NormalizedCameraCapture:
    rng = np.random.default_rng(camera_index)
    raw = rng.integers(0, 60000, size=(n_frames, h, w), dtype=np.uint16)
    frames = raw[:, image_row_slice(telemetry_row_count), :]
    telemetry = raw[:, telemetry_row_slice(telemetry_row_count), :]
    timestamps = 1_700_000_000.0 + np.arange(n_frames, dtype=np.float64) / 30.0
    return NormalizedCameraCapture(
        camera_index=camera_index,
        camera_label=None,
        frames=frames,
        telemetry=telemetry,
        timestamps=timestamps,
        telemetry_row_count=telemetry_row_count,
        timestamp_kind="absolute_utc",
    )


def test_write_and_read_round_trip(tmp_path):
    cam0 = _capture(0, n_frames=10, h=12, w=16)
    store_path = tmp_path / "raw.zarr"
    zarr_store.write(store_path, [cam0], "npz", target_chunk_bytes=1024 * 1024)

    assert zarr_store.camera_indices(store_path) == [0]

    np.testing.assert_array_equal(zarr_store.read_frame_image(store_path, 0, 3), cam0.frames[3])
    np.testing.assert_array_equal(zarr_store.read_frame_telemetry(store_path, 0, 3), cam0.telemetry[3])

    timestamps = zarr_store.read_timestamps(store_path, 0)
    np.testing.assert_allclose(timestamps, cam0.timestamps)


@pytest.mark.parametrize("telemetry_row_count", [2, 0, -2])
def test_telemetry_is_stored_separately_from_image(tmp_path, telemetry_row_count):
    cam = _capture(0, n_frames=5, h=12, w=16, telemetry_row_count=telemetry_row_count)
    store_path = tmp_path / "raw.zarr"
    zarr_store.write(store_path, [cam], "npz", target_chunk_bytes=1024 * 1024)

    root = zarr_store.open_group(store_path)
    group = root["camera_0"]
    assert group.attrs["frame_height"] == 12 - abs(telemetry_row_count)
    assert group.attrs["telemetry_row_count"] == telemetry_row_count
    assert group["counts"].shape == (5, 12 - abs(telemetry_row_count), 16)
    assert ("telemetry" in group.array_keys()) == (telemetry_row_count != 0)

    np.testing.assert_array_equal(zarr_store.read_frame_image(store_path, 0, 3), cam.frames[3])
    np.testing.assert_array_equal(zarr_store.read_frame_telemetry(store_path, 0, 3), cam.telemetry[3])
    assert zarr_store.read_frame_telemetry(store_path, 0, 3).shape == (abs(telemetry_row_count), 16)


def test_open_group_rejects_old_schema(tmp_path):
    cam = _capture(0, n_frames=3, h=12, w=16)
    store_path = tmp_path / "raw.zarr"
    zarr_store.write(store_path, [cam], "npz", target_chunk_bytes=1024 * 1024)
    root = zarr.open_group(store_path, mode="a")
    root.attrs["schema_version"] = 1
    with pytest.raises(ValueError, match="re-ingest"):
        zarr_store.open_group(store_path)


def test_new_stores_are_zarr_format_3(tmp_path):
    store_path = tmp_path / "raw.zarr"
    zarr_store.write(store_path, [_capture(0, n_frames=4, h=12, w=16)], "npz", target_chunk_bytes=1024 * 1024)

    assert zarr.open_group(store_path, mode="r").metadata.zarr_format == 3
    assert (store_path / "zarr.json").exists() and not (store_path / ".zgroup").exists()


def test_zarr_format_2_stores_are_still_read(tmp_path):
    """Datasets ingested before the move to zarr 3 keep working, without a rewrite."""
    cam = _capture(0, n_frames=6, h=12, w=16)
    store_path = tmp_path / "raw.zarr"
    root = zarr.open_group(store_path, mode="w", zarr_format=2)
    root.attrs.update({"schema_version": zarr_store.SCHEMA_VERSION, "source_format": "npz", "camera_count": 1})
    group = root.create_group("camera_0")
    group.attrs.update({"telemetry_row_count": 2})
    group.create_array("counts", data=cam.frames, chunks=(4, 10, 16))
    group.create_array("telemetry", data=cam.telemetry, chunks=(4, 2, 16))
    group.create_array("timestamps", data=cam.timestamps, chunks=(6,))

    assert zarr.open_group(store_path, mode="r").metadata.zarr_format == 2
    assert zarr_store.camera_indices(store_path) == [0]
    np.testing.assert_array_equal(zarr_store.read_frame_image(store_path, 0, 5), cam.frames[5])
    np.testing.assert_array_equal(zarr_store.read_frame_telemetry(store_path, 0, 5), cam.telemetry[5])
    np.testing.assert_allclose(zarr_store.read_timestamps(store_path, 0), cam.timestamps)


def test_multi_camera_round_trip(tmp_path):
    cam0 = _capture(0, n_frames=8, h=12, w=16)
    cam1 = _capture(1, n_frames=8, h=12, w=16)
    store_path = tmp_path / "raw.zarr"
    zarr_store.write(store_path, [cam0, cam1], "npz", target_chunk_bytes=1024 * 1024)

    assert zarr_store.camera_indices(store_path) == [0, 1]
    np.testing.assert_array_equal(zarr_store.read_frame_image(store_path, 1, 0), cam1.frames[0])


def test_chunking_targets_byte_budget():
    # 12x16 uint16 frame = 384 bytes; target 1024 bytes -> ~2 frames/chunk
    chunk_frames = zarr_store.compute_time_chunk_size(12, 16, target_bytes=1024)
    assert chunk_frames == 2

    # Target smaller than one frame still yields at least 1 frame/chunk.
    assert zarr_store.compute_time_chunk_size(12, 16, target_bytes=10) == 1
