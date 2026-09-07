import numpy as np
import pytest

from thermal_dataset_manager.ingest.types import NormalizedCameraCapture
from thermal_dataset_manager.storage import zarr_store


def _capture(
    camera_index: int, n_frames: int, h: int, w: int, telemetry_row_count: int = 2
) -> NormalizedCameraCapture:
    rng = np.random.default_rng(camera_index)
    frames = rng.integers(0, 60000, size=(n_frames, h, w), dtype=np.uint16)
    timestamps = 1_700_000_000.0 + np.arange(n_frames, dtype=np.float64) / 30.0
    return NormalizedCameraCapture(
        camera_index=camera_index,
        camera_label=None,
        frames=frames,
        timestamps=timestamps,
        telemetry_row_count=telemetry_row_count,
        timestamp_kind="absolute_utc",
    )


def test_write_and_read_round_trip(tmp_path):
    cam0 = _capture(0, n_frames=10, h=12, w=16)
    store_path = tmp_path / "raw.zarr"
    zarr_store.write(store_path, [cam0], "npz", target_chunk_bytes=1024 * 1024)

    assert zarr_store.camera_indices(store_path) == [0]

    frame = zarr_store.read_frame(store_path, 0, 3)
    np.testing.assert_array_equal(frame, cam0.frames[3])

    image = zarr_store.read_frame_image(store_path, 0, 3)
    np.testing.assert_array_equal(image, cam0.frames[3, 2:, :])

    timestamps = zarr_store.read_timestamps(store_path, 0)
    np.testing.assert_allclose(timestamps, cam0.timestamps)


@pytest.mark.parametrize(
    "telemetry_row_count, expected_image_slice",
    [
        (2, slice(2, None)),   # top 2 rows are telemetry
        (0, slice(None)),      # no telemetry rows
        (-2, slice(None, -2)),  # bottom 2 rows are telemetry
    ],
)
def test_telemetry_row_count_slices_top_bottom_or_none(tmp_path, telemetry_row_count, expected_image_slice):
    cam = _capture(0, n_frames=5, h=12, w=16, telemetry_row_count=telemetry_row_count)
    store_path = tmp_path / "raw.zarr"
    zarr_store.write(store_path, [cam], "npz", target_chunk_bytes=1024 * 1024)

    root = zarr_store.open_group(store_path)
    assert root["camera_0"].attrs["frame_height"] == 12 - abs(telemetry_row_count)
    assert root["camera_0"].attrs["telemetry_row_count"] == telemetry_row_count

    # Raw frame is always stored whole; only the image view drops telemetry.
    np.testing.assert_array_equal(zarr_store.read_frame(store_path, 0, 3), cam.frames[3])
    np.testing.assert_array_equal(
        zarr_store.read_frame_image(store_path, 0, 3), cam.frames[3, expected_image_slice, :]
    )


def test_multi_camera_round_trip(tmp_path):
    cam0 = _capture(0, n_frames=8, h=12, w=16)
    cam1 = _capture(1, n_frames=8, h=12, w=16)
    store_path = tmp_path / "raw.zarr"
    zarr_store.write(store_path, [cam0, cam1], "npz", target_chunk_bytes=1024 * 1024)

    assert zarr_store.camera_indices(store_path) == [0, 1]
    np.testing.assert_array_equal(zarr_store.read_frame(store_path, 1, 0), cam1.frames[0])


def test_chunking_targets_byte_budget():
    # 12x16 uint16 frame = 384 bytes; target 1024 bytes -> ~2 frames/chunk
    chunk_frames = zarr_store.compute_time_chunk_size(12, 16, target_bytes=1024)
    assert chunk_frames == 2

    # Target smaller than one frame still yields at least 1 frame/chunk.
    assert zarr_store.compute_time_chunk_size(12, 16, target_bytes=10) == 1
