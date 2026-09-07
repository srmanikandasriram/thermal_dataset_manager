"""Zarr v2 read/write helpers.

Layout (one group per dataset, one subgroup per camera stream, plus an
`extra/` group for any source-file arrays that weren't mapped to a camera):

    raw.zarr/
        .zattrs: {schema_version, source_format, camera_count}
        camera_<n>/
            counts      (n_frames, H, W) uint16   -- includes telemetry rows
            timestamps  (n_frames,) float64
            .zattrs: {telemetry_row_count, frame_height, frame_width,
                      camera_index, camera_label, timestamp_kind}
        extra/
            <original key name>   -- verbatim, native dtype

All zarr access in the app goes through this module so a future v2->v3
migration, or a different chunking/codec policy, is a contained change.
"""

from pathlib import Path

import numcodecs
import numpy as np
import zarr

from thermal_dataset_manager.ingest.types import NormalizedCameraCapture

SCHEMA_VERSION = 1
_EXTRA_GROUP = "extra"


def compute_time_chunk_size(frame_height: int, frame_width: int, target_bytes: int) -> int:
    """Number of frames per chunk, aiming for ~target_bytes uncompressed per chunk."""
    bytes_per_frame = frame_height * frame_width * np.dtype("uint16").itemsize
    if bytes_per_frame <= 0:
        return 1
    return max(1, target_bytes // bytes_per_frame)


def write(
    store_path: Path,
    cameras: list[NormalizedCameraCapture],
    source_format: str,
    target_chunk_bytes: int,
    extra_arrays: dict[str, np.ndarray] | None = None,
) -> None:
    store_path.parent.mkdir(parents=True, exist_ok=True)
    store = zarr.DirectoryStore(str(store_path))
    root = zarr.group(store=store, overwrite=True)
    root.attrs["schema_version"] = SCHEMA_VERSION
    root.attrs["source_format"] = source_format
    root.attrs["camera_count"] = len(cameras)

    for cam in cameras:
        n_frames, h_total, w = cam.frames.shape
        chunk_frames = compute_time_chunk_size(h_total, w, target_chunk_bytes)

        cam_group = root.create_group(f"camera_{cam.camera_index}")
        cam_group.attrs["camera_index"] = cam.camera_index
        cam_group.attrs["camera_label"] = cam.camera_label
        cam_group.attrs["telemetry_row_count"] = cam.telemetry_row_count
        cam_group.attrs["frame_height"] = h_total - cam.telemetry_row_count
        cam_group.attrs["frame_width"] = w
        cam_group.attrs["timestamp_kind"] = cam.timestamp_kind

        cam_group.create_dataset(
            "counts",
            data=cam.frames,
            chunks=(chunk_frames, h_total, w),
            dtype="uint16",
            compressor=numcodecs.Blosc(cname="zstd", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE),
        )
        cam_group.create_dataset(
            "timestamps",
            data=cam.timestamps.astype("float64"),
            chunks=(n_frames,),
            dtype="float64",
            compressor=numcodecs.Blosc(cname="zstd", clevel=5),
        )

    if extra_arrays:
        extra_group = root.create_group(_EXTRA_GROUP)
        for key, value in extra_arrays.items():
            extra_group.create_dataset(
                key,
                data=value,
                compressor=numcodecs.Blosc(cname="zstd", clevel=5),
            )


def open_group(store_path: Path) -> zarr.hierarchy.Group:
    store = zarr.DirectoryStore(str(store_path))
    return zarr.open_group(store=store, mode="r")


def camera_indices(store_path: Path) -> list[int]:
    root = open_group(store_path)
    return sorted(
        int(name.removeprefix("camera_")) for name in root.group_keys() if name.startswith("camera_")
    )


def read_frame(store_path: Path, camera_index: int, frame_index: int) -> np.ndarray:
    """Read one raw frame (including telemetry rows) for a given camera."""
    root = open_group(store_path)
    counts = root[f"camera_{camera_index}"]["counts"]
    return counts[frame_index]


def read_frame_image(store_path: Path, camera_index: int, frame_index: int) -> np.ndarray:
    """Read one frame's image data (telemetry rows sliced off)."""
    root = open_group(store_path)
    cam_group = root[f"camera_{camera_index}"]
    telemetry_rows = int(cam_group.attrs["telemetry_row_count"])
    frame = cam_group["counts"][frame_index]
    return frame[telemetry_rows:, :]


def read_timestamps(store_path: Path, camera_index: int) -> np.ndarray:
    root = open_group(store_path)
    return root[f"camera_{camera_index}"]["timestamps"][:]


def read_extra_array(store_path: Path, key: str) -> np.ndarray:
    root = open_group(store_path)
    if _EXTRA_GROUP not in root.group_keys():
        raise KeyError(key)
    extra_group = root[_EXTRA_GROUP]
    if key not in extra_group.array_keys():
        raise KeyError(key)
    # `[:]` fails on a 0-d (scalar) array -- zarr treats it as "too many
    # indices". `[...]` reads the full array regardless of dimensionality.
    return extra_group[key][...]
