"""Zarr v2 read/write helpers.

Layout (one group per dataset; a dataset is a single camera stream, so there is
exactly one camera subgroup, numbered by the camera's index in the source file):

    raw.zarr/
        .zattrs: {schema_version, source_format, camera_count}
        camera_<n>/
            counts      (n_frames, H, W) uint16   -- image rows only
            telemetry   (n_frames, R, W) uint16   -- the stripped telemetry rows, verbatim,
                                                     top-to-bottom; absent when R == 0
            timestamps  (n_frames,) float64
            .zattrs: {telemetry_row_count, frame_height, frame_width,
                      camera_index, camera_label, timestamp_kind}
                     telemetry_row_count is signed: where the R rows sat in the raw
                     frame (positive = top, negative = bottom).

Arrays are created without an explicit `compressor=`, so they use zarr's own default
(`zarr.storage.default_compressor`; Blosc lz4 level 5 with byte-shuffle in zarr 2.x).

All zarr access in the app goes through this module so a future v2->v3
migration, or a different chunking/codec policy, is a contained change.
"""

from pathlib import Path

import numpy as np
import zarr

from thermal_dataset_manager.ingest.types import NormalizedCameraCapture

# v2: telemetry rows are stripped out of `counts` into a separate `telemetry` array.
SCHEMA_VERSION = 2


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
) -> None:
    store_path.parent.mkdir(parents=True, exist_ok=True)
    store = zarr.DirectoryStore(str(store_path))
    root = zarr.group(store=store, overwrite=True)
    root.attrs["schema_version"] = SCHEMA_VERSION
    root.attrs["source_format"] = source_format
    root.attrs["camera_count"] = len(cameras)

    for cam in cameras:
        n_frames, h, w = cam.frames.shape
        chunk_frames = compute_time_chunk_size(h, w, target_chunk_bytes)

        cam_group = root.create_group(f"camera_{cam.camera_index}")
        cam_group.attrs["camera_index"] = cam.camera_index
        cam_group.attrs["camera_label"] = cam.camera_label
        cam_group.attrs["telemetry_row_count"] = cam.telemetry_row_count
        cam_group.attrs["frame_height"] = h
        cam_group.attrs["frame_width"] = w
        cam_group.attrs["timestamp_kind"] = cam.timestamp_kind

        cam_group.create_dataset(
            "counts",
            data=cam.frames,
            chunks=(chunk_frames, h, w),
            dtype="uint16",
        )
        n_telemetry_rows = cam.telemetry.shape[1]
        if n_telemetry_rows > 0:
            cam_group.create_dataset(
                "telemetry",
                data=cam.telemetry,
                chunks=(compute_time_chunk_size(n_telemetry_rows, w, target_chunk_bytes), n_telemetry_rows, w),
                dtype="uint16",
            )
        cam_group.create_dataset(
            "timestamps",
            data=cam.timestamps.astype("float64"),
            chunks=(n_frames,),
            dtype="float64",
        )


def open_group(store_path: Path) -> zarr.hierarchy.Group:
    store = zarr.DirectoryStore(str(store_path))
    root = zarr.open_group(store=store, mode="r")
    version = root.attrs.get("schema_version", 1)
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"{store_path} uses zarr schema v{version}, expected v{SCHEMA_VERSION}; "
            "re-ingest it from the source capture"
        )
    return root


def camera_indices(store_path: Path) -> list[int]:
    root = open_group(store_path)
    return sorted(
        int(name.removeprefix("camera_")) for name in root.group_keys() if name.startswith("camera_")
    )


def read_frame_image(store_path: Path, camera_index: int, frame_index: int) -> np.ndarray:
    """Read one frame's image data (telemetry rows are stored separately)."""
    root = open_group(store_path)
    return root[f"camera_{camera_index}"]["counts"][frame_index]


def read_frame_telemetry(store_path: Path, camera_index: int, frame_index: int) -> np.ndarray:
    """Read one frame's raw telemetry rows, shape (R, W); R == 0 when the
    stream has no telemetry."""
    root = open_group(store_path)
    cam_group = root[f"camera_{camera_index}"]
    if "telemetry" not in cam_group.array_keys():
        return np.zeros((0, cam_group["counts"].shape[2]), dtype="uint16")
    return cam_group["telemetry"][frame_index]


def read_timestamps(store_path: Path, camera_index: int) -> np.ndarray:
    root = open_group(store_path)
    return root[f"camera_{camera_index}"]["timestamps"][:]
