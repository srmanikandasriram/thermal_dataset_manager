from collections.abc import Mapping
from pathlib import Path

import numpy as np

from thermal_dataset_manager.ingest.key_registry import CameraKeyPair, discover_camera_keys
from thermal_dataset_manager.ingest.readers import h5_reader, npz_reader
from thermal_dataset_manager.ingest.types import (
    NormalizedCameraCapture,
    NormalizedCapture,
    TimestampKind,
)
from thermal_dataset_manager.models.metadata import SourceFormat

# Timestamps larger than this are assumed to be absolute epoch seconds
# (roughly year 2001+); smaller values are assumed relative/monotonic.
_ABSOLUTE_EPOCH_THRESHOLD = 1_000_000_000.0


def infer_timestamp_kind(timestamps: np.ndarray) -> TimestampKind:
    if timestamps.size == 0:
        return "relative"
    return "absolute_utc" if float(timestamps[0]) >= _ABSOLUTE_EPOCH_THRESHOLD else "relative"


def _squeeze_trailing_singleton_dims(frames: np.ndarray) -> np.ndarray:
    """Some sources emit frames with a trailing singleton channel dimension,
    e.g. (n_frames, H, W, 1) instead of (n_frames, H, W) -- squeeze it out
    so the rest of the pipeline (which assumes 3-D frames) sees a clean shape.
    """
    while frames.ndim > 3 and frames.shape[-1] == 1:
        frames = frames.squeeze(axis=-1)
    return frames


def load_array_map(path: Path, source_format: SourceFormat) -> dict[str, np.ndarray]:
    """Read every array/dataset in a source file into a flat name -> ndarray map,
    dispatching to the format-specific reader. This is the shared first step for
    both auto-detected and explicitly user-mapped ingest.
    """
    if source_format in ("npz", "npz.zst"):
        return npz_reader.load_array_map(path)
    if source_format == "h5":
        return h5_reader.load_array_map(path)
    raise ValueError(f"Unsupported source format: {source_format}")


def build_camera_capture(
    array_map: Mapping[str, np.ndarray],
    pair: CameraKeyPair,
    telemetry_row_count: int,
) -> NormalizedCameraCapture:
    frames = _squeeze_trailing_singleton_dims(np.asarray(array_map[pair.frames_key]))
    timestamps = np.asarray(array_map[pair.timestamps_key]).astype("float64")
    return NormalizedCameraCapture(
        camera_index=pair.camera_index,
        camera_label=pair.camera_label,
        frames=frames.astype("uint16", copy=False),
        timestamps=timestamps,
        telemetry_row_count=telemetry_row_count,
        timestamp_kind=infer_timestamp_kind(timestamps),
    )


def build_normalized_capture_from_pairs(
    array_map: Mapping[str, np.ndarray],
    pairs: list[CameraKeyPair],
    telemetry_row_count: int,
) -> NormalizedCapture:
    used_keys = {pair.frames_key for pair in pairs} | {pair.timestamps_key for pair in pairs}
    extra_arrays = {key: np.asarray(value) for key, value in array_map.items() if key not in used_keys}
    return NormalizedCapture(
        cameras=[build_camera_capture(array_map, pair, telemetry_row_count) for pair in pairs],
        extra_arrays=extra_arrays,
    )


def build_normalized_capture(
    array_map: Mapping[str, np.ndarray],
    telemetry_row_count: int,
) -> NormalizedCapture:
    """Auto-detect camera streams via the key registry and build a NormalizedCapture."""
    pairs = discover_camera_keys(list(array_map.keys()))
    return build_normalized_capture_from_pairs(array_map, pairs, telemetry_row_count)
