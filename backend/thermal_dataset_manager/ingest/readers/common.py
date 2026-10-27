from pathlib import Path

import numpy as np

from thermal_dataset_manager.ingest.content_hash import frames_content_sha256
from thermal_dataset_manager.ingest.key_registry import CameraKeyPair
from thermal_dataset_manager.ingest.readers.h5_reader import H5Source
from thermal_dataset_manager.ingest.readers.npz_reader import NpzSource
from thermal_dataset_manager.ingest.readers.source import ArraySource
from thermal_dataset_manager.ingest.types import (
    NormalizedCameraCapture,
    NormalizedCapture,
    TimestampKind,
    image_row_slice,
    telemetry_row_slice,
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


def open_source(path: Path, source_format: SourceFormat) -> ArraySource:
    """Open a source file lazily, dispatching to the format-specific reader.
    Use as a context manager; no array data is read until `read` is called."""
    if source_format in ("npz", "npz.zst"):
        return NpzSource(path)
    if source_format == "h5":
        return H5Source(path)
    raise ValueError(f"Unsupported source format: {source_format}")


def describe_other_variables(source: ArraySource, used_keys: set[str]) -> str:
    """Free-text summary (name, shape, dtype; one line each) of the arrays that
    are not ingested, built from headers alone."""
    return "\n".join(
        f"{key}: shape={info.shape}, dtype={info.dtype}"
        for key, info in source.infos.items()
        if key not in used_keys
    )


def build_camera_capture(
    source: ArraySource,
    pair: CameraKeyPair,
    telemetry_row_count: int,
) -> NormalizedCameraCapture:
    frames = _squeeze_trailing_singleton_dims(source.read(pair.frames_key))
    timestamps = source.read(pair.timestamps_key).astype("float64")
    content_sha256 = frames_content_sha256(frames) if frames.ndim == 3 else None
    if frames.ndim == 3:
        # Split the telemetry rows off the raw frames. (Other shapes are left
        # alone for validation to reject.) A telemetry_row_count at least as
        # large as the frame height leaves no image rows; validation catches it.
        telemetry = frames[:, telemetry_row_slice(telemetry_row_count), :]
        frames = frames[:, image_row_slice(telemetry_row_count), :]
    else:
        telemetry = np.zeros((0, 0, 0), dtype="uint16")
    return NormalizedCameraCapture(
        camera_index=pair.camera_index,
        camera_label=pair.camera_label,
        frames=frames,
        telemetry=telemetry,
        timestamps=timestamps,
        telemetry_row_count=telemetry_row_count,
        timestamp_kind=infer_timestamp_kind(timestamps),
        content_sha256=content_sha256,
    )


def build_normalized_capture_from_pairs(
    source: ArraySource,
    pairs: list[CameraKeyPair],
    default_telemetry_row_count: int,
    other_variables: str = "",
) -> NormalizedCapture:
    """Loads only the given pairs' frames and timestamps. Each pair's own
    `telemetry_row_count` wins; pairs that leave it unset (e.g. everything
    from auto-detection) fall back to the default."""
    return NormalizedCapture(
        cameras=[
            build_camera_capture(
                source,
                pair,
                pair.telemetry_row_count
                if pair.telemetry_row_count is not None
                else default_telemetry_row_count,
            )
            for pair in pairs
        ],
        other_variables=other_variables,
    )
