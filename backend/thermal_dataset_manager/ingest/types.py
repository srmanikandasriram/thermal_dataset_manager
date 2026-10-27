from dataclasses import dataclass
from typing import Literal

import numpy as np

TimestampKind = Literal["absolute_utc", "relative"]


def image_row_slice(telemetry_row_count: int) -> slice:
    """Row slice selecting the image rows of a raw frame.

    Positive N -> the top N rows are telemetry; 0 -> no telemetry rows;
    negative N -> the bottom |N| rows are telemetry.
    """
    n = telemetry_row_count
    if n > 0:
        return slice(n, None)
    if n < 0:
        return slice(None, n)
    return slice(None)


def telemetry_row_slice(telemetry_row_count: int) -> slice:
    """Row slice selecting the telemetry rows of a raw frame (the complement
    of `image_row_slice`), in top-to-bottom order."""
    n = telemetry_row_count
    if n > 0:
        return slice(None, n)
    if n < 0:
        return slice(n, None)
    return slice(0, 0)


@dataclass
class NormalizedCameraCapture:
    """One camera stream extracted from a source file, in a format-agnostic shape.

    The telemetry rows are split off the raw frames at normalization time:
    `frames` holds the image rows only, `telemetry` the stripped rows
    (top-to-bottom, shape (n_frames, |telemetry_row_count|, W)). The signed
    `telemetry_row_count` records where they sat in the raw frame: positive
    -> that many rows at the top, negative -> at the bottom, 0 -> none.
    """

    camera_index: int
    camera_label: str | None
    frames: np.ndarray  # (n_frames, H, W) uint16, image rows only
    telemetry: np.ndarray  # (n_frames, |telemetry_row_count|, W) uint16
    timestamps: np.ndarray  # (n_frames,) float64
    telemetry_row_count: int
    timestamp_kind: TimestampKind
    content_sha256: str | None = None  # of the raw frames, before the telemetry split


@dataclass
class NormalizedCapture:
    """All camera streams found in one source file. Any other arrays in the
    file are not ingested; `other_variables` is a free-text summary of them
    (name, shape, dtype) kept in the dataset metadata.
    """

    cameras: list[NormalizedCameraCapture]
    other_variables: str = ""
