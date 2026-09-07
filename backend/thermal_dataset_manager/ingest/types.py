from dataclasses import dataclass
from typing import Literal

import numpy as np

TimestampKind = Literal["absolute_utc", "relative"]


def image_row_slice(telemetry_row_count: int) -> slice:
    """Row slice that drops the telemetry rows from a raw frame.

    Positive N -> the top N rows are telemetry; 0 -> no telemetry rows;
    negative N -> the bottom |N| rows are telemetry.
    """
    n = telemetry_row_count
    if n > 0:
        return slice(n, None)
    if n < 0:
        return slice(None, n)
    return slice(None)


def image_height(total_height: int, telemetry_row_count: int) -> int:
    """Image-row count once the telemetry rows (top or bottom) are removed."""
    return total_height - abs(telemetry_row_count)


@dataclass
class NormalizedCameraCapture:
    """One camera stream extracted from a source file, in a format-agnostic shape.

    `frames` includes the telemetry rows (raw fidelity is preserved as-is);
    `telemetry_row_count` tells downstream code which rows are telemetry:
    positive -> that many rows at the top, negative -> that many at the
    bottom, 0 -> none. Use `image_row_slice` / `image_height` to apply it.
    """

    camera_index: int
    camera_label: str | None
    frames: np.ndarray  # (n_frames, H_total, W) uint16
    timestamps: np.ndarray  # (n_frames,) float64
    telemetry_row_count: int
    timestamp_kind: TimestampKind


@dataclass
class NormalizedCapture:
    """All camera streams found in one source file, plus any other arrays in
    the source file that weren't mapped to a camera's frames/timestamps --
    preserved as-is rather than silently dropped (e.g. telemetry-offset
    arrays alongside frame data).
    """

    cameras: list[NormalizedCameraCapture]
    extra_arrays: dict[str, np.ndarray]
