from dataclasses import dataclass
from typing import Literal

import numpy as np

TimestampKind = Literal["absolute_utc", "relative"]


@dataclass
class NormalizedCameraCapture:
    """One camera stream extracted from a source file, in a format-agnostic shape.

    `frames` includes the telemetry rows (raw fidelity is preserved as-is);
    `telemetry_row_count` tells downstream code how many leading rows to skip
    to get at image data.
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
