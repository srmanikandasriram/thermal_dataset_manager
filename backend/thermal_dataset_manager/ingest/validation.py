import numpy as np

from thermal_dataset_manager.ingest.types import NormalizedCapture

# Fraction of consecutive timestamp pairs allowed to be non-increasing
# before we hard-fail instead of just noting it (small jitter is common
# with camera hardware clocks).
_MAX_NONMONOTONIC_FRACTION = 0.01


class IngestValidationError(ValueError):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


def validate(capture: NormalizedCapture) -> None:
    errors: list[str] = []

    if not capture.cameras:
        errors.append("No camera streams found in source file.")
        raise IngestValidationError(errors)

    for cam in capture.cameras:
        prefix = f"camera {cam.camera_index}"

        if cam.frames.ndim != 3:
            errors.append(f"{prefix}: frames array must be 3-D (n_frames, H, W), got shape {cam.frames.shape}")
            continue

        n_frames, h_total, w = cam.frames.shape

        if cam.frames.dtype != np.uint16:
            errors.append(f"{prefix}: frames dtype must be uint16, got {cam.frames.dtype}")

        if len(cam.timestamps) != n_frames:
            errors.append(
                f"{prefix}: frame_count ({n_frames}) does not match timestamp_count ({len(cam.timestamps)})"
            )

        if h_total <= cam.telemetry_row_count:
            errors.append(
                f"{prefix}: frame height ({h_total}) must exceed telemetry_row_count "
                f"({cam.telemetry_row_count})"
            )

        if w <= 0:
            errors.append(f"{prefix}: frame width must be positive, got {w}")

        if n_frames == 0:
            errors.append(f"{prefix}: capture has zero frames")

        if len(cam.timestamps) >= 2:
            deltas = np.diff(cam.timestamps)
            nonmonotonic = int(np.sum(deltas <= 0))
            fraction = nonmonotonic / len(deltas)
            if fraction > _MAX_NONMONOTONIC_FRACTION:
                errors.append(
                    f"{prefix}: timestamps are non-monotonic in {nonmonotonic}/{len(deltas)} "
                    f"consecutive pairs ({fraction:.1%}), exceeding tolerance"
                )

    if errors:
        raise IngestValidationError(errors)
