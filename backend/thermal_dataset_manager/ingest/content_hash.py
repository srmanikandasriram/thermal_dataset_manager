"""Content hash of a camera's raw frames, for duplicate detection that
survives re-compression or re-packaging of the source file.

The hash covers the full raw frames (telemetry rows included, in their
original position) as little-endian uint16 in (n_frames, height, width)
order, plus that shape. It is independent of how the source file was
compressed or how many telemetry rows were split off at ingest.
"""

import hashlib

import numpy as np

_CHUNK_FRAMES = 32


def frames_content_sha256(frames) -> str:
    """`frames`: array-like of shape (n_frames, height, width) supporting
    slicing along the first axis (ndarray, zarr array)."""
    hasher = hashlib.sha256()
    hasher.update(repr(tuple(frames.shape)).encode())
    for start in range(0, frames.shape[0], _CHUNK_FRAMES):
        chunk = np.asarray(frames[start : start + _CHUNK_FRAMES]).astype("<u2", copy=False)
        hasher.update(np.ascontiguousarray(chunk).data)
    return hasher.hexdigest()


class _RawFramesFromZarr:
    """Re-joins a stored camera's image and telemetry rows into the raw
    frame layout, so it hashes the same as the source array did."""

    def __init__(self, counts, telemetry, telemetry_row_count: int) -> None:
        self._counts = counts
        self._telemetry = telemetry
        self._trc = telemetry_row_count
        n, h, w = counts.shape
        self.shape = (n, h + abs(telemetry_row_count), w)

    def __getitem__(self, sl: slice) -> np.ndarray:
        image = self._counts[sl]
        if self._trc == 0 or self._telemetry is None:
            return image
        tele = self._telemetry[sl]
        return np.concatenate([tele, image] if self._trc > 0 else [image, tele], axis=1)


def stored_camera_content_sha256(counts, telemetry, telemetry_row_count: int) -> str:
    return frames_content_sha256(_RawFramesFromZarr(counts, telemetry, telemetry_row_count))
