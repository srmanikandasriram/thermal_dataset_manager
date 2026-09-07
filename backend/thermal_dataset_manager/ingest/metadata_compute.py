import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import numpy as np

from thermal_dataset_manager.ingest.types import NormalizedCapture, image_height, image_row_slice
from thermal_dataset_manager.models.metadata import (
    AutoMetadata,
    CameraStreamMetadata,
    ExtraArrayMetadata,
    SourceFormat,
    TimestampKind,
)

_HASH_CHUNK_SIZE = 1024 * 1024


class FrameArray(Protocol):
    """Anything shaped (n_frames, H, W) that supports combined fancy+slice
    indexing and .shape -- an in-memory ndarray during ingest, or a lazy
    zarr array when recomputing metadata from an already-ingested store.
    """

    shape: tuple[int, ...]

    def __getitem__(self, key: object) -> np.ndarray: ...


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _sampled_count_stats(
    frames: FrameArray, telemetry_row_count: int, sample_frames: int
) -> tuple[int, int, float, float, float, float]:
    """Returns (min, max, mean, std, preview_vmin, preview_vmax) from a
    sampled subset of frames, for speed on long captures.
    """
    n_frames = frames.shape[0]
    if n_frames == 0:
        return 0, 0, 0.0, 0.0, 0.0, 1.0
    indices = np.linspace(0, n_frames - 1, num=min(sample_frames, n_frames), dtype=int)
    image = np.asarray(frames[indices, image_row_slice(telemetry_row_count), :])
    vmin, vmax = np.percentile(image, [2, 98])
    if vmax <= vmin:
        vmax = vmin + 1
    return (
        int(image.min()),
        int(image.max()),
        float(image.mean()),
        float(image.std()),
        float(vmin),
        float(vmax),
    )


def compute_camera_metadata(
    *,
    camera_index: int,
    camera_label: str | None,
    frames: FrameArray,
    timestamps: np.ndarray,
    telemetry_row_count: int,
    timestamp_kind: TimestampKind,
    stats_sample_frames: int,
    compute_stats: bool = True,
) -> CameraStreamMetadata:
    """The single source of truth for per-camera computed metadata. Used
    both at ingest time (frames = the freshly-read in-memory array) and by
    ingest.recompute (frames = the camera's zarr `counts` array), so a new
    stat added here is backfillable onto already-ingested datasets via
    recompute rather than requiring re-ingestion from the source file.

    frame_count/shape/fps/duration are cheap (shape + a small timestamps
    array); the count stats (min/max/mean/std) require scanning sampled
    pixel data and are the part `compute_stats=False` skips -- used for the
    fast "raw data written" step, with a background recompute filling
    these in shortly after (see ingest.pipeline.write_raw_dataset).
    """
    n_frames, h_total, w = frames.shape
    fps_estimate = None
    duration_seconds = None
    start_time_utc = None

    if len(timestamps) >= 2:
        deltas = np.diff(timestamps)
        median_delta = float(np.median(deltas))
        if median_delta > 0:
            fps_estimate = 1.0 / median_delta
        duration_seconds = float(timestamps[-1] - timestamps[0])

    if timestamp_kind == "absolute_utc" and len(timestamps) > 0:
        start_time_utc = datetime.fromtimestamp(float(timestamps[0]), tz=timezone.utc)

    if compute_stats:
        counts_min, counts_max, counts_mean, counts_std, preview_vmin, preview_vmax = _sampled_count_stats(
            frames, telemetry_row_count, stats_sample_frames
        )
    else:
        counts_min = counts_max = counts_mean = counts_std = preview_vmin = preview_vmax = None

    return CameraStreamMetadata(
        camera_index=camera_index,
        camera_label=camera_label,
        frame_count=n_frames,
        frame_height=image_height(h_total, telemetry_row_count),
        frame_width=w,
        telemetry_row_count=telemetry_row_count,
        fps_estimate=fps_estimate,
        duration_seconds=duration_seconds,
        timestamp_kind=timestamp_kind,
        start_time_utc=start_time_utc,
        counts_min=counts_min,
        counts_max=counts_max,
        counts_mean=counts_mean,
        counts_std=counts_std,
        preview_vmin=preview_vmin,
        preview_vmax=preview_vmax,
    )


def compute_auto_metadata(
    dataset_id: str,
    source_path: Path,
    source_format: SourceFormat,
    capture: NormalizedCapture,
    ingest_tool_version: str,
    stats_sample_frames: int,
    *,
    compute_stats: bool = True,
    source_sha256: str | None = None,
) -> AutoMetadata:
    return AutoMetadata(
        dataset_id=dataset_id,
        source_format=source_format,
        source_filename=source_path.name,
        source_sha256=source_sha256 if source_sha256 is not None else sha256_of_file(source_path),
        ingested_at=datetime.now(timezone.utc),
        ingest_tool_version=ingest_tool_version,
        metadata_status="ready" if compute_stats else "pending",
        camera_count=len(capture.cameras),
        cameras=[
            compute_camera_metadata(
                camera_index=cam.camera_index,
                camera_label=cam.camera_label,
                frames=cam.frames,
                timestamps=cam.timestamps,
                telemetry_row_count=cam.telemetry_row_count,
                timestamp_kind=cam.timestamp_kind,
                stats_sample_frames=stats_sample_frames,
                compute_stats=compute_stats,
            )
            for cam in capture.cameras
        ],
        extra_arrays=[
            ExtraArrayMetadata(key=key, shape=list(value.shape), dtype=str(value.dtype))
            for key, value in capture.extra_arrays.items()
        ],
    )
