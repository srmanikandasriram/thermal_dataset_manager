import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import numpy as np

from thermal_dataset_manager.ingest.telemetry import parse_telemetry_rows
from thermal_dataset_manager.ingest.types import NormalizedCapture
from thermal_dataset_manager.models.metadata import (
    AutoMetadata,
    CameraStreamMetadata,
    CameraTelemetryMetadata,
    SourceFormat,
    TimestampKind,
)

_TELEMETRY_SAMPLE_FRAMES = 8

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
    frames: FrameArray, sample_frames: int
) -> tuple[int, int, float, float, float, float]:
    """Returns (min, max, mean, std, preview_vmin, preview_vmax) from a
    sampled subset of frames, for speed on long captures.
    """
    n_frames = frames.shape[0]
    if n_frames == 0:
        return 0, 0, 0.0, 0.0, 0.0, 1.0
    indices = np.linspace(0, n_frames - 1, num=min(sample_frames, n_frames), dtype=int)
    image = np.asarray(frames[indices, :, :])
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


def _static_telemetry(telemetry: FrameArray | None) -> CameraTelemetryMetadata | None:
    """Decode the Boson+ telemetry line from a handful of evenly-spaced
    frames and keep the fields that agree across all of them. Cheap: reads
    only the stored telemetry rows of ~8 frames.
    """
    if telemetry is None or telemetry.shape[1] == 0:
        return None
    n_frames = telemetry.shape[0]
    if n_frames == 0:
        return None

    indices = np.unique(
        np.linspace(0, n_frames - 1, num=min(_TELEMETRY_SAMPLE_FRAMES, n_frames), dtype=int)
    )
    parsed = []
    for i in indices:
        decoded = parse_telemetry_rows(np.asarray(telemetry[int(i)]))
        if decoded is not None:
            parsed.append(decoded)
    if not parsed:
        return None

    def agreed(getter):
        values = [getter(t) for t in parsed]
        return values[0] if all(v == values[0] for v in values) else None

    pipeline = agreed(lambda t: t.pipeline_enabled)
    software_revision = agreed(lambda t: t.software_revision)
    roi = agreed(lambda t: t.roi)
    return CameraTelemetryMetadata(
        telemetry_revision=agreed(lambda t: t.revision),
        part_number=agreed(lambda t: t.part_number) or None,
        camera_serial=agreed(lambda t: t.camera_serial),
        sensor_serial=agreed(lambda t: t.sensor_serial),
        software_revision=list(software_revision) if software_revision is not None else None,
        frame_rate_hz=agreed(lambda t: t.frame_rate_hz),
        gain_mode=agreed(lambda t: t.gain_mode),
        nuc_table=agreed(lambda t: t.current_nuc_table),
        image_stats_roi=list(roi) if roi is not None else None,
        external_sync_slave=agreed(lambda t: t.external_sync_slave),
        pipeline_enabled=(
            sorted(name for name, on in pipeline.items() if on) if pipeline is not None else None
        ),
        frames_sampled=len(parsed),
    )


def compute_camera_metadata(
    *,
    camera_index: int,
    camera_label: str | None,
    frames: FrameArray,
    telemetry: FrameArray | None,
    timestamps: np.ndarray,
    telemetry_row_count: int,
    timestamp_kind: TimestampKind,
    stats_sample_frames: int,
    compute_stats: bool = True,
) -> CameraStreamMetadata:
    """The single source of truth for per-camera computed metadata. Used
    both at ingest time (frames = the freshly-read in-memory array) and by
    ingest.recompute (frames/telemetry = the camera's zarr `counts` and
    `telemetry` arrays; `frames` is image rows only), so a new
    stat added here is backfillable onto already-ingested datasets via
    recompute rather than requiring re-ingestion from the source file.

    frame_count/shape/fps/duration are cheap (shape + a small timestamps
    array); the count stats (min/max/mean/std) require scanning sampled
    pixel data and are the part `compute_stats=False` skips -- used for the
    fast "raw data written" step, with a background recompute filling
    these in shortly after (see ingest.pipeline.write_raw_dataset).
    """
    n_frames, h, w = frames.shape
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
            frames, stats_sample_frames
        )
    else:
        counts_min = counts_max = counts_mean = counts_std = preview_vmin = preview_vmax = None

    return CameraStreamMetadata(
        camera_index=camera_index,
        camera_label=camera_label,
        frame_count=n_frames,
        frame_height=h,
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
        telemetry=_static_telemetry(telemetry),
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
                telemetry=cam.telemetry,
                timestamps=cam.timestamps,
                telemetry_row_count=cam.telemetry_row_count,
                timestamp_kind=cam.timestamp_kind,
                stats_sample_frames=stats_sample_frames,
                compute_stats=compute_stats,
            )
            for cam in capture.cameras
        ],
        other_variables=capture.other_variables,
    )
