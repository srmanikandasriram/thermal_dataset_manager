from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

SourceFormat = Literal["npz", "npz.zst", "h5"]
TimestampKind = Literal["absolute_utc", "relative"]
MetadataStatus = Literal["pending", "ready"]


class CameraTelemetryMetadata(BaseModel):
    """Static fields decoded from the Boson+ telemetry line embedded in each
    frame (datasheet section 6.7 / Table 5). Only quantities that stay
    constant across the capture are kept here -- identity, build, and
    capture-configuration fields; per-frame values (frame counter, timestamp,
    ROI stats, temperatures) are not.

    Absent (the whole object is None) when the stream has no telemetry rows
    or no telemetry row validated. Individual fields are None when they could
    not be read or -- unexpectedly -- disagreed across the sampled frames.
    """

    telemetry_revision: int | None = None
    part_number: str | None = None
    camera_serial: int | None = None
    sensor_serial: int | None = None
    software_revision: list[int] | None = None  # [major, minor, patch] words as stored
    frame_rate_hz: int | None = None
    gain_mode: str | None = None  # high | low | automatic | reserved
    nuc_table: int | None = None
    image_stats_roi: list[int] | None = None  # 4 words: the ROI the mean/min/max cover
    external_sync_slave: bool | None = None
    pipeline_enabled: list[str] | None = None  # names of enabled processing stages
    frames_sampled: int = 0


class CameraStreamMetadata(BaseModel):
    """Auto-computed, per-camera-stream metadata. Immutable after ingest."""

    # Index of this stream in the source file. A multi-camera source file is
    # split into one dataset per camera, so this is the dataset's only
    # camera and keeps the source numbering (the zarr group is camera_<n>).
    camera_index: int
    camera_label: str | None = None
    frame_count: int
    frame_height: int  # image rows only, excludes telemetry rows
    frame_width: int
    # Signed: positive N = top N rows are telemetry, negative N = bottom
    # |N| rows, 0 = no telemetry rows.
    # The rows themselves are stored in the zarr `telemetry` array.
    telemetry_row_count: int
    dtype: str = "uint16"

    fps_estimate: float | None = None
    duration_seconds: float | None = None
    timestamp_kind: TimestampKind
    start_time_utc: datetime | None = None

    counts_min: int | None = None
    counts_max: int | None = None
    counts_mean: float | None = None
    counts_std: float | None = None

    # Percentile-clipped normalization range used for colormapped preview
    # rendering, fixed at ingest time so frame-to-frame scrubbing doesn't
    # flicker from per-frame min/max normalization.
    preview_vmin: float | None = None
    preview_vmax: float | None = None

    telemetry: CameraTelemetryMetadata | None = None

    # sha256 of the raw frames (see ingest.content_hash); detects the same
    # video ingested from differently packaged files. None until computed.
    content_sha256: str | None = None


class AutoMetadata(BaseModel):
    """Computed at ingest time (or recomputed later). Never edited by users;
    the underlying raw data is immutable, though this document itself gets
    rewritten as metadata computation finishes or is redone.
    """

    dataset_id: str
    source_format: SourceFormat
    source_filename: str
    source_sha256: str
    ingested_at: datetime
    ingest_tool_version: str

    # "pending" right after raw data ingestion (cheap fields only -- frame
    # counts/shapes/fps/duration, all derivable without scanning pixel
    # data); flips to "ready" once full stats/thumbnails are computed,
    # which happens in the background, decoupled from the upload itself.
    metadata_status: MetadataStatus = "ready"

    camera_count: int
    cameras: list[CameraStreamMetadata]
    # Free-text summary (one "name: shape=..., dtype=..." line each) of the
    # source-file arrays that weren't ingested -- everything but the camera
    # frames/timestamps. Informational only; the arrays themselves are dropped.
    other_variables: str = ""


class UserMetadata(BaseModel):
    """User-editable metadata. PATCH-able via the API."""

    name: str = ""
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    notes: str = ""
    # Datasets that belong to the same capture (e.g. the cameras of one
    # multi-camera file, or files grouped with ingest_cli --capture) share this.
    capture_id: str | None = None
    capture_location: str | None = None
    scene_category: str | None = None
    custom_fields: dict[str, str] = Field(default_factory=dict)


class UserMetadataUpdate(BaseModel):
    """Partial update payload for PATCH /api/datasets/{id}/metadata.

    All fields optional; only fields explicitly set are merged in.
    """

    name: str | None = None
    description: str | None = None
    tags: list[str] | None = None
    notes: str | None = None
    capture_id: str | None = None
    capture_location: str | None = None
    scene_category: str | None = None
    custom_fields: dict[str, str] | None = None


class DatasetMetadata(BaseModel):
    auto: AutoMetadata
    user: UserMetadata
