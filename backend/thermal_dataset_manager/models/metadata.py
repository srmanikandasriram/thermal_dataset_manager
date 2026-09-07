from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

SourceFormat = Literal["npz", "npz.zst", "h5"]
TimestampKind = Literal["absolute_utc", "relative"]
MetadataStatus = Literal["pending", "ready"]


class CameraStreamMetadata(BaseModel):
    """Auto-computed, per-camera-stream metadata. Immutable after ingest."""

    camera_index: int
    camera_label: str | None = None
    frame_count: int
    frame_height: int  # image rows only, excludes telemetry rows
    frame_width: int
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


class ExtraArrayMetadata(BaseModel):
    """An array from the source file that wasn't mapped to any camera's
    frames/timestamps. Preserved verbatim in the zarr store's `extra/`
    group rather than being silently dropped -- e.g. a telemetry-offset
    array alongside frame data.
    """

    key: str
    shape: list[int]
    dtype: str


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
    extra_arrays: list[ExtraArrayMetadata] = Field(default_factory=list)


class UserMetadata(BaseModel):
    """User-editable metadata. PATCH-able via the API."""

    name: str = ""
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    notes: str = ""
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
    capture_location: str | None = None
    scene_category: str | None = None
    custom_fields: dict[str, str] | None = None


class DatasetMetadata(BaseModel):
    auto: AutoMetadata
    user: UserMetadata
