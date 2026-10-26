// Mirrors backend Pydantic response shapes
// (backend/thermal_dataset_manager/models/metadata.py, api/datasets.py)

export type TimestampKind = 'absolute_utc' | 'relative'
export type SourceFormat = 'npz' | 'npz.zst' | 'h5'
export type MetadataStatus = 'pending' | 'ready'

export interface CameraTelemetryMetadata {
  telemetry_revision: number | null
  part_number: string | null
  camera_serial: number | null
  sensor_serial: number | null
  software_revision: number[] | null
  frame_rate_hz: number | null
  gain_mode: string | null
  nuc_table: number | null
  image_stats_roi: number[] | null
  external_sync_slave: boolean | null
  pipeline_enabled: string[] | null
  frames_sampled: number
}

export interface CameraStreamMetadata {
  camera_index: number
  camera_label: string | null
  frame_count: number
  frame_height: number
  frame_width: number
  telemetry_row_count: number
  dtype: string
  fps_estimate: number | null
  duration_seconds: number | null
  timestamp_kind: TimestampKind
  start_time_utc: string | null
  counts_min: number | null
  counts_max: number | null
  counts_mean: number | null
  counts_std: number | null
  preview_vmin: number | null
  preview_vmax: number | null
  telemetry: CameraTelemetryMetadata | null
}

export interface AutoMetadata {
  dataset_id: string
  source_format: SourceFormat
  source_filename: string
  source_sha256: string
  ingested_at: string
  ingest_tool_version: string
  metadata_status: MetadataStatus
  camera_count: number
  cameras: CameraStreamMetadata[]
  other_variables: string
}

export interface UserMetadata {
  name: string
  description: string
  tags: string[]
  notes: string
  capture_id: string | null
  capture_location: string | null
  scene_category: string | null
  custom_fields: Record<string, string>
}

export interface UserMetadataUpdate {
  name?: string
  description?: string
  tags?: string[]
  notes?: string
  capture_id?: string | null
  capture_location?: string | null
  scene_category?: string | null
  custom_fields?: Record<string, string>
}

export interface DatasetMetadata {
  auto: AutoMetadata
  user: UserMetadata
}

export interface DatasetSummary {
  dataset_id: string
  name: string
  description: string
  tags: string[]
  ingested_at: string
  source_filename: string
  capture_id: string | null
  camera_count: number
  total_frame_count: number
  max_duration_seconds: number | null
  scene_category: string | null
  capture_location: string | null
  thumbnail_url: string
}

export type DatasetSortField =
  | 'name'
  | 'description'
  | 'ingested_at'
  | 'source_filename'
  | 'capture_id'
  | 'camera_count'
  | 'total_frame_count'
  | 'max_duration_seconds'
  | 'scene_category'
  | 'capture_location'

export type SortOrder = 'asc' | 'desc'

export interface DatasetListResponse {
  items: DatasetSummary[]
  limit: number
  offset: number
  total: number
}

export interface DatasetSearchParams {
  q?: string
  tags?: string[]
  date_from?: string
  date_to?: string
  capture_id?: string
  sort?: DatasetSortField
  order?: SortOrder
  limit?: number
  offset?: number
}
