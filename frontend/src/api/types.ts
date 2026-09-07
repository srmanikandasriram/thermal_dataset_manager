// Mirrors backend Pydantic response shapes
// (backend/thermal_dataset_manager/models/metadata.py, api/datasets.py, api/ingest.py)

export type TimestampKind = 'absolute_utc' | 'relative'
export type SourceFormat = 'npz' | 'npz.zst' | 'h5'
export type MetadataStatus = 'pending' | 'ready'

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
}

export interface ExtraArrayMetadata {
  key: string
  shape: number[]
  dtype: string
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
  extra_arrays: ExtraArrayMetadata[]
}

export interface UserMetadata {
  name: string
  description: string
  tags: string[]
  notes: string
  capture_location: string | null
  scene_category: string | null
  custom_fields: Record<string, string>
}

export interface UserMetadataUpdate {
  name?: string
  description?: string
  tags?: string[]
  notes?: string
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
  camera_count: number
  total_frame_count: number
  max_duration_seconds: number | null
  thumbnail_url: string
}

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
  limit?: number
  offset?: number
}

export type JobStatus = 'pending' | 'running' | 'succeeded' | 'failed'

export interface JobStatusResponse {
  job_id: string
  status: JobStatus
  dataset_id: string | null
  errors: string[]
}

export interface KeyInfo {
  key: string
  shape: number[]
  dtype: string
}

export interface CameraMapping {
  camera_label: string | null
  frames_key: string
  timestamps_key: string
  // Signed telemetry-row count: positive N = top N rows are telemetry,
  // negative N = bottom |N| rows, 0 = none.
  telemetry_row_count: number
}

export interface UploadInspectionResponse {
  upload_token: string
  filename: string
  keys: KeyInfo[]
  auto_mapping: CameraMapping[] | null
  auto_mapping_error: string | null
}
