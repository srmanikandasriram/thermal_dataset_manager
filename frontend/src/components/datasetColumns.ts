import type { DatasetSortField, DatasetSummary, SortOrder } from '../api/types'

export type ColumnKey = DatasetSortField | 'tags' | 'dataset_id'

export type DatasetColumn = {
  key: ColumnKey
  label: string
  numeric?: boolean
  mono?: boolean
  /** false = display-only, no header sort control */
  sortable: boolean
  /** shown out of the box; others are opt-in via the column picker */
  defaultVisible: boolean
  /** always shown, not offered in the picker */
  locked?: boolean
  render: (d: DatasetSummary) => string | number
}

function formatDuration(seconds: number | null): string {
  return seconds == null ? '—' : `${seconds.toFixed(1)}s`
}

export const COLUMNS: DatasetColumn[] = [
  { key: 'name', label: 'Name', sortable: true, defaultVisible: true, locked: true, render: (d) => d.name },
  {
    key: 'camera_count',
    label: 'Cameras',
    numeric: true,
    sortable: true,
    defaultVisible: true,
    render: (d) => d.camera_count,
  },
  {
    key: 'total_frame_count',
    label: 'Frames',
    numeric: true,
    sortable: true,
    defaultVisible: true,
    render: (d) => d.total_frame_count.toLocaleString(),
  },
  {
    key: 'max_duration_seconds',
    label: 'Duration',
    numeric: true,
    sortable: true,
    defaultVisible: true,
    render: (d) => formatDuration(d.max_duration_seconds),
  },
  { key: 'scene_category', label: 'Scene', sortable: true, defaultVisible: true, render: (d) => d.scene_category ?? '—' },
  {
    key: 'capture_location',
    label: 'Location',
    sortable: true,
    defaultVisible: true,
    render: (d) => d.capture_location ?? '—',
  },
  {
    key: 'ingested_at',
    label: 'Ingested',
    sortable: true,
    defaultVisible: true,
    render: (d) => new Date(d.ingested_at).toLocaleDateString(),
  },
  {
    key: 'description',
    label: 'Description',
    sortable: true,
    defaultVisible: false,
    render: (d) => d.description || '—',
  },
  {
    key: 'source_filename',
    label: 'Source file',
    sortable: true,
    defaultVisible: false,
    render: (d) => d.source_filename || '—',
  },
  {
    key: 'capture_id',
    label: 'Capture',
    sortable: true,
    defaultVisible: false,
    render: (d) => d.capture_id || '—',
  },
  {
    key: 'tags',
    label: 'Tags',
    sortable: false,
    defaultVisible: false,
    render: (d) => (d.tags.length ? d.tags.join(', ') : '—'),
  },
  {
    key: 'dataset_id',
    label: 'Dataset ID',
    mono: true,
    sortable: false,
    defaultVisible: false,
    render: (d) => d.dataset_id,
  },
]

export const COLUMN_KEYS: ReadonlySet<string> = new Set(COLUMNS.map((c) => c.key))

export const DEFAULT_VISIBLE_COLUMNS: ColumnKey[] = COLUMNS.filter((c) => c.defaultVisible).map((c) => c.key)

/** Default sort direction when a column is first clicked: text ascending,
 *  numbers and dates descending (largest / newest first). */
export function defaultSortOrder(column: DatasetColumn): SortOrder {
  return column.numeric || column.key === 'ingested_at' ? 'desc' : 'asc'
}
