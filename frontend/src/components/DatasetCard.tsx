import type { DatasetSummary } from '../api/types'

function formatDuration(seconds: number | null): string {
  if (seconds == null) return '—'
  return `${seconds.toFixed(1)}s`
}

export function DatasetCard({ dataset, onClick }: { dataset: DatasetSummary; onClick: () => void }) {
  return (
    <button className="dataset-card" onClick={onClick}>
      <img className="dataset-card-thumb" src={dataset.thumbnail_url} alt={dataset.name} loading="lazy" />
      <div className="dataset-card-body">
        <div className="dataset-card-name">{dataset.name}</div>
        <div className="dataset-card-meta">
          {dataset.camera_count > 1 ? `${dataset.camera_count} cameras · ` : ''}
          {dataset.total_frame_count} frames · {formatDuration(dataset.max_duration_seconds)}
        </div>
        {dataset.tags.length > 0 && (
          <div className="tag-list">
            {dataset.tags.map((tag) => (
              <span className="tag-pill" key={tag}>
                {tag}
              </span>
            ))}
          </div>
        )}
      </div>
    </button>
  )
}
