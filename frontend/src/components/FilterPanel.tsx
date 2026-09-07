export function FilterPanel({
  availableTags,
  selectedTags,
  onToggleTag,
}: {
  availableTags: string[]
  selectedTags: string[]
  onToggleTag: (tag: string) => void
}) {
  if (availableTags.length === 0) return null

  return (
    <div className="filter-panel">
      <div className="filter-panel-label">Tags</div>
      <div className="tag-list">
        {availableTags.map((tag) => (
          <button
            key={tag}
            className={`tag-pill tag-filter ${selectedTags.includes(tag) ? 'tag-filter-active' : ''}`}
            onClick={() => onToggleTag(tag)}
          >
            {tag}
          </button>
        ))}
      </div>
    </div>
  )
}
