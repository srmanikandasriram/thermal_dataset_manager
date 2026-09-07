import { useQuery } from '@tanstack/react-query'
import { useMemo, useState } from 'react'
import { listDatasets } from '../api/client'
import { DatasetCard } from '../components/DatasetCard'
import { FilterPanel } from '../components/FilterPanel'
import { SearchBar } from '../components/SearchBar'

export function BrowsePage({ onSelectDataset }: { onSelectDataset: (id: string) => void }) {
  const [q, setQ] = useState('')
  const [selectedTags, setSelectedTags] = useState<string[]>([])

  const { data, isLoading, error } = useQuery({
    queryKey: ['datasets', q, selectedTags],
    queryFn: () => listDatasets({ q: q || undefined, tags: selectedTags.length ? selectedTags : undefined }),
  })

  const availableTags = useMemo(() => {
    const tags = new Set<string>()
    for (const item of data?.items ?? []) {
      for (const tag of item.tags) tags.add(tag)
    }
    return Array.from(tags).sort()
  }, [data])

  function toggleTag(tag: string) {
    setSelectedTags((prev) => (prev.includes(tag) ? prev.filter((t) => t !== tag) : [...prev, tag]))
  }

  return (
    <div className="browse-page">
      <div className="browse-page-toolbar">
        <SearchBar value={q} onChange={setQ} />
      </div>
      <FilterPanel availableTags={availableTags} selectedTags={selectedTags} onToggleTag={toggleTag} />

      {isLoading && <p className="muted">Loading…</p>}
      {error && <p className="error">Failed to load datasets: {(error as Error).message}</p>}
      {data && data.items.length === 0 && <p className="muted">No datasets found. Ingest a capture to get started.</p>}

      <div className="dataset-grid">
        {data?.items.map((dataset) => (
          <DatasetCard key={dataset.dataset_id} dataset={dataset} onClick={() => onSelectDataset(dataset.dataset_id)} />
        ))}
      </div>
    </div>
  )
}
