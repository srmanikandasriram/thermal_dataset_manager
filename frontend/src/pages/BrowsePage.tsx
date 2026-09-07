import { useInfiniteQuery } from '@tanstack/react-query'
import { useMemo, useState } from 'react'
import { listDatasets } from '../api/client'
import { DatasetCard } from '../components/DatasetCard'
import { FilterPanel } from '../components/FilterPanel'
import { SearchBar } from '../components/SearchBar'

const PAGE_SIZE = 50

export function BrowsePage({ onSelectDataset }: { onSelectDataset: (id: string) => void }) {
  const [q, setQ] = useState('')
  const [selectedTags, setSelectedTags] = useState<string[]>([])

  const { data, isLoading, error, hasNextPage, fetchNextPage, isFetchingNextPage } = useInfiniteQuery({
    queryKey: ['datasets', q, selectedTags],
    queryFn: ({ pageParam }) =>
      listDatasets({
        q: q || undefined,
        tags: selectedTags.length ? selectedTags : undefined,
        limit: PAGE_SIZE,
        offset: pageParam,
      }),
    initialPageParam: 0,
    getNextPageParam: (lastPage, allPages) => {
      const loaded = allPages.reduce((n, page) => n + page.items.length, 0)
      return loaded < lastPage.total ? loaded : undefined
    },
  })

  const items = useMemo(() => data?.pages.flatMap((page) => page.items) ?? [], [data])
  const total = data?.pages.at(-1)?.total ?? 0

  const availableTags = useMemo(() => {
    const tags = new Set<string>()
    for (const item of items) {
      for (const tag of item.tags) tags.add(tag)
    }
    return Array.from(tags).sort()
  }, [items])

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
      {data && items.length === 0 && <p className="muted">No datasets found. Ingest a capture to get started.</p>}

      {items.length > 0 && (
        <p className="muted browse-count">
          Showing {items.length} of {total}
        </p>
      )}

      <div className="dataset-grid">
        {items.map((dataset) => (
          <DatasetCard key={dataset.dataset_id} dataset={dataset} onClick={() => onSelectDataset(dataset.dataset_id)} />
        ))}
      </div>

      {hasNextPage && (
        <button
          type="button"
          className="load-more-button"
          onClick={() => fetchNextPage()}
          disabled={isFetchingNextPage}
        >
          {isFetchingNextPage ? 'Loading…' : `Load more (${total - items.length} remaining)`}
        </button>
      )}
    </div>
  )
}
