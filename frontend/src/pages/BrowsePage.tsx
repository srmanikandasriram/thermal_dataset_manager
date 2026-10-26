import { useInfiniteQuery } from '@tanstack/react-query'
import { useMemo, useState } from 'react'
import { listDatasets } from '../api/client'
import type { DatasetSortField, SortOrder } from '../api/types'
import { ColumnPicker } from '../components/ColumnPicker'
import { DatasetCard } from '../components/DatasetCard'
import { DatasetTable } from '../components/DatasetTable'
import { COLUMN_KEYS, DEFAULT_VISIBLE_COLUMNS, type ColumnKey } from '../components/datasetColumns'
import { FilterPanel } from '../components/FilterPanel'
import { SearchBar } from '../components/SearchBar'

const PAGE_SIZE = 50
const VIEW_STORAGE_KEY = 'tdm.browseView'
const COLUMNS_STORAGE_KEY = 'tdm.browseColumns'

type ViewMode = 'grid' | 'list'

function initialView(): ViewMode {
  try {
    return localStorage.getItem(VIEW_STORAGE_KEY) === 'list' ? 'list' : 'grid'
  } catch {
    return 'grid'
  }
}

function initialColumns(): Set<ColumnKey> {
  try {
    const stored = JSON.parse(localStorage.getItem(COLUMNS_STORAGE_KEY) ?? '') as unknown
    if (Array.isArray(stored)) {
      const valid = stored.filter((k): k is ColumnKey => typeof k === 'string' && COLUMN_KEYS.has(k))
      // an explicitly-empty stored list is a valid choice (only Name showing)
      if (valid.length || stored.length === 0) return new Set(valid)
    }
  } catch {
    // fall through to defaults
  }
  return new Set(DEFAULT_VISIBLE_COLUMNS)
}

export function BrowsePage({ onSelectDataset }: { onSelectDataset: (id: string) => void }) {
  const [q, setQ] = useState('')
  const [selectedTags, setSelectedTags] = useState<string[]>([])
  const [view, setView] = useState<ViewMode>(initialView)
  const [sort, setSort] = useState<DatasetSortField>('ingested_at')
  const [order, setOrder] = useState<SortOrder>('desc')
  const [visibleColumns, setVisibleColumns] = useState<Set<ColumnKey>>(initialColumns)

  function chooseView(next: ViewMode) {
    setView(next)
    try {
      localStorage.setItem(VIEW_STORAGE_KEY, next)
    } catch {
      // ignore unavailable storage
    }
  }

  function changeSort(field: DatasetSortField, nextOrder: SortOrder) {
    setSort(field)
    setOrder(nextOrder)
  }

  function toggleColumn(key: ColumnKey, visible: boolean) {
    setVisibleColumns((prev) => {
      const next = new Set(prev)
      if (visible) next.add(key)
      else next.delete(key)
      try {
        localStorage.setItem(COLUMNS_STORAGE_KEY, JSON.stringify([...next]))
      } catch {
        // ignore unavailable storage
      }
      return next
    })
    // if the column driving the current sort is hidden, fall back to default
    if (!visible && key === sort) {
      setSort('ingested_at')
      setOrder('desc')
    }
  }

  const { data, isLoading, error, hasNextPage, fetchNextPage, isFetchingNextPage } = useInfiniteQuery({
    queryKey: ['datasets', q, selectedTags, sort, order],
    queryFn: ({ pageParam }) =>
      listDatasets({
        q: q || undefined,
        tags: selectedTags.length ? selectedTags : undefined,
        sort,
        order,
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
        {view === 'list' && <ColumnPicker visibleColumns={visibleColumns} onToggle={toggleColumn} />}
        <div className="view-toggle" role="group" aria-label="View mode">
          <button
            type="button"
            className={view === 'grid' ? 'active' : ''}
            aria-pressed={view === 'grid'}
            onClick={() => chooseView('grid')}
          >
            Grid
          </button>
          <button
            type="button"
            className={view === 'list' ? 'active' : ''}
            aria-pressed={view === 'list'}
            onClick={() => chooseView('list')}
          >
            List
          </button>
        </div>
      </div>
      <FilterPanel availableTags={availableTags} selectedTags={selectedTags} onToggleTag={toggleTag} />

      {isLoading && <p className="muted">Loading…</p>}
      {error && <p className="error">Failed to load datasets: {(error as Error).message}</p>}
      {data && items.length === 0 && <p className="muted">No datasets found. Ingest a capture with backend/scripts/ingest_cli.py to get started.</p>}

      {items.length > 0 && (
        <p className="muted browse-count">
          Showing {items.length} of {total}
        </p>
      )}

      {items.length > 0 &&
        (view === 'grid' ? (
          <div className="dataset-grid">
            {items.map((dataset) => (
              <DatasetCard
                key={dataset.dataset_id}
                dataset={dataset}
                onClick={() => onSelectDataset(dataset.dataset_id)}
              />
            ))}
          </div>
        ) : (
          <DatasetTable
            datasets={items}
            visibleColumns={visibleColumns}
            sort={sort}
            order={order}
            onSort={changeSort}
            onSelectDataset={onSelectDataset}
          />
        ))}

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
