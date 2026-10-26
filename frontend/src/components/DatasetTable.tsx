import type { DatasetSortField, DatasetSummary, SortOrder } from '../api/types'
import { COLUMNS, type ColumnKey, type DatasetColumn, defaultSortOrder } from './datasetColumns'

export function DatasetTable({
  datasets,
  visibleColumns,
  sort,
  order,
  onSort,
  onSelectDataset,
}: {
  datasets: DatasetSummary[]
  visibleColumns: ReadonlySet<ColumnKey>
  sort: DatasetSortField
  order: SortOrder
  onSort: (field: DatasetSortField, order: SortOrder) => void
  onSelectDataset: (id: string) => void
}) {
  const columns = COLUMNS.filter((c) => c.locked || visibleColumns.has(c.key))

  function headerClick(column: DatasetColumn) {
    if (!column.sortable) return
    const field = column.key as DatasetSortField
    if (field === sort) {
      onSort(field, order === 'asc' ? 'desc' : 'asc')
    } else {
      onSort(field, defaultSortOrder(column))
    }
  }

  return (
    <div className="dataset-table-wrap">
      <table className="dataset-table">
        <thead>
          <tr>
            {columns.map((column) => {
              const active = column.sortable && column.key === sort
              return (
                <th
                  key={column.key}
                  className={column.numeric ? 'numeric' : undefined}
                  aria-sort={active ? (order === 'asc' ? 'ascending' : 'descending') : 'none'}
                >
                  {column.sortable ? (
                    <button type="button" onClick={() => headerClick(column)}>
                      {column.label}
                      <span className="sort-caret">{active ? (order === 'asc' ? ' ▲' : ' ▼') : ''}</span>
                    </button>
                  ) : (
                    column.label
                  )}
                </th>
              )
            })}
          </tr>
        </thead>
        <tbody>
          {datasets.map((dataset) => (
            <tr
              key={dataset.dataset_id}
              tabIndex={0}
              onClick={() => onSelectDataset(dataset.dataset_id)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') onSelectDataset(dataset.dataset_id)
              }}
            >
              {columns.map((column) => (
                <td
                  key={column.key}
                  className={[column.numeric && 'numeric', column.mono && 'mono'].filter(Boolean).join(' ') || undefined}
                >
                  {column.render(dataset)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
