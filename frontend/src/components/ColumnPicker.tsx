import { COLUMNS, type ColumnKey } from './datasetColumns'

/** Dropdown of checkboxes for toggling optional table columns. Locked columns
 *  (Name) are not offered. Native <details> so it needs no outside-click glue. */
export function ColumnPicker({
  visibleColumns,
  onToggle,
}: {
  visibleColumns: ReadonlySet<ColumnKey>
  onToggle: (key: ColumnKey, visible: boolean) => void
}) {
  const toggleable = COLUMNS.filter((c) => !c.locked)
  const extraCount = toggleable.filter((c) => visibleColumns.has(c.key)).length

  return (
    <details className="column-picker">
      <summary>Columns ({extraCount})</summary>
      <div className="column-picker-menu">
        {toggleable.map((column) => (
          <label key={column.key}>
            <input
              type="checkbox"
              checked={visibleColumns.has(column.key)}
              onChange={(e) => onToggle(column.key, e.target.checked)}
            />
            {column.label}
          </label>
        ))}
      </div>
    </details>
  )
}
