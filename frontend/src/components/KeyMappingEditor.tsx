import type { CameraMapping, KeyInfo } from '../api/types'

function formatKeyLabel(key: KeyInfo): string {
  return `${key.key}  (${key.shape.join('×')}, ${key.dtype})`
}

const TELEMETRY_OPTIONS: { value: number; label: string }[] = [
  { value: 2, label: 'Top 2 rows' },
  { value: 0, label: 'None' },
  { value: -2, label: 'Bottom 2 rows' },
]

export const DEFAULT_TELEMETRY_ROW_COUNT = 2

/** Per-camera telemetry-row selector. Signed: +N top rows, -N bottom rows, 0 none. */
export function TelemetryRowsSelect({
  value,
  onChange,
}: {
  value: number | null | undefined
  onChange: (value: number) => void
}) {
  return (
    <select value={value ?? DEFAULT_TELEMETRY_ROW_COUNT} onChange={(e) => onChange(Number(e.target.value))}>
      {TELEMETRY_OPTIONS.map((opt) => (
        <option key={opt.value} value={opt.value}>
          {opt.label}
        </option>
      ))}
    </select>
  )
}

export function KeyMappingEditor({
  keys,
  mapping,
  onChange,
}: {
  keys: KeyInfo[]
  mapping: CameraMapping[]
  onChange: (mapping: CameraMapping[]) => void
}) {
  function updateRow(index: number, patch: Partial<CameraMapping>) {
    onChange(mapping.map((row, i) => (i === index ? { ...row, ...patch } : row)))
  }

  function removeRow(index: number) {
    onChange(mapping.filter((_, i) => i !== index))
  }

  function addRow() {
    onChange([
      ...mapping,
      { camera_label: null, frames_key: '', timestamps_key: '', telemetry_row_count: DEFAULT_TELEMETRY_ROW_COUNT },
    ])
  }

  function moveRow(index: number, direction: -1 | 1) {
    const target = index + direction
    if (target < 0 || target >= mapping.length) return
    const next = [...mapping]
    ;[next[index], next[target]] = [next[target], next[index]]
    onChange(next)
  }

  return (
    <div className="key-mapping-editor">
      <p className="key-mapping-note">
        Row order sets each camera's internal index (0, 1, 2…) — the storage path and API use that index, not
        the label below. If you're comparing this dataset against others later, rely on the <strong>camera
        label</strong> to identify which physical camera is which, not its position here — the same camera can
        end up at a different index in a different capture depending on row order.
      </p>
      <table>
        <thead>
          <tr>
            <th />
            <th>Camera label</th>
            <th>Frames key</th>
            <th>Timestamps key</th>
            <th>Telemetry rows</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {mapping.map((row, i) => (
            <tr key={i}>
              <td className="key-mapping-order">
                <span className="key-mapping-index">{i}</span>
                <span className="key-mapping-reorder">
                  <button
                    type="button"
                    className="icon-button"
                    onClick={() => moveRow(i, -1)}
                    disabled={i === 0}
                    aria-label="Move camera up"
                  >
                    ↑
                  </button>
                  <button
                    type="button"
                    className="icon-button"
                    onClick={() => moveRow(i, 1)}
                    disabled={i === mapping.length - 1}
                    aria-label="Move camera down"
                  >
                    ↓
                  </button>
                </span>
              </td>
              <td>
                <input
                  placeholder="e.g. A"
                  value={row.camera_label ?? ''}
                  onChange={(e) => updateRow(i, { camera_label: e.target.value || null })}
                />
              </td>
              <td>
                <select value={row.frames_key} onChange={(e) => updateRow(i, { frames_key: e.target.value })}>
                  <option value="">Select key…</option>
                  {keys.map((k) => (
                    <option key={k.key} value={k.key}>
                      {formatKeyLabel(k)}
                    </option>
                  ))}
                </select>
              </td>
              <td>
                <select
                  value={row.timestamps_key}
                  onChange={(e) => updateRow(i, { timestamps_key: e.target.value })}
                >
                  <option value="">Select key…</option>
                  {keys.map((k) => (
                    <option key={k.key} value={k.key}>
                      {formatKeyLabel(k)}
                    </option>
                  ))}
                </select>
              </td>
              <td>
                <TelemetryRowsSelect
                  value={row.telemetry_row_count}
                  onChange={(v) => updateRow(i, { telemetry_row_count: v })}
                />
              </td>
              <td>
                <button type="button" className="icon-button" onClick={() => removeRow(i)} aria-label="Remove camera">
                  ✕
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <button type="button" onClick={addRow}>
        + Add camera
      </button>
    </div>
  )
}
