import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { getExtraArray } from '../api/client'
import type { ExtraArrayMetadata } from '../api/types'

function ExtraArrayRow({ datasetId, info }: { datasetId: string; info: ExtraArrayMetadata }) {
  const [expanded, setExpanded] = useState(false)
  const { data, isLoading, error } = useQuery({
    queryKey: ['extra-array', datasetId, info.key],
    queryFn: () => getExtraArray(datasetId, info.key),
    enabled: expanded,
  })

  return (
    <div className="extra-array-row">
      <button className="extra-array-toggle" onClick={() => setExpanded((e) => !e)}>
        {expanded ? '▾' : '▸'} <code>{info.key}</code>
        <span className="muted"> ({info.shape.join('×')}, {info.dtype})</span>
      </button>
      {expanded && (
        <div className="extra-array-values">
          {isLoading && <p className="muted">Loading…</p>}
          {error && <p className="error">{(error as Error).message}</p>}
          {data && <pre>{JSON.stringify(data.values, null, 1)}</pre>}
        </div>
      )}
    </div>
  )
}

export function ExtraArraysPanel({ datasetId, extraArrays }: { datasetId: string; extraArrays: ExtraArrayMetadata[] }) {
  if (extraArrays.length === 0) return null

  return (
    <div className="extra-arrays-panel">
      <h3>Auxiliary data</h3>
      <p className="muted">
        Arrays from the source file not used as any camera's frames/timestamps, preserved as-is.
      </p>
      {extraArrays.map((info) => (
        <ExtraArrayRow key={info.key} datasetId={datasetId} info={info} />
      ))}
    </div>
  )
}
