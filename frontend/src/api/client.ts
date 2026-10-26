import type {
  DatasetListResponse,
  DatasetMetadata,
  DatasetSearchParams,
  UserMetadataUpdate,
} from './types'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    headers: init?.body instanceof FormData ? undefined : { 'Content-Type': 'application/json' },
    ...init,
  })
  if (!res.ok) {
    const body = await res.text()
    throw new Error(`${res.status} ${res.statusText}: ${body}`)
  }
  return res.json() as Promise<T>
}

export function listDatasets(params: DatasetSearchParams): Promise<DatasetListResponse> {
  const search = new URLSearchParams()
  if (params.q) search.set('q', params.q)
  if (params.tags?.length) search.set('tags', params.tags.join(','))
  if (params.date_from) search.set('date_from', params.date_from)
  if (params.date_to) search.set('date_to', params.date_to)
  if (params.capture_id) search.set('capture_id', params.capture_id)
  if (params.sort) search.set('sort', params.sort)
  if (params.order) search.set('order', params.order)
  search.set('limit', String(params.limit ?? 50))
  search.set('offset', String(params.offset ?? 0))
  return request(`/api/datasets?${search.toString()}`)
}

export function getDataset(datasetId: string): Promise<DatasetMetadata> {
  return request(`/api/datasets/${datasetId}`)
}

export function updateDatasetMetadata(datasetId: string, update: UserMetadataUpdate): Promise<DatasetMetadata> {
  return request(`/api/datasets/${datasetId}/metadata`, {
    method: 'PATCH',
    body: JSON.stringify(update),
  })
}

export function recomputeMetadata(datasetId: string): Promise<DatasetMetadata> {
  return request(`/api/datasets/${datasetId}/recompute-metadata`, { method: 'POST' })
}

export function frameUrl(datasetId: string, cameraIndex: number, frameIndex: number): string {
  return `/api/datasets/${datasetId}/cameras/${cameraIndex}/frames/${frameIndex}.png`
}

export function thumbnailUrl(datasetId: string, cameraIndex: number): string {
  return `/api/datasets/${datasetId}/cameras/${cameraIndex}/thumbnail.png`
}
