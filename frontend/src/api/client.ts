import type {
  CameraMapping,
  DatasetListResponse,
  DatasetMetadata,
  DatasetSearchParams,
  JobStatusResponse,
  UploadInspectionResponse,
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

/**
 * `fetch` doesn't expose upload progress, so this uses XMLHttpRequest
 * directly -- the only way to report bytes-sent for a large file upload.
 */
export function inspectUpload(file: File, onProgress?: (fraction: number) => void): Promise<UploadInspectionResponse> {
  const form = new FormData()
  form.append('file', file)

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('POST', '/api/ingest/upload')

    if (onProgress) {
      xhr.upload.addEventListener('progress', (e) => {
        if (e.lengthComputable) onProgress(e.loaded / e.total)
      })
    }

    xhr.addEventListener('load', () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText) as UploadInspectionResponse)
        } catch {
          reject(new Error('Failed to parse response'))
        }
      } else {
        reject(new Error(`${xhr.status} ${xhr.statusText}: ${xhr.responseText}`))
      }
    })
    xhr.addEventListener('error', () => reject(new Error('Network error during upload')))
    xhr.addEventListener('abort', () => reject(new Error('Upload aborted')))

    xhr.send(form)
  })
}

export function confirmIngest(
  uploadToken: string,
  filename: string,
  mapping: CameraMapping[],
): Promise<JobStatusResponse> {
  return request('/api/ingest/confirm', {
    method: 'POST',
    body: JSON.stringify({ upload_token: uploadToken, filename, mapping }),
  })
}

export function getJobStatus(jobId: string): Promise<JobStatusResponse> {
  return request(`/api/ingest/jobs/${jobId}`)
}

export function cancelUpload(uploadToken: string): Promise<void> {
  return request(`/api/ingest/upload/${uploadToken}`, { method: 'DELETE' })
}

export interface ExtraArrayValues {
  key: string
  shape: number[]
  dtype: string
  values: unknown
}

export function getExtraArray(datasetId: string, key: string): Promise<ExtraArrayValues> {
  return request(`/api/datasets/${datasetId}/extra/${encodeURIComponent(key)}`)
}

export function frameUrl(datasetId: string, cameraIndex: number, frameIndex: number): string {
  return `/api/datasets/${datasetId}/cameras/${cameraIndex}/frames/${frameIndex}.png`
}

export function thumbnailUrl(datasetId: string, cameraIndex: number): string {
  return `/api/datasets/${datasetId}/cameras/${cameraIndex}/thumbnail.png`
}
