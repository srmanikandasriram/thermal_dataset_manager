import { useEffect, useRef, useState } from 'react'
import { cancelUpload, confirmIngest, getJobStatus, inspectUpload } from '../api/client'
import { KeyMappingEditor } from '../components/KeyMappingEditor'
import { ProgressBar } from '../components/ProgressBar'
import type { CameraMapping, JobStatusResponse, UploadInspectionResponse } from '../api/types'

const POLL_INTERVAL_MS = 1000

type Stage = 'idle' | 'uploading' | 'inspecting' | 'review' | 'ingesting' | 'error'

function mappingIsComplete(mapping: CameraMapping[]): boolean {
  return mapping.length > 0 && mapping.every((row) => row.frames_key && row.timestamps_key)
}

/**
 * A dropped folder shows up in `DataTransfer.files` as a bogus zero-byte
 * "file" -- uploading it doesn't fail cleanly, it just breaks confusingly
 * partway through. `DataTransferItem.webkitGetAsEntry()` (despite the
 * name, supported in all major browsers) is the only way to actually tell
 * a dropped file from a dropped directory before attempting the upload.
 */
function pickDroppedFile(dataTransfer: DataTransfer): { file: File } | { error: string } {
  const item = dataTransfer.items?.[0]
  if (item && item.kind === 'file') {
    const entry = typeof item.webkitGetAsEntry === 'function' ? item.webkitGetAsEntry() : null
    if (entry?.isDirectory) {
      return { error: 'That\'s a folder — drop a single .npz, .npz.zst, or .h5 file instead.' }
    }
    const file = item.getAsFile()
    if (file) return { file }
  }
  const file = dataTransfer.files[0]
  return file ? { file } : { error: 'No file found in the dropped item.' }
}

export function IngestPage({ onIngested }: { onIngested: (datasetId: string) => void }) {
  const [stage, setStage] = useState<Stage>('idle')
  const [dragOver, setDragOver] = useState(false)
  const [inspection, setInspection] = useState<UploadInspectionResponse | null>(null)
  const [mapping, setMapping] = useState<CameraMapping[]>([])
  const [manualMode, setManualMode] = useState(false)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [job, setJob] = useState<JobStatusResponse | null>(null)
  const [uploadProgress, setUploadProgress] = useState(0)
  const pollRef = useRef<number | null>(null)
  // Tracks a staged upload that hasn't been confirmed (or cancelled) yet,
  // so it can be cleaned up if the user navigates away mid-review instead
  // of clicking Cancel or Ingest.
  const pendingUploadTokenRef = useRef<string | null>(null)

  useEffect(() => {
    return () => {
      if (pendingUploadTokenRef.current) {
        cancelUpload(pendingUploadTokenRef.current).catch(() => {})
      }
    }
  }, [])

  function reset() {
    setStage('idle')
    setInspection(null)
    setMapping([])
    setManualMode(false)
    setErrorMessage(null)
    setJob(null)
    setUploadProgress(0)
    pendingUploadTokenRef.current = null
    if (pollRef.current) window.clearInterval(pollRef.current)
  }

  function handleCancelReview() {
    if (inspection) {
      cancelUpload(inspection.upload_token).catch(() => {})
      pendingUploadTokenRef.current = null
    }
    reset()
  }

  async function handleFile(file: File) {
    setStage('uploading')
    setUploadProgress(0)
    setErrorMessage(null)
    try {
      const result = await inspectUpload(file, (fraction) => {
        setUploadProgress(fraction)
        // The last stretch is the server reading the file's keys, with no
        // progress signal of its own -- switch the label once bytes are sent.
        if (fraction >= 1) setStage('inspecting')
      })
      setInspection(result)
      setMapping(result.auto_mapping ?? [])
      setManualMode(result.auto_mapping === null)
      pendingUploadTokenRef.current = result.upload_token
      setStage('review')
    } catch (e) {
      setErrorMessage((e as Error).message)
      setStage('error')
    }
  }

  function pollJob(jobId: string) {
    pollRef.current = window.setInterval(async () => {
      const status = await getJobStatus(jobId)
      setJob(status)
      if (status.status === 'succeeded' || status.status === 'failed') {
        if (pollRef.current) window.clearInterval(pollRef.current)
        if (status.status === 'succeeded' && status.dataset_id) {
          onIngested(status.dataset_id)
        }
      }
    }, POLL_INTERVAL_MS)
  }

  async function handleConfirm() {
    if (!inspection) return
    setStage('ingesting')
    // Ownership of the staged upload passes to /confirm's background job
    // from here -- it cleans up its own staging dir, so stop tracking it.
    pendingUploadTokenRef.current = null
    try {
      const status = await confirmIngest(inspection.upload_token, inspection.filename, mapping)
      setJob(status)
      pollJob(status.job_id)
    } catch (e) {
      setErrorMessage((e as Error).message)
      setStage('error')
    }
  }

  return (
    <div className="ingest-page">
      {stage === 'idle' && (
        <div
          className={`dropzone ${dragOver ? 'dropzone-active' : ''}`}
          onDragOver={(e) => {
            e.preventDefault()
            setDragOver(true)
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault()
            setDragOver(false)
            const result = pickDroppedFile(e.dataTransfer)
            if ('error' in result) {
              setErrorMessage(result.error)
              setStage('error')
              return
            }
            handleFile(result.file)
          }}
        >
          <p>Drag and drop a .npz, .npz.zst, or .h5 capture here</p>
          <p className="muted">or</p>
          <label className="file-picker">
            Choose file
            <input
              type="file"
              accept=".npz,.zst,.h5,.hdf5"
              onChange={(e) => {
                const file = e.target.files?.[0]
                if (file) handleFile(file)
              }}
              hidden
            />
          </label>
        </div>
      )}

      {stage === 'uploading' && (
        <div className="upload-progress">
          <p className="muted">Uploading…</p>
          <ProgressBar fraction={uploadProgress} />
        </div>
      )}

      {stage === 'inspecting' && <p className="muted">Reading file…</p>}

      {stage === 'review' && inspection && (
        <div className="ingest-review">
          <p>
            <strong>{inspection.filename}</strong> — found {inspection.keys.length} key(s).
          </p>

          {inspection.auto_mapping ? (
            <p>
              Auto-detected {inspection.auto_mapping.length} camera
              {inspection.auto_mapping.length > 1 ? 's' : ''}:{' '}
              {inspection.auto_mapping.map((m) => m.camera_label ?? '(single)').join(', ')}
            </p>
          ) : (
            <p className="error">Couldn't auto-detect a camera mapping: {inspection.auto_mapping_error}</p>
          )}

          <label className="manual-mapping-toggle">
            <input
              type="checkbox"
              checked={manualMode}
              disabled={!inspection.auto_mapping}
              onChange={(e) => {
                setManualMode(e.target.checked)
                // Revert any in-progress manual edits when switching back to auto-detected.
                if (!e.target.checked) setMapping(inspection.auto_mapping ?? [])
              }}
            />
            Map keys manually
          </label>

          {manualMode && (
            <KeyMappingEditor keys={inspection.keys} mapping={mapping} onChange={setMapping} />
          )}

          <div className="ingest-review-actions">
            <button onClick={handleCancelReview}>Cancel</button>
            <button disabled={!mappingIsComplete(mapping)} onClick={handleConfirm}>
              Ingest
            </button>
          </div>
        </div>
      )}

      {stage === 'ingesting' && job && job.status !== 'succeeded' && job.status !== 'failed' && (
        <p className="muted">Ingesting ({job.status})…</p>
      )}

      {stage === 'ingesting' && job?.status === 'succeeded' && <p>Ingest complete — opening dataset…</p>}

      {(stage === 'error' || job?.status === 'failed') && (
        <div className="error">
          <p>{job?.status === 'failed' ? 'Ingest failed:' : 'Something went wrong:'}</p>
          <ul>{(job?.errors.length ? job.errors : [errorMessage ?? 'Unknown error']).map((err, i) => <li key={i}>{err}</li>)}</ul>
          <button onClick={reset}>Try again</button>
        </div>
      )}
    </div>
  )
}
