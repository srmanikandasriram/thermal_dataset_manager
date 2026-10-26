import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Fragment, useState } from 'react'
import { getDataset, recomputeMetadata, updateDatasetMetadata } from '../api/client'
import { FrameScrubber } from '../components/FrameScrubber'
import { MetadataEditor } from '../components/MetadataEditor'
import type { CameraTelemetryMetadata, UserMetadataUpdate } from '../api/types'

const PENDING_POLL_INTERVAL_MS = 2000

function formatBytes(hash: string): string {
  return `${hash.slice(0, 12)}…`
}

function formatStat(value: number | null, digits = 0): string {
  return value == null ? '—' : value.toFixed(digits)
}

/** Static Boson+ telemetry fields, rendered as <dt>/<dd> pairs inside the
 *  per-camera stats <dl>. Rows with no value are omitted. */
function TelemetryRows({ telemetry }: { telemetry: CameraTelemetryMetadata }) {
  const rows: [string, string | null][] = [
    ['Part number', telemetry.part_number],
    ['Camera serial', telemetry.camera_serial?.toString() ?? null],
    ['Sensor serial', telemetry.sensor_serial?.toString() ?? null],
    ['Firmware', telemetry.software_revision?.join('.') ?? null],
    ['Telemetry rev.', telemetry.telemetry_revision?.toString() ?? null],
    ['Frame rate', telemetry.frame_rate_hz != null ? `${telemetry.frame_rate_hz} Hz` : null],
    ['Gain mode', telemetry.gain_mode],
    ['NUC table', telemetry.nuc_table?.toString() ?? null],
    ['External sync', telemetry.external_sync_slave == null ? null : telemetry.external_sync_slave ? 'slave' : 'free-running'],
    ['Pipeline', telemetry.pipeline_enabled?.length ? telemetry.pipeline_enabled.join(', ') : null],
  ]
  return (
    <>
      {rows
        .filter(([, value]) => value != null)
        .map(([label, value]) => (
          <Fragment key={label}>
            <dt>{label}</dt>
            <dd>{value}</dd>
          </Fragment>
        ))}
    </>
  )
}

export function DatasetDetailPage({ datasetId, onBack }: { datasetId: string; onBack: () => void }) {
  const queryClient = useQueryClient()
  const [activeCameraIndex, setActiveCameraIndex] = useState(0)

  const { data, isLoading, error } = useQuery({
    queryKey: ['dataset', datasetId],
    queryFn: () => getDataset(datasetId),
    // While metadata is still being computed in the background, poll until
    // it flips to "ready" so stats/thumbnails appear without a manual refresh.
    refetchInterval: (query) => (query.state.data?.auto.metadata_status === 'pending' ? PENDING_POLL_INTERVAL_MS : false),
  })

  const mutation = useMutation({
    mutationFn: (update: UserMetadataUpdate) => updateDatasetMetadata(datasetId, update),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['dataset', datasetId] })
      queryClient.invalidateQueries({ queryKey: ['datasets'] })
    },
  })

  const recompute = useMutation({
    mutationFn: () => recomputeMetadata(datasetId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['dataset', datasetId] })
      queryClient.invalidateQueries({ queryKey: ['datasets'] })
    },
  })

  if (isLoading) return <p className="muted">Loading…</p>
  if (error) return <p className="error">Failed to load dataset: {(error as Error).message}</p>
  if (!data) return null

  const { auto, user } = data
  const activeCamera = auto.cameras.find((c) => c.camera_index === activeCameraIndex) ?? auto.cameras[0]
  const pending = auto.metadata_status === 'pending'

  return (
    <div className="detail-page">
      <button className="back-link" onClick={onBack}>
        ← Back to browse
      </button>

      {pending && (
        <p className="metadata-pending-banner">Computing metadata (stats, thumbnail)… this updates automatically.</p>
      )}

      <div className="detail-layout">
        <div className="detail-metadata">
          <MetadataEditor user={user} saving={mutation.isPending} onSave={(update) => mutation.mutate(update)} />
          {mutation.isError && <p className="error">Save failed: {(mutation.error as Error).message}</p>}

          <div className="auto-metadata">
            <div className="auto-metadata-header">
              <h3>Capture info</h3>
              <button
                className="recompute-button"
                disabled={recompute.isPending}
                onClick={() => recompute.mutate()}
              >
                {recompute.isPending ? 'Recomputing…' : 'Recompute metadata'}
              </button>
            </div>
            {recompute.isError && <p className="error">Recompute failed: {(recompute.error as Error).message}</p>}
            <dl>
              <dt>Source file</dt>
              <dd>{auto.source_filename}</dd>
              {data.user.capture_id && (
                <>
                  <dt>Capture</dt>
                  <dd>{data.user.capture_id}</dd>
                </>
              )}
              <dt>Source format</dt>
              <dd>{auto.source_format}</dd>
              <dt>Ingested</dt>
              <dd>{new Date(auto.ingested_at).toLocaleString()}</dd>
              <dt>Checksum</dt>
              <dd title={auto.source_sha256}>{formatBytes(auto.source_sha256)}</dd>
              <dt>Cameras</dt>
              <dd>{auto.camera_count}</dd>
            </dl>
          </div>

          {auto.other_variables && (
            <div className="other-variables">
              <h3>Other variables in source file (not ingested)</h3>
              <pre>{auto.other_variables}</pre>
            </div>
          )}
        </div>

        <div className="detail-preview">
          {auto.cameras.length > 1 && (
            <div className="camera-tabs">
              {auto.cameras.map((cam) => (
                <button
                  key={cam.camera_index}
                  className={cam.camera_index === activeCameraIndex ? 'camera-tab-active' : ''}
                  onClick={() => setActiveCameraIndex(cam.camera_index)}
                >
                  Camera {cam.camera_index}
                </button>
              ))}
            </div>
          )}

          {activeCamera && (
            <>
              <FrameScrubber datasetId={datasetId} camera={activeCamera} />
              <dl className="camera-stats">
                <dt>Resolution</dt>
                <dd>
                  {activeCamera.frame_width}×{activeCamera.frame_height}
                </dd>
                <dt>Frames</dt>
                <dd>{activeCamera.frame_count}</dd>
                <dt>FPS (est.)</dt>
                <dd>{formatStat(activeCamera.fps_estimate, 2)}</dd>
                <dt>Duration</dt>
                <dd>{activeCamera.duration_seconds != null ? `${activeCamera.duration_seconds.toFixed(2)}s` : '—'}</dd>
                <dt>Counts range</dt>
                <dd>
                  {activeCamera.counts_min == null ? '—' : `${activeCamera.counts_min} – ${activeCamera.counts_max}`}
                </dd>
                {activeCamera.telemetry && <TelemetryRows telemetry={activeCamera.telemetry} />}
              </dl>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
