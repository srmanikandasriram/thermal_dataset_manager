import { useEffect, useRef, useState } from 'react'
import { frameUrl } from '../api/client'
import type { CameraStreamMetadata } from '../api/types'

const PREFETCH_WINDOW = 5

export function FrameScrubber({ datasetId, camera }: { datasetId: string; camera: CameraStreamMetadata }) {
  const [frameIndex, setFrameIndex] = useState(0)
  const [playing, setPlaying] = useState(false)
  const intervalRef = useRef<number | null>(null)

  const lastFrame = Math.max(camera.frame_count - 1, 0)

  useEffect(() => {
    // Prefetch a small neighboring window so scrubbing/playback feels smooth
    // (the browser's own image cache does the rest).
    for (let offset = -PREFETCH_WINDOW; offset <= PREFETCH_WINDOW; offset++) {
      const idx = frameIndex + offset
      if (idx < 0 || idx > lastFrame) continue
      new Image().src = frameUrl(datasetId, camera.camera_index, idx)
    }
  }, [frameIndex, datasetId, camera.camera_index, lastFrame])

  useEffect(() => {
    if (!playing) {
      if (intervalRef.current) window.clearInterval(intervalRef.current)
      return
    }
    const fps = camera.fps_estimate && camera.fps_estimate > 0 ? camera.fps_estimate : 10
    intervalRef.current = window.setInterval(() => {
      setFrameIndex((prev) => (prev >= lastFrame ? 0 : prev + 1))
    }, 1000 / fps)
    return () => {
      if (intervalRef.current) window.clearInterval(intervalRef.current)
    }
  }, [playing, camera.fps_estimate, lastFrame])

  if (camera.frame_count === 0) {
    return <div className="frame-scrubber-empty">No frames.</div>
  }

  return (
    <div className="frame-scrubber">
      <img
        className="frame-scrubber-image"
        src={frameUrl(datasetId, camera.camera_index, frameIndex)}
        alt={`Frame ${frameIndex}`}
      />
      <div className="frame-scrubber-controls">
        <button onClick={() => setPlaying((p) => !p)}>{playing ? 'Pause' : 'Play'}</button>
        <input
          type="range"
          min={0}
          max={lastFrame}
          value={frameIndex}
          onChange={(e) => {
            setPlaying(false)
            setFrameIndex(Number(e.target.value))
          }}
        />
        <span className="frame-scrubber-counter">
          {frameIndex + 1} / {camera.frame_count}
        </span>
      </div>
    </div>
  )
}
