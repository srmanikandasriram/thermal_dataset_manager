export function ProgressBar({ fraction }: { fraction: number }) {
  const pct = Math.round(Math.min(Math.max(fraction, 0), 1) * 100)
  return (
    <div className="progress-bar" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
      <div className="progress-bar-track">
        <div className="progress-bar-fill" style={{ width: `${pct}%` }} />
      </div>
      <span className="progress-bar-label">{pct}%</span>
    </div>
  )
}
