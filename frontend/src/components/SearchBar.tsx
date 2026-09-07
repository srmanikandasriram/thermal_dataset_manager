import { useEffect, useState } from 'react'

export function SearchBar({ value, onChange }: { value: string; onChange: (q: string) => void }) {
  const [draft, setDraft] = useState(value)

  useEffect(() => {
    const timeout = setTimeout(() => onChange(draft), 300)
    return () => clearTimeout(timeout)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [draft])

  return (
    <input
      className="search-bar"
      type="search"
      placeholder="Search name, description, notes…"
      value={draft}
      onChange={(e) => setDraft(e.target.value)}
    />
  )
}
