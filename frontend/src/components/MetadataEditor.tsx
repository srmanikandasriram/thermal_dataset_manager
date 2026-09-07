import { useState } from 'react'
import type { UserMetadata, UserMetadataUpdate } from '../api/types'

export function MetadataEditor({
  user,
  saving,
  onSave,
}: {
  user: UserMetadata
  saving: boolean
  onSave: (update: UserMetadataUpdate) => void
}) {
  const [name, setName] = useState(user.name)
  const [description, setDescription] = useState(user.description)
  const [tagsText, setTagsText] = useState(user.tags.join(', '))
  const [notes, setNotes] = useState(user.notes)
  const [captureLocation, setCaptureLocation] = useState(user.capture_location ?? '')
  const [sceneCategory, setSceneCategory] = useState(user.scene_category ?? '')

  const dirty =
    name !== user.name ||
    description !== user.description ||
    tagsText !== user.tags.join(', ') ||
    notes !== user.notes ||
    captureLocation !== (user.capture_location ?? '') ||
    sceneCategory !== (user.scene_category ?? '')

  function handleSave() {
    onSave({
      name,
      description,
      tags: tagsText
        .split(',')
        .map((t) => t.trim())
        .filter(Boolean),
      notes,
      capture_location: captureLocation || null,
      scene_category: sceneCategory || null,
    })
  }

  return (
    <div className="metadata-editor">
      <label>
        Name
        <input value={name} onChange={(e) => setName(e.target.value)} />
      </label>
      <label>
        Description
        <textarea value={description} onChange={(e) => setDescription(e.target.value)} rows={2} />
      </label>
      <label>
        Tags (comma-separated)
        <input value={tagsText} onChange={(e) => setTagsText(e.target.value)} />
      </label>
      <div className="metadata-editor-row">
        <label>
          Capture location
          <input value={captureLocation} onChange={(e) => setCaptureLocation(e.target.value)} />
        </label>
        <label>
          Scene category
          <input value={sceneCategory} onChange={(e) => setSceneCategory(e.target.value)} />
        </label>
      </div>
      <label>
        Notes
        <textarea value={notes} onChange={(e) => setNotes(e.target.value)} rows={3} />
      </label>
      <button disabled={!dirty || saving} onClick={handleSave}>
        {saving ? 'Saving…' : 'Save changes'}
      </button>
    </div>
  )
}
