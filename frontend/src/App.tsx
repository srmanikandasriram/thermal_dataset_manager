import { useState } from 'react'
import { BrowsePage } from './pages/BrowsePage'
import { DatasetDetailPage } from './pages/DatasetDetailPage'

type View = { name: 'browse' } | { name: 'detail'; datasetId: string }

function App() {
  const [view, setView] = useState<View>({ name: 'browse' })
  const goHome = () => setView({ name: 'browse' })

  return (
    <div className="app">
      <header className="app-header">
        <h1
          onClick={goHome}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault()
              goHome()
            }
          }}
          role="link"
          tabIndex={0}
          title="Back to browse"
        >
          Thermal Dataset Manager
        </h1>
      </header>

      <main>
        {view.name === 'browse' && (
          <BrowsePage onSelectDataset={(datasetId) => setView({ name: 'detail', datasetId })} />
        )}
        {view.name === 'detail' && (
          <DatasetDetailPage datasetId={view.datasetId} onBack={() => setView({ name: 'browse' })} />
        )}
      </main>
    </div>
  )
}

export default App
