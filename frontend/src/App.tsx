import { useState } from 'react'
import { BrowsePage } from './pages/BrowsePage'
import { DatasetDetailPage } from './pages/DatasetDetailPage'
import { IngestPage } from './pages/IngestPage'

type View = { name: 'browse' } | { name: 'detail'; datasetId: string } | { name: 'ingest' }

function App() {
  const [view, setView] = useState<View>({ name: 'browse' })

  return (
    <div className="app">
      <header className="app-header">
        <h1 onClick={() => setView({ name: 'browse' })}>Thermal Dataset Manager</h1>
        <nav>
          <button onClick={() => setView({ name: 'browse' })}>Browse</button>
          <button onClick={() => setView({ name: 'ingest' })}>Ingest capture</button>
        </nav>
      </header>

      <main>
        {view.name === 'browse' && (
          <BrowsePage onSelectDataset={(datasetId) => setView({ name: 'detail', datasetId })} />
        )}
        {view.name === 'detail' && (
          <DatasetDetailPage datasetId={view.datasetId} onBack={() => setView({ name: 'browse' })} />
        )}
        {view.name === 'ingest' && (
          <IngestPage onIngested={(datasetId) => setView({ name: 'detail', datasetId })} />
        )}
      </main>
    </div>
  )
}

export default App
