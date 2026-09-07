# thermal_dataset_manager

A lightweight browser based app to manage, preview, organize raw thermal video captures.

Raw captures (`.npz`, `.npz.zst`, `.h5`) are ingested, validated, and converted into a canonical
[Zarr](https://zarr.dev/) layout with per-dataset metadata — so downstream tooling has one
uniform format to read instead of bespoke loaders per source file. Datasets are immutable once
ingested; only the user-editable half of the metadata (name, description, tags, notes...) can be
changed afterward.

## Layout

- `backend/thermal_dataset_manager/` — FastAPI app: ingest pipeline, Zarr storage, SQLite search
  catalog, HTTP API.
- `backend/scripts/` — command-line tools, usable without the API/UI running (see below).
- `frontend/` — React + Vite + TypeScript browse/search/preview/edit UI.

## Setup

Requires Python >= 3.10 and Node >= 18.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cd frontend
npm install
```

Copy `.env.example` to `.env` and adjust `THERMAL_DATA_ROOT` (where ingested datasets live on
disk). `THERMAL_CATALOG_DB` (the SQLite search index) is optional — it defaults to
`<THERMAL_DATA_ROOT>/catalog.sqlite`, so it can't silently end up pointing somewhere else than
your data. The catalog is a derived index (never a source of truth) and is safe to delete; rebuild
it with `python backend/scripts/rebuild_catalog_cli.py` or `POST /api/admin/rebuild-catalog`,
either of which rescans every dataset's `metadata.json` on disk.

## Ingesting a capture

Without any server running:

```bash
source .venv/bin/activate
python backend/scripts/ingest_cli.py path/to/capture.npz [more_captures...]
```

Each source file's arrays are matched by name against the registry in
`backend/thermal_dataset_manager/ingest/key_registry.py` (currently `raw_thr_frames`/
`raw_thr_tstamps`, optionally suffixed `_N` per camera for multi-camera captures — letter
suffixes `_A`/`_B` are accepted and normalized to `_0`/`_1`). Add a new regex pair there to
support another naming convention — no other pipeline code needs to change.

Ingest is deduplicated by the SHA-256 of the **raw source file bytes** (computed before the file
is decoded, so the camera-key mapping/normalization above does not affect it). Submitting a file
byte-identical to an already-ingested capture is rejected with an error naming the existing
dataset (id, name, source filename, ingest time) rather than creating a redundant copy.

## Maintenance

All three are safe to run at any time (they never touch the immutable raw zarr data) and work
without a server running:

```bash
source .venv/bin/activate

# Rebuild the search catalog from disk (see "Copy .env.example" above for when you'd need this).
python backend/scripts/rebuild_catalog_cli.py

# Recompute a dataset's auto-metadata (stats, thumbnails, extra-array inventory) from its
# already-ingested zarr store -- e.g. after extending what gets computed, or to retry if
# background metadata computation after an ingest failed. Omit the IDs to recompute everything.
python backend/scripts/recompute_metadata_cli.py [dataset_id ...]
```

The same operations are available via the API (`POST /api/admin/rebuild-catalog`,
`POST /api/admin/recompute-all-metadata`, `POST /api/datasets/{id}/recompute-metadata`) and, for
per-dataset metadata recompute, a button on the dataset detail page in the UI.

## Running the app

```bash
# terminal 1
source .venv/bin/activate
uvicorn thermal_dataset_manager.app:app --reload --reload-dir backend --port 8000

# terminal 2
cd frontend
npm run dev
```

`--reload-dir backend` matters: without it, `--reload` watches the whole repo root by default,
which includes `THERMAL_DATA_ROOT`'s default location (`./datasets`). Every file an ingest
writes there (a zarr store is hundreds of small chunk files) then triggers a server restart
mid-ingest, which can abort a background ingest job partway through.

Open the URL Vite prints (defaults to http://localhost:5173). The dev server proxies `/api/*` to
the backend on port 8000 (see `frontend/vite.config.ts`).

## Tests

```bash
source .venv/bin/activate
pytest backend/tests
```
