# thermal_dataset_manager

A lightweight browser based app to browse, preview, and organize raw thermal video captures. Ingest is CLI-only (see below).

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

# a capture with no telemetry rows (applies to every camera in every file this run)
python backend/scripts/ingest_cli.py --telemetry-rows 0 path/to/capture.npz
```

To keep track of what has been ingested, add `--move-ingested-to DIR`: each source file is moved into
`DIR` (keeping its path relative to the folder you gave) once it ingests successfully. Skipped and
failed files stay where they are, so what's left behind is what still needs attention. Add
`--move-duplicates` to also move files that were already ingested. Don't move or rename files yourself
while a run is in progress; it works from the list of files it found at startup.

A source file with several cameras is split at ingest into **one dataset per camera**, so a dataset
is always a single thermal video. Its Zarr group keeps the camera's index from the source file
(`camera_1` for `raw_thr_frames_1`). The datasets from one file share a `capture_id` (generated); use
`--capture ID` to group datasets from separate files under one capture id. Re-ingesting a file skips
cameras that were already ingested (deduplicated on source hash + camera index).

Only the camera frames and timestamps are ingested, and only frame arrays of dtype `uint16` (others are
skipped; a file with none is rejected). The readers inspect array headers first and load just the
frames and timestamps, so unrelated arrays in the file are never read into memory. Any other arrays in the source file (auxiliary
sensors, etc.) are not stored; a free-text summary of them (name, shape, dtype) is kept in the
dataset's metadata as `other_variables`.

Each raw frame may carry non-image **telemetry rows**. `--telemetry-rows` controls which rows those
are: `2` (default) = the top 2 rows, `0` = none, `-2` = the bottom 2 rows. The default is `THERMAL_DEFAULT_TELEMETRY_ROW_COUNT`.
Telemetry rows are stripped from the frames at ingest and stored verbatim in a separate `telemetry`
array alongside `counts` in the same Zarr store (`counts` holds image rows only). The Boson+ telemetry
line is also parsed during ingest, and its static fields (serials, part number, frame rate, gain mode,
enabled pipeline stages...) are recorded in the dataset's metadata. Stores written by older versions
(telemetry inside `counts`, schema v1) are rejected with a message to re-ingest.

Each source file's arrays are matched by name against the registry in
`backend/thermal_dataset_manager/ingest/key_registry.py` (currently `raw_thr_frames`/
`raw_thr_tstamps`, optionally suffixed `_N` per camera for multi-camera captures — letter
suffixes `_A`/`_B` are accepted and normalized to `_0`/`_1`). Add a new regex pair there to
support another naming convention — no other pipeline code needs to change.

Ingest is deduplicated by the SHA-256 of the **raw source file bytes** (computed before the file
is decoded, so the camera-key mapping/normalization above does not affect it). Submitting a file
byte-identical to an already-ingested capture is rejected with an error naming the existing
dataset (id, name, source filename, ingest time) rather than creating a redundant copy.

Each camera is also deduplicated on a **content hash** of its raw frames (telemetry rows included,
independent of file compression or `--telemetry-rows`), so the same video packaged differently
(e.g. `.npz` and `.npz.zst`) is caught too. Datasets ingested before this existed get their hash
from `python backend/scripts/backfill_content_hash_cli.py`, which also lists any datasets that hold
identical frames.

## Maintenance

All three are safe to run at any time (they never touch the immutable raw zarr data) and work
without a server running:

```bash
source .venv/bin/activate

# Rebuild the search catalog from disk (see "Copy .env.example" above for when you'd need this).
python backend/scripts/rebuild_catalog_cli.py

# Recompute a dataset's auto-metadata (stats, thumbnails) from its
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

## Loading videos for downstream processing

`thermal_dataset_manager.loader` reads the library straight off disk (no server needed): the catalog
finds videos, and frames are only read from the zarr stores when you ask for them.

```python
from thermal_dataset_manager.loader import Library

lib = Library()   # uses THERMAL_DATA_ROOT
videos = lib.find(tags=["bunny"], min_frames=500, shape=(512, 640), pipeline_disabled=["temporal_filter"])
v = videos[0]
v.frames(slice(100, 200))        # (100, H, W) uint16, image rows only
v.frames(time=(2.0, 4.0))        # by seconds since the first frame
v.frames([0, 50, 99])            # arbitrary frame indices
v.frames(raw=True)               # telemetry rows re-attached, as in the source
for batch in v.iter_batches(32): ...   # whole video with bounded memory
v.parse_telemetry(0)             # decoded Boson+ telemetry of one frame
lib.capture(v.capture_id)        # every camera of the same capture
```

`find` filters: `q` (full text), `tags`, `capture_id`, `date_from/to`, `scene_category`,
`capture_location`, `min_frames`, `max_frames`, `min_duration`, `shape`, `has_telemetry`, `gain_mode`,
`pipeline_enabled`/`pipeline_disabled`, `where=` (any predicate on the metadata) and `limit`.
