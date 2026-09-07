#!/usr/bin/env bash
# Start the FastAPI backend (dev mode, auto-reload).
# Run in its own terminal:  ./start_backend.sh
set -euo pipefail

cd "$(dirname "$0")"

# shellcheck disable=SC1091
source .venv/bin/activate

# --reload-dir backend: only watch source, not THERMAL_DATA_ROOT (./datasets),
# so ingest writes don't trigger a restart mid-job. See README "Running the app".
exec uvicorn thermal_dataset_manager.app:app \
  --reload --reload-dir backend --port "${THERMAL_PORT:-8000}"
