#!/usr/bin/env bash
# Start the Vite dev server for the React frontend.
# Run in its own terminal:  ./start_frontend.sh
# It proxies /api/* to the backend on :8000, so start_backend.sh should be running too.
set -euo pipefail

cd "$(dirname "$0")/frontend"

if [ ! -d node_modules ]; then
  echo "node_modules missing -- running npm install..."
  npm install
fi

exec npm run dev
