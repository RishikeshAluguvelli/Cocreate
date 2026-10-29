#!/usr/bin/env bash
# Local dev: backend on :8000 (reload) + Vite on :5173 (proxies /api and /files)
set -euo pipefail
cd "$(dirname "$0")"
( cd backend && uv run uvicorn app.main:app --reload --port 8000 ) &
( cd frontend && npm run dev ) &
wait
