#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_ROOT="${HOME}/.conda/envs/fundus-dx"
PORT="${FUNDUS_DX_PORT:-8000}"

export PATH="${ENV_ROOT}/bin:${PATH}"
cd "${PROJECT_ROOT}"

if [[ ! -f frontend/dist/index.html ]]; then
  echo "frontend/dist is missing; run npm ci && npm run build in frontend/ first" >&2
  exit 1
fi

exec python -m uvicorn api.main:app --host 0.0.0.0 --port "${PORT}" --workers 1
