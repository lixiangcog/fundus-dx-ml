#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_ROOT="${HOME}/.conda/envs/fundus-dx"
PORT="${FUNDUS_DX_PORT:-8000}"

export PATH="${ENV_ROOT}/bin:${PATH}"
cd "${PROJECT_ROOT}"
mkdir -p logs runtime

if [[ ! -f frontend/dist/index.html ]]; then
  echo "frontend/dist is missing; run npm ci && npm run build in frontend/ first" >&2
  exit 1
fi

watchdog_pid=""
web_pid=""

cleanup() {
  trap - EXIT INT TERM HUP
  if [[ -n "${watchdog_pid}" ]] && kill -0 "${watchdog_pid}" 2>/dev/null; then
    kill "${watchdog_pid}" 2>/dev/null || true
  fi
  if [[ -n "${web_pid}" ]] && kill -0 "${web_pid}" 2>/dev/null; then
    kill "${web_pid}" 2>/dev/null || true
  fi
  [[ -z "${watchdog_pid}" ]] || wait "${watchdog_pid}" 2>/dev/null || true
  [[ -z "${web_pid}" ]] || wait "${web_pid}" 2>/dev/null || true
}

trap cleanup EXIT INT TERM HUP

bash scripts/gpu_stack_watchdog.sh >> "logs/gpu-watchdog-${SLURM_JOB_ID:-local}.log" 2>&1 &
watchdog_pid=$!

python -m uvicorn api.main:app --host 0.0.0.0 --port "${PORT}" --workers 1 &
web_pid=$!
wait "${web_pid}"
