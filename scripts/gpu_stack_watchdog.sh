#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POLL_SECONDS="${GPU_STACK_POLL_SECONDS:-45}"

mkdir -p "${PROJECT_ROOT}/runtime" "${PROJECT_ROOT}/logs"
exec 9>"${PROJECT_ROOT}/runtime/gpu_stack_watchdog.instance.lock"
if ! flock -n 9; then
  echo "GPU stack watchdog already active; this instance will exit."
  exit 0
fi

ensure_stack() {
  if ! python -m api.gpu_scheduler ensure; then
    echo "GPU stack status check failed; retrying later." >&2
  fi
}

ensure_stack
while sleep "${POLL_SECONDS}"; do
  ensure_stack
done
