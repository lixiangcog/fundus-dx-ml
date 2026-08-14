#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SBATCH_FILE="${PROJECT_ROOT}/slurm/retina-gpu-stack.sbatch"
JOB_NAME=retina-gpu-stack
POLL_SECONDS="${GPU_STACK_POLL_SECONDS:-45}"
SUBMIT_COOLDOWN_SECONDS="${GPU_STACK_SUBMIT_COOLDOWN_SECONDS:-180}"
last_submit=0

mkdir -p "${PROJECT_ROOT}/runtime" "${PROJECT_ROOT}/logs"
exec 9>"${PROJECT_ROOT}/runtime/gpu_stack_watchdog.lock"
if ! flock -n 9; then
  echo "GPU stack watchdog already active; this instance will exit."
  exit 0
fi

ensure_stack() {
  local active now job_id
  if ! active=$(squeue -h -u "${USER}" -n "${JOB_NAME}" -o "%i %T" 2>/dev/null); then
    echo "Unable to query Slurm queue; retrying later." >&2
    return 0
  fi
  if [[ -n "${active//[[:space:]]/}" ]]; then
    return 0
  fi

  now=$(date +%s)
  if (( now - last_submit < SUBMIT_COOLDOWN_SECONDS )); then
    return 0
  fi
  if job_id=$(sbatch --parsable "${SBATCH_FILE}" 2>&1); then
    last_submit=${now}
    echo "Submitted independent GPU stack job ${job_id}."
  else
    echo "GPU stack submission failed: ${job_id}" >&2
    last_submit=${now}
  fi
}

ensure_stack
while sleep "${POLL_SECONDS}"; do
  ensure_stack
done
