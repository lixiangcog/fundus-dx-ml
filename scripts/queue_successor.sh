#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: queue_successor.sh <service-key> <current-job-id> <sbatch-file>" >&2
  exit 2
fi

service_key=$1
current_job_id=$2
sbatch_file=$3
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
state_dir="${project_root}/runtime/successors"
marker="${state_dir}/${service_key}-${current_job_id}.job"

mkdir -p "${state_dir}"
exec 8>"${marker}.lock"
flock -x 8

if [[ -s "${marker}" ]]; then
  successor_id="$(tr -d '[:space:]' < "${marker}")"
  if [[ -n "${successor_id}" ]] && squeue -h -j "${successor_id}" 2>/dev/null | grep -q .; then
    echo "Successor ${successor_id} already queued for ${service_key} job ${current_job_id}."
    exit 0
  fi
fi

successor_id="$(sbatch --parsable --dependency="afterany:${current_job_id}" "${sbatch_file}")"
successor_id="${successor_id%%;*}"
printf '%s\n' "${successor_id}" > "${marker}"
echo "Queued successor ${successor_id} for ${service_key} job ${current_job_id}."
