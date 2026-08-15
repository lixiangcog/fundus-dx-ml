#!/usr/bin/env bash
#SBATCH --job-name=retina-gpu-stack
#SBATCH --partition=A800-N
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --requeue
#SBATCH --output=/data/user/hd66945/fundus-dx-ml/logs/gpu-stack-%j.out
#SBATCH --error=/data/user/hd66945/fundus-dx-ml/logs/gpu-stack-%j.err

set -euo pipefail

PROJECT_ROOT=/data/user/hd66945/fundus-dx-ml
QWEN_PYTHON=/data/user/hd66945/.conda/envs/med-tiv-eye/bin/python
RETINA_PYTHON=/data/user/hd66945/.venvs/retina-imaging/bin/python

export QWEN_VL_PORT="${QWEN_VL_PORT:-8011}"
export VISIONUNITE_PORT="${VISIONUNITE_PORT:-8012}"
export IMAGING_PORT="${IMAGING_PORT:-8013}"
export QWEN_VL_MODEL="${QWEN_VL_MODEL:-/data/user/hd66945/models/Qwen2.5-VL-3B-Instruct}"
export VISIONUNITE_CHECKPOINT="${VISIONUNITE_CHECKPOINT:-${PROJECT_ROOT}/models/visionunite/checkpoint-VisionUniteV1.pth}"
export VISIONUNITE_LLAMA_ROOT="${VISIONUNITE_LLAMA_ROOT:-${PROJECT_ROOT}/models/visionunite/llama_model_weights}"
export HF_HOME=/data/user/hd66945/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4

cd "${PROJECT_ROOT}"
mkdir -p logs runtime
bash scripts/queue_successor.sh retina-gpu "${SLURM_JOB_ID}" "${PROJECT_ROOT}/slurm/retina-gpu-stack.sbatch" || \
  echo "Unable to queue the next GPU stack job; the web watchdog remains active." >&2

declare -a service_pids=()
declare -a service_names=()
declare -a service_logs=()

cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM HUP
  for pid in "${service_pids[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
  for pid in "${service_pids[@]:-}"; do
    wait "${pid}" 2>/dev/null || true
  done
  exit "${exit_code}"
}

trap cleanup EXIT
trap 'exit 143' INT TERM HUP

start_service() {
  local name=$1
  local python_bin=$2
  local app_module=$3
  local port=$4
  local log_file="${PROJECT_ROOT}/logs/gpu-stack-${name}-${SLURM_JOB_ID}.log"

  "${python_bin}" -m uvicorn "${app_module}" --host 0.0.0.0 --port "${port}" > "${log_file}" 2>&1 &
  service_pids+=("$!")
  service_names+=("${name}")
  service_logs+=("${log_file}")
}

wait_ready() {
  local index=$1
  local port=$2
  local pid=${service_pids[$index]}
  local name=${service_names[$index]}
  local log_file=${service_logs[$index]}

  for _ in $(seq 1 180); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "${name} exited during startup" >&2
      tail -80 "${log_file}" >&2 || true
      return 1
    fi
    if curl -fsS --max-time 2 "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      echo "${name} ready on port ${port}"
      return 0
    fi
    sleep 2
  done

  echo "${name} did not become ready" >&2
  tail -80 "${log_file}" >&2 || true
  return 1
}

# Stagger model loading to limit peak CPU and disk pressure. All three services
# share the single GPU granted to this Slurm job through CUDA_VISIBLE_DEVICES.
start_service imaging "${RETINA_PYTHON}" services.imaging_server:app "${IMAGING_PORT}"
wait_ready 0 "${IMAGING_PORT}"

start_service qwen "${QWEN_PYTHON}" services.qwen_vl_server:app "${QWEN_VL_PORT}"
wait_ready 1 "${QWEN_VL_PORT}"

start_service visionunite "${RETINA_PYTHON}" services.visionunite_server:app "${VISIONUNITE_PORT}"
wait_ready 2 "${VISIONUNITE_PORT}"

echo "GPU stack ready: job=${SLURM_JOB_ID} host=$(hostname) cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

while sleep 15; do
  for index in "${!service_pids[@]}"; do
    if ! kill -0 "${service_pids[$index]}" 2>/dev/null; then
      echo "${service_names[$index]} stopped unexpectedly" >&2
      tail -80 "${service_logs[$index]}" >&2 || true
      exit 1
    fi
  done
done
