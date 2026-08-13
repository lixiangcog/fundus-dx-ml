#!/usr/bin/env bash
#SBATCH --job-name=amd-qwen-vl
#SBATCH --partition=A800-N
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --output=/data/user/hd66945/fundus-dx-ml/logs/qwen-vl-%j.out
#SBATCH --error=/data/user/hd66945/fundus-dx-ml/logs/qwen-vl-%j.err

set -euo pipefail

export QWEN_VL_PORT="${QWEN_VL_PORT:-8011}"
export QWEN_VL_MODEL="${QWEN_VL_MODEL:-/data/user/hd66945/models/Qwen2.5-VL-3B-Instruct}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /data/user/hd66945/fundus-dx-ml
exec /data/user/hd66945/.conda/envs/med-tiv-eye/bin/python -m uvicorn services.qwen_vl_server:app --host 0.0.0.0 --port "${QWEN_VL_PORT}"

