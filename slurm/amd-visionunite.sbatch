#!/usr/bin/env bash
#SBATCH --job-name=amd-visionunite
#SBATCH --partition=A800-N
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --output=/data/user/hd66945/fundus-dx-ml/logs/visionunite-%j.out
#SBATCH --error=/data/user/hd66945/fundus-dx-ml/logs/visionunite-%j.err

set -euo pipefail

export VISIONUNITE_PORT="${VISIONUNITE_PORT:-8012}"
export VISIONUNITE_CHECKPOINT="${VISIONUNITE_CHECKPOINT:-/data/user/hd66945/fundus-dx-ml/models/visionunite/checkpoint-VisionUniteV1.pth}"
export VISIONUNITE_LLAMA_ROOT="${VISIONUNITE_LLAMA_ROOT:-/data/user/hd66945/fundus-dx-ml/models/visionunite/llama_model_weights}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/user/hd66945/.cache/huggingface

cd /data/user/hd66945/fundus-dx-ml
exec /data/user/hd66945/.conda/envs/visionunite/bin/python -m uvicorn services.visionunite_server:app --host 0.0.0.0 --port "${VISIONUNITE_PORT}"
