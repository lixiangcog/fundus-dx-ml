#!/usr/bin/env bash
#SBATCH --job-name=retina-imaging
#SBATCH --partition=A800-N
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --output=/data/user/hd66945/fundus-dx-ml/logs/imaging-%j.out
#SBATCH --error=/data/user/hd66945/fundus-dx-ml/logs/imaging-%j.err

set -euo pipefail
export IMAGING_PORT="${IMAGING_PORT:-8013}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /data/user/hd66945/fundus-dx-ml
mkdir -p logs runtime
exec /data/user/hd66945/.venvs/retina-imaging/bin/python -m uvicorn \
  services.imaging_server:app --host 0.0.0.0 --port "${IMAGING_PORT}"
