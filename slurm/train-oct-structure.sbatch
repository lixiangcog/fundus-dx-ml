#!/usr/bin/env bash
#SBATCH --job-name=oct-structure-cal
#SBATCH --partition=A800-N
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/data/user/hd66945/fundus-dx-ml/logs/oct-structure-%j.out
#SBATCH --error=/data/user/hd66945/fundus-dx-ml/logs/oct-structure-%j.err

set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
cd /data/user/hd66945/fundus-dx-ml
mkdir -p logs models/oct-structure
exec /data/user/hd66945/.venvs/retina-imaging/bin/python scripts/train_oct_structure.py
