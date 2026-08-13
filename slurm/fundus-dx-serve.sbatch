#!/usr/bin/env bash
#SBATCH --job-name=fundus-dx
#SBATCH --partition=Intel-8358
#SBATCH --nodelist=node10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=7-00:00:00
#SBATCH --output=/data/user/hd66945/fundus-dx-ml/logs/server-%j.out
#SBATCH --error=/data/user/hd66945/fundus-dx-ml/logs/server-%j.err

set -euo pipefail

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export FUNDUS_DX_PORT="${FUNDUS_DX_PORT:-8000}"

cd /data/user/hd66945/fundus-dx-ml
exec ./scripts/start_remote.sh
