#!/usr/bin/env bash
#SBATCH --job-name=fundus-dx
#SBATCH --partition=Intel-8358
#SBATCH --nodelist=node10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=7-00:00:00
#SBATCH --requeue
#SBATCH --output=/data/user/hd66945/fundus-dx-ml/logs/server-%j.out
#SBATCH --error=/data/user/hd66945/fundus-dx-ml/logs/server-%j.err

set -euo pipefail

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export FUNDUS_DX_PORT="${FUNDUS_DX_PORT:-8000}"
export FUNDUS_ENABLE_GPU_AUTOSUBMIT=1
export GPU_STACK_START_TIMEOUT_SECONDS="${GPU_STACK_START_TIMEOUT_SECONDS:-900}"

cd /data/user/hd66945/fundus-dx-ml
bash scripts/queue_successor.sh web "${SLURM_JOB_ID}" "/data/user/hd66945/fundus-dx-ml/slurm/fundus-dx-serve.sbatch" || \
  echo "Unable to queue the next web job; manual recovery may be required." >&2
exec ./scripts/start_remote.sh
