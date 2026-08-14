# Independent GPU service stack

The web application does not borrow a GPU from an unrelated training job. Its
three inference services run inside one dedicated Slurm job:

- job name: `retina-gpu-stack`
- partition: `A800-N`
- allocation: `1 GPU`, `16 CPU cores`, `64 GB host memory`
- time limit: 7 days
- services: multimodal inference on port 8011, fundus specialist inference on
  port 8012, and image enhancement plus pixel segmentation on port 8013

The CPU web job starts `scripts/gpu_stack_watchdog.sh`. The watchdog checks the
Slurm queue every 45 seconds and submits the GPU stack when no running or
pending stack job exists. A lock file prevents duplicate watchdogs and Slurm
job-name checks prevent duplicate GPU allocations.

## Manual operations

Submit the stack manually when needed:

```bash
cd /data/user/hd66945/fundus-dx-ml
sbatch slurm/retina-gpu-stack.sbatch
```

Inspect the allocation and service health:

```bash
squeue -u hd66945 -n retina-gpu-stack
curl http://gpu01:8011/health
curl http://gpu01:8012/health
curl http://gpu01:8013/health
```

The stack is healthy only when all three endpoints report `ready` and share the
same Slurm job id. If the stack is cancelled while the CPU web job is running,
the watchdog will submit a replacement. A pending job means the application is
waiting for the cluster to grant its own GPU; it never falls back to another
user job or returns simulated inference.
