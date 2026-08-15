"""Slurm-backed GPU job lifecycle management for production inference."""
from __future__ import annotations

import fcntl
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SBATCH_FILE = PROJECT_ROOT / "slurm" / "retina-gpu-stack.sbatch"
RUNTIME_DIR = PROJECT_ROOT / "runtime"
SUBMIT_LOCK = RUNTIME_DIR / "gpu_stack_submit.lock"
JOB_NAME = "retina-gpu-stack"
ACTIVE_STATES = "PENDING,RUNNING,CONFIGURING,COMPLETING"


def autosubmit_enabled() -> bool:
    return os.getenv("FUNDUS_ENABLE_GPU_AUTOSUBMIT", "0").lower() in {"1", "true", "yes"}


def _run(command: list[str], timeout: int = 15) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)


def active_jobs() -> list[dict[str, str]]:
    user = os.getenv("USER") or os.getenv("LOGNAME") or "hd66945"
    try:
        result = _run([
            "squeue", "--noheader", "--user", user, "--name", JOB_NAME,
            f"--states={ACTIVE_STATES}", "--format=%i|%T|%R|%N",
        ])
    except (OSError, subprocess.SubprocessError) as exc:
        return [{"job_id": "", "state": "UNKNOWN", "reason": str(exc), "node": ""}]
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip() or "squeue failed"
        return [{"job_id": "", "state": "UNKNOWN", "reason": detail, "node": ""}]
    jobs: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        parts = line.strip().split("|", 3)
        if len(parts) == 4:
            jobs.append({"job_id": parts[0], "state": parts[1], "reason": parts[2], "node": parts[3]})
    return jobs


def gpu_job_status() -> dict[str, Any]:
    jobs = active_jobs()
    known = [job for job in jobs if job.get("job_id")]
    if known:
        running = next((job for job in known if job["state"] == "RUNNING"), known[0])
        return {"status": running["state"].lower(), "job_id": running["job_id"], "jobs": known}
    if jobs and jobs[0].get("state") == "UNKNOWN":
        return {"status": "unknown", "job_id": "", "detail": jobs[0].get("reason", "")}
    return {"status": "absent", "job_id": "", "jobs": []}


def ensure_gpu_stack() -> dict[str, Any]:
    """Return an active job or atomically submit one when production autosubmit is enabled."""
    current = gpu_job_status()
    if current["status"] != "absent":
        return {**current, "submitted": False}
    if not autosubmit_enabled():
        return {**current, "submitted": False, "autosubmit": False}

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    with SUBMIT_LOCK.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        current = gpu_job_status()
        if current["status"] != "absent":
            return {**current, "submitted": False, "autosubmit": True}
        try:
            result = _run(["sbatch", "--parsable", str(SBATCH_FILE)])
        except (OSError, subprocess.SubprocessError) as exc:
            return {"status": "error", "job_id": "", "submitted": False, "autosubmit": True, "detail": str(exc)}
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip() or "sbatch failed"
            return {"status": "error", "job_id": "", "submitted": False, "autosubmit": True, "detail": detail}
        job_id = result.stdout.strip().split(";", 1)[0]
        record = {"job_id": job_id, "submitted_at": time.time(), "script": str(SBATCH_FILE)}
        (RUNTIME_DIR / "last_gpu_submission.json").write_text(json.dumps(record), encoding="utf-8")
        return {"status": "submitted", "job_id": job_id, "submitted": True, "autosubmit": True}


def wait_for_service(
    status_check: Callable[[], dict[str, Any]],
    service_label: str,
) -> dict[str, Any]:
    """Ensure the GPU job exists and wait for one registered service to become ready."""
    service = status_check()
    if service.get("status") == "ready" or not autosubmit_enabled():
        return service

    job = ensure_gpu_stack()
    timeout = max(30, int(os.getenv("GPU_STACK_START_TIMEOUT_SECONDS", "900")))
    poll = max(1, int(os.getenv("GPU_STACK_READY_POLL_SECONDS", "5")))
    deadline = time.monotonic() + timeout
    last_ensure = time.monotonic()
    while time.monotonic() < deadline:
        time.sleep(poll)
        service = status_check()
        if service.get("status") == "ready":
            return service
        if time.monotonic() - last_ensure >= 30:
            job = ensure_gpu_stack()
            last_ensure = time.monotonic()

    job_id = job.get("job_id") or "待分配"
    job_state = job.get("status", "unknown")
    raise RuntimeError(
        f"{service_label} GPU 作业已自动申请（作业 {job_id}，状态 {job_state}），"
        "但模型仍在排队或加载，请稍后重试。"
    )


if __name__ == "__main__":
    import sys

    command = sys.argv[1] if len(sys.argv) > 1 else "status"
    output = ensure_gpu_stack() if command == "ensure" else gpu_job_status()
    print(json.dumps(output, ensure_ascii=False))
