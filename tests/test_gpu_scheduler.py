import subprocess

from api import gpu_scheduler


def test_ensure_reuses_running_job(monkeypatch):
    monkeypatch.setattr(gpu_scheduler, "active_jobs", lambda: [
        {"job_id": "123", "state": "RUNNING", "reason": "gpu02", "node": "gpu02"}
    ])
    monkeypatch.setattr(gpu_scheduler, "autosubmit_enabled", lambda: True)

    result = gpu_scheduler.ensure_gpu_stack()

    assert result["job_id"] == "123"
    assert result["status"] == "running"
    assert result["submitted"] is False


def test_ensure_does_not_submit_when_disabled(monkeypatch):
    monkeypatch.setattr(gpu_scheduler, "active_jobs", lambda: [])
    monkeypatch.setattr(gpu_scheduler, "autosubmit_enabled", lambda: False)

    result = gpu_scheduler.ensure_gpu_stack()

    assert result["status"] == "absent"
    assert result["submitted"] is False
    assert result["autosubmit"] is False


def test_wait_returns_immediately_for_ready_service(monkeypatch):
    monkeypatch.setattr(gpu_scheduler, "autosubmit_enabled", lambda: True)

    result = gpu_scheduler.wait_for_service(lambda: {"status": "ready", "host": "gpu02"}, "影像分析")

    assert result["host"] == "gpu02"


def test_ensure_submits_when_no_job_exists(monkeypatch, tmp_path):
    monkeypatch.setattr(gpu_scheduler, "autosubmit_enabled", lambda: True)
    monkeypatch.setattr(gpu_scheduler, "gpu_job_status", lambda: {"status": "absent", "job_id": "", "jobs": []})
    monkeypatch.setattr(gpu_scheduler, "RUNTIME_DIR", tmp_path)
    monkeypatch.setattr(gpu_scheduler, "SUBMIT_LOCK", tmp_path / "submit.lock")
    monkeypatch.setattr(
        gpu_scheduler,
        "_run",
        lambda command, timeout=15: subprocess.CompletedProcess(command, 0, stdout="456\n", stderr=""),
    )

    result = gpu_scheduler.ensure_gpu_stack()

    assert result["status"] == "submitted"
    assert result["job_id"] == "456"
    assert result["submitted"] is True


def test_ensure_never_submits_when_queue_state_is_unknown(monkeypatch):
    monkeypatch.setattr(gpu_scheduler, "autosubmit_enabled", lambda: True)
    monkeypatch.setattr(
        gpu_scheduler,
        "gpu_job_status",
        lambda: {"status": "unknown", "job_id": "", "detail": "squeue unavailable"},
    )

    result = gpu_scheduler.ensure_gpu_stack()

    assert result["status"] == "unknown"
    assert result["submitted"] is False
