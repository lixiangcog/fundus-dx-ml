"""Client for the private calibrated pixel-segmentation service."""
from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATUS_FILE = PROJECT_ROOT / "runtime/imaging_service.json"
TOKEN_FILE = PROJECT_ROOT / "runtime/agent_token"
NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def status() -> dict[str, Any]:
    if not STATUS_FILE.is_file():
        return {"status": "offline", "detail": "Imaging GPU service has not registered"}
    try:
        result = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"status": "offline", "detail": "Invalid imaging service status"}
    result["status_age_seconds"] = round(time.time() - float(result.get("updated_at", 0)), 1)
    if result.get("status") == "ready":
        try:
            request = urllib.request.Request(
                f"http://{result['host']}:{result['port']}/health",
                headers={"Accept": "application/json"},
            )
            with NO_PROXY_OPENER.open(request, timeout=2.5) as response:
                live = json.loads(response.read().decode("utf-8"))
            result["live"] = live
            result["status"] = live.get("status", "offline")
        except Exception as exc:
            result.update(status="offline", detail=f"Imaging health check failed: {exc}")
    return result


def infer(task: str, image_path: Path) -> dict[str, Any]:
    service = status()
    if service.get("status") != "ready":
        from api.gpu_scheduler import wait_for_service

        service = wait_for_service(status, "影像分析")
    if service.get("status") != "ready":
        raise RuntimeError(service.get("detail") or "Imaging GPU service is not ready")
    if not TOKEN_FILE.is_file():
        raise RuntimeError("Internal imaging service token is missing")
    payload = json.dumps({"task": task, "image": str(image_path.resolve())}).encode("utf-8")
    request = urllib.request.Request(
        f"http://{service['host']}:{service['port']}/infer",
        data=payload, method="POST",
        headers={
            "Content-Type": "application/json",
            "X-Agent-Token": TOKEN_FILE.read_text(encoding="utf-8").strip(),
        },
    )
    try:
        with NO_PROXY_OPENER.open(request, timeout=420) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Imaging inference failed ({exc.code}): {detail}") from exc
    except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
        raise RuntimeError(f"Imaging inference unavailable: {exc}") from exc
    if not result.get("real_inference"):
        raise RuntimeError("Imaging service returned no verified inference output")
    return result
