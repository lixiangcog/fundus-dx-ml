"""Private GPU service for the public VisionUnite V1 fundus checkpoint."""
from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import numpy as np
import torch
import torchvision.transforms as transforms
from fastapi import FastAPI, Header, HTTPException
from PIL import Image, ImageEnhance
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VISIONUNITE_ROOT = PROJECT_ROOT / "third_party" / "VisionUnite"
CHECKPOINT = Path(os.getenv("VISIONUNITE_CHECKPOINT", PROJECT_ROOT / "models/visionunite/checkpoint-VisionUniteV1.pth"))
LLAMA_ROOT = Path(os.getenv("VISIONUNITE_LLAMA_ROOT", PROJECT_ROOT / "models/visionunite/llama_model_weights"))
RUNTIME_DIR = PROJECT_ROOT / "runtime"
STATUS_FILE = RUNTIME_DIR / "visionunite_service.json"
TOKEN_FILE = RUNTIME_DIR / "agent_token"
PORT = int(os.getenv("VISIONUNITE_PORT", "8012"))
INFERENCE_LOCK = threading.Lock()
MODEL = None

sys.path.insert(0, str(VISIONUNITE_ROOT))
import llama  # noqa: E402

SIGNAL_NAMES = ["other_abnormality", "hemorrhage_exudation", "optic_cup_disc", "fundus_color_boundary", "macular_abnormality", "arteriovenous_abnormality"]

class InferenceRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=12000)
    images: List[str] = Field(min_length=1, max_length=4)
    max_new_tokens: int = Field(default=256, ge=32, le=320)
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)

def _token() -> str:
    return TOKEN_FILE.read_text(encoding="utf-8").strip() if TOKEN_FILE.is_file() else ""

def _write_status(status: str, detail: str = "") -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"status": status, "host": socket.gethostname(), "port": PORT, "model": "VisionUnite V1 (public checkpoint)", "model_path": str(CHECKPOINT), "job_id": os.getenv("SLURM_JOB_ID", ""), "updated_at": time.time(), "detail": detail}
    STATUS_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

def _load() -> None:
    global MODEL
    if not CHECKPOINT.is_file():
        raise RuntimeError(f"VisionUnite checkpoint not found at {CHECKPOINT}")
    base = LLAMA_ROOT / "7B"
    for required in (base / "params.json", base / "tokenizer.model"):
        if not required.is_file():
            raise RuntimeError(f"VisionUnite model scaffold is missing: {required}")
    MODEL = llama.load(str(CHECKPOINT), str(LLAMA_ROOT), device="cuda")
    if isinstance(MODEL, RuntimeError):
        raise MODEL
    MODEL.eval()

def _prepare(paths: list[Path]) -> torch.Tensor:
    transform = transforms.Compose([transforms.Resize(448, interpolation=transforms.InterpolationMode.BICUBIC), transforms.CenterCrop(448), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    outputs = []
    for path in paths:
        with path.open("rb") as stream:
            image = Image.open(stream).convert("RGB")
        image = ImageEnhance.Contrast(image).enhance(1.3)
        array = np.asarray(image, dtype=np.int16).copy()
        for channel in range(3):
            array[:, :, channel] = array[:, :, channel] - array[:, :, channel].min() + 1
        image = Image.fromarray(np.clip(array, 0, 255).astype("uint8"), mode="RGB")
        outputs.append(transform(image))
    return torch.stack(outputs, dim=0).cuda(non_blocking=True)

@asynccontextmanager
async def lifespan(_app: FastAPI):
    _write_status("loading")
    try:
        _load()
        _write_status("ready")
    except Exception as exc:
        _write_status("error", str(exc))
        raise
    yield
    _write_status("stopped")

app = FastAPI(title="RetinaScope VisionUnite Service", version="1.0.0", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ready" if MODEL is not None else "loading", "model": "VisionUnite V1 (public checkpoint)", "device": str(next(MODEL.parameters()).device) if MODEL is not None else None, "job_id": os.getenv("SLURM_JOB_ID", "")}

@app.post("/infer")
def infer(request: InferenceRequest, x_agent_token: str = Header(default="")):
    if not _token() or x_agent_token != _token():
        raise HTTPException(status_code=401, detail="invalid internal service token")
    if MODEL is None:
        raise HTTPException(status_code=503, detail="model is not ready")
    paths = []
    for raw_path in request.images:
        path = Path(raw_path).resolve()
        if not path.is_file() or PROJECT_ROOT not in path.parents:
            raise HTTPException(status_code=400, detail=f"invalid image path: {raw_path}")
        paths.append(path)
    visit_labels = ["Baseline visit", "Follow-up visit"]
    prompts = [llama.format_prompt(f"{visit_labels[index] if index < 2 else f'Image {index + 1}'}. {request.prompt}") for index in range(len(paths))]
    images = _prepare(paths)
    started = time.perf_counter()
    with INFERENCE_LOCK, torch.inference_mode():
        texts, flags = MODEL.generate(images, prompts, input_type="vision", max_gen_len=request.max_new_tokens, temperature=request.temperature, top_p=0.75)
    observations = [{"text": text.strip(), "signal_flags": {name: bool(value) for name, value in zip(SIGNAL_NAMES, row)}} for text, row in zip(texts, flags)]
    return {"observations": observations, "runtime_ms": round((time.perf_counter() - started) * 1000, 1), "model": "VisionUnite V1 (public checkpoint)", "device": str(next(MODEL.parameters()).device), "input_images": len(paths), "real_inference": True}
