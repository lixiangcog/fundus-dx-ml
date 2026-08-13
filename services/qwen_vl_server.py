"""Private GPU service for real Qwen2.5-VL multimodal inference."""
from __future__ import annotations

import json
import os
import socket
import time
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import torch
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.getenv("QWEN_VL_MODEL", "/data/user/hd66945/models/Qwen2.5-VL-3B-Instruct"))
RUNTIME_DIR = PROJECT_ROOT / "runtime"
STATUS_FILE = RUNTIME_DIR / "qwen_service.json"
TOKEN_FILE = RUNTIME_DIR / "agent_token"
PORT = int(os.getenv("QWEN_VL_PORT", "8011"))
INFERENCE_LOCK = threading.Lock()
MODEL = None
PROCESSOR = None


class InferenceRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=30000)
    images: List[str] = Field(default_factory=list, max_length=8)
    max_new_tokens: int = Field(default=700, ge=32, le=1400)
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)


def _token() -> str:
    return TOKEN_FILE.read_text(encoding="utf-8").strip() if TOKEN_FILE.is_file() else ""


def _write_status(status: str, detail: str = "") -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "host": socket.gethostname(),
        "port": PORT,
        "model": "Qwen2.5-VL-3B-Instruct",
        "model_path": str(MODEL_PATH),
        "job_id": os.getenv("SLURM_JOB_ID", ""),
        "updated_at": time.time(),
        "detail": detail,
    }
    STATUS_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load() -> None:
    global MODEL, PROCESSOR
    if not MODEL_PATH.joinpath("model-00001-of-00002.safetensors").is_file():
        raise RuntimeError(f"Model weights not found at {MODEL_PATH}")
    PROCESSOR = AutoProcessor.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        min_pixels=256 * 28 * 28,
        max_pixels=640 * 28 * 28,
    )
    MODEL = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        low_cpu_mem_usage=True,
    )
    MODEL.eval()


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


app = FastAPI(title="RetinaScope Qwen-VL Service", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": "ready" if MODEL is not None else "loading",
        "model": "Qwen2.5-VL-3B-Instruct",
        "device": str(next(MODEL.parameters()).device) if MODEL is not None else None,
        "job_id": os.getenv("SLURM_JOB_ID", ""),
    }


@app.post("/infer")
def infer(request: InferenceRequest, x_agent_token: str = Header(default="")):
    if not _token() or x_agent_token != _token():
        raise HTTPException(status_code=401, detail="invalid internal service token")
    if MODEL is None or PROCESSOR is None:
        raise HTTPException(status_code=503, detail="model is not ready")

    image_paths = []
    for raw_path in request.images:
        path = Path(raw_path).resolve()
        if not path.is_file() or PROJECT_ROOT not in path.parents:
            raise HTTPException(status_code=400, detail=f"invalid image path: {raw_path}")
        image_paths.append(path)

    content = [
        {"type": "image", "image": f"file://{path}"}
        for path in image_paths
    ]
    content.append({"type": "text", "text": request.prompt})
    messages = [{"role": "user", "content": content}]
    rendered = PROCESSOR.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = PROCESSOR(
        text=[rendered],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(MODEL.device)

    started = time.perf_counter()
    with torch.inference_mode():
        generated = MODEL.generate(
            **inputs,
            max_new_tokens=request.max_new_tokens,
            do_sample=request.temperature > 0,
            temperature=max(request.temperature, 0.01),
            top_p=0.9,
            repetition_penalty=1.03,
        )
    trimmed = [output[len(source):] for source, output in zip(inputs.input_ids, generated)]
    text = PROCESSOR.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    return {
        "text": text,
        "runtime_ms": round((time.perf_counter() - started) * 1000, 1),
        "model": "Qwen2.5-VL-3B-Instruct",
        "device": str(MODEL.device),
        "input_images": len(image_paths),
        "generated_tokens": int(trimmed[0].numel()),
        "real_inference": True,
    }

