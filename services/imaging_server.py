"""Private GPU pixel-segmentation service for calibrated imaging pipelines."""
from __future__ import annotations

import base64
import json
import os
import socket
import threading
import time
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, Header, HTTPException
from monai.networks.nets import DynUNet, UNet
from PIL import Image
from pydantic import BaseModel, Field
from safetensors.torch import load_file
import torchseg

from services.oct_ddpm import OCTDiffusionDenoiser

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNTIME_DIR = PROJECT_ROOT / "runtime"
STATUS_FILE = RUNTIME_DIR / "imaging_service.json"
TOKEN_FILE = RUNTIME_DIR / "agent_token"
PORT = int(os.getenv("IMAGING_PORT", "8013"))
FUNDUS_CHECKPOINT = PROJECT_ROOT / "models/fundus-lesions/model.safetensors"
OCTA_CHECKPOINT = PROJECT_ROOT / "models/octa-vessels/30_model.pth"
OCT_CHECKPOINT = PROJECT_ROOT / "models/oct-structure/duke_unet_v1.pth"
OCT_ENHANCEMENT_CHECKPOINT = PROJECT_ROOT / "models/oct-enhancement/DDPM_oct_dataset2_2021-07-08.pt"
INFERENCE_LOCK = threading.Lock()
MODELS: dict[str, torch.nn.Module] = {}

FUNDUS_NAMES = ["BG", "CTW", "EX", "HE", "MA"]
FUNDUS_COLORS = np.array(
    [[0, 0, 0], [236, 166, 63], [140, 241, 142], [68, 152, 240], [93, 71, 201]],
    dtype=np.uint8,
)
OCT_NAMES = ["Background", "ILM", "NFL", "IPL", "INL", "OPL", "ISM", "OS", "BM", "Fluid"]
OCT_COLORS = np.array(
    [[8, 14, 20], [0, 235, 255], [38, 166, 255], [111, 255, 179], [255, 219, 77],
     [255, 121, 198], [174, 108, 255], [255, 134, 76], [75, 215, 135], [255, 62, 95]],
    dtype=np.uint8,
)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class ImagingRequest(BaseModel):
    image: str = Field(min_length=1)
    task: str = Field(pattern="^(fundus_lesions|octa_vessels|oct_structure|oct_enhancement)$")


def _token() -> str:
    return TOKEN_FILE.read_text(encoding="utf-8").strip() if TOKEN_FILE.is_file() else ""


def _write_status(status: str, detail: str = "") -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "host": socket.gethostname(),
        "port": PORT,
        "model": "Fundus U-Net + OCTA DynUNet + Duke OCT U-Net + OCT diffusion enhancement",
        "job_id": os.getenv("SLURM_JOB_ID", ""),
        "updated_at": time.time(),
        "detail": detail,
    }
    STATUS_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_oct() -> UNet:
    return UNet(
        spatial_dims=2, in_channels=1, out_channels=10,
        channels=(24, 48, 96, 192, 320), strides=(2, 2, 2, 2),
        num_res_units=2, norm="INSTANCE",
    )


def _load() -> None:
    missing = [
        str(path) for path in
        (FUNDUS_CHECKPOINT, OCTA_CHECKPOINT, OCT_CHECKPOINT, OCT_ENHANCEMENT_CHECKPOINT)
        if not path.is_file()
    ]
    if missing:
        raise RuntimeError("Missing calibrated checkpoint(s): " + ", ".join(missing))

    fundus = torchseg.create_model(
        arch="unet", encoder_name="seresnext50_32x4d", encoder_weights=None,
        in_channels=3, classes=5,
    )
    fundus.load_state_dict({
        key.removeprefix("model."): value for key, value in load_file(FUNDUS_CHECKPOINT).items()
    })
    fundus.eval().cuda()

    octa = DynUNet(
        spatial_dims=2, in_channels=1, out_channels=1,
        kernel_size=[3, 3, 3, 3, 3], strides=[1, 2, 2, 2, 1],
        upsample_kernel_size=[1, 2, 2, 2, 1],
    )
    octa.load_state_dict(torch.load(OCTA_CHECKPOINT, map_location="cpu", weights_only=True)["model"])
    octa.eval().cuda()

    oct_model = _build_oct()
    oct_model.load_state_dict(torch.load(OCT_CHECKPOINT, map_location="cpu", weights_only=False)["model"])
    oct_model.eval().cuda()
    oct_enhancement = OCTDiffusionDenoiser(OCT_ENHANCEMENT_CHECKPOINT, timestep=14)
    oct_enhancement.eval().cuda()
    MODELS.update(
        fundus_lesions=fundus, octa_vessels=octa,
        oct_structure=oct_model, oct_enhancement=oct_enhancement,
    )


def _allowed_path(raw_path: str) -> Path:
    path = Path(raw_path).resolve()
    if not path.is_file() or PROJECT_ROOT not in path.parents:
        raise HTTPException(status_code=400, detail="Image path is outside the project runtime")
    return path


def _png64(array: np.ndarray, mode: str | None = None) -> str:
    array = np.clip(array, 0, 255).astype(np.uint8)
    image = Image.fromarray(array, mode=mode)
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _autofit_fundus(image: np.ndarray, size: int = 1024) -> tuple[np.ndarray, dict]:
    roi = (image.max(axis=2) > 0.05 * max(float(image.max()), 1)).astype(np.uint8)
    points = cv2.findNonZero(roi)
    if points is None:
        raise HTTPException(status_code=400, detail="No fundus field of view detected")
    x, y, width, height = cv2.boundingRect(points)
    cropped = image[y:y + height, x:x + width]
    scale = size / max(cropped.shape[:2])
    resized = cv2.resize(cropped, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    pad_y, pad_x = size - resized.shape[0], size - resized.shape[1]
    top, left = pad_y // 2, pad_x // 2
    fitted = np.pad(resized, ((top, pad_y - top), (left, pad_x - left), (0, 0)))
    transform = {
        "x": x, "y": y, "width": width, "height": height,
        "top": top, "left": left,
        "fit_height": resized.shape[0], "fit_width": resized.shape[1], "size": size,
    }
    return fitted, transform


def _fundus_infer(image: Image.Image) -> dict:
    original = np.asarray(image.convert("RGB"))
    fitted, transform = _autofit_fundus(original)
    tensor = torch.from_numpy(fitted.astype(np.float32) / 255.0).permute(2, 0, 1)
    tensor = ((tensor - IMAGENET_MEAN) / IMAGENET_STD)[None].cuda()
    with torch.inference_mode():
        probability = torch.softmax(MODELS["fundus_lesions"](tensor), dim=1)[0].cpu().numpy()
    labels = probability.argmax(0).astype(np.uint8)
    overlay = cv2.addWeighted(fitted, 0.72, FUNDUS_COLORS[labels], 0.52, 0)
    masks = {
        name: _png64(np.uint8(labels == class_id) * 255, "L")
        for class_id, name in enumerate(FUNDUS_NAMES) if class_id > 0
    }
    return {
        "model": "ClementP U-Net / SE-ResNeXt50", "labels": FUNDUS_NAMES,
        "label_map_png": _png64(labels, "L"), "mask_pngs": masks,
        "overlay_png": _png64(overlay), "transform": transform,
    }


def _octa_infer(image: Image.Image) -> dict:
    gray = np.asarray(image.convert("L").resize((1216, 1216), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    upstream = np.flip(np.rot90(gray, 1), axis=0).copy()
    tensor = torch.from_numpy(upstream)[None, None].cuda()
    with torch.inference_mode():
        probability = torch.sigmoid(MODELS["octa_vessels"](tensor))[0, 0].cpu().numpy()
    mask = (probability >= 0.5).astype(np.uint8)
    count, components, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    clean = np.zeros_like(mask)
    for label_id in range(1, count):
        if stats[label_id, cv2.CC_STAT_AREA] >= 128:
            clean[components == label_id] = 1
    clean = np.rot90(np.flip(clean, axis=0), -1).copy()
    probability = np.rot90(np.flip(probability, axis=0), -1).copy()
    base = cv2.cvtColor(np.uint8(gray * 255), cv2.COLOR_GRAY2RGB)
    color = base.copy()
    color[clean > 0] = (0, 236, 255)
    overlay = cv2.addWeighted(base, 0.5, color, 0.5, 0)
    return {
        "model": "OCTA DynUNet S-GAN epoch 30",
        "mask_png": _png64(clean * 255, "L"),
        "probability_png": _png64(np.uint8(probability * 255), "L"),
        "overlay_png": _png64(overlay), "input_size": [1216, 1216],
    }


def _oct_infer(image: Image.Image) -> dict:
    scan = np.asarray(image.convert("L").resize((512, 512), Image.Resampling.BICUBIC), dtype=np.float32) / 255.0
    normalized = (scan - float(scan.mean())) / max(float(scan.std()), 0.08)
    tensor = torch.from_numpy(normalized)[None, None].cuda()
    with torch.inference_mode():
        labels = MODELS["oct_structure"](tensor).argmax(1)[0].cpu().numpy().astype(np.uint8)
    base = cv2.cvtColor(np.uint8(scan * 255), cv2.COLOR_GRAY2RGB)
    overlay = cv2.addWeighted(base, 0.58, OCT_COLORS[labels], 0.42, 0)
    boundary = np.zeros_like(labels, dtype=bool)
    boundary[1:] |= labels[1:] != labels[:-1]
    overlay[boundary] = (0, 245, 255)
    return {
        "model": "Duke DME residual U-Net v1", "labels": OCT_NAMES,
        "label_map_png": _png64(labels, "L"), "overlay_png": _png64(overlay),
        "input_size": [512, 512],
    }


def _oct_enhancement_infer(image: Image.Image) -> dict:
    with torch.inference_mode():
        enhanced = MODELS["oct_enhancement"].enhance(image)
    return {
        "model": "OCT DDPM public checkpoint",
        "enhanced_png": _png64(enhanced, "L"),
        "timestep": MODELS["oct_enhancement"].timestep,
        "input_size": [image.width, image.height],
    }


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


app = FastAPI(title="RetinaScope calibrated imaging service", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": "ready" if len(MODELS) == 4 else "loading",
        "models": sorted(MODELS), "device": "cuda",
        "job_id": os.getenv("SLURM_JOB_ID", ""),
    }


@app.post("/infer")
def infer(request: ImagingRequest, x_agent_token: str = Header(default="")):
    if not _token() or x_agent_token != _token():
        raise HTTPException(status_code=401, detail="invalid internal service token")
    if request.task not in MODELS:
        raise HTTPException(status_code=503, detail=f"{request.task} is not ready")
    started = time.perf_counter()
    with Image.open(_allowed_path(request.image)) as source:
        image = source.convert("RGB")
    with INFERENCE_LOCK:
        result = {
            "fundus_lesions": _fundus_infer,
            "octa_vessels": _octa_infer,
            "oct_structure": _oct_infer,
            "oct_enhancement": _oct_enhancement_infer,
        }[request.task](image)
    return {
        "task": request.task, **result,
        "runtime_ms": round((time.perf_counter() - started) * 1000, 1),
        "real_inference": True,
    }
