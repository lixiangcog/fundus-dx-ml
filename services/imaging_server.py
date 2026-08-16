"""Private GPU pixel-segmentation service for calibrated imaging pipelines."""
from __future__ import annotations

import base64
import csv
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, Header, HTTPException
from monai.networks.nets import DynUNet, UNet
from PIL import Image
from pydantic import BaseModel, Field
from safetensors.torch import load_file
import torchseg
import timm
from torchvision import models, transforms

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
EYE_AGE_CHECKPOINT = PROJECT_ROOT / "models/eye-age/resnet101-nonfiltered.pth"
AMD_PATHOLOGY_DIR = PROJECT_ROOT / "models/amd-pathology"
OCT_AMD_CHECKPOINT = AMD_PATHOLOGY_DIR / "raw/oct_classifier/pytorch_model.bin"
OCT_FLUID_SUBTYPE_ONNX = {
    "slot1": AMD_PATHOLOGY_DIR / "oct-fluid/slot1_v2l_seed2024.onnx",
    "slot2": AMD_PATHOLOGY_DIR / "oct-fluid/slot2_v2l_seed123.onnx",
}
FUNDUS_AMD_ONNX = {
    "drusen": AMD_PATHOLOGY_DIR / "drusen.onnx",
    "pigment": AMD_PATHOLOGY_DIR / "pigment.onnx",
    "advanced_amd": AMD_PATHOLOGY_DIR / "advanced_amd.onnx",
    "ga": AMD_PATHOLOGY_DIR / "ga.onnx",
    "central_ga": AMD_PATHOLOGY_DIR / "central_ga.onnx",
}
VASCX_MODEL_DIR = PROJECT_ROOT / "models/vascx"
VASCX_BIN = Path(sys.executable).with_name("vascx")
INFERENCE_LOCK = threading.Lock()
MODELS: dict[str, object] = {}
READY_TASKS = {
    "fundus_lesions", "octa_vessels", "oct_structure", "oct_enhancement",
    "eye_age", "retinal_vascular", "oct_amd_pathology", "oct_fluid_subtypes",
    "fundus_amd_pathology",
}

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
OCT_AMD_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]
OCT_FLUID_NAMES = ["BG", "IRF", "SRF", "PED"]
OCT_FLUID_COLORS = np.array(
    [[0, 0, 0], [35, 122, 245], [255, 137, 55], [63, 190, 110]], dtype=np.uint8
)
FUNDUS_AMD_LABELS = {
    "drusen": ["无/小玻璃膜疣", "中等玻璃膜疣", "大玻璃膜疣"],
    "pigment": ["未见色素异常", "色素异常"],
    "advanced_amd": ["未见晚期 AMD", "晚期 AMD"],
    "ga": ["未见地图样萎缩", "地图样萎缩"],
    "central_ga": ["未见中心性地图样萎缩", "中心性地图样萎缩"],
}
FUNDUS_AMD_FINDING_LABELS = {
    "drusen": "中/大玻璃膜疣",
    "pigment": "色素异常",
    "advanced_amd": "晚期 AMD",
    "ga": "地图样萎缩",
    "central_ga": "中心性地图样萎缩",
}


class OCTAMDClassifier(nn.Module):
    """EfficientNet-B3 checkpoint released for CNV/DME/drusen OCT screening."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b3", pretrained=False, num_classes=0, global_pool=""
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.3), nn.Linear(1536, 512), nn.ReLU(inplace=True),
            nn.Dropout(0.15), nn.Linear(512, 4),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        features = self.backbone(tensor)
        return self.classifier(self.global_pool(features))


class ImagingRequest(BaseModel):
    image: str = Field(min_length=1)
    task: str = Field(pattern="^(fundus_lesions|octa_vessels|oct_structure|oct_enhancement|eye_age|retinal_vascular|oct_amd_pathology|oct_fluid_subtypes|fundus_amd_pathology)$")


def _token() -> str:
    return TOKEN_FILE.read_text(encoding="utf-8").strip() if TOKEN_FILE.is_file() else ""


def _write_status(status: str, detail: str = "") -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "host": socket.gethostname(),
        "port": PORT,
        "model": "Nine ophthalmic imaging tasks",
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
        (
            FUNDUS_CHECKPOINT, OCTA_CHECKPOINT, OCT_CHECKPOINT, OCT_ENHANCEMENT_CHECKPOINT,
            EYE_AGE_CHECKPOINT, OCT_AMD_CHECKPOINT, *OCT_FLUID_SUBTYPE_ONNX.values(),
            *FUNDUS_AMD_ONNX.values(), VASCX_MODEL_DIR / "artery_vein/av.pt",
            VASCX_MODEL_DIR / "vessels/vessels.pt", VASCX_MODEL_DIR / "disc/disc.pt",
            VASCX_MODEL_DIR / "fovea/fovea.pt", VASCX_BIN,
        )
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

    oct_amd = OCTAMDClassifier()
    oct_amd.load_state_dict(torch.load(OCT_AMD_CHECKPOINT, map_location="cpu", weights_only=True))
    oct_amd.eval().cuda()

    onnx_options = ort.SessionOptions()
    onnx_options.intra_op_num_threads = 2
    onnx_options.inter_op_num_threads = 1
    onnx_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    fundus_amd = {
        name: ort.InferenceSession(
            str(path), sess_options=onnx_options, providers=["CPUExecutionProvider"]
        )
        for name, path in FUNDUS_AMD_ONNX.items()
    }
    fluid_options = ort.SessionOptions()
    fluid_options.intra_op_num_threads = 4
    fluid_options.inter_op_num_threads = 1
    fluid_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    oct_fluid_subtypes = {
        name: ort.InferenceSession(
            str(path), sess_options=fluid_options, providers=["CPUExecutionProvider"]
        )
        for name, path in OCT_FLUID_SUBTYPE_ONNX.items()
    }

    eye_age_checkpoint = torch.load(EYE_AGE_CHECKPOINT, map_location="cpu", weights_only=False)
    eye_age = models.resnet101(weights=None)
    eye_age.fc = nn.Sequential(
        nn.Linear(2048, 512), nn.BatchNorm1d(512, momentum=0.1), nn.ReLU(inplace=True),
        nn.Dropout(0.4), nn.Linear(512, 128), nn.BatchNorm1d(128, momentum=0.1),
        nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(128, 1),
    )
    eye_age.load_state_dict(eye_age_checkpoint["model_state_dict"])
    eye_age.eval().cuda()
    eye_age.age_mean = float(eye_age_checkpoint["mean_age"])
    eye_age.age_std = float(eye_age_checkpoint["std_age"])
    MODELS.update(
        fundus_lesions=fundus, octa_vessels=octa,
        oct_structure=oct_model, oct_enhancement=oct_enhancement, eye_age=eye_age,
        retinal_vascular=True, oct_amd_pathology=oct_amd,
        oct_fluid_subtypes=oct_fluid_subtypes, fundus_amd_pathology=fundus_amd,
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


def _percentile_normalize(
    array: np.ndarray, region: np.ndarray | None = None, low: float = 5, high: float = 99
) -> np.ndarray:
    values = array[region] if region is not None and np.any(region) else array.reshape(-1)
    minimum, maximum = np.percentile(values, [low, high])
    return np.clip((array - minimum) / max(float(maximum - minimum), 1e-6), 0, 1)


def _octa_cnv_candidate(
    vessel_mask: np.ndarray, probability: np.ndarray, gray: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Create a vessel-model-derived CNV candidate from abnormal central flow density."""
    height, width = vessel_mask.shape
    yy, xx = np.ogrid[:height, :width]
    radius = np.sqrt((xx - width / 2) ** 2 + (yy - height / 2) ** 2)
    central = radius < min(height, width) * 0.38
    radial_prior = np.exp(-((radius / (min(height, width) * 0.48)) ** 2))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(
        np.uint8(np.clip(gray, 0, 1) * 255)
    ).astype(np.float32) / 255.0
    local_density = cv2.GaussianBlur(vessel_mask.astype(np.float32), (0, 0), 24)
    local_detail = cv2.GaussianBlur(clahe, (0, 0), 7)
    score = (
        0.58 * _percentile_normalize(local_density, central)
        + 0.24 * _percentile_normalize(local_detail, central)
        + 0.18 * probability
    ) * (0.65 + 0.35 * radial_prior)
    threshold = float(np.quantile(score[central], 0.94))
    candidate = np.uint8((score >= threshold) & central)
    candidate = cv2.morphologyEx(
        candidate, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    )
    candidate = cv2.morphologyEx(
        candidate, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    )
    count, components, stats, centroids = cv2.connectedComponentsWithStats(candidate, 8)
    clean = np.zeros_like(candidate)
    for label_id in range(1, count):
        distance = np.linalg.norm(centroids[label_id] - np.array([width / 2, height / 2]))
        if stats[label_id, cv2.CC_STAT_AREA] >= 700 and distance < min(height, width) * 0.35:
            clean[components == label_id] = 1
    return clean, np.clip(score, 0, 1)


def _octa_infer(image: Image.Image) -> dict:
    gray = np.asarray(
        image.convert("L").resize((1216, 1216), Image.Resampling.BILINEAR), dtype=np.float32
    ) / 255.0
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
    cnv_candidate, cnv_score = _octa_cnv_candidate(clean, probability, gray)

    base = cv2.cvtColor(np.uint8(gray * 255), cv2.COLOR_GRAY2RGB)
    vessel_color = base.copy()
    vessel_color[clean > 0] = (0, 236, 255)
    vessel_overlay = cv2.addWeighted(base, 0.5, vessel_color, 0.5, 0)

    heatmap = cv2.cvtColor(
        cv2.applyColorMap(np.uint8(cnv_score * 255), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB
    )
    cnv_overlay = cv2.addWeighted(base, 0.72, heatmap, 0.28, 0)
    cnv_overlay[cnv_candidate > 0] = (
        0.45 * cnv_overlay[cnv_candidate > 0] + 0.55 * np.array([255, 55, 180])
    ).astype(np.uint8)
    contours, _ = cv2.findContours(cnv_candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(cnv_overlay, contours, -1, (255, 220, 45), 4)
    return {
        "model": "OCTA vessel model with central CNV candidate extraction",
        "mask_png": _png64(clean * 255, "L"),
        "probability_png": _png64(np.uint8(probability * 255), "L"),
        "overlay_png": _png64(vessel_overlay),
        "cnv_candidate_mask_png": _png64(cnv_candidate * 255, "L"),
        "cnv_probability_png": _png64(np.uint8(cnv_score * 255), "L"),
        "cnv_overlay_png": _png64(cnv_overlay),
        "cnv_candidate_pixels": int(cnv_candidate.sum()),
        "cnv_candidate_ratio_percent": round(float(cnv_candidate.mean() * 100), 4),
        "cnv_candidate_components": len(contours),
        "input_size": [1216, 1216],
    }


def _oct_infer(image: Image.Image) -> dict:
    scan = np.asarray(image.convert("L").resize((512, 512), Image.Resampling.BICUBIC), dtype=np.float32) / 255.0
    normalized = (scan - float(scan.mean())) / max(float(scan.std()), 0.08)
    tensor = torch.from_numpy(normalized)[None, None].cuda()
    with torch.inference_mode():
        labels = MODELS["oct_structure"](tensor).argmax(1)[0].cpu().numpy().astype(np.uint8)
    # The public layer model can hallucinate labels in the black scan margins.
    # Estimate a smooth curved top boundary plus an adaptive lower tissue limit.
    # This keeps the retinal contour while removing upper/lower background layers.
    row_profile = cv2.GaussianBlur(
        scan.mean(axis=1).reshape(-1, 1), (1, 31), 0
    ).ravel()
    lower_threshold = max(0.18, float(np.percentile(row_profile, 60)) * 0.95)
    band_rows = row_profile >= lower_threshold
    band_rows[:50] = False
    ids = np.flatnonzero(band_rows)
    if ids.size:
        lower_limit = int(ids[-1])
    else:
        lower_limit = int(scan.shape[0] * 0.72)
    tissue_profile = cv2.GaussianBlur(scan, (1, 21), 0)
    tissue_threshold = max(0.10, float(np.percentile(row_profile, 45)) * 0.90)
    tissue_rows = tissue_profile >= tissue_threshold
    tissue_rows[:50, :] = False  # ignore annotations/reflections at the scan edge
    row_ids = np.arange(scan.shape[0])[:, None]
    top = np.where(
        tissue_rows.any(axis=0), tissue_rows.argmax(axis=0), scan.shape[0] // 6
    ).astype(np.float32)
    top = cv2.GaussianBlur(top.reshape(1, -1), (0, 0), 8).ravel()
    retinal_band = (
        (row_ids >= top[None, :])
        & (row_ids <= lower_limit)
        & (lower_limit - top[None, :] > 80)
    )
    labels[~retinal_band] = 0
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


def _oct_amd_pathology_infer(image: Image.Image) -> dict:
    resized = image.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
    rgb = np.asarray(resized)
    tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(resized)[None].cuda()
    model = MODELS["oct_amd_pathology"]
    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        features = model.backbone(tensor)
        features.retain_grad()
        logits = model.classifier(model.global_pool(features))
        probabilities = torch.softmax(logits, dim=1)[0]
        prediction = int(torch.argmax(probabilities))
        logits[0, prediction].backward()
        weights = features.grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * features).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
    cam = cam.detach().cpu().numpy()
    cam = (cam - float(cam.min())) / max(float(cam.max() - cam.min()), 1e-7)
    heatmap = cv2.cvtColor(
        cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB
    )
    overlay = cv2.addWeighted(rgb, 0.62, heatmap, 0.38, 0)
    scores = {
        name: round(float(probabilities[index].detach().cpu()), 7)
        for index, name in enumerate(OCT_AMD_NAMES)
    }
    return {
        "model": "Public OCT EfficientNet-B3",
        "prediction": OCT_AMD_NAMES[prediction],
        "confidence": scores[OCT_AMD_NAMES[prediction]],
        "probabilities": scores,
        "heatmap_png": _png64(overlay),
        "input_size": [224, 224],
    }


def _clahe_oct(image: Image.Image) -> np.ndarray:
    gray = np.asarray(image.convert("L").resize((512, 512), Image.Resampling.BILINEAR), dtype=np.float32)
    minimum, maximum = float(gray.min()), float(gray.max())
    scaled = np.uint8((gray - minimum) / max(maximum - minimum, 1e-6) * 255)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(scaled)


def _oct_fluid_probability(session: ort.InferenceSession, tensor: np.ndarray) -> np.ndarray:
    logits = session.run(None, {session.get_inputs()[0].name: tensor})[0]
    shifted = logits - logits.max(axis=1, keepdims=True)
    probability = np.exp(shifted)
    return probability / np.maximum(probability.sum(axis=1, keepdims=True), 1e-8)


def _oct_fluid_subtypes_infer(image: Image.Image) -> dict:
    processed = _clahe_oct(image)
    tensor = (processed.astype(np.float32) / 255.0)[None, None]
    model_probabilities = {}
    for name, session in MODELS["oct_fluid_subtypes"].items():
        direct = _oct_fluid_probability(session, tensor)
        flipped = _oct_fluid_probability(session, tensor[:, :, :, ::-1].copy())[:, :, :, ::-1]
        model_probabilities[name] = (direct + flipped) / 2.0

    probability = 0.92 * model_probabilities["slot1"] + 0.08 * model_probabilities["slot2"]
    labels = probability.argmax(axis=1)[0].astype(np.uint8)
    for class_id, minimum_area in ((1, 12), (2, 24), (3, 24)):
        class_mask = np.uint8(labels == class_id)
        count, components, stats, _ = cv2.connectedComponentsWithStats(class_mask, 8)
        for label_id in range(1, count):
            if stats[label_id, cv2.CC_STAT_AREA] < minimum_area:
                labels[components == label_id] = 0

    slot1_labels = model_probabilities["slot1"].argmax(axis=1)[0]
    slot2_labels = model_probabilities["slot2"].argmax(axis=1)[0]
    slot1_fluid = slot1_labels > 0
    slot2_fluid = slot2_labels > 0
    agreement = float(
        2 * np.logical_and(slot1_fluid, slot2_fluid).sum()
        / max(int(slot1_fluid.sum() + slot2_fluid.sum()), 1)
    )
    confidence = probability.max(axis=1)[0]
    lesion_confidence = confidence[labels > 0]

    original_gray = np.asarray(image.convert("L"), dtype=np.uint8)
    display_gray = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(12, 6)).apply(original_gray)
    display_labels = cv2.resize(
        labels, (image.width, image.height), interpolation=cv2.INTER_NEAREST
    )
    base = cv2.cvtColor(display_gray, cv2.COLOR_GRAY2RGB)
    overlay = base.copy()
    for class_id in range(1, len(OCT_FLUID_NAMES)):
        region = display_labels == class_id
        overlay[region] = (
            0.42 * base[region] + 0.58 * OCT_FLUID_COLORS[class_id]
        ).astype(np.uint8)
        contours, _ = cv2.findContours(
            np.uint8(region), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, tuple(map(int, OCT_FLUID_COLORS[class_id])), 2)

    masks = {
        name: _png64(np.uint8(display_labels == class_id) * 255, "L")
        for class_id, name in enumerate(OCT_FLUID_NAMES) if class_id > 0
    }
    return {
        "model": "Calibrated dual-model OCT fluid ensemble",
        "labels": OCT_FLUID_NAMES,
        "label_map_png": _png64(labels, "L"),
        "mask_pngs": masks,
        "overlay_png": _png64(overlay),
        "mean_confidence": round(float(lesion_confidence.mean()), 6)
        if lesion_confidence.size else round(float(confidence.mean()), 6),
        "ensemble_agreement_dice": round(agreement, 6),
        "input_size": [512, 512],
        "display_size": [image.width, image.height],
    }


def _fundus_occlusion_attention(
    cropped: Image.Image, session: ort.InferenceSession, finding_id: str,
    baseline_probability: float, grid: int = 9,
) -> np.ndarray:
    size = int(session.get_inputs()[0].shape[1])
    array = np.asarray(cropped.resize((size, size), Image.Resampling.BICUBIC), dtype=np.float32)
    tensor = array / 127.5 - 1.0
    patch_size = max(12, int(size * 0.20))
    centers_y = np.linspace(size * 0.10, size * 0.90, grid).astype(int)
    centers_x = np.linspace(size * 0.10, size * 0.90, grid).astype(int)
    batch = np.repeat(tensor[None], grid * grid, axis=0)
    index = 0
    for center_y in centers_y:
        for center_x in centers_x:
            y0, y1 = max(0, center_y - patch_size // 2), min(size, center_y + patch_size // 2)
            x0, x1 = max(0, center_x - patch_size // 2), min(size, center_x + patch_size // 2)
            batch[index, y0:y1, x0:x1] = 0
            index += 1
    output = session.run(None, {session.get_inputs()[0].name: batch})[0]
    occluded = output[:, 1:].sum(axis=1) if finding_id == "drusen" else output[:, 1]
    drops = np.maximum(baseline_probability - occluded, 0).reshape(grid, grid)
    heatmap = cv2.resize(drops, (512, 512), interpolation=cv2.INTER_CUBIC)
    heatmap = np.maximum(cv2.GaussianBlur(heatmap, (0, 0), 12), 0)
    return heatmap / max(float(heatmap.max()), 1e-8)


def _fundus_amd_pathology_infer(image: Image.Image) -> dict:
    source = image.convert("RGB")
    side = min(source.size)
    left = (source.width - side) // 2
    top = (source.height - side) // 2
    cropped = source.crop((left, top, left + side, top + side))
    findings = []
    attention_maps = []
    attention_weights = []
    sessions = MODELS["fundus_amd_pathology"]
    for name, session in sessions.items():
        shape = session.get_inputs()[0].shape
        size = int(shape[1])
        array = np.asarray(cropped.resize((size, size), Image.Resampling.BICUBIC), dtype=np.float32)
        tensor = array[None] / 127.5 - 1.0
        output = session.run(None, {session.get_inputs()[0].name: tensor})[0][0]
        labels = FUNDUS_AMD_LABELS[name]
        probabilities = {label: round(float(output[index]), 7) for index, label in enumerate(labels)}
        prediction = int(np.argmax(output))
        positive_probability = float(output[1:].sum()) if name == "drusen" else float(output[1])
        findings.append({
            "id": name,
            "label": FUNDUS_AMD_FINDING_LABELS[name],
            "prediction_label": labels[prediction],
            "status": "positive" if prediction > 0 else "negative",
            "confidence": round(float(output[prediction]), 7),
            "positive_probability": round(positive_probability, 7),
            "probabilities": probabilities,
        })
        if positive_probability >= 0.05:
            attention_maps.append(
                _fundus_occlusion_attention(cropped, session, name, positive_probability)
            )
            attention_weights.append(max(positive_probability, 0.08))

    if attention_maps:
        attention = np.max(
            np.stack([heatmap * weight for heatmap, weight in zip(attention_maps, attention_weights)]),
            axis=0,
        )
        attention /= max(float(attention.max()), 1e-8)
    else:
        attention = np.zeros((512, 512), dtype=np.float32)

    model_rgb = np.asarray(cropped.resize((512, 512), Image.Resampling.LANCZOS), dtype=np.float32)
    lightness = cv2.cvtColor(np.uint8(model_rgb), cv2.COLOR_RGB2LAB)[:, :, 0].astype(np.float32)
    yy, xx = np.ogrid[:512, :512]
    macular_region = (xx - 256) ** 2 + (yy - 256) ** 2 < (512 * 0.34) ** 2
    bright_residual = np.maximum(lightness - cv2.GaussianBlur(lightness, (0, 0), 6), 0)
    yellow_signal = (model_rgb[:, :, 0] + model_rgb[:, :, 1]) * 0.5 - model_rgb[:, :, 2]
    yellow_residual = np.maximum(yellow_signal - cv2.GaussianBlur(yellow_signal, (0, 0), 12), 0)
    feature = (
        0.72 * _percentile_normalize(bright_residual, macular_region)
        + 0.28 * _percentile_normalize(yellow_residual, macular_region)
    )
    candidate_score = feature * (0.28 + 0.72 * attention)
    threshold = float(np.percentile(candidate_score[macular_region], 96.8))
    candidate = np.uint8(
        (candidate_score >= threshold) & macular_region & (attention > 0.35)
    )
    candidate = cv2.morphologyEx(
        candidate, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    candidate = cv2.dilate(
        candidate, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )
    count, components, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    clean = np.zeros_like(candidate)
    for label_id in range(1, count):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if 12 <= area <= 2500:
            clean[components == label_id] = 1

    attention_side = cv2.resize(attention, (side, side), interpolation=cv2.INTER_CUBIC)
    candidate_side = cv2.resize(clean, (side, side), interpolation=cv2.INTER_NEAREST)
    original = np.asarray(source)
    overlay = original.copy()
    crop_rgb = original[top:top + side, left:left + side]
    crop_heatmap = cv2.cvtColor(
        cv2.applyColorMap(np.uint8(attention_side * 255), cv2.COLORMAP_TURBO),
        cv2.COLOR_BGR2RGB,
    )
    overlay_crop = cv2.addWeighted(crop_rgb, 0.78, crop_heatmap, 0.22, 0)
    overlay_crop[candidate_side > 0] = (
        0.35 * overlay_crop[candidate_side > 0] + 0.65 * np.array([177, 71, 255])
    ).astype(np.uint8)
    overlay[top:top + side, left:left + side] = overlay_crop
    full_mask = np.zeros(original.shape[:2], dtype=np.uint8)
    full_mask[top:top + side, left:left + side] = candidate_side
    contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 230, 80), max(2, side // 500))
    return {
        "model": "AMD finding ensemble with occlusion localization",
        "findings": findings,
        "overlay_png": _png64(overlay),
        "attention_png": _png64(np.uint8(attention_side * 255), "L"),
        "candidate_mask_png": _png64(full_mask * 255, "L"),
        "candidate_pixels": int(clean.sum()),
        "candidate_ratio_percent": round(float(clean.mean() * 100), 4),
        "candidate_components": len(contours),
        "input_crop": [side, side],
        "display_size": [source.width, source.height],
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


def _crop_eye_age(image: Image.Image, size: int = 224) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    points = cv2.findNonZero(np.uint8(gray > 25) * 255)
    if points is None:
        raise HTTPException(status_code=400, detail="No retinal field of view detected")
    x, y, width, height = cv2.boundingRect(points)
    scale = size / max(width, height)
    region = rgb[y:y + height, x:x + width]
    resized = cv2.resize(
        region, (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_LANCZOS4,
    )
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    top = (size - resized.shape[0]) // 2
    left = (size - resized.shape[1]) // 2
    canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    return canvas


def _eye_age_infer(image: Image.Image) -> dict:
    fitted = _crop_eye_age(image)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tensor = transform(Image.fromarray(fitted))[None].cuda()
    model = MODELS["eye_age"]
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []
    forward_handle = model.layer4[-1].register_forward_hook(
        lambda _module, _inputs, output: activations.append(output)
    )
    backward_handle = model.layer4[-1].register_full_backward_hook(
        lambda _module, _grad_input, grad_output: gradients.append(grad_output[0])
    )
    try:
        model.zero_grad(set_to_none=True)
        normalized_age = model(tensor)[0, 0]
        normalized_age.backward()
        activation = activations[0].detach()[0]
        gradient = gradients[0].detach()[0]
    finally:
        forward_handle.remove()
        backward_handle.remove()
    weights = gradient.mean(dim=(1, 2), keepdim=True)
    heatmap = torch.relu((weights * activation).sum(dim=0)).cpu().numpy()
    heatmap -= float(heatmap.min())
    heatmap /= max(float(heatmap.max()), 1e-8)
    heatmap = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_CUBIC)
    colored = cv2.cvtColor(cv2.applyColorMap(np.uint8(heatmap * 255), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(fitted, 0.66, colored, 0.34, 0)
    prediction = float(normalized_age.detach().cpu()) * model.age_std + model.age_mean
    gray = cv2.cvtColor(fitted, cv2.COLOR_RGB2GRAY)
    field = gray > 25
    return {
        "model": "Retinal age ResNet101 non-filtered",
        "predicted_age": round(prediction, 2),
        "overlay_png": _png64(overlay),
        "preprocessed_png": _png64(fitted),
        "heatmap_png": _png64(np.uint8(heatmap * 255), "L"),
        "quality": {
            "field_coverage_percent": round(float(field.mean() * 100), 1),
            "sharpness": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1),
            "status": "passed" if field.mean() >= 0.45 else "review",
        },
    }


def _read_one_row(path: Path) -> dict[str, float | None]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    row.pop("", None)
    result: dict[str, float | None] = {}
    for key, value in row.items():
        try:
            parsed = float(value)
            result[key] = parsed if np.isfinite(parsed) else None
        except (TypeError, ValueError):
            result[key] = None
    return result


def _retinal_vascular_infer(image: Image.Image) -> dict:
    work_root = RUNTIME_DIR / "systemic_gpu"
    work_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vascx-", dir=work_root) as temporary:
        work = Path(temporary)
        inputs = work / "input"
        output = work / "output"
        inputs.mkdir()
        image.save(inputs / "case.png", format="PNG")
        command = [
            str(VASCX_BIN), "run-models", str(inputs), str(output),
            "--no-quality", "--no-overlay", "--device", "cuda:0", "--n-jobs", "1",
            "--av-model", str(VASCX_MODEL_DIR / "artery_vein/av.pt"),
            "--vessels-model", str(VASCX_MODEL_DIR / "vessels/vessels.pt"),
            "--disc-model", str(VASCX_MODEL_DIR / "disc/disc.pt"),
            "--fovea-model", str(VASCX_MODEL_DIR / "fovea/fovea.pt"),
        ]
        run = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=180)
        if run.returncode:
            raise RuntimeError(f"Vascular model pipeline failed: {run.stderr[-1200:]}")
        biomarker_csv = work / "biomarkers.csv"
        quantify = subprocess.run(
            [str(VASCX_BIN), "calc-biomarkers", str(output), str(biomarker_csv),
             "--feature_set", "full_v3", "--n-jobs", "1"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=120,
        )
        if quantify.returncode:
            raise RuntimeError(f"Vascular quantification failed: {quantify.stderr[-1200:]}")

        base = np.asarray(Image.open(output / "preprocessed_rgb/case.png").convert("RGB"))
        vessels = np.asarray(Image.open(output / "vessels/case.png").convert("L")) > 0
        artery_vein = np.asarray(Image.open(output / "artery_vein/case.png").convert("L"))
        disc_small = np.asarray(Image.open(output / "disc/case.png").convert("L")) > 0
        disc = cv2.resize(np.uint8(disc_small), (base.shape[1], base.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        with (output / "fovea.csv").open("r", encoding="utf-8", newline="") as handle:
            fovea_row = next(csv.DictReader(handle))
        fovea = (round(float(fovea_row["x_fovea"])), round(float(fovea_row["y_fovea"])))

        tint = base.copy()
        tint[artery_vein == 1] = (255, 75, 76)
        tint[artery_vein == 2] = (69, 146, 255)
        tint[artery_vein == 3] = (0, 225, 204)
        overlay = cv2.addWeighted(base, 0.63, tint, 0.52, 0)
        contours, _ = cv2.findContours(np.uint8(disc), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 238, 95), 3)
        cv2.circle(overlay, fovea, 10, (255, 255, 255), 2)
        cv2.circle(overlay, fovea, 3, (255, 238, 95), -1)

        biomarkers = _read_one_row(biomarker_csv)
        field = base.max(axis=2) > 25
        gray = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY)
        checks = {
            "retinal_field": bool(field.mean() >= 0.45),
            "vessel_map": bool(vessels.sum() >= 2000),
            "optic_disc": bool(disc.sum() >= 400),
            "fovea": bool(0 <= fovea[0] < base.shape[1] and 0 <= fovea[1] < base.shape[0]),
        }
        return {
            "model": "VascX full_v3",
            "overlay_png": _png64(overlay),
            "preprocessed_png": _png64(base),
            "vessels_png": _png64(np.uint8(vessels) * 255, "L"),
            "artery_vein_png": _png64(artery_vein, "L"),
            "disc_png": _png64(np.uint8(disc) * 255, "L"),
            "fovea": [fovea[0], fovea[1]],
            "biomarkers": biomarkers,
            "quality": {
                "status": "passed" if all(checks.values()) else "review",
                "checks": checks,
                "completed_checks": sum(checks.values()),
                "field_coverage_percent": round(float(field.mean() * 100), 1),
                "sharpness": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1),
                "vessel_pixels": int(vessels.sum()),
            },
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
        "status": "ready" if READY_TASKS.issubset(MODELS) else "loading",
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
            "oct_amd_pathology": _oct_amd_pathology_infer,
            "oct_fluid_subtypes": _oct_fluid_subtypes_infer,
            "fundus_amd_pathology": _fundus_amd_pathology_infer,
            "eye_age": _eye_age_infer,
            "retinal_vascular": _retinal_vascular_infer,
        }[request.task](image)
    return {
        "task": request.task, **result,
        "runtime_ms": round((time.perf_counter() - started) * 1000, 1),
        "real_inference": True,
    }
