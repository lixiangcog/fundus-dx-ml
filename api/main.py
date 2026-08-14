import asyncio
import json
import shutil
import uuid

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import torch
from PIL import Image, ImageChops, UnidentifiedImageError
import io
import sys
import os
import time
from pathlib import Path
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared import CLASS_NAMES, get_device, get_inference_transform, build_resnet18
from api.pipelines_v3 import CAPABILITIES, PIPELINES, PIPELINE_INDEX
from api.imaging_client import status as imaging_status
from api.samples import get_pipeline_reference, get_sample, public_catalog
from api.amd_agent import DEFAULT_CASE, public_status as amd_agent_status
from api.amd_agent import run_case as run_amd_case, run_default_case as run_default_amd_case
from api.systemic import SYSTEMIC_MODULES, public_config as systemic_public_config
from api.systemic import run_module as run_systemic_module, sample_path as systemic_sample_path

APP_VERSION = "5.0.0"
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"

app = FastAPI(title="Ophthalmic Multimodal Analysis API", version=APP_VERSION,
              description="Research-only OCT, OCTA and color fundus imaging workbench.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

class_names = CLASS_NAMES
device = get_device()


def load_model():
    loaded = build_resnet18(len(class_names))
    loaded.load_state_dict(torch.load(PROJECT_ROOT / "best_model.pth", map_location=device))
    loaded = loaded.to(device)
    loaded.eval()
    return loaded


model = load_model()
transform = get_inference_transform()


async def read_image(file: UploadFile) -> Image.Image:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="文件必须是 JPG、PNG 或 WebP 图像")
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="图像不能超过 12 MB")
    try:
        return Image.open(io.BytesIO(contents)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="无法解析上传的图像")


def _normalize_amd_case(payload: object) -> dict:
    """Validate user-entered longitudinal fields before GPU inference."""
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="病例信息必须是 JSON 对象")

    def object_field(name: str) -> dict:
        value = payload.get(name)
        if not isinstance(value, dict):
            raise HTTPException(status_code=400, detail=f"病例缺少{name}信息")
        return value

    def text_value(value: object, label: str, limit: int) -> str:
        text = str(value or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail=f"请填写{label}")
        if len(text) > limit:
            raise HTTPException(status_code=400, detail=f"{label}不能超过 {limit} 个字符")
        return text

    def number_value(value: object, label: str, minimum: float, maximum: float) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail=f"{label}必须是数字")
        if not minimum <= number <= maximum:
            raise HTTPException(status_code=400, detail=f"{label}应在 {minimum:g} 至 {maximum:g} 之间")
        return number

    patient = object_field("patient")
    treatment = object_field("treatment")
    visits = payload.get("visits")
    if not isinstance(visits, list) or len(visits) != 2 or not all(isinstance(item, dict) for item in visits):
        raise HTTPException(status_code=400, detail="病例必须包含基线和随访两次就诊")

    normalized = dict(payload)
    normalized["patient"] = {
        **patient,
        "age": int(number_value(patient.get("age"), "年龄", 18, 110)),
        "sex": text_value(patient.get("sex"), "性别", 20),
        "eye": text_value(patient.get("eye"), "眼别", 20),
        "diagnosis": text_value(patient.get("diagnosis"), "诊断", 160),
    }
    normalized["treatment"] = {
        **treatment,
        "agent": text_value(treatment.get("agent"), "治疗方式", 160),
        "injections": int(number_value(treatment.get("injections"), "治疗次数", 0, 200)),
        "current_interval_weeks": text_value(treatment.get("current_interval_weeks"), "治疗间隔", 40),
    }
    normalized["context"] = text_value(payload.get("context"), "病例记录", 1200)
    normalized["visits"] = [
        {
            **visit,
            "id": text_value(visit.get("id") or f"V{index}", "就诊编号", 20),
            "label": text_value(visit.get("label") or ("基线" if index == 0 else "随访"), "就诊名称", 40),
            "date": text_value(visit.get("date"), "就诊日期", 40),
            "bcva_decimal": number_value(visit.get("bcva_decimal"), "视力", 0, 2),
        }
        for index, visit in enumerate(visits)
    ]
    return normalized


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None, "device": str(device),
            "version": APP_VERSION, "pipelines_ready": len(PIPELINES),
            "imaging_service": imaging_status()}


@app.get("/capabilities")
async def capabilities():
    return {"modalities": ["OCT", "OCTA", "眼底彩照"], "capabilities": CAPABILITIES,
            "samples": public_catalog(), "research_only": True, "version": APP_VERSION}


@app.get("/systemic/config")
async def systemic_config():
    return {"modules": systemic_public_config(), "research_only": True, "version": APP_VERSION}


@app.get("/systemic/sample/{module_id}")
async def systemic_sample(module_id: str):
    path = systemic_sample_path(module_id)
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="研究样本不存在")
    return FileResponse(path, headers={"Cache-Control": "no-store, max-age=0"})


@app.post("/systemic/analyze/{module_id}")
async def analyze_systemic(
    module_id: str,
    file: UploadFile = File(...),
    chronological_age: float | None = Form(None),
):
    if module_id not in SYSTEMIC_MODULES:
        raise HTTPException(status_code=404, detail="未知研究模块")
    if chronological_age is not None and not 18 <= chronological_age <= 100:
        raise HTTPException(status_code=400, detail="实际年龄应在 18 至 100 岁之间")
    image = await read_image(file)
    upload_dir = PROJECT_ROOT / "runtime" / "systemic_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    image_path = upload_dir / f"{uuid.uuid4().hex}.png"
    image.save(image_path, format="PNG")
    try:
        return await asyncio.to_thread(run_systemic_module, module_id, image_path, chronological_age)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    finally:
        image_path.unlink(missing_ok=True)


@app.get("/research-samples/{sample_id}")
async def research_sample(sample_id: str):
    sample = get_sample(sample_id)
    if not sample or not sample["path"].is_file():
        raise HTTPException(status_code=404, detail="研究样本不存在")
    return FileResponse(sample["path"], headers={"Cache-Control":"no-store, max-age=0"})


@app.get("/model-info")
async def model_info():
    return {"name": "FundusDx ResNet18", "modality": "color_fundus_photograph",
            "classes": class_names, "input_size": [224, 224], "validation_accuracy": 0.977,
            "clinical_use": False}


@app.post("/analyze/{pipeline_id}")
async def analyze(pipeline_id: str, file: UploadFile = File(...), sample_id: str | None = Form(None)):
    if pipeline_id not in PIPELINES:
        raise HTTPException(status_code=404, detail="未知分析功能")
    image = await read_image(file)
    upload_dir = PROJECT_ROOT / "runtime/imaging_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    image_path = upload_dir / f"{uuid.uuid4().hex}.png"
    image.save(image_path, format="PNG")
    reference = get_pipeline_reference(sample_id, pipeline_id)
    if reference:
        registered = Image.open(reference["path"]).convert("RGB")
        if registered.size != image.size or ImageChops.difference(registered, image).getbbox() is not None:
            reference = None
    try:
        result = PIPELINES[pipeline_id](image=image, model=model, transform=transform,
                                        class_names=class_names, image_path=image_path, reference=reference)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    finally:
        image_path.unlink(missing_ok=True)
    return {"pipeline": PIPELINE_INDEX[pipeline_id], "input": {"filename": file.filename,
            "width": image.width, "height": image.height, "sample_id": sample_id,
            "reference_applied": reference is not None}, **result, "model_version": APP_VERSION,
            "real_inference": True}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Compatibility endpoint retained for clients of v1."""
    image = await read_image(file)
    started = time.perf_counter()
    with torch.inference_mode():
        probabilities = torch.softmax(model(transform(image).unsqueeze(0).to(device)), dim=1)[0]
        index = int(torch.argmax(probabilities))
    return {"prediction": class_names[index], "confidence": float(probabilities[index]),
            "probabilities": {class_names[i]: float(probabilities[i]) for i in range(len(class_names))},
            "inference_ms": round((time.perf_counter()-started)*1000, 1), "model_version": APP_VERSION}


@app.get("/amd-agent/config")
async def amd_config():
    return {
        "name": "Longitudinal AMD Evidence Agent",
        "default_case": DEFAULT_CASE,
        "service": amd_agent_status(),
        "required_images": ["baseline_oct", "baseline_octa", "baseline_fundus",
                            "followup_oct", "followup_octa", "followup_fundus"],
        "outputs": [
            "OCT 层结构与液体分割",
            "OCTA 血管分割与微血管定量",
            "眼底彩照病灶定位与面积量化",
            "基线—随访量化变化",
            "循证随访建议与结构化操作规划",
        ],
        "research_only": True,
    }


@app.get("/amd-agent/status")
async def amd_status():
    return amd_agent_status()


@app.post("/amd-agent/analyze-default")
async def analyze_default_amd():
    try:
        return await asyncio.to_thread(run_default_amd_case, model, transform, class_names)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/amd-agent/analyze")
async def analyze_amd(
    case_json: str = Form(...),
    baseline_oct: UploadFile = File(...), baseline_octa: UploadFile = File(...),
    baseline_fundus: UploadFile = File(...), followup_oct: UploadFile = File(...),
    followup_octa: UploadFile = File(...), followup_fundus: UploadFile = File(...),
):
    try:
        case = _normalize_amd_case(json.loads(case_json))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"病例 JSON 格式无效：{exc.msg}")
    upload_dir = PROJECT_ROOT / "runtime" / "amd_uploads" / uuid.uuid4().hex
    upload_dir.mkdir(parents=True, exist_ok=False)
    uploads = {
        "baseline_oct": baseline_oct, "baseline_octa": baseline_octa,
        "baseline_fundus": baseline_fundus, "followup_oct": followup_oct,
        "followup_octa": followup_octa, "followup_fundus": followup_fundus,
    }
    paths = {}
    try:
        for key, upload in uploads.items():
            image = await read_image(upload)
            path = upload_dir / f"{key}.png"
            image.save(path, format="PNG")
            paths[key] = path
        return await asyncio.to_thread(run_amd_case, case, paths, model, transform, class_names)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    finally:
        shutil.rmtree(upload_dir, ignore_errors=True)



@app.get("/", response_class=HTMLResponse)
async def root():
    index = FRONTEND_DIST / "index.html"
    if index.is_file():
        return FileResponse(index)
    return "<h1>Ophthalmic workbench frontend is not built.</h1>"


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True, check_dir=False), name="frontend")
