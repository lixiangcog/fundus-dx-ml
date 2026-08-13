from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import torch
from PIL import Image, UnidentifiedImageError
import io
import sys
import os
import time
from pathlib import Path
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared import CLASS_NAMES, get_device, get_inference_transform, build_resnet18
from api.pipelines import CAPABILITIES, PIPELINES, PIPELINE_INDEX

APP_VERSION = "2.0.0"
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


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None, "device": str(device),
            "version": APP_VERSION, "pipelines_ready": len(PIPELINES)}


@app.get("/capabilities")
async def capabilities():
    return {"modalities": ["OCT", "OCTA", "眼底彩照"], "capabilities": CAPABILITIES,
            "research_only": True, "version": APP_VERSION}


@app.get("/model-info")
async def model_info():
    return {"name": "FundusDx ResNet18", "modality": "color_fundus_photograph",
            "classes": class_names, "input_size": [224, 224], "validation_accuracy": 0.977,
            "clinical_use": False}


@app.post("/analyze/{pipeline_id}")
async def analyze(pipeline_id: str, file: UploadFile = File(...)):
    if pipeline_id not in PIPELINES:
        raise HTTPException(status_code=404, detail="未知分析功能")
    image = await read_image(file)
    try:
        result = PIPELINES[pipeline_id](image=image, model=model, transform=transform,
                                        class_names=class_names)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return {"pipeline": PIPELINE_INDEX[pipeline_id], "input": {"filename": file.filename,
            "width": image.width, "height": image.height}, **result, "model_version": APP_VERSION}


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


@app.get("/", response_class=HTMLResponse)
async def root():
    index = FRONTEND_DIST / "index.html"
    if index.is_file():
        return FileResponse(index)
    return "<h1>Ophthalmic workbench frontend is not built.</h1>"


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True, check_dir=False), name="frontend")
