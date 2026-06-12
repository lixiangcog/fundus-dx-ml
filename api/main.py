from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
from PIL import Image, UnidentifiedImageError
import io
import sys
import os
import uvicorn

# Add parent dir to path so 'shared' resolves when running from api/ or project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared import CLASS_NAMES, get_device, get_inference_transform, build_resnet18

app = FastAPI(title="Fundus Classification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class_names = CLASS_NAMES
device = get_device()

def load_model():
    model = build_resnet18(len(class_names))

    model_path = "best_model.pth"
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()
    return model

model = load_model()

transform = get_inference_transform()

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fundus Classification API</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            color: #e2e8f0;
        }
        .container {
            text-align: center;
            padding: 3rem;
            max-width: 500px;
        }
        .icon {
            font-size: 4rem;
            margin-bottom: 1.5rem;
        }
        h1 {
            font-size: 1.75rem;
            font-weight: 600;
            margin-bottom: 0.75rem;
            color: #f8fafc;
        }
        .subtitle {
            color: #94a3b8;
            margin-bottom: 2rem;
            line-height: 1.6;
        }
        .status {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            background: rgba(34, 197, 94, 0.1);
            border: 1px solid rgba(34, 197, 94, 0.3);
            padding: 0.5rem 1rem;
            border-radius: 9999px;
            font-size: 0.875rem;
            color: #4ade80;
            margin-bottom: 2rem;
        }
        .status::before {
            content: '';
            width: 8px;
            height: 8px;
            background: #4ade80;
            border-radius: 50%;
        }
        .btn {
            display: inline-block;
            background: linear-gradient(135deg, #f43f5e 0%, #e11d48 100%);
            color: white;
            text-decoration: none;
            padding: 0.875rem 2rem;
            border-radius: 0.5rem;
            font-weight: 500;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(244, 63, 94, 0.3);
        }
        .classes {
            margin-top: 2.5rem;
            padding-top: 2rem;
            border-top: 1px solid #334155;
        }
        .classes-title {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: #64748b;
            margin-bottom: 1rem;
        }
        .tags {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            justify-content: center;
        }
        .tag {
            background: #334155;
            padding: 0.375rem 0.75rem;
            border-radius: 0.375rem;
            font-size: 0.8rem;
            color: #cbd5e1;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="icon">👁️</div>
        <h1>Fundus Classification API</h1>
        <p class="subtitle">
            Deep learning model for classifying retinal fundus images.
            Built with PyTorch and FastAPI.
        </p>
        <div class="status">API Running</div>
        <br><br>
        <a href="https://fundus-dx-ml.vercel.app" class="btn">Open Demo App</a>
        <div class="classes">
            <div class="classes-title">Supported Classifications</div>
            <div class="tags">
                <span class="tag">AMD</span>
                <span class="tag">Cataract</span>
                <span class="tag">Diabetic Retinopathy</span>
                <span class="tag">Normal</span>
            </div>
        </div>
    </div>
</body>
</html>
"""

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

        top_prob, top_catid = torch.topk(probabilities, 1)

        confidence = top_prob.item()
        predicted_class = class_names[top_catid.item()]

        all_probs = {class_names[i]: probabilities[0][i].item() for i in range(len(class_names))}

    return {
        "prediction": predicted_class,
        "confidence": confidence,
        "probabilities": all_probs
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
