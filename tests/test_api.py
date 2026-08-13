import io
import math

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app
from shared import CLASS_NAMES

client = TestClient(app)


def synthetic_image_bytes():
    image = Image.new("RGB", (512, 512), (170, 80, 30))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    return buf


def test_root_returns_landing_page():
    response = client.get("/")
    assert response.status_code == 200
    assert "Fundus Classification API" in response.text or "Fundus DX" in response.text


def test_health_reports_loaded_model():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["version"]


def test_model_info_describes_supported_scope():
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert body["modality"] == "color_fundus_photograph"
    assert body["classes"] == CLASS_NAMES
    assert body["clinical_use"] is False


def test_predict_returns_valid_prediction():
    response = client.post(
        "/predict",
        files={"file": ("sample.png", synthetic_image_bytes(), "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in CLASS_NAMES
    assert 0.0 <= body["confidence"] <= 1.0
    assert set(body["probabilities"]) == set(CLASS_NAMES)
    assert math.isclose(sum(body["probabilities"].values()), 1.0, abs_tol=1e-4)
    assert body["confidence"] == pytest.approx(max(body["probabilities"].values()))
    assert body["inference_ms"] >= 0
    assert body["model_version"]


def test_predict_rejects_non_image_content_type():
    response = client.post(
        "/predict",
        files={"file": ("notes.txt", io.BytesIO(b"not an image"), "text/plain")},
    )
    assert response.status_code == 400


def test_predict_rejects_corrupt_image():
    response = client.post(
        "/predict",
        files={"file": ("fake.png", io.BytesIO(b"\x89PNG but not really"), "image/png")},
    )
    assert response.status_code == 400
