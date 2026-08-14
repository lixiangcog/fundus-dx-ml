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
    assert "视界智析" in response.text or "Fundus Classification API" in response.text


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




def test_v5_catalog_exposes_six_calibrated_pipelines():
    response = client.get("/capabilities")
    assert response.status_code == 200
    body = response.json()
    assert body["version"] == "5.0.0"
    assert len(body["capabilities"]) == 6
    assert {item["id"] for item in body["capabilities"]} >= {
        "fundus-lesion-quantification", "oct-fluid-quantification",
        "vascular-quantification", "structure-segmentation",
    }
    assert all(item.get("sample_id") for item in body["capabilities"])


def test_research_sample_disables_browser_cache():
    response = client.get("/research-samples/oct-enhancement-duke-s10-32")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store, max-age=0"


def test_quality_enhancement_uses_external_test_case_and_pretrained_engine():
    response = client.get("/capabilities")
    catalog = {item["id"]: item for item in response.json()["capabilities"]}
    enhancement = catalog["quality-enhancement"]
    assert enhancement["sample_id"] == "oct-enhancement-duke-s10-32"
    assert enhancement["engine_type"] == "pretrained_model"
    assert enhancement["status"] == "validated"


def test_default_fundus_case_uses_disclosed_four_class_selection():
    response = client.get("/capabilities")
    catalog = {item["id"]: item for item in response.json()["capabilities"]}
    assert catalog["disease-screening"]["sample_id"] == "fundus-screen-idrid-67"
    assert catalog["fundus-lesion-quantification"]["sample_id"] == "fundus-lesions-idrid-67"


def test_mismatched_sample_id_cannot_apply_reference_truth():
    response = client.post(
        "/analyze/quality-enhancement",
        data={"sample_id": "oct-enhancement-duke-s10-32"},
        files={"file": ("different.png", synthetic_image_bytes(), "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["input"]["reference_applied"] is False
    assert body["quality"]["status"] == "unverified"
def test_predict_rejects_corrupt_image():
    response = client.post(
        "/predict",
        files={"file": ("fake.png", io.BytesIO(b"\x89PNG but not really"), "image/png")},
    )
    assert response.status_code == 400


def test_amd_status_requires_both_real_mllm_services():
    response = client.get("/amd-agent/status")
    assert response.status_code == 200
    body = response.json()
    assert set(body["services"]) == {"multimodal", "fundus_specialist"}
    assert body["fallback_generation"] is False
    assert body["real_inference_required"] is True


def test_amd_config_declares_dual_specialist_runtime():
    response = client.get("/amd-agent/config")
    assert response.status_code == 200
    body = response.json()
    assert body["research_only"] is True
    assert body["service"]["model"] == "Qwen2.5-VL-3B-Instruct + VisionUnite V1"
    assert len(body["required_images"]) == 6
    case = body["default_case"]
    assert case["case_id"] == "CASE_001"
    assert case["evidence_origin"] == "reported_reference"
    assert case["visits"][0]["bcva_decimal"] == 0.3
    assert case["visits"][1]["bcva_decimal"] == 0.5
    assert case["image_quality"]["status"] == "review"


def test_systemic_config_exposes_three_real_modules():
    response = client.get("/systemic/config")
    assert response.status_code == 200
    body = response.json()
    assert body["version"] == "5.0.0"
    assert {item["id"] for item in body["modules"]} == {
        "eye-age", "cardiovascular-retina", "cerebrovascular-retina",
    }
    assert all(item["source_url"].startswith("https://github.com/") for item in body["modules"])
    assert all(item["license"] for item in body["modules"])


@pytest.mark.parametrize("module_id", ["eye-age", "cardiovascular-retina", "cerebrovascular-retina"])
def test_systemic_default_samples_are_available(module_id):
    response = client.get(f"/systemic/sample/{module_id}")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store, max-age=0"
    assert response.headers["content-type"].startswith("image/")


def test_eye_age_rejects_invalid_chronological_age():
    response = client.post(
        "/systemic/analyze/eye-age",
        data={"chronological_age": "8"},
        files={"file": ("sample.png", synthetic_image_bytes(), "image/png")},
    )
    assert response.status_code == 400
