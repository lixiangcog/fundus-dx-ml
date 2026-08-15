from pathlib import Path

from fastapi.testclient import TestClient

from api import coronary_agent
from api.main import app


client = TestClient(app)


def test_coronary_agent_runs_dual_image_model_chain(monkeypatch):
    monkeypatch.setattr(coronary_agent, "_visionunite_infer", lambda *args, **kwargs: {
        "observations": [
            {"text": "彩照可见血管轻度迂曲"},
            {"text": "OCTA 可见局部毛细血管稀疏"},
        ],
        "runtime_ms": 12.0,
        "model": "VisionUnite V1",
        "real_inference": True,
    })
    monkeypatch.setattr(coronary_agent, "_qwen_infer", lambda *args, **kwargs: {
        "text": (
            '{"risk_level":"high","cfp_summary":"血管轻度迂曲",'
            '"octa_summary":"局部毛细血管稀疏","integrated_summary":"双模态风险信号一致",'
            '"recommended_checks":["复核血压、血脂和血糖"]}'
        ),
        "runtime_ms": 18.0,
        "model": "Qwen2.5-VL",
        "real_inference": True,
    })
    cfp = {
        "runtime_ms": 8.0,
        "biomarkers": {
            "full_cre_arteries": 120,
            "full_cre_veins": 160,
            "vd_disc_full_arteries": 0.12,
            "vd_disc_full_veins": 0.16,
            "lw_tort_dist_arteries": 1.1,
            "lw_tort_dist_veins": 1.2,
            "mn_bifangle_arteries": 70,
            "mn_bifangle_veins": 74,
            "mean_sparsity_vessels": 0.08,
        },
    }
    octa = {
        "runtime_ms": 9.0,
        "metrics": [
            {"label": "血管密度", "value": 38.5},
            {"label": "中央无血管候选核心", "value": 220},
            {"label": "候选等效直径", "value": 16.7},
            {"label": "骨架总长度", "value": 3200},
            {"label": "分支点", "value": 45},
            {"label": "平均管径代理", "value": 2.3},
            {"label": "分形维数", "value": 1.62},
        ],
    }

    result = coronary_agent.analyze_coronary_risk(
        Path("cfp.jpg"), Path("octa.png"), cfp, octa
    )

    assert result["risk_level"] == "high"
    assert result["risk_label"] == "较高"
    assert result["real_inference"] is True
    assert {item["tool"] for item in result["trace"]} == {
        "cfp_vascular_quantification",
        "octa_microvascular_quantification",
        "multimodal_retinal_specialist",
        "coronary_risk_synthesis",
    }


def test_cardiovascular_endpoint_requires_octa():
    response = client.post(
        "/systemic/analyze/cardiovascular-retina",
        files={"file": ("cfp.png", b"not-reached", "image/png")},
    )
    assert response.status_code == 400
    assert "OCTA" in response.json()["detail"]


def test_cardiovascular_config_exposes_both_samples():
    response = client.get("/systemic/config")
    module = next(
        item for item in response.json()["modules"]
        if item["id"] == "cardiovascular-retina"
    )
    assert module["sample_url"]
    assert "modality=octa" in module["sample_octa_url"]
