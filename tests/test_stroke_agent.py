from pathlib import Path

from fastapi.testclient import TestClient

from api import stroke_agent
from api.main import app


client = TestClient(app)


def test_revised_fsrp_is_bounded_and_rises_with_risk_factors():
    low = dict(stroke_agent.DEFAULT_STROKE_PROFILE)
    low.update(age=55, systolic_bp=110, diabetes=False, antihypertensive=False)
    high = dict(stroke_agent.DEFAULT_STROKE_PROFILE)
    high.update(age=80, systolic_bp=190, smoker=True, diabetes=True,
                atrial_fibrillation=True, cardiovascular_disease=True)

    low_result = stroke_agent.calculate_revised_fsrp(low)
    high_result = stroke_agent.calculate_revised_fsrp(high)

    assert 0 <= low_result["probability"] <= 1
    assert 0 <= high_result["probability"] <= 1
    assert high_result["percent"] > low_result["percent"]
    assert high_result["outcome"] == "首次卒中"


def test_report_placeholders_are_rejected_and_punctuation_is_cleaned():
    fallback = ["实际影像信号"]
    assert stroke_agent._as_list(["可见影像表现", "第二项表现"], fallback) == fallback
    assert stroke_agent._as_list(["可见视网膜血管改变。"], fallback) == ["可见视网膜血管改变"]


def test_stroke_agent_runs_both_real_model_steps(monkeypatch):

    monkeypatch.setattr(stroke_agent, "_visionunite_infer", lambda *args, **kwargs: {
        "observations": [{"text": "可见血管轻度迂曲", "signal_flags": {}}],
        "runtime_ms": 12.0, "model": "VisionUnite V1", "real_inference": True,
    })
    monkeypatch.setattr(stroke_agent, "_qwen_infer", lambda *args, **kwargs: {
        "text": '{"image_findings":["血管轻度迂曲"],"recommended_checks":["复测血压"],"integrated_interpretation":"风险数值与眼底证据已综合。","uncertainty":""}',
        "runtime_ms": 18.0, "model": "Qwen2.5-VL", "real_inference": True,
    })
    vascular = {
        "runtime_ms": 8.0,
        "biomarkers": {
            "full_cre_arteries": 120, "full_cre_veins": 160,
            "vd_disc_full_arteries": 0.12, "vd_disc_full_veins": 0.16,
            "lw_tort_dist_arteries": 1.1, "lw_tort_dist_veins": 1.2,
            "mean_sparsity_vessels": 0.08,
        },
    }

    result = stroke_agent.analyze_stroke_risk(Path("sample.png"), vascular, stroke_agent.DEFAULT_STROKE_PROFILE)

    assert result["real_inference"] is True
    assert result["image_findings"] == ["未见明确出血或渗出信号", "未见明确动静脉异常信号"]
    assert result["model_synthesis"]["image_findings"] == ["血管轻度迂曲"]
    assert {item["tool"] for item in result["trace"]} == {
        "retinal_vascular_quantification", "fundus_specialist",
        "multimodal_report", "stroke_risk_formula",
    }


def test_cerebrovascular_endpoint_requires_risk_inputs():
    response = client.post(
        "/systemic/analyze/cerebrovascular-retina",
        files={"file": ("sample.png", b"not-reached", "image/png")},
    )
    assert response.status_code == 400
    assert "年龄" in response.json()["detail"]
