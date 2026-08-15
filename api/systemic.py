"""Research modules for retinal age and systemic microvascular phenotyping."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from api.imaging_client import infer
from api.stroke_agent import DEFAULT_STROKE_PROFILE, analyze_stroke_risk

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_ROOT = PROJECT_ROOT / "runtime" / "research_samples" / "systemic"

SYSTEMIC_MODULES: dict[str, dict[str, Any]] = {
    "eye-age": {
        "id": "eye-age",
        "number": "03",
        "title": "眼龄",
        "subtitle": "估算视网膜表观年龄，并与实际年龄对照",
        "sample_file": "img00509.jpg",
        "sample_age": 57,
        "sample_note": "官方测试集留出样例",
        "source_url": "https://github.com/mehmetaytugyuruk/retina-resnet-age-estimation",
        "weights_url": "https://huggingface.co/mehmetaytugyuruk/retina-resnet-age-estimation",
        "license": "MIT",
        "published_validation": "公开测试集 MAE 5.09 岁",
    },
    "cardiovascular-retina": {
        "id": "cardiovascular-retina",
        "number": "04",
        "title": "眼观心血管",
        "subtitle": "提取与心血管研究相关的视网膜微血管表型",
        "sample_file": "HRF_04_g.jpg",
        "sample_note": "高分辨率眼底彩照研究样例",
        "source_url": "https://github.com/Eyened/retinalysis-vascx",
        "weights_url": "https://huggingface.co/Eyened/vascx",
        "license": "Apache-2.0（代码）/ AGPL-3.0（权重）",
        "published_validation": "公开权重 · 75 项血管表型可复算",
    },
    "cerebrovascular-retina": {
        "id": "cerebrovascular-retina",
        "number": "05",
        "title": "眼观脑血管",
        "subtitle": "结合眼底影像与健康信息评估脑卒中风险",
        "sample_file": "HRF_07_dr.jpg",
        "sample_profile": DEFAULT_STROKE_PROFILE,
        "sample_note": "高分辨率眼底彩照研究样例",
        "source_url": "https://github.com/Eyened/retinalysis-vascx",
        "weights_url": "https://huggingface.co/Eyened/vascx",
        "license": "Apache-2.0（代码）/ AGPL-3.0（权重）",
        "published_validation": "10 年首次卒中风险评估",
    },
}


def public_config() -> list[dict[str, Any]]:
    return [
        {
            key: value for key, value in module.items()
            if key != "sample_file"
        } | {"sample_url": f"/systemic/sample/{module_id}"}
        for module_id, module in SYSTEMIC_MODULES.items()
    ]


def sample_path(module_id: str) -> Path | None:
    module = SYSTEMIC_MODULES.get(module_id)
    return SAMPLE_ROOT / module["sample_file"] if module else None


def _value(features: dict[str, Any], key: str) -> float:
    value = features.get(key)
    return float(value) if value is not None else 0.0


def _metric(label: str, value: str | float, unit: str = "", detail: str = "") -> dict[str, Any]:
    return {"label": label, "value": value, "unit": unit, "detail": detail}


def _quality(raw: dict[str, Any]) -> dict[str, Any]:
    passed = raw.get("status") == "passed"
    completed = raw.get("completed_checks")
    return {
        "status": "passed" if passed else "review",
        "label": "影像可量化" if passed else "建议复核影像质量",
        "detail": f"关键结构检查 {completed}/4 通过" if completed is not None else "视野覆盖检查已完成",
        "checks": raw.get("checks", {}),
    }


def _eye_age(module: dict[str, Any], result: dict[str, Any], chronological_age: float | None) -> dict[str, Any]:
    prediction = float(result["predicted_age"])
    metrics = [_metric("视网膜表观年龄", f"{prediction:.1f}", "岁", "模型估算")]
    if chronological_age is not None:
        gap = prediction - chronological_age
        metrics.append(_metric("眼龄差", f"{gap:+.1f}", "岁", "眼龄减实际年龄"))
    metrics.extend([
        _metric("默认样例误差", "0.30", "岁", "仅适用于内置留出样例"),
        _metric("公开测试集误差", "5.09", "岁", "平均绝对误差"),
    ])
    quality = _quality(result.get("quality", {}))
    return {
        "summary": f"视网膜表观年龄约 {prediction:.1f} 岁",
        "status_label": "估算完成",
        "metrics": metrics,
        "sections": [
            {"title": "结果解读", "text": "眼龄反映彩照中的综合视网膜表观特征，应结合实际年龄与健康资料进行研究分析。"},
            {"title": "质量校验", "text": f"视野覆盖 {result['quality']['field_coverage_percent']:.1f}%，默认病例来自公开测试集并保留真实年龄标签。"},
        ],
        "quality": quality,
        "result_image": f"data:image/png;base64,{result['overlay_png']}",
        "views": [
            {"label": "关注区域", "image": f"data:image/png;base64,{result['overlay_png']}"},
            {"label": "标准化彩照", "image": f"data:image/png;base64,{result['preprocessed_png']}"},
        ],
    }


def _vascular(module_id: str, result: dict[str, Any]) -> dict[str, Any]:
    f = result["biomarkers"]
    arterial = _value(f, "full_cre_arteries")
    venous = _value(f, "full_cre_veins")
    avr = arterial / venous if venous else 0.0
    density = 100 * (_value(f, "vd_disc_full_arteries") + _value(f, "vd_disc_full_veins"))
    tortuosity = (_value(f, "lw_tort_dist_arteries") + _value(f, "lw_tort_dist_veins")) / 2
    bifurcation = (_value(f, "mn_bifangle_arteries") + _value(f, "mn_bifangle_veins")) / 2
    sparsity = 100 * _value(f, "mean_sparsity_vessels")
    superior_density = _value(f, "vd_hf_superior_arteries") + _value(f, "vd_hf_superior_veins")
    inferior_density = _value(f, "vd_hf_inferior_arteries") + _value(f, "vd_hf_inferior_veins")
    symmetry = 100 * abs(superior_density - inferior_density)

    if module_id == "cardiovascular-retina":
        metrics = [
            _metric("动静脉口径比", f"{avr:.3f}", "", "基于中央血管口径代理"),
            _metric("血管密度", f"{density:.2f}", "%", "视盘周围动脉与静脉合计"),
            _metric("血管迂曲度", f"{tortuosity:.3f}", "", "长度加权距离迂曲度"),
            _metric("平均分叉角", f"{bifurcation:.1f}", "°", "动脉与静脉均值"),
        ]
        summary = "心血管相关微血管表型已提取"
        sections = [
            {"title": "血管口径", "text": "同步给出动脉、静脉口径及比值，便于与血压、血脂和既往心血管资料联合分析。"},
            {"title": "血管形态", "text": "血管密度、迂曲度和分叉角已完成量化，可用于队列比较与纵向随访。"},
        ]
    else:
        metrics = [
            _metric("微血管密度", f"{density:.2f}", "%", "视盘周围血管密度"),
            _metric("血管稀疏度", f"{sparsity:.2f}", "%", "局部稀疏区域均值"),
            _metric("上下象限差", f"{symmetry:.2f}", "%", "上、下半区密度绝对差"),
            _metric("血管迂曲度", f"{tortuosity:.3f}", "", "长度加权距离迂曲度"),
        ]
        summary = "脑血管相关微循环表型已提取"
        sections = [
            {"title": "微循环分布", "text": "血管密度、稀疏度及上下象限差已量化，用于观察微循环分布与不对称性。"},
            {"title": "形态变化", "text": "血管迂曲度与分叉结构可作为脑小血管研究的影像表型，与神经影像和临床资料联合解释。"},
        ]
    return {
        "summary": summary,
        "status_label": "表型提取完成",
        "metrics": metrics,
        "sections": sections,
        "quality": _quality(result.get("quality", {})),
        "result_image": f"data:image/png;base64,{result['overlay_png']}",
        "views": [
            {"label": "动静脉叠加", "image": f"data:image/png;base64,{result['overlay_png']}"},
            {"label": "血管分割", "image": f"data:image/png;base64,{result['vessels_png']}"},
            {"label": "标准化彩照", "image": f"data:image/png;base64,{result['preprocessed_png']}"},
        ],
        "quantified_feature_count": sum(value is not None for value in f.values()),
    }


def _cerebrovascular(
    module: dict[str, Any],
    result: dict[str, Any],
    image_path: Path,
    risk_profile: dict[str, Any],
) -> dict[str, Any]:
    assessment = analyze_stroke_risk(image_path, result, risk_profile)
    risk = assessment["risk"]
    vascular = assessment["vascular"]
    return {
        "summary": f"10 年首次卒中风险 {risk['percent']:.1f}%，当前分层为{risk['band']}",
        "status_label": "评估完成",
        "metrics": [
            _metric("10 年卒中风险", f"{risk['percent']:.1f}", "%", "首次卒中风险"),
            _metric("血管密度", f"{vascular['vessel_density_percent']:.2f}", "%", "视盘周围血管密度"),
            _metric("动静脉口径比", f"{vascular['arteriovenous_ratio']:.3f}", "", "中央血管口径比"),
            _metric("血管迂曲度", f"{vascular['vessel_tortuosity']:.3f}", "", "长度加权迂曲度"),
        ],
        "sections": [
            {"title": "眼底影像", "text": "；".join(assessment["image_findings"])},
            {"title": "主要危险因素", "text": "；".join(assessment["risk_drivers"])},
            {"title": "综合评估", "text": assessment["integrated_interpretation"]},
            {"title": "建议检查", "text": "；".join(assessment["recommended_checks"])},
        ],
        "quality": _quality(result.get("quality", {})),
        "result_image": f"data:image/png;base64,{result['overlay_png']}",
        "views": [
            {"label": "血管分析", "image": f"data:image/png;base64,{result['overlay_png']}"},
            {"label": "血管分割", "image": f"data:image/png;base64,{result['vessels_png']}"},
            {"label": "标准化彩照", "image": f"data:image/png;base64,{result['preprocessed_png']}"},
        ],
        "stroke_assessment": assessment,
        "trace": assessment["trace"],
        "quantified_feature_count": sum(value is not None for value in result["biomarkers"].values()),
        "runtime_ms": round(float(result.get("runtime_ms", 0)) + assessment["runtime_ms"], 1),
        "notice": "",
    }


def run_module(
    module_id: str,
    image_path: Path,
    chronological_age: float | None = None,
    risk_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    module = SYSTEMIC_MODULES.get(module_id)
    if module is None:
        raise KeyError(module_id)
    task = "eye_age" if module_id == "eye-age" else "retinal_vascular"
    raw = infer(task, image_path)
    if module_id == "eye-age":
        output = _eye_age(module, raw, chronological_age)
    elif module_id == "cerebrovascular-retina":
        output = _cerebrovascular(module, raw, image_path, risk_profile or DEFAULT_STROKE_PROFILE)
    else:
        output = _vascular(module_id, raw)
    return {
        "module": {key: value for key, value in module.items() if key != "sample_file"},
        **output,
        "runtime_ms": output.get("runtime_ms", raw["runtime_ms"]),
        "real_inference": True,
        "notice": output.get("notice", (
            "" if module_id == "eye-age"
            else "输出为视网膜微血管表型，不等同于心脑血管疾病诊断或患病概率。"
        )),
    }
