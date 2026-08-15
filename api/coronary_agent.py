"""CFP + OCTA multimodal coronary-risk signal assessment."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from api.amd_agent import _json_from_text, _qwen_infer, _visionunite_infer


LEVELS = {"low": "较低", "medium": "中等", "high": "较高"}


def _metric(result: dict[str, Any], label: str) -> float:
    for item in result.get("metrics", []):
        if item.get("label") == label:
            try:
                return float(item.get("value"))
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def _cfp_context(result: dict[str, Any]) -> dict[str, float]:
    features = result.get("biomarkers", {})

    def value(key: str) -> float:
        raw = features.get(key)
        return float(raw) if raw is not None else 0.0

    artery = value("full_cre_arteries")
    vein = value("full_cre_veins")
    return {
        "arteriovenous_ratio": round(artery / vein, 3) if vein else 0.0,
        "vessel_density_percent": round(
            100 * (value("vd_disc_full_arteries") + value("vd_disc_full_veins")), 2
        ),
        "vessel_tortuosity": round(
            (value("lw_tort_dist_arteries") + value("lw_tort_dist_veins")) / 2, 3
        ),
        "bifurcation_angle_degrees": round(
            (value("mn_bifangle_arteries") + value("mn_bifangle_veins")) / 2, 1
        ),
        "vessel_sparsity_percent": round(100 * value("mean_sparsity_vessels"), 2),
    }


def _octa_context(result: dict[str, Any]) -> dict[str, float]:
    return {
        "vessel_density_percent": _metric(result, "血管密度"),
        "central_avascular_candidate_pixels": _metric(result, "中央无血管候选核心"),
        "central_candidate_diameter_pixels": _metric(result, "候选等效直径"),
        "skeleton_length_pixels": _metric(result, "骨架总长度"),
        "branch_points": _metric(result, "分支点"),
        "mean_caliber_pixels": _metric(result, "平均管径代理"),
        "fractal_dimension": _metric(result, "分形维数"),
    }


def _clean_text(value: Any, fallback: str, limit: int = 260) -> str:
    text = re.sub(r"s+", " ", str(value or "")).strip(" ；;。")
    placeholders = {"", "待填写", "具体内容", "影像表现", "分析结果"}
    return fallback if text in placeholders else text[:limit]


def _clean_list(value: Any, fallback: list[str]) -> list[str]:
    if not isinstance(value, list):
        return fallback
    items = []
    for raw in value:
        text = re.sub(r"s+", " ", str(raw)).strip(" ；;。")
        if text and text not in {"建议检查", "检查项目", "具体建议"}:
            items.append(text[:120])
    return items[:4] or fallback


def _specialist_prompt() -> str:
    return (
        "Two retinal images from one assessment are provided in this order: first a color fundus photograph (CFP), "
        "then an OCT angiography image (OCTA). For CFP, describe visible vessel caliber, tortuosity, branching, "
        "hemorrhage or exudation and image quality. For OCTA, describe capillary density, rarefaction, central "
        "avascular region and network continuity. Describe only visible retinal evidence. Do not diagnose coronary "
        "heart disease and do not invent measurements. Return one concise observation per image."
    )


def _report_prompt(cfp: dict[str, float], octa: dict[str, float], observations: list[dict[str, Any]]) -> str:
    return f"""你是视网膜多模态心血管风险分析助手。附件顺序为：第一张 CFP 眼底彩照，第二张 OCTA。
CFP 定量：{json.dumps(cfp, ensure_ascii=False)}
OCTA 定量：{json.dumps(octa, ensure_ascii=False)}
视觉模型观察：{json.dumps(observations, ensure_ascii=False)}

任务是判断“冠心病相关视网膜风险信号”，不是确诊冠心病。综合规则：
- OCTA 血管密度或分形维数下降、中央无血管区扩大、毛细血管稀疏是主要风险信号；
- CFP 口径比、血管稀疏或迂曲改变只能作为辅助证据；
- 两种模态证据一致时可提高风险等级；证据不足或图像不清时选择 medium，不得虚构概率。
返回严格 JSON：
{{
  "risk_level": "low 或 medium 或 high",
  "cfp_summary": "当前彩照的具体表现",
  "octa_summary": "当前OCTA的具体表现",
  "integrated_summary": "两种影像与量化指标的综合依据",
  "recommended_checks": ["与当前风险等级对应的具体检查"]
}}"""


def analyze_coronary_risk(
    cfp_path: Path,
    octa_path: Path,
    cfp_result: dict[str, Any],
    octa_result: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    cfp = _cfp_context(cfp_result)
    octa = _octa_context(octa_result)
    trace = [
        {
            "tool": "cfp_vascular_quantification",
            "status": "completed",
            "runtime_ms": cfp_result.get("runtime_ms", 0),
            "real_execution": True,
        },
        {
            "tool": "octa_microvascular_quantification",
            "status": "completed",
            "runtime_ms": octa_result.get("runtime_ms", 0),
            "real_execution": True,
        },
    ]

    specialist_result = _visionunite_infer(_specialist_prompt(), [cfp_path, octa_path], max_new_tokens=260)
    observations = specialist_result["observations"]
    trace.append({
        "tool": "multimodal_retinal_specialist",
        "status": "completed",
        "runtime_ms": specialist_result["runtime_ms"],
        "model": specialist_result["model"],
        "real_execution": True,
    })

    report_result = _qwen_infer(
        _report_prompt(cfp, octa, observations),
        [cfp_path, octa_path],
        max_new_tokens=520,
    )
    report = _json_from_text(report_result["text"])
    trace.append({
        "tool": "coronary_risk_synthesis",
        "status": "completed",
        "runtime_ms": report_result["runtime_ms"],
        "model": report_result["model"],
        "real_execution": True,
    })

    level = str(report.get("risk_level") or "medium").strip().lower()
    if level not in LEVELS:
        level = "medium"
    cfp_summary = (
        f"动静脉口径比 {cfp['arteriovenous_ratio']:.3f}，"
        f"血管密度 {cfp['vessel_density_percent']:.2f}%，"
        f"迂曲度 {cfp['vessel_tortuosity']:.3f}，血管稀疏度 {cfp['vessel_sparsity_percent']:.2f}%"
    )
    octa_summary = (
        f"血管密度 {octa['vessel_density_percent']:.2f}%，"
        f"中央无血管候选区 {octa['central_avascular_candidate_pixels']:.0f} px²，"
        f"分形维数 {octa['fractal_dimension']:.3f}，"
        f"分支点 {octa['branch_points']:.0f} 个"
    )
    integrated = (
        f"联合模型将冠心病相关视网膜风险信号分为{LEVELS[level]}；"
        "判断同时采用 CFP 口径比与迂曲度、OCTA 血管密度与网络复杂度，"
        "未使用单一指标直接下结论"
    )
    checks = [
        "复核血压、空腹血脂、空腹血糖和糖化血红蛋白等心血管危险因素",
        "风险信号持续或存在胸痛等症状时进行心血管专科评估",
    ]


    return {
        "risk_level": level,
        "risk_label": LEVELS[level],
        "cfp_metrics": cfp,
        "octa_metrics": octa,
        "cfp_summary": cfp_summary,
        "octa_summary": octa_summary,
        "integrated_summary": integrated,
        "recommended_checks": checks,
        "trace": trace,
        "runtime_ms": round((time.perf_counter() - started) * 1000, 1),
        "real_inference": True,
    }
