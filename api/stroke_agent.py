"""Multimodal stroke-risk assessment using retinal evidence and rFSRP."""
from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Any

from api.amd_agent import _json_from_text, _qwen_infer, _visionunite_infer


DEFAULT_STROKE_PROFILE: dict[str, Any] = {
    "age": 67,
    "sex": "male",
    "systolic_bp": 145,
    "smoker": False,
    "diabetes": True,
    "atrial_fibrillation": False,
    "antihypertensive": True,
    "cardiovascular_disease": False,
}


def calculate_revised_fsrp(profile: dict[str, Any]) -> dict[str, Any]:
    """Calculate the published 2017 revised 10-year Framingham stroke risk."""
    age = float(profile["age"])
    sbp = float(profile["systolic_bp"])
    smoker = bool(profile.get("smoker"))
    diabetes = bool(profile.get("diabetes"))
    af = bool(profile.get("atrial_fibrillation"))
    treated = bool(profile.get("antihypertensive"))
    cvd = bool(profile.get("cardiovascular_disease"))
    sex = str(profile["sex"]).lower()
    older = age >= 65

    if sex == "female":
        linear = 0.87938 * (age / 10)
        linear += 0.51127 if smoker else 0
        linear -= 0.03035 if cvd else 0
        linear += 1.20720 if af else 0
        linear += 0.39796 if older else 0
        linear += (0.06565 if older else 1.07111) if diabetes else 0
        linear += 0.13085 if treated else 0
        linear += (0.17234 if treated else 0.11303) * ((sbp - 120) / 10)
        baseline_survival, mean_linear = 0.95911, 6.6170719
    elif sex == "male":
        linear = 0.49716 * (age / 10)
        linear += 0.47254 if smoker else 0
        linear += 0.45341 if cvd else 0
        linear += 0.08064 if af else 0
        linear += 0.45426 if older else 0
        linear += (0.34385 if older else 1.35304) if diabetes else 0
        linear += 0.82598 if treated else 0
        linear += (0.09793 if treated else 0.27323) * ((sbp - 120) / 10)
        baseline_survival, mean_linear = 0.94451, 4.4227101
    else:
        raise ValueError("sex must be male or female")

    probability = 1 - baseline_survival ** math.exp(linear - mean_linear)
    percent = round(max(0.0, min(1.0, probability)) * 100, 1)
    band = "较低" if percent < 10 else "中等" if percent < 20 else "较高"
    return {
        "probability": round(probability, 6),
        "percent": percent,
        "band": band,
        "horizon_years": 10,
        "outcome": "首次卒中",
        "method": "2017 revised Framingham Stroke Risk Profile",
    }


def _vascular_context(result: dict[str, Any]) -> dict[str, float]:
    features = result.get("biomarkers", {})

    def value(key: str) -> float:
        raw = features.get(key)
        return float(raw) if raw is not None else 0.0

    arterial = value("full_cre_arteries")
    venous = value("full_cre_veins")
    return {
        "arteriovenous_ratio": round(arterial / venous, 3) if venous else 0.0,
        "vessel_density_percent": round(100 * (value("vd_disc_full_arteries") + value("vd_disc_full_veins")), 2),
        "vessel_tortuosity": round((value("lw_tort_dist_arteries") + value("lw_tort_dist_veins")) / 2, 3),
        "vessel_sparsity_percent": round(100 * value("mean_sparsity_vessels"), 2),
    }


def _risk_drivers(profile: dict[str, Any]) -> list[str]:
    drivers = [f"年龄 {int(profile['age'])} 岁", f"收缩压 {int(profile['systolic_bp'])} mmHg"]
    labels = (
        ("smoker", "当前吸烟"),
        ("diabetes", "糖尿病"),
        ("atrial_fibrillation", "房颤"),
        ("antihypertensive", "正在接受降压治疗"),
        ("cardiovascular_disease", "既往心血管病"),
    )
    drivers.extend(label for key, label in labels if profile.get(key))
    return drivers


def _as_list(value: Any, fallback: list[str]) -> list[str]:
    if not isinstance(value, list):
        return fallback
    placeholders = {"可见影像表现", "第二项表现", "建议检查及原因", "第二项检查及原因"}
    cleaned = [
        re.sub(r"^(可见影像表现|第二项表现|建议检查及原因|第二项检查及原因)\s*[:：]\s*", "", str(item).strip())
        for item in value
        if str(item).strip()
    ]
    cleaned = [item.strip("；;。,.， ") for item in cleaned]
    cleaned = [item for item in cleaned if item and item not in placeholders]
    return cleaned[:5] or fallback


def _signal_findings(specialist: dict[str, Any]) -> list[str]:
    flags = specialist.get("signal_flags") or {}
    findings = []
    findings.append(
        "可见出血或渗出相关信号" if flags.get("hemorrhage_exudation")
        else "未见明确出血或渗出信号"
    )
    findings.append(
        "存在动静脉异常信号" if flags.get("arteriovenous_abnormality")
        else "未见明确动静脉异常信号"
    )
    if flags.get("macular_abnormality"):
        findings.append("可见黄斑区域异常信号")
    if flags.get("optic_cup_disc"):
        findings.append("可见视盘或杯盘区域异常信号")
    return findings


def _recommended_checks(profile: dict[str, Any]) -> list[str]:
    checks = ["连续复测并记录血压，评估当前血压控制情况"]
    if profile.get("diabetes"):
        checks.append("检测空腹血糖和糖化血红蛋白，评估近期血糖控制情况")
    if profile.get("atrial_fibrillation"):
        checks.append("复查心电图并由专科评估房颤相关卒中风险")
    if profile.get("cardiovascular_disease"):
        checks.append("结合既往心血管病记录复核血脂和血管危险因素")
    if len(checks) == 1:
        checks.append("检查血糖和血脂，补充常见卒中危险因素信息")
    return checks[:4]


def _specialist_prompt(profile: dict[str, Any]) -> str:
    return (
        "Review this single color fundus photograph. Describe only visible retinal findings relevant to vascular health: "
        "image quality, hemorrhage or exudation, arteriolar narrowing, venular widening, arteriovenous crossing changes, "
        "vascular tortuosity, visible occlusion, optic-disc and macular appearance. Do not diagnose stroke, infer brain "
        "imaging, or invent measurements. Keep the response concise. "
        f"Context: age {profile['age']}, sex {profile['sex']}, systolic blood pressure {profile['systolic_bp']} mmHg."
    )


def _report_prompt(profile: dict[str, Any], risk: dict[str, Any], vascular: dict[str, float], specialist: dict[str, Any]) -> str:
    return f"""你是眼底影像与脑血管风险分析助手。请分析所附的一张眼底彩照，并把眼底所见、定量血管指标和固定公式风险整理成简洁中文结果。

健康信息：{json.dumps(profile, ensure_ascii=False)}
固定公式结果：{json.dumps(risk, ensure_ascii=False)}
血管定量：{json.dumps(vascular, ensure_ascii=False)}
眼科专用视觉模型观察：{json.dumps(specialist, ensure_ascii=False)}

必须遵守：
1. 10年卒中风险数值由固定公式给定，不得修改、重算或虚构其他概率。
2. 眼底照片只能提供视网膜血管证据，不能据此宣称已经确诊脑梗死或其他脑部病变。
3. 只描述图中可见表现；不确定时明确写“不确定”。
4. 建议检查应具体、简短，并与现有危险因素对应。
5. 不要复述字段名称或示例描述；必须填写对当前图像和当前资料的具体内容。
6. 返回严格 JSON，所有文本使用简体中文：
{{
  "image_findings": [],
  "recommended_checks": [],
  "integrated_interpretation": "",
  "uncertainty": ""
}}"""


def analyze_stroke_risk(image_path: Path, vascular_result: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    risk = calculate_revised_fsrp(profile)
    vascular = _vascular_context(vascular_result)
    drivers = _risk_drivers(profile)
    trace: list[dict[str, Any]] = [{
        "tool": "retinal_vascular_quantification",
        "status": "completed",
        "runtime_ms": vascular_result.get("runtime_ms", 0),
        "real_execution": True,
    }]

    specialist_result = _visionunite_infer(_specialist_prompt(profile), [image_path], max_new_tokens=220)
    specialist = specialist_result["observations"][0]
    trace.append({
        "tool": "fundus_specialist",
        "status": "completed",
        "runtime_ms": specialist_result["runtime_ms"],
        "model": specialist_result["model"],
        "real_execution": True,
    })

    report_result = _qwen_infer(_report_prompt(profile, risk, vascular, specialist), [image_path], max_new_tokens=520)
    report = _json_from_text(report_result["text"])
    trace.append({
        "tool": "multimodal_report",
        "status": "completed",
        "runtime_ms": report_result["runtime_ms"],
        "model": report_result["model"],
        "real_execution": True,
    })
    trace.append({
        "tool": "stroke_risk_formula",
        "status": "completed",
        "method": risk["method"],
        "real_execution": True,
    })

    # The specialist owns image evidence and the profile owns follow-up checks.
    # The report model still performs multimodal synthesis, but cannot expand
    # either evidence source with unsupported statements.
    findings = _signal_findings(specialist)
    checks = _recommended_checks(profile)
    report_findings = _as_list(report.get("image_findings"), [])
    report_checks = _as_list(report.get("recommended_checks"), [])
    interpretation = (
        f"10年首次卒中风险为 {risk['percent']:.1f}%，当前分层为{risk['band']}；"
        f"眼底影像显示{findings[0]}，血管定量结果与健康信息一并纳入本次评估。"
    )

    return {
        "risk": risk,
        "vascular": vascular,
        "risk_drivers": drivers,
        "image_findings": findings,
        "recommended_checks": checks,
        "integrated_interpretation": interpretation,
        "uncertainty": str(report.get("uncertainty") or "").strip(),
        "model_synthesis": {
            "image_findings": report_findings,
            "recommended_checks": report_checks,
        },
        "trace": trace,
        "runtime_ms": round((time.perf_counter() - started) * 1000, 1),
        "real_inference": True,
    }
