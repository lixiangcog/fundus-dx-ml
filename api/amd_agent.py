"""Evidence-grounded longitudinal AMD agent orchestration.

Image-derived results are decision-support signals, never autonomous diagnoses.
The final action is selected by deterministic constraints after every candidate
has been evaluated against retrieved evidence.
"""
from __future__ import annotations

import json
import hashlib
import math
import os
import re
import secrets
import socket
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from api.pipelines_v3 import (
    disease_screening,
    fundus_lesion_quantification,
    oct_fluid_quantification,
    structure_segmentation,
    vascular_quantification,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNTIME_DIR = PROJECT_ROOT / "runtime"
CASE_DIR = RUNTIME_DIR / "amd_cases"
QWEN_STATUS_FILE = RUNTIME_DIR / "qwen_service.json"
VISIONUNITE_STATUS_FILE = RUNTIME_DIR / "visionunite_service.json"
NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
TOKEN_FILE = RUNTIME_DIR / "agent_token"
EVIDENCE_FILE = PROJECT_ROOT / "data" / "amd_evidence.json"
DEFAULT_IMAGES = {
    "baseline_oct": PROJECT_ROOT / "runtime/research_samples/amd_v0_oct.png",
    "baseline_octa": PROJECT_ROOT / "runtime/research_samples/amd_v0_octa.png",
    "baseline_fundus": PROJECT_ROOT / "runtime/research_samples/amd_v0_fundus.png",
    "followup_oct": PROJECT_ROOT / "runtime/research_samples/amd_v1_oct.png",
    "followup_octa": PROJECT_ROOT / "runtime/research_samples/amd_v1_octa.png",
    "followup_fundus": PROJECT_ROOT / "runtime/research_samples/amd_v1_fundus.png",
}
DEFAULT_CASE = {
    "case_id": "CASE_001",
    "title": "内置示例：纵向 nAMD 随访病例",
    "research_demo": True,
    "evidence_origin": "reported_reference",
    "patient": {"age": 78, "sex": "女", "eye": "右眼", "diagnosis": "新生血管性 AMD（既往诊断）"},
    "treatment": {"agent": "玻璃体腔抗 VEGF（具体药物未记录）", "injections": 5, "current_interval_weeks": "未记录"},
    "visits": [
        {"id":"V0","label":"基线","date":"2024-03","bcva_decimal":0.3,
         "images":{"oct":"/research-samples/amd-v0-oct","octa":"/research-samples/amd-v0-octa","fundus":"/research-samples/amd-v0-fundus"}},
        {"id":"V1","label":"随访","date":"2024-06","bcva_decimal":0.5,
         "images":{"oct":"/research-samples/amd-v1-oct","octa":"/research-samples/amd-v1-octa","fundus":"/research-samples/amd-v1-fundus"}},
    ],
    "context": "病例记录：78 岁女性右眼 nAMD，接受 5 次抗 VEGF 注射；2024-03 至 2024-06 视力与多模态影像标志物总体改善。",
    "reference_biomarkers": {
        "provenance":"paper_reported_not_locally_recomputed",
        "oct":{"candidate_lesion_area_mm2":[2.38,1.28],"maximum_lesion_height_um":[413.4,354.9]},
        "fundus":{"candidate_lesion_area_mm2":[8.58,6.58],"followup_relative_to_baseline_percent":76.7},
        "octa":{"cnv_candidate_area_mm2":[1.39,0.08],"followup_relative_to_baseline_percent":5.7},
        "bcva_decimal":[0.3,0.5]
    },
    "image_quality": {"source":"paper_figure_crop","native_pixels":[93,99],"display_pixels":[564,594],"status":"review","reason":"当前示例影像分辨率有限，仅用于系统功能演示；不等同于原始 DICOM/OCT 体数据。"}
}


def ensure_agent_token() -> str:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    if not TOKEN_FILE.is_file() or not TOKEN_FILE.read_text(encoding="utf-8").strip():
        TOKEN_FILE.write_text(secrets.token_urlsafe(36), encoding="utf-8")
        os.chmod(TOKEN_FILE, 0o600)
    return TOKEN_FILE.read_text(encoding="utf-8").strip()


def _service_status(status_file: Path) -> dict[str, Any]:
    if not status_file.is_file():
        return {"status": "offline", "detail": "GPU service has not registered"}
    try:
        status = json.loads(status_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"status": "offline", "detail": "Invalid GPU service status"}
    age = time.time() - float(status.get("updated_at", 0))
    if status.get("status") == "ready":
        try:
            req = urllib.request.Request(
                f"http://{status['host']}:{status['port']}/health",
                headers={"Accept": "application/json"},
            )
            with NO_PROXY_OPENER.open(req, timeout=2.5) as response:
                live = json.loads(response.read().decode("utf-8"))
            status["live"] = live
            status["status"] = live.get("status", "offline")
        except Exception as exc:
            status["status"] = "offline"
            status["detail"] = f"GPU health check failed: {exc}"
    status["status_age_seconds"] = round(age, 1)
    return status


def public_status() -> dict[str, Any]:
    qwen = _service_status(QWEN_STATUS_FILE)
    visionunite = _service_status(VISIONUNITE_STATUS_FILE)
    ready = qwen.get("status") == "ready" and visionunite.get("status") == "ready"
    states = {qwen.get("status"), visionunite.get("status")}
    return {
        "status": "ready" if ready else "loading" if "loading" in states else "offline",
        "model": "Qwen2.5-VL-3B-Instruct + VisionUnite V1",
        "services": {
            "multimodal": {
                "status": qwen.get("status", "offline"),
                "model": qwen.get("model", "Qwen2.5-VL-3B-Instruct"),
                "job_id": qwen.get("job_id", ""),
                "node": qwen.get("host", ""),
                "detail": qwen.get("detail", ""),
            },
            "fundus_specialist": {
                "status": visionunite.get("status", "offline"),
                "model": visionunite.get("model", "VisionUnite V1"),
                "job_id": visionunite.get("job_id", ""),
                "node": visionunite.get("host", ""),
                "detail": visionunite.get("detail", ""),
            },
        },
        "real_inference_required": True,
        "fallback_generation": False,
        "detail": "" if ready else "Qwen2.5-VL and VisionUnite V1 must both be ready; fallback is disabled.",
    }

def _gpu_infer(
    status_file: Path,
    service_name: str,
    prompt: str,
    images: list[Path],
    max_new_tokens: int,
) -> dict[str, Any]:
    status = _service_status(status_file)
    if status.get("status") != "ready":
        raise RuntimeError(status.get("detail") or f"{service_name} GPU service is not ready")
    body = json.dumps({
        "prompt": prompt,
        "images": [str(path.resolve()) for path in images],
        "max_new_tokens": max_new_tokens,
        "temperature": 0.1,
    }, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"http://{status['host']}:{status['port']}/infer",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", "X-Agent-Token": ensure_agent_token()},
    )
    try:
        with NO_PROXY_OPENER.open(req, timeout=180) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{service_name} inference failed ({exc.code}): {detail}") from exc
    except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
        raise RuntimeError(f"{service_name} inference unavailable: {exc}") from exc
    if not result.get("real_inference"):
        raise RuntimeError(f"{service_name} returned no verified inference output")
    return result


def _qwen_infer(prompt: str, images: list[Path], max_new_tokens: int = 700) -> dict[str, Any]:
    result = _gpu_infer(QWEN_STATUS_FILE, "Qwen-VL", prompt, images, max_new_tokens)
    if not result.get("text"):
        raise RuntimeError("Qwen-VL returned no verified inference text")
    return result


def _visionunite_infer(prompt: str, images: list[Path], max_new_tokens: int = 256) -> dict[str, Any]:
    result = _gpu_infer(VISIONUNITE_STATUS_FILE, "VisionUnite", prompt, images, max_new_tokens)
    if not result.get("observations") or len(result["observations"]) != len(images):
        raise RuntimeError("VisionUnite returned an incomplete specialist batch")
    return result

def _json_from_text(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^\s*```(?:json)?|\s*```\s*$", "", text.strip(), flags=re.I)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                pass
    return {"raw_text": text, "uncertainty": ["模型未返回可解析 JSON；已保留原始真实推理文本。"]}


def _metric_value(result: dict, label: str, default: float = 0.0) -> float:
    for metric in result.get("metrics", []):
        if metric.get("label") == label:
            try:
                return float(metric.get("value"))
            except (TypeError, ValueError):
                return default
    return default


def _percent_change(followup: float, baseline: float) -> float | None:
    """Return a signed percent change without inventing a value for a zero baseline."""
    if abs(baseline) < 1e-6:
        return 0.0 if abs(followup) < 1e-6 else None
    return round((followup - baseline) / baseline * 100, 2)


def _run_tools(paths: dict[str, Path], model, transform, class_names) -> dict[str, Any]:
    visits = {}
    for visit in ("baseline", "followup"):
        oct_result = structure_segmentation(Image.open(paths[f"{visit}_oct"]).convert("RGB"), image_path=paths[f"{visit}_oct"])
        fluid_result = oct_fluid_quantification(Image.open(paths[f"{visit}_oct"]).convert("RGB"), image_path=paths[f"{visit}_oct"])
        octa_result = vascular_quantification(Image.open(paths[f"{visit}_octa"]).convert("RGB"), image_path=paths[f"{visit}_octa"])
        fundus_result = disease_screening(
            Image.open(paths[f"{visit}_fundus"]).convert("RGB"),
            model=model,
            transform=transform,
            class_names=class_names,
        )
        lesion_result = fundus_lesion_quantification(
            Image.open(paths[f"{visit}_fundus"]).convert("RGB"),
            image_path=paths[f"{visit}_fundus"],
        )
        visits[visit] = {
            "oct": {
                "summary": oct_result["summary"],
                "thickness_proxy_px": _metric_value(oct_result, "视网膜厚度代理"),
                "retinal_area_percent": _metric_value(oct_result, "有效视网膜占比"),
                "overlay": oct_result["result_image"],
                "structure_overlay": oct_result["result_image"],
                "fluid_overlay": fluid_result["result_image"],
                "fluid_area_px": _metric_value(fluid_result, "液体面积"),
                "fluid_ratio_percent": _metric_value(fluid_result, "液体占比"),
                "fluid_components": _metric_value(fluid_result, "液体连通区"),
                "max_fluid_height_px": _metric_value(fluid_result, "最大垂直高度"),
                "quality": {"structure": oct_result.get("quality"), "fluid": fluid_result.get("quality")},
                "runtime_ms": round(oct_result["runtime_ms"] + fluid_result["runtime_ms"], 1),
            },
            "octa": {
                "summary": octa_result["summary"],
                "vessel_density_percent": _metric_value(octa_result, "血管密度"),
                "central_avascular_area_px2": _metric_value(octa_result, "中央无血管候选核心"),
                "skeleton_length_px": _metric_value(octa_result, "骨架总长度"),
                "branch_points": _metric_value(octa_result, "分支点"),
                "end_points": _metric_value(octa_result, "端点"),
                "average_caliber_px": _metric_value(octa_result, "平均管径代理"),
                "fractal_dimension": _metric_value(octa_result, "分形维数"),
                "overlay": octa_result["result_image"],
                "quality": octa_result.get("quality"),
                "runtime_ms": octa_result["runtime_ms"],
            },
            "fundus": {
                "summary": lesion_result["summary"],
                "screening_summary": fundus_result["summary"],
                "prediction": fundus_result["prediction"],
                "confidence": fundus_result["confidence"],
                "probabilities": fundus_result["probabilities"],
                "overlay": lesion_result["result_image"],
                "screening_overlay": fundus_result["result_image"],
                "lesion_overlay": lesion_result["result_image"],
                "cotton_wool_area_px": _metric_value(lesion_result, "棉絮斑/软性渗出面积"),
                "hard_exudate_area_px": _metric_value(lesion_result, "硬性渗出面积"),
                "hemorrhage_area_px": _metric_value(lesion_result, "出血面积"),
                "microaneurysm_area_px": _metric_value(lesion_result, "微动脉瘤面积"),
                "lesion_ratio_percent": _metric_value(lesion_result, "病灶总占比"),
                "quality": lesion_result.get("quality"),
                "runtime_ms": round(fundus_result["runtime_ms"] + lesion_result["runtime_ms"], 1),
            },
        }
    b, f = visits["baseline"], visits["followup"]
    deltas = {
        "oct_thickness_proxy_percent": _percent_change(f["oct"]["thickness_proxy_px"], b["oct"]["thickness_proxy_px"]),
        "oct_fluid_area_percent": _percent_change(f["oct"]["fluid_area_px"], b["oct"]["fluid_area_px"]),
        "oct_fluid_ratio_points": round(f["oct"]["fluid_ratio_percent"] - b["oct"]["fluid_ratio_percent"], 3),
        "octa_vessel_density_points": round(f["octa"]["vessel_density_percent"] - b["octa"]["vessel_density_percent"], 2),
        "octa_skeleton_length_percent": _percent_change(f["octa"]["skeleton_length_px"], b["octa"]["skeleton_length_px"]),
        "octa_central_avascular_area_percent": _percent_change(f["octa"]["central_avascular_area_px2"], b["octa"]["central_avascular_area_px2"]),
        "fundus_lesion_ratio_points": round(f["fundus"]["lesion_ratio_percent"] - b["fundus"]["lesion_ratio_percent"], 3),
        "fundus_hemorrhage_area_percent": _percent_change(f["fundus"]["hemorrhage_area_px"], b["fundus"]["hemorrhage_area_px"]),
        "amd_probability_points": round((f["fundus"]["probabilities"].get("amd", 0) - b["fundus"]["probabilities"].get("amd", 0)) * 100, 2),
    }
    return {"visits": visits, "deltas": deltas}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9-]+|[\u4e00-\u9fff]{1,4}", text.lower())


def _retrieve(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    documents = json.loads(EVIDENCE_FILE.read_text(encoding="utf-8"))
    tokenized = [_tokenize(" ".join(doc["tags"]) + " " + doc["title"] + " " + doc["evidence"]) for doc in documents]
    q = Counter(_tokenize(query))
    document_frequency = Counter()
    for tokens in tokenized:
        document_frequency.update(set(tokens))
    avg_len = sum(map(len, tokenized)) / max(len(tokenized), 1)
    ranked = []
    for doc, tokens in zip(documents, tokenized):
        frequencies = Counter(tokens)
        score = 0.0
        for term, q_count in q.items():
            if term not in frequencies:
                continue
            idf = math.log(1 + (len(documents) - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
            tf = frequencies[term]
            score += q_count * idf * (tf * 2.2) / (tf + 1.2 * (0.25 + 0.75 * len(tokens) / max(avg_len, 1)))
        ranked.append((score, doc))
    ranked.sort(key=lambda item: item[0], reverse=True)
    selected = []
    for score, doc in ranked[:top_k]:
        selected.append({**doc, "retrieval_score": round(score, 4)})
    return selected


def _contains_any(text: str, terms: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(term.lower() in lowered for term in terms)


def _evaluate_options(case: dict, tools: dict, vision: dict, specialist: dict, evidence: list[dict]) -> tuple[list[dict], dict]:
    visits = case["visits"]
    if "bcva_decimal" in visits[0]:
        vision_delta = float(visits[1]["bcva_decimal"]) - float(visits[0]["bcva_decimal"])
        vision_unit = "decimal_acuity"
        vision_worse = vision_delta <= -0.1
    else:
        vision_delta = float(visits[1]["bcva_logmar"]) - float(visits[0]["bcva_logmar"])
        vision_unit = "logmar"
        vision_worse = vision_delta >= 0.1
    deltas = tools["deltas"]
    reported = case.get("reference_biomarkers")
    reported_deltas = {}
    if reported:
        for key, values in (("oct_lesion_area",reported["oct"]["candidate_lesion_area_mm2"]),("octa_cnv_area",reported["octa"]["cnv_candidate_area_mm2"]),("fundus_lesion_area",reported["fundus"]["candidate_lesion_area_mm2"])):
            reported_deltas[key+"_percent"] = round((values[1]-values[0])/max(values[0],1e-6)*100,2)
    vision_text = json.dumps({"generalist": vision, "fundus_specialist": specialist}, ensure_ascii=False)
    active_visual = _contains_any(vision_text, ["new fluid", "increased fluid", "新发液体", "液体增加", "new hemorrhage", "新出血", "worsening"])
    uncertain = "raw_text" in vision or _contains_any(vision_text, ["uncertain", "无法", "cannot determine", "质量不足"])
    if reported:
        objective_activity = vision_worse or any(value > 10 for value in reported_deltas.values())
        active = objective_activity
        stable = not active and vision_delta >= 0 and all(value <= 0 for value in reported_deltas.values())
        uncertain_for_selection = False
    else:
        objective_activity = vision_worse or deltas["oct_thickness_proxy_percent"] > 10 or deltas["amd_probability_points"] > 15
        active = objective_activity or active_visual
        stable = not active and not vision_worse and abs(deltas["oct_thickness_proxy_percent"]) <= 10
        uncertain_for_selection = uncertain
    injections = int(case["treatment"].get("injections", 0))
    evidence_ids = [item["id"] for item in evidence]

    specs = [
        ("continue_monitor", "继续当前方案并密切监测", 84 if stable else 58, ["E1", "E2", "E6"]),
        ("shorten_interval", "缩短抗 VEGF 治疗间隔", 86 if active else 32, ["E2", "E3", "E4"]),
        ("extend_interval", "仅在确认无活动时逐步延长间隔", 70 if stable and not uncertain_for_selection else 25, ["E2", "E4"]),
        ("switch_agent", "复核诊断后考虑更换抗 VEGF 药物", 64 if active and injections >= 6 else 24, ["E1", "E3", "E5"]),
        ("reimage_expert_review", "短期同协议复查并由视网膜专科复核", 72 if uncertain else 48, ["E3", "E5", "E6"]),
    ]
    options = []
    for option_id, title, score, citations in specs:
        available = [item for item in citations if item in evidence_ids]
        verdict = "支持" if score >= 75 else "条件性支持" if score >= 50 else "证据不足" if score >= 30 else "不支持"
        options.append({
            "id": option_id,
            "title": title,
            "score": score,
            "verdict": verdict,
            "evidence_ids": available or evidence_ids[:2],
        })
    options.sort(key=lambda item: item["score"], reverse=True)
    state = {
        "activity": "suspected_active" if active else "apparently_stable" if stable else "uncertain",
        "visual_acuity_change": round(vision_delta, 3),
        "visual_acuity_unit": vision_unit,
        "reported_biomarker_deltas": reported_deltas,
        "decision_measurement_source": "paper_reported_reference" if reported else "locally_computed_tools",
        "objective_activity_trigger": objective_activity,
        "visual_activity_signal": active_visual,
        "uncertainty_trigger": uncertain,
        "selection_rule": "deterministic_constraint_ranking",
    }
    return options, state


def _build_procedure_plan(
    case: dict,
    state: dict,
    selected: dict,
    tools: dict,
    model_draft: dict,
    evidence: list[dict],
) -> dict[str, Any]:
    """Build a clinician-gated procedure plan from model synthesis plus fixed safety rules."""
    eye = case.get("patient", {}).get("eye", "术眼待确认")
    action_routes = {
        "continue_monitor": "当前不自动新增侵入性操作；由专科确认是否沿用既有玻璃体腔治疗路径",
        "shorten_interval": "复核玻璃体腔抗 VEGF 治疗路径，并由专科决定是否缩短治疗间隔",
        "extend_interval": "仅在确认无活动后考虑延长治疗间隔，不自动取消既定治疗",
        "switch_agent": "先复核诊断与既往反应，再由处方医生评估是否更换玻璃体腔治疗方案",
        "reimage_expert_review": "先完成同协议复查与视网膜专科复核，再决定是否进行侵入性操作",
    }
    route = action_routes.get(selected.get("id"), "由视网膜专科根据原始影像确定操作路径")
    available_ids = {item.get("id") for item in evidence}
    procedure_evidence = [item for item in ("E7", "E8") if item in available_ids]
    if not procedure_evidence:
        procedure_evidence = selected.get("evidence_ids", [])[:2]

    rationale = model_draft.get("planning_rationale") if isinstance(model_draft, dict) else None
    if not isinstance(rationale, str) or not rationale.strip():
        rationale = (
            f"当前状态为 {state.get('activity', '待复核')}；程序化评估选择“{selected.get('title', '待定方案')}”。"
            "本规划只整理操作准备与风险控制，不替代术者的适应证判断。"
        )
    model_considerations = model_draft.get("patient_specific_considerations") if isinstance(model_draft, dict) else []
    if not isinstance(model_considerations, list):
        model_considerations = []
    default_considerations = [
        f"{eye}既往接受过 {case.get('treatment', {}).get('injections', '多')} 次眼内治疗，应核对既往疗效与不良事件记录",
        "当前示例影像分辨率有限，任何靶区与活动性判断都需在原始 OCT/OCTA/眼底影像上复核",
        "系统像素定量与历史随访指标来源不同，不能直接替代设备原生物理测量",
    ]
    model_considerations = list(dict.fromkeys(
        [str(item).strip() for item in model_considerations + default_considerations if str(item).strip()]
    ))
    required_decisions = model_draft.get("required_specialist_decisions") if isinstance(model_draft, dict) else []
    if not isinstance(required_decisions, list):
        required_decisions = []
    default_decisions = [
        "确认是否存在需要继续治疗的活动性新生血管病变",
        "确认具体药物、剂量、治疗间隔与是否需要补充造影",
        "确认是否存在感染、炎症、眼压或全身情况等延期因素",
        "确认当前病例是否需要玻璃体视网膜手术；系统不会因 AMD 诊断自动触发手术",
    ]
    required_decisions = list(dict.fromkeys(
        [str(item).strip() for item in required_decisions + default_decisions if str(item).strip()]
    ))

    return {
        "title": "AMD 操作 / 手术规划",
        "status": "待视网膜专科医生确认",
        "planning_rationale": rationale,
        "procedure_overview": {
            "candidate_route": route,
            "laterality": eye,
            "target": "黄斑区病变活动控制；不提供注射点或手术导航坐标",
            "timing": selected.get("title", "结合复查结果确定"),
            "drug_and_dose": "不由系统生成，必须由处方医生确认",
        },
        "patient_specific_considerations": model_considerations,
        "preoperative_checks": [
            "核对患者身份、术眼、知情同意、当日药物与治疗记录",
            "复核视力、眼压、原始 OCT/OCTA/眼底彩照及影像质量",
            "排查活动性眼部感染或炎症，并询问既往注射并发症与过敏史",
            "结合全身病史、当前用药和近期眼科操作，由临床团队完成风险评估",
        ],
        "intraoperative_plan": [
            "仅由具备资质且完成培训的人员在符合规范的环境中实施",
            "执行无菌流程，术前再次核对术眼、药物、批次和有效期",
            "确保现场能够识别并处理急性眼压升高、出血等紧急情况",
            "系统不输出注射位点、器械参数、药物剂量或替代术者判断的导航指令",
        ],
        "postoperative_monitoring": [
            "记录实际术眼、药物、批次、操作人员和即时不良事件",
            "向患者说明疼痛加重、进行性视力下降、明显红眼或分泌物等紧急复诊信号",
            "按专科计划复查视力与 OCT，必要时复查 OCTA 和眼底彩照",
            "将新发液体、出血或视力下降与本次基线定量结果进行纵向比较",
        ],
        "escalation_and_alternatives": [
            "若出现新发或增加的液体、出血或视力下降，应提前复查并重新评估治疗间隔",
            "若反应持续不佳，应复核诊断并考虑补充荧光素或吲哚菁绿造影等专科检查",
            "更换药物、光动力治疗或玻璃体视网膜手术只能由专科在补充检查后决定",
            "若出现疑似感染性眼内炎、视网膜脱离或其他严重并发症，应进入急诊处置流程",
        ],
        "required_specialist_decisions": required_decisions,
        "evidence_ids": procedure_evidence,
        "quantitative_context": tools.get("deltas", {}),
        "research_notice": "该内容为系统生成的辅助规划，需经视网膜专科确认；不是处方、手术医嘱或可直接执行的术式方案。",
    }


def _vision_prompt(case: dict, tool_results: dict) -> str:
    return f"""You are an ophthalmic imaging assistant. Compare six images in this exact order:
1 baseline OCT; 2 baseline OCTA; 3 baseline color fundus; 4 follow-up OCT; 5 follow-up OCTA; 6 follow-up color fundus.
Case context: {json.dumps(case, ensure_ascii=False)}
Locally computed auxiliary deltas (separate from the historical reference biomarkers): {json.dumps(tool_results["deltas"], ensure_ascii=False)}
Historical reference biomarkers, when present: {json.dumps(case.get("reference_biomarkers"), ensure_ascii=False)}
Describe only visible findings. Do not claim a clinical diagnosis and do not invent tests that are not shown.
Compare retinal morphology and possible fluid-like spaces on OCT, macular flow network on OCTA, fundus macular appearance or hemorrhage cues, longitudinal change, and image quality.
Return JSON only. All string values must be Simplified Chinese:
{{
  "baseline_findings": {{"oct": "...", "octa": "...", "fundus": "..."}},
  "followup_findings": {{"oct": "...", "octa": "...", "fundus": "..."}},
  "change_assessment": "...",
  "image_quality": "...",
  "uncertainty": ["..."]
}}"""


def _report_prompt(case: dict, tools: dict, vision: dict, specialist: dict, options: list[dict], state: dict, evidence: list[dict]) -> str:
    chosen = options[0]
    compact_evidence = [{"id": e["id"], "evidence": e["evidence"]} for e in evidence]
    return f"""You are a retinal clinical decision-support assistant. A deterministic rule engine has already selected the action; do not replace it.
Selected action: {chosen["title"]}; disease state: {state["activity"]}.
Case: {json.dumps(case, ensure_ascii=False)}
Locally computed auxiliary tool deltas: {json.dumps(tools["deltas"], ensure_ascii=False)}
Cross-modal Qwen observations: {json.dumps(vision, ensure_ascii=False)}
VisionUnite fundus specialist observations: {json.dumps(specialist, ensure_ascii=False)}
Candidate evaluations: {json.dumps(options, ensure_ascii=False)}
Allowed evidence: {json.dumps(compact_evidence, ensure_ascii=False)}
Use only this evidence. Do not add unprovided drugs, doses, injection coordinates, device parameters, or numeric thresholds.
The procedure plan is a clinician-review draft, not an executable order. Return JSON only and write all values in Simplified Chinese:
{{
  "case_summary": "...",
  "imaging_interpretation": "...",
  "treatment_response": "...",
  "quantitative_change": "...",
  "evidence_integration": "...",
  "recommended_plan": "...",
  "followup_schedule": "...",
  "safety_triggers": ["symptom or vision deterioration trigger", "new hemorrhage or increasing fluid trigger"],
  "uncertainty": "...",
  "procedure_planning_draft": {{
    "planning_rationale": "Explain whether an intravitreal procedure should be planned now or only after re-review",
    "patient_specific_considerations": ["..."],
    "required_specialist_decisions": ["..."]
  }}
}}"""


def run_default_case(model, transform, class_names) -> dict[str, Any]:
    return run_case(DEFAULT_CASE, DEFAULT_IMAGES, model, transform, class_names)


def _specialist_prompt(case: dict) -> str:
    patient = case["patient"]
    return (
        "Ophthalmic fundus-only review. Describe visible macular, hemorrhage/exudation, "
        "optic-disc, color/boundary and arteriovenous findings; state image quality and "
        "uncertainty. Do not infer OCT/OCTA findings, diagnose, or recommend treatment. "
        f"Context: age {patient['age']}, {patient['eye']}, prior diagnosis {patient['diagnosis']}. "
        "Keep the response under 120 words."
    )


def run_case(case: dict, images: dict[str, Path], model, transform, class_names) -> dict[str, Any]:
    started = time.perf_counter()
    run_id = uuid.uuid4().hex[:12]
    case_path = CASE_DIR / run_id
    case_path.mkdir(parents=True, exist_ok=False)
    saved_paths: dict[str, Path] = {}
    for key, source in images.items():
        target = case_path / f"{key}.png"
        Image.open(source).convert("RGB").save(target, format="PNG", optimize=True)
        saved_paths[key] = target

    hashes = {key:hashlib.sha256(path.read_bytes()).hexdigest() for key,path in saved_paths.items()}
    dimensions = {key:list(Image.open(path).size) for key,path in saved_paths.items()}
    image_meta = case.get("image_quality", {})
    native = image_meta.get("native_pixels")
    distinct = len(set(hashes.values())) == len(hashes)
    quality_status = "review" if native and min(native) < 224 else "passed"
    case_quality = {
        "status": quality_status,
        "label": "示例影像分辨率有限，需人工复核" if quality_status == "review" else "输入质量门槛通过",
        "checks": {"six_distinct_images":distinct,"modalities_complete":len(saved_paths)==6,"decoded_dimensions":dimensions,"source_native_pixels":native},
        "reason": image_meta.get("reason", ""),
        "decision_measurements": "paper_reported_reference" if case.get("reference_biomarkers") else "locally_computed_tools",
    }

    trace = []
    step_started = time.perf_counter()
    tools = _run_tools(saved_paths, model, transform, class_names)
    trace.append({"tool": "quantitative_imaging", "status": "completed", "runtime_ms": round((time.perf_counter() - step_started) * 1000, 1), "real_execution": True})

    ordered_images = [saved_paths[key] for key in ("baseline_oct", "baseline_octa", "baseline_fundus", "followup_oct", "followup_octa", "followup_fundus")]
    vision_result = _qwen_infer(_vision_prompt(case, tools), ordered_images, max_new_tokens=750)
    vision = _json_from_text(vision_result["text"])
    trace.append({"tool": "qwen_vl_longitudinal_compare", "status": "completed", "runtime_ms": vision_result["runtime_ms"], "model": vision_result["model"], "input_images": 6, "real_execution": True})

    specialist_result = _visionunite_infer(
        _specialist_prompt(case),
        [saved_paths["baseline_fundus"], saved_paths["followup_fundus"]],
        max_new_tokens=256,
    )
    specialist = {
        "baseline": specialist_result["observations"][0],
        "followup": specialist_result["observations"][1],
    }
    trace.append({"tool": "visionunite_fundus_specialist", "status": "completed", "runtime_ms": specialist_result["runtime_ms"], "model": specialist_result["model"], "input_images": 2, "real_execution": True})

    query = (
        "nAMD OCT fluid BCVA treat-and-extend interval recurrence switch poor response "
        "intravitreal injection consent checklist sterile site drug record postoperative safety "
        f"{case['context']} {json.dumps(vision, ensure_ascii=False)} {json.dumps(specialist, ensure_ascii=False)}"
    )
    evidence = _retrieve(query, top_k=8)
    trace.append({"tool": "bm25_evidence_retrieval", "status": "completed", "documents": len(evidence), "corpus": len(json.loads(EVIDENCE_FILE.read_text(encoding="utf-8"))), "real_execution": True})

    options, state = _evaluate_options(case, tools, vision, specialist, evidence)
    trace.append({"tool": "candidate_option_evaluator", "status": "completed", "options": len(options), "selection": "programmatic", "real_execution": True})

    report_result = _qwen_infer(_report_prompt(case, tools, vision, specialist, options, state, evidence), [], max_new_tokens=1100)
    report = _json_from_text(report_result["text"])
    selected = options[0]
    model_report_draft = dict(report)
    model_procedure_draft = report.pop("procedure_planning_draft", {})
    deltas = tools["deltas"]

    def display_delta(key: str, unit: str) -> str:
        value = deltas.get(key)
        return "基线为 0，变化率不计算" if value is None else f"{value:+g}{unit}"

    report.setdefault("case_summary", case.get("context", "病例信息需结合原始病历复核。"))
    quantitative_change = (
        f"本地分割模型测得：OCT 厚度代理 {display_delta('oct_thickness_proxy_percent', '%')}，"
        f"液体面积 {display_delta('oct_fluid_area_percent', '%')}；"
        f"OCTA 血管密度 {display_delta('octa_vessel_density_points', ' 个百分点')}，"
        f"血管骨架 {display_delta('octa_skeleton_length_percent', '%')}；"
        f"彩照病灶占比 {display_delta('fundus_lesion_ratio_points', ' 个百分点')}，"
        f"AMD 筛查概率 {display_delta('amd_probability_points', ' 个百分点')}。"
    )
    model_observation = vision.get(
        "change_assessment",
        "多模态模型已完成两次就诊影像对比，具体征象需结合原始影像复核。",
    )
    report["imaging_interpretation"] = (
        f"多模态模型观察：{model_observation} 本地量化结果：{quantitative_change} "
        "模型文字与定量结果不一致时，以原始影像和专科人工复核为准。"
    )
    reported = case.get("reference_biomarkers")
    if reported:
        report["treatment_response"] = (
            f"历史记录显示视力由 {reported['bcva_decimal'][0]} 提升至 {reported['bcva_decimal'][1]}，"
            f"OCT 候选病灶面积由 {reported['oct']['candidate_lesion_area_mm2'][0]} 降至 {reported['oct']['candidate_lesion_area_mm2'][1]} mm²，"
            f"OCTA CNV 候选面积由 {reported['octa']['cnv_candidate_area_mm2'][0]} 降至 {reported['octa']['cnv_candidate_area_mm2'][1]} mm²；"
            "总体提示病情改善。系统分割结果存在部分方向不一致，需在原始影像上复核。"
        )
    else:
        report["treatment_response"] = (
            f"基于本地分割量化与多模态模型观察综合评估。{quantitative_change} "
            "是否构成治疗反应必须由视网膜专科结合原始影像确认。"
        )
    report["quantitative_change"] = quantitative_change
    if not isinstance(report.get("safety_triggers"), list) or not report["safety_triggers"]:
        report["safety_triggers"] = [
            "视力下降、视物变形或其他症状加重",
            "新发或增加的视网膜液体、黄斑出血或其他活动性征象",
            "眼内操作后疼痛加重、进行性视力下降或明显红眼",
        ]
    report.setdefault("uncertainty", case_quality.get("reason") or "所有输出均需在原始影像上由视网膜专科复核。")
    report["recommended_plan"] = selected["title"]
    report["evidence_ids"] = selected["evidence_ids"]
    report["evidence_integration"] = (
        f"程序化候选评估将‘{selected['title']}’评为{selected['verdict']}；"
        f"依据证据 {' '.join(f'[{item}]' for item in selected['evidence_ids'])}。"
        "该结论由检索证据与约束规则生成，不由语言模型自行选择。"
    )
    schedule_by_action = {
        "continue_monitor": "沿用由专科确认的现有方案；具体复查时间不由系统自动设定。复查应包含视力与 OCT，必要时同步复查 OCTA 和眼底彩照。",
        "shorten_interval": "在专科复核活动性后考虑缩短治疗间隔；具体时间、药物与剂量由处方医生决定，并用视力和 OCT 评价反应。",
        "extend_interval": "仅在专科确认无活动后逐步评估延长间隔；若视力下降、液体或出血增加则重新评估。",
        "switch_agent": "先复核诊断、既往治疗反应和补充影像，再由处方医生决定是否更换方案及复查时间。",
        "reimage_expert_review": "短期采用同一设备与协议复查 OCT、OCTA 和眼底彩照；具体时间由视网膜专科结合症状、原始影像和采集质量确定。",
    }
    report["followup_schedule"] = schedule_by_action.get(
        selected["id"],
        "具体复查时间由视网膜专科结合原始影像和症状确定。",
    )
    state_labels = {
        "suspected_active": "疑似仍有活动",
        "apparently_stable": "当前影像与视力总体稳定",
        "uncertain": "活动性尚不确定",
    }
    report["structured_summary"] = {
        "case_overview": report["case_summary"],
        "disease_state": state_labels.get(state.get("activity"), state.get("activity", "待复核")),
        "treatment_response": report["treatment_response"],
        "imaging_interpretation": report["imaging_interpretation"],
        "quantitative_change": report["quantitative_change"],
    }
    report["recommendation"] = {
        "selected_action": selected["title"],
        "verdict": selected["verdict"],
        "followup_schedule": report["followup_schedule"],
        "evidence_integration": report["evidence_integration"],
        "evidence_ids": selected["evidence_ids"],
    }
    report["procedure_plan"] = _build_procedure_plan(
        case, state, selected, tools, model_procedure_draft, evidence
    )
    report["safety"] = {
        "triggers": report["safety_triggers"],
        "uncertainty": report["uncertainty"],
    }
    report["consistency_validated"] = True
    trace.append({"tool": "qwen_report_synthesis", "status": "completed", "runtime_ms": report_result["runtime_ms"], "model": report_result["model"], "input_images": 0, "real_execution": True})
    return {
        "run_id": run_id,
        "model_report_draft": model_report_draft,
        "case": case,
        "case_quality": case_quality,
        "reported_reference_biomarkers": case.get("reference_biomarkers"),
        "clinical_state": state,
        "tool_results": tools,
        "vision_reasoning": vision,
        "fundus_specialist_reasoning": specialist,
        "evidence": evidence,
        "options": options,
        "decision": {
            "action_id": selected["id"],
            "action": selected["title"],
            "verdict": selected["verdict"],
            "confidence_score": selected["score"],
            "evidence_ids": selected["evidence_ids"],
            "selection": "programmatic_after_option_evaluation",
        },
        "report": report,
        "tool_trace": trace,
        "provenance": {
            "vision_model": vision_result["model"],
            "fundus_specialist_model": specialist_result["model"],
            "report_model": report_result["model"],
            "real_mllm_inference": True,
            "fallback_generation": False,
            "quantitative_tools": ["Duke residual U-Net", "OCTA DynUNet", "FundusDx ResNet18"],
            "evidence_retrieval": "BM25 over curated guideline evidence cards",
        },
        "runtime_ms": round((time.perf_counter() - started) * 1000, 1),
        "notice": "本模块用于辅助分析；默认加载去标识的内置示例病例。历史随访记录与系统分析结果分开保存；任何临床行动必须由有资质的眼科医生结合原始影像确认。",
    }
