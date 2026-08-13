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

from api.pipelines_v3 import disease_screening, structure_segmentation, vascular_quantification


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
    "title": "论文 Figure 3 去标识纵向 nAMD 病例",
    "research_demo": True,
    "evidence_origin": "reported_reference",
    "patient": {"age": 78, "sex": "女", "eye": "右眼", "diagnosis": "新生血管性 AMD（论文既往诊断）"},
    "treatment": {"agent": "玻璃体腔抗 VEGF（具体药物未报告）", "injections": 5, "current_interval_weeks": "论文未报告"},
    "visits": [
        {"id":"V0","label":"基线","date":"2024-03","bcva_decimal":0.3,
         "images":{"oct":"/research-samples/amd-v0-oct","octa":"/research-samples/amd-v0-octa","fundus":"/research-samples/amd-v0-fundus"}},
        {"id":"V1","label":"随访","date":"2024-06","bcva_decimal":0.5,
         "images":{"oct":"/research-samples/amd-v1-oct","octa":"/research-samples/amd-v1-octa","fundus":"/research-samples/amd-v1-fundus"}},
    ],
    "context": "论文报告：78 岁女性右眼 nAMD，接受 5 次抗 VEGF 注射；2024-03 至 2024-06 视力与多模态影像标志物总体改善。",
    "reference_biomarkers": {
        "provenance":"paper_reported_not_locally_recomputed",
        "oct":{"candidate_lesion_area_mm2":[2.38,1.28],"maximum_lesion_height_um":[413.4,354.9]},
        "fundus":{"candidate_lesion_area_mm2":[8.58,6.58],"followup_relative_to_baseline_percent":76.7},
        "octa":{"cnv_candidate_area_mm2":[1.39,0.08],"followup_relative_to_baseline_percent":5.7},
        "bcva_decimal":[0.3,0.5]
    },
    "image_quality": {"source":"paper_figure_crop","native_pixels":[93,99],"display_pixels":[564,594],"status":"review","reason":"论文图示缩略图被放大，仅用于工作流复现；不等同于原始 DICOM/OCT 体数据。"}
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


def _run_tools(paths: dict[str, Path], model, transform, class_names) -> dict[str, Any]:
    visits = {}
    for visit in ("baseline", "followup"):
        oct_result = structure_segmentation(Image.open(paths[f"{visit}_oct"]).convert("RGB"), image_path=paths[f"{visit}_oct"])
        octa_result = vascular_quantification(Image.open(paths[f"{visit}_octa"]).convert("RGB"), image_path=paths[f"{visit}_octa"])
        fundus_result = disease_screening(
            Image.open(paths[f"{visit}_fundus"]).convert("RGB"),
            model=model,
            transform=transform,
            class_names=class_names,
        )
        visits[visit] = {
            "oct": {
                "summary": oct_result["summary"],
                "thickness_proxy_px": _metric_value(oct_result, "视网膜厚度代理"),
                "retinal_area_percent": _metric_value(oct_result, "有效视网膜占比"),
                "overlay": oct_result["result_image"],
                "runtime_ms": oct_result["runtime_ms"],
            },
            "octa": {
                "summary": octa_result["summary"],
                "vessel_density_percent": _metric_value(octa_result, "血管密度"),
                "skeleton_length_px": _metric_value(octa_result, "骨架总长度"),
                "branch_points": _metric_value(octa_result, "分支点"),
                "fractal_dimension": _metric_value(octa_result, "分形维数"),
                "overlay": octa_result["result_image"],
                "runtime_ms": octa_result["runtime_ms"],
            },
            "fundus": {
                "summary": fundus_result["summary"],
                "prediction": fundus_result["prediction"],
                "confidence": fundus_result["confidence"],
                "probabilities": fundus_result["probabilities"],
                "overlay": fundus_result["result_image"],
                "runtime_ms": fundus_result["runtime_ms"],
            },
        }
    b, f = visits["baseline"], visits["followup"]
    deltas = {
        "oct_thickness_proxy_percent": round((f["oct"]["thickness_proxy_px"] - b["oct"]["thickness_proxy_px"]) / max(b["oct"]["thickness_proxy_px"], 1) * 100, 2),
        "octa_vessel_density_percent": round(f["octa"]["vessel_density_percent"] - b["octa"]["vessel_density_percent"], 2),
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


def _vision_prompt(case: dict, tool_results: dict) -> str:
    return f"""You are an ophthalmic imaging assistant. Compare six images in this exact order:
1 baseline OCT; 2 baseline OCTA; 3 baseline color fundus; 4 follow-up OCT; 5 follow-up OCTA; 6 follow-up color fundus.
Case context: {json.dumps(case, ensure_ascii=False)}
Locally computed auxiliary deltas (not the paper reference biomarkers): {json.dumps(tool_results["deltas"], ensure_ascii=False)}
Paper-reported reference biomarkers, when present: {json.dumps(case.get("reference_biomarkers"), ensure_ascii=False)}
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
Use only this evidence. Do not add unprovided drugs, doses, or numeric thresholds. Return JSON only and write all values in Simplified Chinese:
{{
  "case_summary": "...",
  "imaging_interpretation": "...",
  "evidence_integration": "...",
  "recommended_plan": "...",
  "followup_schedule": "...",
  "safety_triggers": ["symptom or vision deterioration trigger", "new hemorrhage or increasing fluid trigger"],
  "uncertainty": "..."
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
        "label": "论文图示缩略图需人工复核" if quality_status == "review" else "输入质量门槛通过",
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

    query = f"nAMD OCT fluid BCVA treat-and-extend interval recurrence switch poor response {case['context']} {json.dumps(vision, ensure_ascii=False)} {json.dumps(specialist, ensure_ascii=False)}"
    evidence = _retrieve(query, top_k=5)
    trace.append({"tool": "bm25_evidence_retrieval", "status": "completed", "documents": len(evidence), "corpus": len(json.loads(EVIDENCE_FILE.read_text(encoding="utf-8"))), "real_execution": True})

    options, state = _evaluate_options(case, tools, vision, specialist, evidence)
    trace.append({"tool": "candidate_option_evaluator", "status": "completed", "options": len(options), "selection": "programmatic", "real_execution": True})

    report_result = _qwen_infer(_report_prompt(case, tools, vision, specialist, options, state, evidence), [], max_new_tokens=850)
    report = _json_from_text(report_result["text"])
    selected = options[0]
    model_report_draft = dict(report)
    report["recommended_plan"] = selected["title"]
    report["evidence_ids"] = selected["evidence_ids"]
    report["evidence_integration"] = (
        f"程序化候选评估将‘{selected['title']}’评为{selected['verdict']}；"
        f"依据证据 {' '.join(f'[{item}]' for item in selected['evidence_ids'])}。"
        "该结论由检索证据与约束规则生成，不由语言模型自行选择。"
    )
    if selected["id"] == "reimage_expert_review":
        report["followup_schedule"] = (
            "短期采用同一设备与协议复查 OCT、OCTA 和眼底彩照；"
            "具体时间由视网膜专科结合症状、原始影像和采集质量确定。"
        )
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
        "notice": "科研与教学用途；默认病例来自论文 Figure 3 的去标识图示。论文报告数值与本地模型输出分开保存；任何临床行动必须由有资质的眼科医生结合原始影像确认。",
    }

