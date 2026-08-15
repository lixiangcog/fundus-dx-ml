from api.amd_agent import (
    DEFAULT_CASE,
    EVIDENCE_SUMMARIES_ZH,
    _build_procedure_plan,
    _decision_evidence_details,
    _percent_change,
)


def test_percent_change_does_not_invent_zero_baseline_rate():
    assert _percent_change(0.0, 0.0) == 0.0
    assert _percent_change(12.0, 0.0) is None
    assert _percent_change(75.0, 100.0) == -25.0


def test_procedure_plan_is_structured_and_clinician_gated():
    plan = _build_procedure_plan(
        DEFAULT_CASE,
        {"activity": "apparently_stable"},
        {
            "id": "continue_monitor",
            "title": "继续当前方案并密切监测",
            "evidence_ids": ["E1", "E2"],
        },
        {"deltas": {"oct_fluid_area_percent": -20.0}},
        {
            "planning_rationale": "影像总体改善，侵入性操作需结合原始影像复核。",
            "patient_specific_considerations": ["核对既往治疗反应"],
            "required_specialist_decisions": ["确认是否继续眼内治疗"],
        },
        [{"id": "E7"}, {"id": "E8"}],
    )

    assert plan["status"] == "待视网膜专科医生确认"
    assert plan["procedure_overview"]["laterality"] == "右眼"
    assert "处方医生确认" in plan["procedure_overview"]["drug_and_dose"]
    assert "系统" not in plan["procedure_overview"]["drug_and_dose"]
    assert plan["preoperative_checks"]
    assert plan["intraoperative_plan"]
    assert plan["postoperative_monitoring"]
    assert plan["escalation_and_alternatives"]
    assert plan["required_specialist_decisions"]
    assert plan["evidence_ids"] == ["E7", "E8"]
    assert "术前" in plan["research_notice"]


def test_decision_evidence_explains_case_measurements_and_citations():
    details = _decision_evidence_details(
        DEFAULT_CASE,
        {"deltas": {}},
        {"activity": "apparently_stable"},
        {
            "id": "continue_monitor",
            "title": "继续当前方案并密切监测",
            "evidence_ids": ["E1", "E2"],
        },
        [
            {"id": "E1", "source": "NICE", "year": 2018, "summary_zh": EVIDENCE_SUMMARIES_ZH["E1"]},
            {"id": "E2", "source": "APVRS", "year": 2021, "summary_zh": EVIDENCE_SUMMARIES_ZH["E2"]},
        ],
    )

    assert details[0]["label"] == "本病例变化"
    assert "0.3 提高至 0.5" in details[0]["text"]
    assert "2.38 降至 1.28 mm²" in details[0]["text"]
    assert "继续当前方案并密切监测" in details[0]["text"]
    assert details[1]["id"] == "E1"
    assert "活动性湿性 AMD" in details[1]["text"]
    assert "缩短 2–4 周" in details[2]["text"]
