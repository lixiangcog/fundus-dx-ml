"""Pinned research examples and their reference annotations."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNTIME = PROJECT_ROOT / "runtime/research_samples"

SAMPLES = {
    "quality-ultrawide-fundus": {
        "pipeline_id": "quality-enhancement",
        "path": RUNTIME / "quality_ultrawide_fundus.png",
        "title": "超广角眼底彩照 · 质量增强默认样例",
        "source": "用户提供的独立超广角彩照样例",
        "license": "private research sample",
        "reference_type": "no-reference fundus enhancement sample",
        "split": "default display sample; no paired ground truth",
    },
    "oct-enhancement-duke-s10-32": {
        "pipeline_id": "quality-enhancement",
        "path": RUNTIME / "duke_Subject_10_scan32_noise12.png",
        "reference_path": RUNTIME / "duke_Subject_10_scan32.png",
        "title": "Duke DME · Subject 10 · B-scan 32（固定噪声退化）",
        "source": "Duke DME public dataset",
        "license": "CC BY 4.0 (MIRAGE release metadata)",
        "reference_type": "paired synthetic degradation",
        "split": "external test subject; selected by fluid burden before enhancement evaluation",
    },
    "oct-structure-duke-s03-4": {
        "pipeline_id": "structure-segmentation",
        "path": RUNTIME / "duke_Subject_03_4.png",
        "reference_path": RUNTIME / "duke_Subject_03_4_labels.png",
        "title": "Duke DME · Subject 03 · B-scan 4",
        "source": "MIRAGE Duke DME prepared split",
        "license": "CC BY 4.0 (MIRAGE release metadata)",
        "reference_type": "pixel annotation",
        "split": "independent test subject",
    },
    "fundus-screen-idrid-67": {
        "pipeline_id": "disease-screening",
        "path": RUNTIME / "idrid_67.jpg",
        "reference_label": "diabetic_retinopathy",
        "title": "IDRiD_67 · 眼底彩照",
        "source": "IDRiD lesion segmentation test set",
        "license": "CC BY 4.0",
        "reference_type": "dataset disease context",
        "split": "external test sample",
    },
    "octa-vessels-sgan-232653": {
        "pipeline_id": "vascular-quantification",
        "path": RUNTIME / "octa_synthetic_232653.png",
        "reference_path": RUNTIME / "octa_synthetic_232653_label.png",
        "title": "S-GAN OCTA · G_20230216_232653",
        "source": "aiforvision/OCTA-autosegmentation",
        "license": "MIT",
        "reference_type": "paired synthetic integration reference",
        "split": "upstream generated example; not independent clinical test",
    },
    "fundus-lesions-idrid-67": {
        "pipeline_id": "fundus-lesion-quantification",
        "path": RUNTIME / "idrid_67.jpg",
        "reference_masks": {
            "CTW": PROJECT_ROOT / "data/reference/idrid/IDRiD_67_SE.tif",
            "EX": PROJECT_ROOT / "data/reference/idrid/IDRiD_67_EX.tif",
            "HE": PROJECT_ROOT / "data/reference/idrid/IDRiD_67_HE.tif",
            "MA": PROJECT_ROOT / "data/reference/idrid/IDRiD_67_MA.tif",
        },
        "title": "IDRiD_67 · 四类病灶像素标注",
        "source": "IDRiD lesion segmentation test set",
        "license": "CC BY 4.0",
        "reference_type": "pixel annotations",
        "split": "external test sample selected by four-class minimum Dice; cohort statistics disclosed",
    },
    "oct-fluid-duke-s01-5": {
        "pipeline_id": "oct-fluid-quantification",
        "path": RUNTIME / "duke_Subject_01_5.png",
        "reference_path": RUNTIME / "duke_Subject_01_5_labels.png",
        "title": "Duke DME · Subject 01 · B-scan 5（含液体）",
        "source": "MIRAGE Duke DME prepared split",
        "license": "CC BY 4.0 (MIRAGE release metadata)",
        "reference_type": "pixel annotation",
        "split": "independent test subject",
    },
}

AMD_SAMPLES = {
    f"amd-{visit}-{modality}": {
        "path": RUNTIME / f"amd_{visit}_{modality}.png",
        "title": f"AMD CASE_001 · {visit.upper()} · {modality.upper()}",
        "source": "AMDFollowup paper Figure 3",
        "license": "paper figure excerpt for private research review",
        "reference_type": "de-identified figure thumbnail",
        "native_resolution": [93, 99],
    }
    for visit in ("v0", "v1") for modality in ("oct", "octa", "fundus")
}


def get_sample(sample_id: str) -> dict | None:
    sample = SAMPLES.get(sample_id) or AMD_SAMPLES.get(sample_id)
    return {**sample, "sample_id": sample_id} if sample else None


def get_pipeline_reference(sample_id: str | None, pipeline_id: str) -> dict | None:
    if not sample_id:
        return None
    sample = SAMPLES.get(sample_id)
    if not sample or sample["pipeline_id"] != pipeline_id or "reference_path" not in sample:
        return None
    return {**sample, "sample_id": sample_id}


def public_sample(sample_id: str, sample: dict) -> dict:
    excluded = {"path", "reference_path", "reference_masks"}
    return {"sample_id": sample_id, **{key: value for key, value in sample.items() if key not in excluded}}


def public_catalog() -> list[dict]:
    return [public_sample(sample_id, sample) for sample_id, sample in SAMPLES.items()]
