
from __future__ import annotations

from typing import Any, Dict, Optional

from trader.services import artifact_store


def load_xai_bundle_for_run(run_id: str) -> Optional[Dict[str, Any]]:
    bundle = artifact_store.load_full_run_bundle(run_id)
    if not bundle:
        return None
    return {
        "manifest": bundle.get("manifest") or {},
        "xai_text": bundle.get("xai_text") or "",
        "shap": bundle.get("shap") or {},
        "lime": bundle.get("lime") or {},
        "rule_summary": bundle.get("rule_summary") or {},
        "explanation_bundle": bundle.get("explanation_bundle") or {},
        "explanation_lab": bundle.get("explanation_lab") or {},
    }


def _first_non_empty(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def build_quick_xai_summary(bundle: Dict[str, Any]) -> str:
    explanation_bundle = bundle.get("explanation_bundle") or {}
    rule_summary = bundle.get("rule_summary") or {}
    shap = bundle.get("shap") or {}
    return _first_non_empty(
        explanation_bundle.get("advisor_summary"),
        rule_summary.get("summary_text"),
        bundle.get("xai_text"),
        shap.get("summary_text"),
        "No XAI summary available.",
    )


def build_technical_xai_summary(bundle: Dict[str, Any]) -> str:
    explanation_bundle = bundle.get("explanation_bundle") or {}
    shap = bundle.get("shap") or {}
    lime = bundle.get("lime") or {}
    rule_summary = bundle.get("rule_summary") or {}

    parts = [
        str(explanation_bundle.get("technical_summary") or "").strip(),
        str(shap.get("summary_text") or "").strip(),
        str(lime.get("summary_text") or "").strip(),
        str(rule_summary.get("summary_text") or "").strip(),
    ]
    text = " ".join(part for part in parts if part)
    return text.strip() or build_quick_xai_summary(bundle)


def build_uncertainty_summary(lab_report: Dict[str, Any]) -> str:
    if not lab_report:
        return ""
    contradictions = lab_report.get("contradictions") or []
    confidence = float(lab_report.get("confidence_score", 0.0) or 0.0)
    open_questions = lab_report.get("open_questions") or []

    if contradictions:
        return "The explanation methods show contradictions, so interpretation should be treated cautiously."
    if 0.0 < confidence < 0.5:
        return "The explanation confidence is low, so conclusions should be treated as tentative."
    if open_questions:
        return "Some explanation questions remain open, so the result should be reviewed with caution."
    return ""
