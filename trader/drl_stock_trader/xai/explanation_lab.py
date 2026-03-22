from __future__ import annotations

from typing import Any, Dict, List


def generate_hypotheses(bundle: Dict[str, Any], risk_snapshot: Dict[str, Any], policy_check: Dict[str, Any]) -> List[str]:
    hypotheses: List[str] = []
    if float((risk_snapshot or {}).get("concentration_hhi", 0.0) or 0.0) >= 0.10:
        hypotheses.append("The portfolio may be more concentrated because a small set of assets dominated the model ranking.")
    if float((risk_snapshot or {}).get("turnover", 0.0) or 0.0) >= 0.35:
        hypotheses.append("The portfolio may be adapting quickly to a regime shift because turnover is elevated.")
    if (policy_check or {}).get("breaches"):
        hypotheses.append("Policy constraints may be shaping the final recommendation because one or more breaches were detected.")
    if (bundle.get("consensus_points") or []):
        hypotheses.append("The most important explanatory signals are likely reliable because multiple explainers agree on them.")
    return hypotheses


def compare_explainer_agreement(bundle: Dict[str, Any]) -> Dict[str, Any]:
    shap_top = set((bundle.get("shap") or {}).get("top_features") or [])
    lime_top = set((bundle.get("lime") or {}).get("top_local_features") or [])
    rule_flags = set((bundle.get("rule_summary") or {}).get("rules_triggered") or [])

    overlap = shap_top.intersection(lime_top)
    denom = max(len(shap_top.union(lime_top)), 1)
    agreement = len(overlap) / denom if (shap_top or lime_top) else (1.0 if rule_flags else 0.0)

    contradictions: List[str] = []
    if shap_top and lime_top and not overlap:
        contradictions.append("Global and local feature explanations point to different dominant drivers.")
    if rule_flags and not shap_top and not lime_top:
        contradictions.append("Only rule-based explanation is populated, so feature-level explanatory confidence is limited.")

    return {
        "cross_method_agreement": float(agreement),
        "contradictions": contradictions,
    }


def build_explanation_lab_report(
    bundle: Dict[str, Any],
    risk_snapshot: Dict[str, Any] | None = None,
    policy_check: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    risk_snapshot = dict(risk_snapshot or {})
    policy_check = dict(policy_check or {})
    agreement_block = compare_explainer_agreement(bundle)
    hypotheses = generate_hypotheses(bundle, risk_snapshot, policy_check)
    agreement = float(agreement_block.get("cross_method_agreement", 0.0))
    confidence = min(1.0, agreement + (0.15 if hypotheses else 0.0))
    contradictions = list(agreement_block.get("contradictions") or [])

    if confidence >= 0.75:
        final_interpretation = "The explanation set is fairly coherent: global, local, and rule-based views are broadly aligned."
    elif confidence >= 0.40:
        final_interpretation = "The explanation set is moderately coherent, but at least one explainer is highlighting a different emphasis."
    else:
        final_interpretation = "The explanation set is weakly aligned, so it should be treated as suggestive rather than definitive."

    open_questions: List[str] = []
    if contradictions:
        open_questions.append("Why do global and local explainers disagree on the dominant drivers?")
    if (policy_check.get("breaches") or []):
        open_questions.append("How much of the final allocation is driven by policy constraints rather than pure model preference?")

    return {
        "hypotheses": hypotheses,
        "cross_method_agreement": agreement,
        "confidence_score": confidence,
        "contradictions": contradictions,
        "final_interpretation": final_interpretation,
        "open_questions": open_questions,
    }
