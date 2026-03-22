from __future__ import annotations

from typing import Any, Dict, List


def _top_features_from_payload(payload: Dict[str, Any], key: str) -> List[str]:
    values = payload.get(key) or []
    return [str(v) for v in values]


def derive_consensus_points(shap_exp: Dict[str, Any], lime_exp: Dict[str, Any], rule_exp: Dict[str, Any]) -> List[str]:
    consensus: List[str] = []
    shap_top = set(_top_features_from_payload(shap_exp, "top_features")[:5])
    lime_top = set(_top_features_from_payload(lime_exp, "top_local_features")[:5])
    overlap = sorted(shap_top.intersection(lime_top))
    if overlap:
        consensus.append("SHAP and LIME agree that these features matter: " + ", ".join(overlap[:5]) + ".")
    if rule_exp.get("rules_triggered"):
        consensus.append("Rule checks identified allocation or risk conditions that can be discussed alongside model attributions.")
    return consensus


def derive_disagreement_points(shap_exp: Dict[str, Any], lime_exp: Dict[str, Any], rule_exp: Dict[str, Any]) -> List[str]:
    disagreements: List[str] = []
    shap_top = set(_top_features_from_payload(shap_exp, "top_features")[:5])
    lime_top = set(_top_features_from_payload(lime_exp, "top_local_features")[:5])
    if shap_top and lime_top and shap_top.isdisjoint(lime_top):
        disagreements.append("SHAP and LIME focus on different drivers, which suggests local and global explanations are diverging.")
    if (rule_exp.get("risk_flags") or []) and not shap_top and not lime_top:
        disagreements.append("Rule-based explanation is available, but feature-based explainers did not surface strong drivers.")
    return disagreements


def build_bundle_summary(bundle: Dict[str, Any]) -> Dict[str, str]:
    shap_summary = str((bundle.get("shap") or {}).get("summary_text", "")).strip()
    lime_summary = str((bundle.get("lime") or {}).get("summary_text", "")).strip()
    rule_summary = str((bundle.get("rule_summary") or {}).get("summary_text", "")).strip()
    advisor_parts = [part for part in [rule_summary, shap_summary] if part]
    technical_parts = [part for part in [shap_summary, lime_summary, rule_summary] if part]
    return {
        "advisor_summary": " ".join(advisor_parts).strip() or "No advisor summary available.",
        "technical_summary": " ".join(technical_parts).strip() or "No technical summary available.",
    }


def build_explanation_bundle(
    shap_exp: Dict[str, Any],
    lime_exp: Dict[str, Any],
    rule_exp: Dict[str, Any],
    risk_snapshot: Dict[str, Any] | None = None,
    policy_check: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    bundle = {
        "shap": dict(shap_exp or {}),
        "lime": dict(lime_exp or {}),
        "rule_summary": dict(rule_exp or {}),
        "risk_snapshot": dict(risk_snapshot or {}),
        "policy_check": dict(policy_check or {}),
    }
    bundle["consensus_points"] = derive_consensus_points(bundle["shap"], bundle["lime"], bundle["rule_summary"])
    bundle["disagreement_points"] = derive_disagreement_points(bundle["shap"], bundle["lime"], bundle["rule_summary"])
    bundle.update(build_bundle_summary(bundle))
    return bundle
