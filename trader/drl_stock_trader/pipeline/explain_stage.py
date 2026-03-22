from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import pandas as pd

from trader.drl_stock_trader.config.app_config import APP_CONFIG
from trader.drl_stock_trader.xai.explanation_bundle import build_explanation_bundle
from trader.drl_stock_trader.xai.explanation_lab import build_explanation_lab_report
from trader.drl_stock_trader.xai.lime_service import build_lime_explainer, explain_allocation_instance
from trader.drl_stock_trader.xai.rule_summary import build_rule_summary
from trader.drl_stock_trader.xai.shap_service import run_shap_explanation
from trader.drl_stock_trader.xai.surrogate_shap import (
    explain_portfolio_decision,
    to_user_friendly_text,
    train_surrogate_for_portfolio,
)


@dataclass
class ExplainStageResult:
    xai_payload: Dict[str, Any]
    summary_text: str
    benchmark_text: str
    policy_text: str
    risk_text: str
    shap_payload: Dict[str, Any]
    lime_payload: Dict[str, Any]
    rule_summary_payload: Dict[str, Any]
    explanation_bundle_payload: Dict[str, Any]
    explanation_lab_payload: Dict[str, Any]


def generate_explanation(
    *,
    current_frame: pd.DataFrame,
    allocation_recommendation: Any,
    previous_weights: Optional[Dict[str, float]] = None,
    benchmark_comparison: Optional[Any] = None,
    risk_snapshot: Optional[Any] = None,
    investor_profile: Optional[Any] = None,
    portfolio_policy: Optional[Any] = None,
    policy_check: Optional[Any] = None,
    feature_columns: Optional[Sequence[str]] = None,
) -> ExplainStageResult:
    benchmark_dict = benchmark_comparison.to_dict() if hasattr(benchmark_comparison, "to_dict") else dict(benchmark_comparison or {})

    legacy_xai_payload = explain_portfolio_decision(
        current_frame=current_frame,
        allocation_recommendation=allocation_recommendation,
        previous_weights=previous_weights,
        benchmark_name=str(benchmark_dict.get("benchmark_name", "equal_weight")),
        policy=portfolio_policy,
        risk_snapshot=risk_snapshot,
        feature_columns=feature_columns,
    )

    surrogate = train_surrogate_for_portfolio(
        current_frame=current_frame,
        allocation_recommendation=allocation_recommendation,
        feature_columns=feature_columns,
    )
    sample_index = int((abs(surrogate.y)).argmax()) if len(surrogate.y) else 0
    sample = surrogate.X.iloc[[sample_index]]

    shap_payload: Dict[str, Any] = {}
    if APP_CONFIG.xai.enable_shap:
        shap_payload = run_shap_explanation(
            model_or_surrogate=surrogate.model,
            background=surrogate.X,
            sample=sample,
            feature_names=surrogate.feature_columns,
            top_k=APP_CONFIG.xai.shap_top_k,
        )

    lime_payload: Dict[str, Any] = {}
    if APP_CONFIG.xai.enable_lime:
        explainer = build_lime_explainer(
            training_data=surrogate.X.to_numpy(dtype=float),
            feature_names=surrogate.feature_columns,
            mode="regression",
        )
        lime_payload = explain_allocation_instance(
            instance=sample.iloc[0].to_numpy(dtype=float),
            predict_fn=surrogate.predict_fn,
            num_features=APP_CONFIG.xai.lime_num_features,
            num_samples=APP_CONFIG.xai.lime_num_samples,
            explainer=explainer,
            feature_names=surrogate.feature_columns,
        )

    rule_summary_payload = build_rule_summary(
        allocation_recommendation=allocation_recommendation,
        benchmark_comparison=benchmark_comparison,
        risk_snapshot=risk_snapshot,
        portfolio_policy=portfolio_policy,
        policy_check=policy_check,
    ) if APP_CONFIG.xai.enable_rule_summary else {}

    explanation_bundle_payload = build_explanation_bundle(
        shap_exp=shap_payload,
        lime_exp=lime_payload,
        rule_exp=rule_summary_payload,
        risk_snapshot=risk_snapshot.to_dict() if hasattr(risk_snapshot, "to_dict") else dict(risk_snapshot or {}),
        policy_check=policy_check.to_dict() if hasattr(policy_check, "to_dict") else dict(policy_check or {}),
    )

    explanation_lab_payload = build_explanation_lab_report(
        explanation_bundle_payload,
        risk_snapshot=risk_snapshot.to_dict() if hasattr(risk_snapshot, "to_dict") else dict(risk_snapshot or {}),
        policy_check=policy_check.to_dict() if hasattr(policy_check, "to_dict") else dict(policy_check or {}),
    ) if APP_CONFIG.xai.enable_explanation_lab else {}

    summary_text = explanation_bundle_payload.get("advisor_summary") or to_user_friendly_text(legacy_xai_payload)
    benchmark_text = str(rule_summary_payload.get("benchmark_flags", [""])[0] if rule_summary_payload.get("benchmark_flags") else benchmark_dict.get("message", ""))
    policy_text = str(rule_summary_payload.get("allocation_flags", [""])[0] if rule_summary_payload.get("allocation_flags") else "")
    risk_text = " ".join(
        filter(
            None,
            [
                str((legacy_xai_payload.get("explanation", {}) or {}).get("risk_posture", {}).get("message", "")),
                str((rule_summary_payload.get("summary_text", "") or "").strip()),
            ],
        )
    ).strip()

    xai_payload = {
        "legacy": legacy_xai_payload,
        "shap": shap_payload,
        "lime": lime_payload,
        "rule_summary": rule_summary_payload,
        "explanation_bundle": explanation_bundle_payload,
        "explanation_lab": explanation_lab_payload,
    }

    return ExplainStageResult(
        xai_payload=xai_payload,
        summary_text=summary_text,
        benchmark_text=benchmark_text,
        policy_text=policy_text,
        risk_text=risk_text,
        shap_payload=shap_payload,
        lime_payload=lime_payload,
        rule_summary_payload=rule_summary_payload,
        explanation_bundle_payload=explanation_bundle_payload,
        explanation_lab_payload=explanation_lab_payload,
    )
