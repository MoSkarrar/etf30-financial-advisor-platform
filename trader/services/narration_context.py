from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from trader.services import artifact_store


DEFAULT_MAX_XAI_CHARS = 0


@dataclass
class AdvisoryNarrationContext:
    session_id: str
    run_id: str
    manifest: Dict[str, Any]
    allocation_recommendation: Dict[str, Any]
    benchmark_comparison: Dict[str, Any]
    risk_snapshot: Dict[str, Any]
    investor_profile: Dict[str, Any]
    portfolio_policy: Dict[str, Any]
    advisory_summary_text: str
    xai_summary: str
    scenario_result: Dict[str, Any] = field(default_factory=dict)
    engine_info: Dict[str, Any] = field(default_factory=dict)
    policy_check: Dict[str, Any] = field(default_factory=dict)
    rule_summary: Dict[str, Any] = field(default_factory=dict)
    explanation_bundle: Dict[str, Any] = field(default_factory=dict)
    explanation_lab: Dict[str, Any] = field(default_factory=dict)
    history: List[Tuple[str, str]] = field(default_factory=list)

    def to_prompt_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "run_id": self.run_id,
            "selected_model": self.manifest.get("selected_model", ""),
            "trade_window": self.manifest.get("trade_window", {}),
            "run_level_metrics": self.manifest.get("run_level_metrics", {}),
            "allocation_recommendation": self.allocation_recommendation,
            "benchmark_comparison": self.benchmark_comparison,
            "risk_snapshot": self.risk_snapshot,
            "investor_profile": self.investor_profile,
            "portfolio_policy": self.portfolio_policy,
            "advisory_summary": self.advisory_summary_text,
            "xai_summary": self.xai_summary,
            "scenario_result": self.scenario_result,
            "engine_info": self.engine_info,
            "policy_check": self.policy_check,
            "rule_summary": self.rule_summary,
            "explanation_bundle": self.explanation_bundle,
            "explanation_lab": self.explanation_lab,
            "stricter_policy_impact": stricter_policy_impact(self),
            "history": self.history,
        }

    def to_prompt(self) -> str:
        return json.dumps(self.to_prompt_dict(), ensure_ascii=False, indent=2)


def trim_text(text: str, max_chars: int = DEFAULT_MAX_XAI_CHARS) -> str:
    text = (text or "").strip()
    if not max_chars or max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def stricter_policy_impact(context: AdvisoryNarrationContext) -> Dict[str, Any]:
    allocation = context.allocation_recommendation or {}
    policy = context.portfolio_policy or {}
    target_weights = allocation.get("target_weights", {}) or {}

    stricter_cap = min(float(policy.get("max_single_position_cap", 0.12) or 0.12), 0.08)
    stricter_cash = max(float(policy.get("min_cash_weight", 0.05) or 0.05), 0.08)
    cash_weight = float(target_weights.get("CASH", 0.0) or 0.0)

    reduction_needed = 0.0
    top_positions = []
    for ticker, weight in target_weights.items():
        if ticker == "CASH":
            continue
        weight = float(weight or 0.0)
        if weight > stricter_cap:
            reduction = weight - stricter_cap
            reduction_needed += reduction
            top_positions.append({"ticker": ticker, "weight": weight, "reduction_needed": reduction})

    additional_cash_needed = max(0.0, stricter_cash - cash_weight)
    estimated_turnover = reduction_needed + additional_cash_needed
    return {
        "stricter_cap": stricter_cap,
        "stricter_cash_floor": stricter_cash,
        "additional_cash_needed": additional_cash_needed,
        "estimated_turnover": estimated_turnover,
        "positions_impacted": sorted(top_positions, key=lambda x: x["reduction_needed"], reverse=True)[:5],
    }


def _empty_explanation_lab() -> Dict[str, Any]:
    return {
        "hypotheses": [],
        "cross_method_agreement": 0,
        "confidence_score": 0,
        "contradictions": [],
        "final_interpretation": "",
        "open_questions": [],
    }


def build_narration_context(run_bundle: Dict[str, Any], max_xai_chars: int = DEFAULT_MAX_XAI_CHARS) -> AdvisoryNarrationContext:
    manifest = run_bundle.get("manifest") or {}
    explanation_bundle = run_bundle.get("explanation_bundle") or manifest.get("explanation_bundle") or {}
    rule_summary = run_bundle.get("rule_summary") or manifest.get("rule_summary") or {}
    explanation_lab = run_bundle.get("explanation_lab") or manifest.get("explanation_lab") or _empty_explanation_lab()
    xai_text = run_bundle.get("xai_text") or explanation_bundle.get("advisor_summary") or rule_summary.get("summary_text") or "No XAI summary available."

    return AdvisoryNarrationContext(
        session_id=str(manifest.get("session_id", "")),
        run_id=str(manifest.get("run_id", "")),
        manifest=manifest,
        allocation_recommendation=run_bundle.get("allocation_recommendation") or manifest.get("allocation_recommendation") or {},
        benchmark_comparison=run_bundle.get("benchmark_comparison") or manifest.get("benchmark_comparison") or {},
        risk_snapshot=run_bundle.get("risk_snapshot") or manifest.get("risk_snapshot") or {},
        investor_profile=run_bundle.get("investor_profile") or manifest.get("investor_profile") or {},
        portfolio_policy=manifest.get("portfolio_policy") or {},
        advisory_summary_text=(run_bundle.get("advisory_summary_text") or manifest.get("explanation_summary") or "").strip(),
        xai_summary=trim_text(xai_text, max_xai_chars),
        scenario_result=run_bundle.get("scenario_result") or {},
        engine_info=run_bundle.get("engine_info") or manifest.get("engine_info") or {},
        policy_check=run_bundle.get("policy_check") or manifest.get("policy_check") or {},
        rule_summary=rule_summary,
        explanation_bundle=explanation_bundle,
        explanation_lab=explanation_lab,
        history=[],
    )


def load_context_for_run(run_id: str, max_xai_chars: int = DEFAULT_MAX_XAI_CHARS) -> Optional[AdvisoryNarrationContext]:
    bundle = artifact_store.load_advisory_bundle(run_id)
    if not bundle:
        return None
    return build_narration_context(bundle, max_xai_chars=max_xai_chars)
