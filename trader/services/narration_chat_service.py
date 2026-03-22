from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from trader.drl_stock_trader.RL_envs.wrappers.ollama_narrator import (
    OllamaNarrationConfig,
    generate_mode_response,
)
from trader.services.narration_context import AdvisoryNarrationContext


PushFn = Callable[[dict], None]
History = List[Tuple[str, str]]


@dataclass(frozen=True)
class AdvisorIntent:
    name: str
    prompt_mode: str
    default_question: str


INTENT_MAP = {
    "explain_allocation": AdvisorIntent("explain_allocation", "advisor_chat", "Why this allocation?"),
    "compare_benchmark": AdvisorIntent("compare_benchmark", "advisor_chat", "How does this compare with the benchmark?"),
    "explain_risk": AdvisorIntent("explain_risk", "risk_summary", "How risky is this allocation?"),
    "compare_scenarios": AdvisorIntent("compare_scenarios", "scenario_compare", "What changes under a stricter policy or scenario?"),
    "policy_breaches": AdvisorIntent("policy_breaches", "risk_committee", "Are there any policy issues or breaches?"),
    "technical_xai": AdvisorIntent("technical_xai", "technical_xai", "Give the technical XAI explanation for this run."),
    "compare_explainers": AdvisorIntent("compare_explainers", "explainer_compare", "Compare SHAP, LIME, and the rule-based explanations for this run."),
    "show_risk_flags": AdvisorIntent("show_risk_flags", "risk_committee", "Summarize the main risk flags and policy issues for this run."),
}


PROFILE_ALIAS = {
    "conservative": {"profile_name": "conservative", "max_single_position_cap": 0.08, "min_cash_weight": 0.10},
    "defensive": {"profile_name": "defensive", "max_single_position_cap": 0.08, "min_cash_weight": 0.10},
    "balanced": {"profile_name": "balanced", "max_single_position_cap": 0.10, "min_cash_weight": 0.06},
    "growth": {"profile_name": "growth", "max_single_position_cap": 0.12, "min_cash_weight": 0.04},
    "aggressive": {"profile_name": "aggressive", "max_single_position_cap": 0.14, "min_cash_weight": 0.03},
}


def _clean_question(question: str) -> str:
    return re.sub(r"\s+", " ", str(question or "").strip())


def _is_trivial_question(question: str) -> bool:
    q = _clean_question(question)
    if not q:
        return True
    if len(q) == 1:
        return True
    alnum = re.findall(r"[A-Za-z0-9]+", q)
    if not alnum:
        return True
    if len(alnum) == 1 and len(alnum[0]) == 1:
        return True
    return False


def detect_intent(question: str) -> AdvisorIntent:
    q = _clean_question(question).lower()
    if any(k in q for k in ["benchmark", "beat", "outperform", "tracking error", "active return"]):
        return INTENT_MAP["compare_benchmark"]
    if any(k in q for k in ["risk", "drawdown", "volatility", "concentration", "correlation", "risk-on", "risk off", "how risky"]):
        return INTENT_MAP["explain_risk"]
    if any(k in q for k in ["scenario", "stricter", "conservative", "max weight", "keep 10% cash", "cash", "what if", "8%", "10%"]):
        return INTENT_MAP["compare_scenarios"]
    if any(k in q for k in ["breach", "policy", "cash floor", "compliance"]):
        return INTENT_MAP["policy_breaches"]
    if any(k in q for k in ["compare explainers", "shap vs lime", "rule-based"]):
        return INTENT_MAP["compare_explainers"]
    if any(k in q for k in ["risk flags", "policy issues"]):
        return INTENT_MAP["show_risk_flags"]
    if any(k in q for k in ["shap", "surrogate", "xai", "feature importance", "technical", "lime"]):
        return INTENT_MAP["technical_xai"]
    return INTENT_MAP["explain_allocation"]


def _extract_percent(question: str, pattern: str) -> Optional[float]:
    m = re.search(pattern, question, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return float(m.group(1)) / 100.0
    except Exception:
        return None


def parse_policy_override_from_question(question: str) -> Dict[str, float | str]:
    q = _clean_question(question).lower()
    out: Dict[str, float | str] = {}

    max_weight = _extract_percent(q, r"max(?:imum)?\s+(?:weight|position(?:\s+cap)?)\s*(?:is|=|to|at|of)?\s*(\d+(?:\.\d+)?)%")
    if max_weight is None:
        max_weight = _extract_percent(q, r"(\d+(?:\.\d+)?)%\s*(?:max\s+weight|position\s+cap)")
    if max_weight is not None:
        out["max_single_position_cap"] = max_weight

    min_cash = _extract_percent(q, r"keep\s*(\d+(?:\.\d+)?)%\s*cash")
    if min_cash is None:
        min_cash = _extract_percent(q, r"(\d+(?:\.\d+)?)%\s*cash")
    if min_cash is not None:
        out["min_cash_weight"] = min_cash

    for alias, settings in PROFILE_ALIAS.items():
        if alias in q and any(k in q for k in ["client", "more", "policy", "what if", "profile"]):
            out.update(settings)
            break

    return out


def simulate_policy_change(context: AdvisoryNarrationContext, overrides: Dict[str, float | str]) -> Dict[str, object]:
    allocation = context.allocation_recommendation or {}
    current_policy = dict(context.portfolio_policy or {})
    current_weights = allocation.get("target_weights", {}) or {}
    cash_weight = float(current_weights.get("CASH", 0.0) or 0.0)

    max_cap_current = float(current_policy.get("max_single_position_cap", 0.12) or 0.12)
    min_cash_current = float(current_policy.get("min_cash_weight", 0.05) or 0.05)

    max_cap_new = float(overrides.get("max_single_position_cap", max_cap_current) or max_cap_current)
    min_cash_new = float(overrides.get("min_cash_weight", min_cash_current) or min_cash_current)
    profile_name = str(overrides.get("profile_name", context.investor_profile.get("profile_name", "current")))

    impacted = []
    current_non_cash = []
    trim_total = 0.0
    for ticker, weight in current_weights.items():
        if ticker == "CASH":
            continue
        w = float(weight or 0.0)
        current_non_cash.append((ticker, w))
        if w > max_cap_new:
            trim = w - max_cap_new
            trim_total += trim
            impacted.append({"ticker": ticker, "current_weight": w, "target_cap": max_cap_new, "trim_needed": trim})

    current_non_cash = sorted(current_non_cash, key=lambda kv: kv[1], reverse=True)
    extra_cash = max(0.0, min_cash_new - cash_weight)
    estimated_turnover = trim_total + extra_cash

    return {
        "profile_name": profile_name,
        "current_max_cap": max_cap_current,
        "new_max_cap": max_cap_new,
        "current_cash_floor": min_cash_current,
        "new_cash_floor": min_cash_new,
        "estimated_turnover": estimated_turnover,
        "additional_cash_needed": extra_cash,
        "positions_impacted": sorted(impacted, key=lambda x: x["trim_needed"], reverse=True)[:5],
        "top_current_positions": [{"ticker": t, "weight": w} for t, w in current_non_cash[:5]],
        "current_cash_weight": cash_weight,
    }


def _fmt_pct(value: object) -> str:
    try:
        return f"{float(value):.2%}"
    except Exception:
        return "0.00%"


def _fallback_answer(context: AdvisoryNarrationContext, question: str, intent: AdvisorIntent) -> str:
    alloc = context.allocation_recommendation or {}
    bench = context.benchmark_comparison or {}
    risk = context.risk_snapshot or {}
    rule_summary = context.rule_summary or {}
    explanation_bundle = context.explanation_bundle or {}
    overrides = parse_policy_override_from_question(question)
    policy_scenario = simulate_policy_change(context, overrides) if overrides else context.to_prompt_dict().get("stricter_policy_impact", {})

    if intent.name == "compare_benchmark":
        return (
            f"Against {bench.get('benchmark_name', 'the benchmark')}, the portfolio shows active return of {_fmt_pct(bench.get('active_return', 0.0))}, "
            f"tracking error of {_fmt_pct(bench.get('tracking_error', 0.0))}, and information ratio of {float(bench.get('information_ratio', 0.0) or 0.0):.2f}. "
            f"Use this as a relative view of whether the allocation is adding value versus simply tracking the benchmark."
        )
    if intent.name in {"explain_risk", "show_risk_flags"}:
        extra = str(rule_summary.get("summary_text") or "").strip()
        return (
            f"The current risk snapshot shows realized volatility {_fmt_pct(risk.get('realized_volatility', 0.0))}, "
            f"max drawdown {_fmt_pct(risk.get('max_drawdown', 0.0))}, concentration HHI {float(risk.get('concentration_hhi', 0.0) or 0.0):.3f}, "
            f"turnover {_fmt_pct(risk.get('turnover', 0.0))}, and cash weight {_fmt_pct(risk.get('cash_weight', 0.0))}. "
            f"{extra}".strip()
        )
    if intent.name == "compare_scenarios":
        impacted = policy_scenario.get("positions_impacted", []) or []
        top_positions = policy_scenario.get("top_current_positions", []) or []
        if impacted:
            top_text = ", ".join(f"{p['ticker']} ({_fmt_pct(p['current_weight'])})" for p in impacted[:5])
            return (
                f"Under the requested stricter policy, estimated turnover is {_fmt_pct(policy_scenario.get('estimated_turnover', 0.0))} "
                f"and extra cash needed is {_fmt_pct(policy_scenario.get('additional_cash_needed', 0.0))}. "
                f"The main positions that would need trimming are {top_text}. "
                f"The new cap would be {_fmt_pct(policy_scenario.get('new_max_cap', 0.0))} and the new cash floor would be {_fmt_pct(policy_scenario.get('new_cash_floor', 0.0))}."
            )
        top_text = ", ".join(f"{p['ticker']} ({_fmt_pct(p['weight'])})" for p in top_positions[:3]) if top_positions else "none"
        return (
            f"Under the requested stricter policy, estimated turnover is {_fmt_pct(policy_scenario.get('estimated_turnover', 0.0))} "
            f"and extra cash needed is {_fmt_pct(policy_scenario.get('additional_cash_needed', 0.0))}. "
            f"No current position appears to exceed the proposed cap, so the change is mostly about the tighter cash rule. "
            f"The largest current positions are {top_text}."
        )
    if intent.name == "compare_explainers":
        return (
            str(explanation_bundle.get("technical_summary") or "").strip()
            or str(rule_summary.get("summary_text") or "").strip()
            or "The stored explanation artifacts are limited for this run, so there is no full explainer comparison to show."
        )
    if intent.name == "technical_xai":
        return str(explanation_bundle.get("technical_summary") or context.xai_summary or alloc.get("rationale_summary", "")).strip()
    return alloc.get("rationale_summary", "The allocation is driven by the latest portfolio features, benchmark trade-offs, and policy rules.")


def answer_question_sync(
    *,
    context: AdvisoryNarrationContext,
    question: str,
    history: Optional[History] = None,
    cfg: Optional[OllamaNarrationConfig] = None,
) -> str:
    question = _clean_question(question)
    if _is_trivial_question(question):
        return "Please ask a fuller question, for example: why this allocation, how risky is it, what if the client is more conservative, or compare it with the benchmark."

    intent = detect_intent(question)
    cfg = cfg or OllamaNarrationConfig.from_app_config()
    overrides = parse_policy_override_from_question(question)
    context_payload = context.to_prompt_dict()
    if overrides:
        context_payload["question_policy_override"] = overrides
        context_payload["question_policy_override_effect"] = simulate_policy_change(context, overrides)

    try:
        answer = generate_mode_response(
            context=context_payload,
            prompt_mode=intent.prompt_mode,
            question=question or intent.default_question,
            history=history,
            cfg=cfg,
        )
        answer = (answer or "").strip()
        return answer or _fallback_answer(context, question, intent)
    except Exception:
        return _fallback_answer(context, question, intent)


def answer_question_async(
    *,
    context: AdvisoryNarrationContext,
    question: str,
    history: Optional[History],
    push: PushFn,
    on_success: Optional[Callable[[str], None]] = None,
    cfg: Optional[OllamaNarrationConfig] = None,
) -> None:
    def worker():
        try:
            answer = answer_question_sync(context=context, question=question, history=history, cfg=cfg)
            answer = (answer or "").strip() or _fallback_answer(context, question, detect_intent(question))
            if on_success:
                on_success(answer)
            push({"type": "advisor_answer", "message": answer})
        except Exception:
            safe = _fallback_answer(context, question, detect_intent(question))
            if on_success:
                on_success(safe)
            push({"type": "advisor_answer", "message": safe})

    threading.Thread(target=worker, daemon=True).start()
