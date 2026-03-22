from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from trader.drl_stock_trader.config.app_config import APP_CONFIG

History = List[Tuple[str, str]]


def _history_block(history: Optional[History], max_turns: Optional[int] = None) -> str:
    turns = max_turns or APP_CONFIG.narration.max_history_turns
    rows = []
    for role, msg in (history or [])[-turns:]:
        rows.append(f"{role.upper()}: {msg}")
    return "\n".join(rows)


def _shrink_long_lists(payload: Any, *, max_items: int = 10) -> Any:
    if isinstance(payload, dict):
        shrunk = {}
        for key, value in payload.items():
            if key in {"feature_values", "local_prediction_context", "run_level_metrics"} and isinstance(value, dict):
                items = list(value.items())[:max_items]
                shrunk[key] = {str(k): _shrink_long_lists(v, max_items=max_items) for k, v in items}
                continue
            shrunk[str(key)] = _shrink_long_lists(value, max_items=max_items)
        return shrunk
    if isinstance(payload, list):
        return [_shrink_long_lists(v, max_items=max_items) for v in payload[:max_items]]
    return payload


def _context_block(context: Dict[str, Any], *, max_chars: Optional[int] = None) -> str:
    limit = max_chars or APP_CONFIG.narration.max_context_chars
    compact = _shrink_long_lists(context)
    text = json.dumps(compact, ensure_ascii=False, indent=2)
    if len(text) <= limit:
        return text

    smaller = dict(compact)
    smaller.pop("run_level_metrics", None)
    smaller.pop("shap_summary", None)
    smaller.pop("lime_summary", None)
    smaller.pop("history", None)
    text = json.dumps(smaller, ensure_ascii=False, indent=2)
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _base_prompt(*, role_instruction: str, context: Dict[str, Any], question: str, history: Optional[History]) -> str:
    return (
        "You are a portfolio-advisor narration assistant for an ETF30 allocation system.\n"
        "Rules:\n"
        "- Use only the supplied context.\n"
        "- Answer the user's exact question directly before adding extra commentary.\n"
        "- Be concrete about weights, risk, benchmark, policy effects, and uncertainty when available.\n"
        "- Prefer short sections and bullets over giant tables unless the user explicitly asks for a table.\n"
        "- If the question is ambiguous, ask for clarification instead of pretending to know.\n"
        "- Do not invent facts missing from the context.\n"
        "- Give enough detail to be useful; do not artificially shorten the answer.\n\n"
        f"ROLE INSTRUCTION:\n{role_instruction}\n\n"
        f"CONTEXT:\n{_context_block(context)}\n\n"
        f"CHAT HISTORY:\n{_history_block(history)}\n\n"
        f"QUESTION:\n{(question or '').strip() or APP_CONFIG.narration.default_question}\n\n"
        "ANSWER:\n"
    )


def build_advisor_prompt(context: Dict[str, Any], question: str = "", history: Optional[History] = None) -> str:
    role_instruction = (
        "Answer like a client-facing advisor. Explain the current allocation, benchmark trade-offs, "
        "and the most important policy or scenario effects in plain language."
    )
    return _base_prompt(role_instruction=role_instruction, context=context, question=question, history=history)


def build_technical_xai_prompt(context: Dict[str, Any], question: str = "", history: Optional[History] = None) -> str:
    role_instruction = (
        "Answer like a technical XAI reviewer. Focus on SHAP, LIME, rule summaries, top factors, "
        "agreement, disagreement, and uncertainty."
    )
    return _base_prompt(role_instruction=role_instruction, context=context, question=question, history=history)


def build_risk_committee_prompt(context: Dict[str, Any], question: str = "", history: Optional[History] = None) -> str:
    role_instruction = (
        "Answer like a risk-committee reviewer. Focus on volatility, drawdown, turnover, concentration, "
        "benchmark-relative posture, policy breaches, and what should be monitored."
    )
    return _base_prompt(role_instruction=role_instruction, context=context, question=question, history=history)


def build_explainer_compare_prompt(context: Dict[str, Any], question: str = "", history: Optional[History] = None) -> str:
    role_instruction = (
        "Answer like an explanation-audit reviewer. Compare SHAP, LIME, and rule-based explanations, "
        "state agreement level, contradictions, and confidence or uncertainty."
    )
    return _base_prompt(role_instruction=role_instruction, context=context, question=question, history=history)
