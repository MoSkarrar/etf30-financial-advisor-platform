from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from trader.domain.session_models import PortfolioPolicy


DEFAULT_CONCENTRATION_LIMIT = 0.12


def check_max_position_weight(weights: Mapping[str, float], limit: float, cash_key: str = "CASH") -> List[Dict[str, Any]]:
    breaches: List[Dict[str, Any]] = []
    limit = float(limit)
    for ticker, raw_weight in (weights or {}).items():
        if str(ticker) == cash_key:
            continue
        weight = float(raw_weight or 0.0)
        if weight > limit:
            breaches.append(
                {
                    "type": "position_cap",
                    "ticker": str(ticker),
                    "observed": weight,
                    "limit": limit,
                    "message": f"{ticker} weight {weight:.2%} exceeds max position cap {limit:.2%}.",
                }
            )
    return breaches


def check_min_cash_weight(cash_weight: float, min_cash: float) -> List[Dict[str, Any]]:
    cash_weight = float(cash_weight or 0.0)
    min_cash = float(min_cash or 0.0)
    if cash_weight < min_cash:
        return [
            {
                "type": "cash_floor",
                "observed": cash_weight,
                "limit": min_cash,
                "message": f"Cash weight {cash_weight:.2%} is below required floor {min_cash:.2%}.",
            }
        ]
    return []


def check_max_turnover(turnover: float, limit: float) -> List[Dict[str, Any]]:
    turnover = float(turnover or 0.0)
    limit = float(limit or 0.0)
    if turnover > limit:
        return [
            {
                "type": "turnover",
                "observed": turnover,
                "limit": limit,
                "message": f"Turnover {turnover:.2%} exceeds turnover budget {limit:.2%}.",
            }
        ]
    return []


def check_concentration(concentration: float, limit: float = DEFAULT_CONCENTRATION_LIMIT) -> List[Dict[str, Any]]:
    concentration = float(concentration or 0.0)
    limit = float(limit or DEFAULT_CONCENTRATION_LIMIT)
    if concentration > limit:
        return [
            {
                "type": "concentration",
                "observed": concentration,
                "limit": limit,
                "message": f"Concentration HHI {concentration:.4f} exceeds comfort limit {limit:.4f}.",
            }
        ]
    return []


def build_policy_check_result(
    *,
    weights: Mapping[str, float],
    turnover: float,
    concentration: float,
    policy: Optional[PortfolioPolicy] = None,
    applied_clips: Optional[List[Dict[str, Any]]] = None,
    cash_key: str = "CASH",
    concentration_limit: Optional[float] = None,
) -> Dict[str, Any]:
    policy = policy or PortfolioPolicy()
    clips = [dict(v) for v in (applied_clips or [])]

    breaches: List[Dict[str, Any]] = []
    breaches.extend(check_max_position_weight(weights, policy.max_single_position_cap, cash_key=cash_key))
    breaches.extend(check_min_cash_weight(float((weights or {}).get(cash_key, 0.0) or 0.0), policy.min_cash_weight))
    breaches.extend(check_max_turnover(turnover, policy.turnover_budget))

    inferred_concentration_limit = concentration_limit
    if inferred_concentration_limit is None:
        inferred_concentration_limit = max(DEFAULT_CONCENTRATION_LIMIT, float(policy.max_single_position_cap) * 1.5)
    breaches.extend(check_concentration(concentration, inferred_concentration_limit))

    messages = [str(v.get("message", "")).strip() for v in breaches if str(v.get("message", "")).strip()]
    severity = "none"
    if messages:
        severity = "high" if len(messages) >= 2 else "medium"

    human_summary = "Portfolio satisfies current policy constraints."
    if messages:
        human_summary = "Policy review found one or more allocation constraints that may need advisor attention."

    return {
        "passed": not messages,
        "breaches": messages,
        "severity": severity,
        "applied_clips": clips,
        "human_summary": human_summary,
        "details": breaches,
    }
