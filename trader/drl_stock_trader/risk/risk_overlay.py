from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from trader.domain.session_models import PortfolioPolicy
from trader.drl_stock_trader.risk.risk_metrics import compute_concentration, compute_turnover

_EPS = 1e-12


def _normalize_weights(weights: Mapping[str, float], cash_key: str = "CASH", long_only: bool = True) -> Dict[str, float]:
    raw = {str(k): float(v) for k, v in (weights or {}).items()}
    if long_only:
        raw = {k: max(v, 0.0) for k, v in raw.items()}
    total = float(sum(raw.values()))
    if total <= _EPS:
        keys = list(raw.keys())
        if cash_key and cash_key in raw:
            return {k: (1.0 if k == cash_key else 0.0) for k in keys}
        return {k: 0.0 for k in keys}
    return {k: v / total for k, v in raw.items()}


def _renormalize_non_cash(weights: Dict[str, float], cash_key: str = "CASH") -> Dict[str, float]:
    cash = float(weights.get(cash_key, 0.0) or 0.0)
    non_cash_keys = [k for k in weights if k != cash_key]
    non_cash_total = float(sum(max(weights.get(k, 0.0), 0.0) for k in non_cash_keys))
    target_non_cash = max(0.0, 1.0 - cash)
    if not non_cash_keys:
        out = dict(weights)
        out[cash_key] = 1.0
        return out
    if non_cash_total <= _EPS:
        per = target_non_cash / max(len(non_cash_keys), 1)
        out = {k: per for k in non_cash_keys}
        out[cash_key] = cash
        return out
    out = {k: max(weights.get(k, 0.0), 0.0) / non_cash_total * target_non_cash for k in non_cash_keys}
    out[cash_key] = cash
    return out


def clip_position_weights(weights: Mapping[str, float], limit: float, cash_key: str = "CASH") -> tuple[Dict[str, float], list[Dict[str, Any]]]:
    adjusted = dict(weights)
    clips = []
    excess = 0.0
    for ticker, raw_weight in list(adjusted.items()):
        if ticker == cash_key:
            continue
        weight = float(raw_weight or 0.0)
        if weight > limit:
            excess += weight - limit
            adjusted[ticker] = float(limit)
            clips.append(
                {
                    "type": "position_cap",
                    "ticker": ticker,
                    "observed": weight,
                    "limit": float(limit),
                    "reduction": float(weight - limit),
                }
            )
    if excess > 0.0:
        adjusted[cash_key] = float(adjusted.get(cash_key, 0.0) or 0.0) + excess
    adjusted = _normalize_weights(adjusted, cash_key=cash_key, long_only=True)
    return adjusted, clips


def enforce_cash_floor(weights: Mapping[str, float], min_cash: float, cash_key: str = "CASH") -> tuple[Dict[str, float], list[Dict[str, Any]]]:
    adjusted = _normalize_weights(weights, cash_key=cash_key, long_only=True)
    current_cash = float(adjusted.get(cash_key, 0.0) or 0.0)
    if current_cash >= min_cash:
        return adjusted, []

    additional_cash = float(min_cash - current_cash)
    adjusted[cash_key] = float(min_cash)
    adjusted = _renormalize_non_cash(adjusted, cash_key=cash_key)
    return adjusted, [
        {
            "type": "cash_floor",
            "observed": current_cash,
            "limit": float(min_cash),
            "reduction": additional_cash,
        }
    ]


def smooth_turnover(
    prev_weights: Optional[Mapping[str, float]],
    new_weights: Mapping[str, float],
    limit: float,
    cash_key: str = "CASH",
) -> tuple[Dict[str, float], list[Dict[str, Any]]]:
    prev_norm = _normalize_weights(prev_weights or new_weights, cash_key=cash_key, long_only=True)
    new_norm = _normalize_weights(new_weights, cash_key=cash_key, long_only=True)
    turnover = compute_turnover(prev_norm, new_norm)
    if turnover <= limit or turnover <= _EPS:
        return new_norm, []

    alpha = max(_EPS, min(1.0, float(limit) / float(turnover)))
    keys = sorted(set(prev_norm) | set(new_norm))
    blended = {k: (1.0 - alpha) * prev_norm.get(k, 0.0) + alpha * new_norm.get(k, 0.0) for k in keys}
    blended = _normalize_weights(blended, cash_key=cash_key, long_only=True)
    return blended, [
        {
            "type": "turnover_smoothing",
            "observed": turnover,
            "limit": float(limit),
            "blend_alpha": alpha,
        }
    ]


def apply_risk_overlay(
    raw_weights: Mapping[str, float],
    policy: Optional[PortfolioPolicy],
    market_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    policy = policy or PortfolioPolicy()
    market_context = market_context or {}
    previous_weights = market_context.get("previous_weights") or {}
    cash_key = str(market_context.get("cash_key") or "CASH")

    adjusted = _normalize_weights(raw_weights, cash_key=cash_key, long_only=bool(policy.long_only))
    clips: list[Dict[str, Any]] = []

    adjusted, c1 = clip_position_weights(adjusted, float(policy.max_single_position_cap), cash_key=cash_key)
    clips.extend(c1)

    adjusted, c2 = enforce_cash_floor(adjusted, float(policy.min_cash_weight), cash_key=cash_key)
    clips.extend(c2)

    adjusted, c3 = smooth_turnover(previous_weights, adjusted, float(policy.turnover_budget), cash_key=cash_key)
    clips.extend(c3)

    adjusted = _normalize_weights(adjusted, cash_key=cash_key, long_only=bool(policy.long_only))

    return {
        "raw_weights": dict(raw_weights),
        "adjusted_weights": adjusted,
        "applied_clips": clips,
        "turnover_before": compute_turnover(previous_weights, raw_weights),
        "turnover_after": compute_turnover(previous_weights, adjusted),
        "concentration_before": compute_concentration(raw_weights),
        "concentration_after": compute_concentration(adjusted),
    }
