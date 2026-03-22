from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from trader.drl_stock_trader.config.app_config import APP_CONFIG


@dataclass
class SelectionResult:
    best_candidate: object
    ranked_candidates: List[object]


def _candidate_score_breakdown(candidate) -> dict:
    summary = candidate.validation_summary
    if summary is None:
        raise ValueError(f"Candidate {getattr(candidate, 'label', '?')} is missing validation_summary.")

    cfg = APP_CONFIG.selection
    parts = {
        "sharpe": cfg.sharpe_weight * float(getattr(summary, "sharpe", 0.0)),
        "active_return": cfg.active_return_weight * float(getattr(summary, "active_return", 0.0)),
        "drawdown": -cfg.drawdown_weight * float(getattr(summary, "max_drawdown", 0.0)),
        "turnover": -cfg.turnover_weight * float(getattr(summary, "turnover", 0.0)),
        "concentration": -cfg.concentration_weight * float(getattr(summary, "concentration", 0.0)),
        "stability": cfg.stability_weight * float(getattr(summary, "stability_score", 0.0)),
        "policy_compliance": cfg.policy_compliance_weight * float(getattr(summary, "policy_compliance_score", 0.0)),
    }
    parts["total"] = float(sum(parts.values()))
    return parts


def _candidate_score(candidate) -> float:
    parts = _candidate_score_breakdown(candidate)
    try:
        setattr(candidate, "selection_score", parts["total"])
        setattr(candidate, "selection_score_breakdown", parts)
    except Exception:
        pass
    return float(parts["total"])


def _has_hard_policy_breach(candidate) -> bool:
    summary = getattr(candidate, "validation_summary", None)
    if summary is None:
        return True

    for attr in ("hard_policy_breach", "has_hard_policy_breach"):
        if hasattr(summary, attr):
            return bool(getattr(summary, attr))

    for attr in ("hard_policy_breaches", "policy_breaches"):
        if hasattr(summary, attr):
            try:
                return len(getattr(summary, attr) or []) > 0
            except Exception:
                return bool(getattr(summary, attr))

    return False


def select_best_candidate(candidates: Iterable[object]) -> SelectionResult:
    candidates = list(candidates)
    if not candidates:
        raise ValueError("No candidate models were produced.")

    compliant = [c for c in candidates if not _has_hard_policy_breach(c)]
    ranked_pool = compliant if compliant else candidates
    ranked = sorted(ranked_pool, key=_candidate_score, reverse=True)
    return SelectionResult(best_candidate=ranked[0], ranked_candidates=ranked)
