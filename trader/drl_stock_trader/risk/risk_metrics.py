from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from trader.domain.session_models import RiskSnapshot

_EPS = 1e-12


def _as_series(values: Optional[Iterable[float]]) -> pd.Series:
    if values is None:
        return pd.Series(dtype=float)
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return pd.to_numeric(pd.Series(list(values), dtype=float), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def _normalize_weights(weights: Optional[Mapping[str, float]]) -> Dict[str, float]:
    raw = {str(k): float(v) for k, v in (weights or {}).items()}
    total = float(sum(max(v, 0.0) for v in raw.values()))
    if total <= _EPS:
        return {k: 0.0 for k in raw}
    return {k: max(v, 0.0) / total for k, v in raw.items()}


def compute_volatility(returns: Optional[Iterable[float]], annualization: float = 252.0) -> float:
    s = _as_series(returns)
    if s.empty:
        return 0.0
    return float(s.std(ddof=0) * np.sqrt(float(annualization)))


def compute_downside_volatility(returns: Optional[Iterable[float]], annualization: float = 252.0) -> float:
    s = _as_series(returns)
    if s.empty:
        return 0.0
    downside = s.where(s < 0.0, 0.0)
    return float(downside.std(ddof=0) * np.sqrt(float(annualization)))


def compute_max_drawdown(account_values: Optional[Iterable[float]]) -> float:
    s = _as_series(account_values)
    if s.empty:
        return 0.0
    peaks = s.cummax().replace(0.0, np.nan)
    drawdowns = (1.0 - s / peaks).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return float(drawdowns.max())


def compute_turnover(prev_weights: Optional[Mapping[str, float]], new_weights: Optional[Mapping[str, float]]) -> float:
    prev = _normalize_weights(prev_weights)
    new = _normalize_weights(new_weights)
    keys = sorted(set(prev) | set(new))
    if not keys:
        return 0.0
    return float(0.5 * sum(abs(new.get(k, 0.0) - prev.get(k, 0.0)) for k in keys))


def compute_concentration(weights: Optional[Mapping[str, float]], exclude_cash: bool = False, cash_key: str = "CASH") -> float:
    norm = _normalize_weights(weights)
    if exclude_cash:
        norm = {k: v for k, v in norm.items() if k != cash_key}
        total = float(sum(norm.values()))
        if total > _EPS:
            norm = {k: v / total for k, v in norm.items()}
    if not norm:
        return 0.0
    return float(sum(v * v for v in norm.values()))


def compute_tracking_error(portfolio_returns: Optional[Iterable[float]], benchmark_returns: Optional[Iterable[float]], annualization: float = 252.0) -> float:
    p = _as_series(portfolio_returns)
    b = _as_series(benchmark_returns)
    if p.empty or b.empty:
        return 0.0
    n = int(min(len(p), len(b)))
    active = p.iloc[-n:].reset_index(drop=True) - b.iloc[-n:].reset_index(drop=True)
    if active.empty:
        return 0.0
    return float(active.std(ddof=0) * np.sqrt(float(annualization)))


def compute_active_return(portfolio_returns: Optional[Iterable[float]], benchmark_returns: Optional[Iterable[float]]) -> float:
    p = _as_series(portfolio_returns)
    b = _as_series(benchmark_returns)
    if p.empty and b.empty:
        return 0.0
    if b.empty:
        return float(p.sum()) if not p.empty else 0.0
    if p.empty:
        return float(-b.sum())
    n = int(min(len(p), len(b)))
    active = p.iloc[-n:].reset_index(drop=True) - b.iloc[-n:].reset_index(drop=True)
    return float(active.sum())


def build_risk_snapshot(
    *,
    history: Optional[pd.DataFrame] = None,
    portfolio_returns: Optional[Iterable[float]] = None,
    benchmark_returns: Optional[Iterable[float]] = None,
    account_values: Optional[Iterable[float]] = None,
    previous_weights: Optional[Mapping[str, float]] = None,
    target_weights: Optional[Mapping[str, float]] = None,
    cash_weight: Optional[float] = None,
) -> RiskSnapshot:
    hist = history.copy() if history is not None else pd.DataFrame()

    port_rets = _as_series(portfolio_returns if portfolio_returns is not None else hist.get("portfolio_return"))
    bench_rets = _as_series(benchmark_returns if benchmark_returns is not None else hist.get("benchmark_return"))
    values = _as_series(account_values if account_values is not None else hist.get("account_value"))

    if cash_weight is None and target_weights is not None:
        cash_weight = float((target_weights or {}).get("CASH", 0.0) or 0.0)

    snapshot = RiskSnapshot(
        realized_volatility=compute_volatility(port_rets),
        downside_volatility=compute_downside_volatility(port_rets),
        max_drawdown=compute_max_drawdown(values),
        concentration_hhi=compute_concentration(target_weights, exclude_cash=False),
        turnover=compute_turnover(previous_weights, target_weights),
        tracking_error=compute_tracking_error(port_rets, bench_rets),
        cash_weight=float(cash_weight or 0.0),
    )
    return snapshot
