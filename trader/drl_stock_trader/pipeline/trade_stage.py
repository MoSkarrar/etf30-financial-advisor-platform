from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv

from trader.domain.session_models import (
    AllocationRecommendation,
    BenchmarkComparison,
    InvestorProfile,
    PortfolioPolicy,
    RiskSnapshot,
)
from trader.drl_stock_trader import algorithms
from trader.drl_stock_trader.RL_envs.EnvMultipleStock_Trade import StockEnvTrade
from trader.drl_stock_trader.config import paths
from trader.drl_stock_trader.risk.policy_checks import build_policy_check_result
from trader.drl_stock_trader.risk.risk_metrics import build_risk_snapshot, compute_active_return
from trader.drl_stock_trader.risk.risk_overlay import apply_risk_overlay


@dataclass
class TradeStageResult:
    allocation: AllocationRecommendation
    benchmark_comparison: BenchmarkComparison
    risk_snapshot: RiskSnapshot
    portfolio_value_series: List[float]
    target_weights: Dict[str, float]
    previous_weights: Dict[str, float]
    rebalance_deltas: Dict[str, float]
    rebalance_trades: List[Dict[str, float]]
    turnover_estimate: float
    cash_weight: float
    policy_breaches: List[str] = field(default_factory=list)
    raw_target_weights: Dict[str, float] = field(default_factory=dict)
    adjusted_target_weights: Dict[str, float] = field(default_factory=dict)
    policy_check: Dict[str, object] = field(default_factory=dict)
    overlay_report: Dict[str, object] = field(default_factory=dict)
    allocation_json_path: str = ""
    benchmark_json_path: str = ""
    advisory_summary_json_path: str = ""
    allocation_snapshot_path: str = ""
    risk_report_path: str = ""


def _to_weight_map(tickers: Sequence[str], weights: Sequence[float]) -> Dict[str, float]:
    return {str(t): float(w) for t, w in zip(tickers, weights)}


def _rebalance_trades_from_delta(deltas: Dict[str, float], portfolio_value: float) -> List[Dict[str, float]]:
    trades = []
    for ticker, delta in deltas.items():
        if ticker == "CASH" or abs(delta) < 1e-6:
            continue
        trades.append(
            {
                "ticker": str(ticker),
                "delta_weight": float(delta),
                "notional_change": float(delta * portfolio_value),
                "action": "BUY" if delta > 0 else "SELL",
            }
        )
    return sorted(trades, key=lambda x: abs(x["delta_weight"]), reverse=True)


def _dump_json(path: str, payload: dict) -> str:
    paths.ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path


def _ensure_numeric_column(hist: pd.DataFrame, column: str, default: float = 0.0) -> None:
    if column not in hist.columns:
        hist[column] = default
    hist[column] = pd.to_numeric(hist[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def _safe_history(history: pd.DataFrame) -> pd.DataFrame:
    hist = history.copy() if history is not None else pd.DataFrame()
    if hist.empty:
        return pd.DataFrame(
            {
                "datadate": [],
                "account_value": [],
                "portfolio_return": [],
                "turnover": [],
                "concentration_hhi": [],
                "cash_weight": [],
                "benchmark_return": [],
                "active_return": [],
            }
        )

    for col in [
        "account_value",
        "portfolio_return",
        "turnover",
        "concentration_hhi",
        "cash_weight",
        "benchmark_return",
        "active_return",
    ]:
        _ensure_numeric_column(hist, col, default=0.0)

    return hist


def _weights_vector_to_map(tickers: Sequence[str], vector: Optional[Sequence[float]]) -> Dict[str, float]:
    if vector is None:
        return {str(t): (1.0 if idx == 0 and str(t) == "CASH" else 0.0) for idx, t in enumerate(tickers)}
    arr = np.asarray(vector, dtype=float).reshape(-1)
    if arr.size != len(tickers):
        arr = np.zeros(len(tickers), dtype=float)
        if len(tickers) > 0 and str(tickers[0]) == "CASH":
            arr[0] = 1.0
    return _to_weight_map(tickers, arr.tolist())


def _build_rebalance_deltas(target_weights: Mapping[str, float], previous_weights_map: Mapping[str, float]) -> Dict[str, float]:
    keys = sorted(set(target_weights) | set(previous_weights_map))
    return {k: float(target_weights.get(k, 0.0) - previous_weights_map.get(k, 0.0)) for k in keys}


def execute_trade_stage(
    socket,
    df: pd.DataFrame,
    covariance_by_date: Dict[int, np.ndarray],
    feature_columns: Sequence[str],
    model,
    run_id: str,
    benchmark_name: str = "equal_weight",
    investor_profile: Optional[InvestorProfile] = None,
    policy: Optional[PortfolioPolicy] = None,
    previous_weights: Optional[Sequence[float]] = None,
    strategy_name: str = "ensemble",
) -> TradeStageResult:
    investor_profile = investor_profile or InvestorProfile()
    policy = policy or PortfolioPolicy()

    trade_env = DummyVecEnv([
        lambda: StockEnvTrade(
            socket=socket,
            df=df,
            covariance_by_date=covariance_by_date,
            feature_columns=feature_columns,
            benchmark_name=benchmark_name,
            investor_profile=investor_profile,
            policy=policy,
            previous_state=previous_weights,
            model_name=strategy_name,
            iteration=run_id,
        )
    ])

    prediction = algorithms.predict_last_allocation(model, trade_env, deterministic=True)
    history = _safe_history(prediction.get("history", pd.DataFrame()))
    tickers = list(prediction.get("weights", {}).keys())
    raw_target_weights = {str(k): float(v) for k, v in prediction.get("weights", {}).items()}

    previous_weights_map = _weights_vector_to_map(tickers, previous_weights)

    overlay_report = apply_risk_overlay(
        raw_target_weights,
        policy,
        market_context={"previous_weights": previous_weights_map, "cash_key": "CASH"},
    )
    adjusted_target_weights = {str(k): float(v) for k, v in overlay_report.get("adjusted_weights", {}).items()}
    target_weights = adjusted_target_weights or raw_target_weights

    rebalance_deltas = _build_rebalance_deltas(target_weights, previous_weights_map)

    latest = prediction.get("latest_diagnostics", {}) or (history.iloc[-1].to_dict() if not history.empty else {})
    metrics_hist = history.iloc[1:].copy() if len(history) > 1 else history.copy()

    cash_weight = float(target_weights.get("CASH", latest.get("cash_weight", 0.0) or 0.0)) if ("CASH" in target_weights or "cash_weight" in latest) else 0.0

    risk_snapshot = build_risk_snapshot(
        history=metrics_hist,
        previous_weights=previous_weights_map,
        target_weights=target_weights,
        cash_weight=cash_weight,
    )

    benchmark_return = float(metrics_hist["benchmark_return"].sum()) if not metrics_hist.empty else 0.0
    portfolio_return = float(metrics_hist["portfolio_return"].sum()) if not metrics_hist.empty else 0.0
    active_return = compute_active_return(
        metrics_hist["portfolio_return"] if not metrics_hist.empty else None,
        metrics_hist["benchmark_return"] if not metrics_hist.empty else None,
    )

    if not metrics_hist.empty:
        active_std = float(metrics_hist["active_return"].std(ddof=0))
        information_ratio = float(metrics_hist["active_return"].mean() / max(active_std, 1e-8) * np.sqrt(252.0))
    else:
        information_ratio = 0.0

    benchmark_comparison = BenchmarkComparison(
        benchmark_name=benchmark_name,
        benchmark_return=benchmark_return,
        portfolio_return=portfolio_return,
        active_return=active_return,
        tracking_error=risk_snapshot.tracking_error,
        information_ratio=information_ratio,
    )

    policy_check = build_policy_check_result(
        weights=target_weights,
        turnover=risk_snapshot.turnover,
        concentration=risk_snapshot.concentration_hhi,
        policy=policy,
        applied_clips=overlay_report.get("applied_clips", []),
    )
    policy_breaches = list(policy_check.get("breaches", []) or [])

    as_of_date = str(int(latest.get("date", latest.get("datadate", df["datadate"].max()))))
    confidence = max(0.0, min(1.0, 1.0 - risk_snapshot.max_drawdown - risk_snapshot.turnover))

    rationale_parts = [
        f"Target portfolio optimized for {investor_profile.profile_name} profile versus {benchmark_name}.",
        f"Cash weight is {cash_weight:.2%}; turnover estimate is {risk_snapshot.turnover:.2%}.",
    ]
    if overlay_report.get("applied_clips"):
        rationale_parts.append(f"Risk overlay applied {len(overlay_report.get('applied_clips', []))} policy adjustment(s).")

    allocation = AllocationRecommendation(
        run_id=run_id,
        as_of_date=as_of_date,
        target_weights={k: float(v) for k, v in target_weights.items()},
        previous_weights=previous_weights_map,
        rebalance_deltas=rebalance_deltas,
        confidence=confidence,
        rationale_summary=" ".join(rationale_parts),
        policy_breaches=policy_breaches,
    )

    latest_account_value = float(latest.get("portfolio_value", latest.get("account_value", 0.0) or 0.0))
    rebalance_trades = _rebalance_trades_from_delta(rebalance_deltas, latest_account_value)

    allocation_json_path = _dump_json(paths.allocation_json_path(run_id), allocation.to_dict())
    benchmark_json_path = _dump_json(paths.benchmark_json_path(run_id), benchmark_comparison.to_dict())
    advisory_summary_json_path = _dump_json(
        paths.advisory_summary_json_path(run_id),
        {
            "run_id": run_id,
            "summary": allocation.rationale_summary,
            "confidence": allocation.confidence,
            "top_trades": rebalance_trades[:10],
            "policy_check": policy_check,
        },
    )
    risk_report_path = _dump_json(
        paths.risk_report_path(run_id),
        {
            **risk_snapshot.to_dict(),
            "policy_check": policy_check,
            "overlay_report": overlay_report,
        },
    )

    allocation_snapshot_path = paths.allocation_snapshot_path(run_id)
    paths.ensure_parent_dir(allocation_snapshot_path)
    pd.DataFrame(
        {
            "ticker": list(target_weights.keys()),
            "target_weight": [float(target_weights[k]) for k in target_weights.keys()],
            "previous_weight": [float(previous_weights_map.get(k, 0.0)) for k in target_weights.keys()],
            "delta_weight": [float(rebalance_deltas.get(k, 0.0)) for k in target_weights.keys()],
            "raw_target_weight": [float(raw_target_weights.get(k, 0.0)) for k in target_weights.keys()],
        }
    ).to_csv(allocation_snapshot_path, index=False)

    return TradeStageResult(
        allocation=allocation,
        benchmark_comparison=benchmark_comparison,
        risk_snapshot=risk_snapshot,
        portfolio_value_series=[float(v) for v in (history.get("account_value", pd.Series(dtype=float)).tolist() if not history.empty else [])],
        target_weights=target_weights,
        previous_weights=previous_weights_map,
        rebalance_deltas=rebalance_deltas,
        rebalance_trades=rebalance_trades,
        turnover_estimate=risk_snapshot.turnover,
        cash_weight=cash_weight,
        policy_breaches=policy_breaches,
        raw_target_weights=raw_target_weights,
        adjusted_target_weights=target_weights,
        policy_check=policy_check,
        overlay_report=overlay_report,
        allocation_json_path=allocation_json_path,
        benchmark_json_path=benchmark_json_path,
        advisory_summary_json_path=advisory_summary_json_path,
        allocation_snapshot_path=allocation_snapshot_path,
        risk_report_path=risk_report_path,
    )
