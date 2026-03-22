from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
from uuid import uuid4

from trader.drl_stock_trader.config.app_config import APP_CONFIG
from trader.drl_stock_trader.models import run_portfolio_advisory_session


class TradingServiceError(ValueError):
    pass


@dataclass(frozen=True)
class AdvisoryMandate:
    session_id: str
    initial_amount: float
    robustness: int
    train_start: str
    trade_start: str
    trade_end: str
    benchmark_choice: str
    scenario_mode: bool
    rebalance_cadence_days: int
    investor_profile: Dict[str, Any]
    policy_settings: Dict[str, Any]
    engine: str = "legacy_rl"
    risk_mode: str = "standard"
    max_position_weight: float = 0.12
    min_cash_weight: float = 0.05
    max_turnover: float = 0.35
    explanation_depth: str = "standard"


DEFAULT_TRAIN_START = "20150101"
DEFAULT_TRADE_START = "20150101"
DEFAULT_TRADE_END = "20250101"


def _normalize_date(value: Any, default: str) -> str:
    raw = str(value or default).strip().replace("-", "")
    if len(raw) != 8 or not raw.isdigit():
        raise TradingServiceError(f"Invalid date value: {value}")
    return raw


def _parse_float(value: Any, name: str) -> float:
    try:
        out = float(value)
    except Exception as exc:
        raise TradingServiceError(f"{name} must be numeric") from exc
    if out <= 0:
        raise TradingServiceError(f"{name} must be greater than zero")
    return out


def _parse_optional_float(value: Any, default: float) -> float:
    if value is None or value == "":
        return float(default)
    try:
        return float(value)
    except Exception:
        return float(default)


def _parse_int(value: Any, name: str, default: int) -> int:
    if value is None or value == "":
        return int(default)
    raw = str(value).strip()
    if "_" in raw:
        raw = raw.split("_")[-1]
    try:
        return int(raw)
    except Exception as exc:
        raise TradingServiceError(f"{name} must be an integer") from exc


def _normalize_investor_profile(payload: Dict[str, Any]) -> Dict[str, Any]:
    defaults = APP_CONFIG.investor_defaults
    profile = payload.get("investor_profile") or {}

    return {
        "profile_name": str(profile.get("profile_name", defaults.profile_name)),
        "target_style": str(profile.get("target_style", defaults.target_style)),
        "risk_tolerance": float(profile.get("risk_tolerance", defaults.risk_tolerance)),
        "target_volatility": float(profile.get("target_volatility", defaults.target_volatility)),
        "max_drawdown_preference": float(profile.get("max_drawdown_preference", defaults.max_drawdown_preference)),
        "turnover_aversion": float(profile.get("turnover_aversion", defaults.turnover_aversion)),
        "min_cash_preference": float(profile.get("min_cash_preference", defaults.min_cash_preference)),
        "benchmark_preference": str(profile.get("benchmark_preference", defaults.benchmark_preference)),
        "advisor_mode": str(profile.get("advisor_mode", defaults.advisor_mode)),
    }


def _normalize_policy(payload: Dict[str, Any]) -> Dict[str, Any]:
    defaults = APP_CONFIG.policy
    risk_flags = APP_CONFIG.risk_flags
    policy = payload.get("policy_settings") or {}

    max_single_position_cap = _parse_optional_float(
        policy.get("max_single_position_cap", payload.get("max_position_weight")),
        defaults.max_single_position_cap,
    )
    min_cash_weight = _parse_optional_float(
        policy.get("min_cash_weight", payload.get("min_cash_weight")),
        defaults.target_cash_floor,
    )
    turnover_budget = _parse_optional_float(
        policy.get("turnover_budget", payload.get("max_turnover")),
        defaults.turnover_budget,
    )

    return {
        "max_single_position_cap": float(max_single_position_cap or risk_flags.max_position_weight),
        "min_cash_weight": float(min_cash_weight or risk_flags.min_cash_weight),
        "turnover_budget": float(turnover_budget or risk_flags.max_turnover_per_rebalance),
        "rebalance_cadence_days": int(policy.get("rebalance_cadence_days", defaults.rebalance_cadence_days)),
        "allow_cash_sleeve": bool(policy.get("allow_cash_sleeve", defaults.allow_cash_sleeve)),
        "long_only": bool(policy.get("long_only", defaults.long_only)),
        "sector_caps": dict(policy.get("sector_caps") or {}),
    }


def _normalize_engine(payload: Dict[str, Any]) -> str:
    raw_engine = str(payload.get("engine") or APP_CONFIG.engine.default_engine).strip().lower()
    allowed = set(APP_CONFIG.engine.allowed_engines)

    if raw_engine == "finrl" and APP_CONFIG.engine.enable_finrl and raw_engine in allowed:
        return raw_engine

    if raw_engine == "legacy_rl" and raw_engine in allowed:
        return raw_engine

    if APP_CONFIG.engine.enable_engine_fallback:
        return APP_CONFIG.engine.default_engine

    raise TradingServiceError(f"Unsupported engine: {raw_engine}")


def build_advisory_mandate(payload: Dict[str, Any]) -> AdvisoryMandate:
    market = str(payload.get("market", "etf30")).strip().lower() or "etf30"
    if market != "etf30":
        raise TradingServiceError("Only ETF30 is supported in this advisory flow.")

    session_id = str(payload.get("session_id") or f"etf30_{uuid4().hex[:12]}")
    initial_amount = _parse_float(payload.get("initial_amount", 0), "initial_amount")
    robustness = _parse_int(payload.get("robustness", 3), "robustness", 3)

    train_start = _normalize_date(payload.get("date_train"), DEFAULT_TRAIN_START)
    trade_start = _normalize_date(payload.get("date_trade_1"), DEFAULT_TRADE_START)
    trade_end = _normalize_date(payload.get("date_trade_2"), DEFAULT_TRADE_END)

    benchmark_choice = str(
        payload.get("benchmark_choice")
        or payload.get("benchmark")
        or APP_CONFIG.benchmarks.default_primary
    ).strip().lower()
    if benchmark_choice not in APP_CONFIG.benchmarks.available:
        benchmark_choice = APP_CONFIG.benchmarks.default_primary

    scenario_mode = bool(payload.get("scenario_mode", False))
    investor_profile = _normalize_investor_profile(payload)
    policy_settings = _normalize_policy(payload)

    rebalance_cadence_days = _parse_int(
        payload.get("rebalance_cadence_days", policy_settings["rebalance_cadence_days"]),
        "rebalance_cadence_days",
        policy_settings["rebalance_cadence_days"],
    )
    policy_settings["rebalance_cadence_days"] = rebalance_cadence_days

    engine = _normalize_engine(payload)
    risk_mode = str(payload.get("risk_mode") or "standard").strip().lower() or "standard"
    explanation_depth = str(payload.get("explanation_depth") or "standard").strip().lower() or "standard"

    max_position_weight = _parse_optional_float(
        payload.get("max_position_weight"),
        policy_settings["max_single_position_cap"],
    )
    min_cash_weight = _parse_optional_float(
        payload.get("min_cash_weight"),
        policy_settings["min_cash_weight"],
    )
    max_turnover = _parse_optional_float(
        payload.get("max_turnover"),
        policy_settings["turnover_budget"],
    )

    policy_settings["max_single_position_cap"] = max_position_weight
    policy_settings["min_cash_weight"] = min_cash_weight
    policy_settings["turnover_budget"] = max_turnover

    return AdvisoryMandate(
        session_id=session_id,
        initial_amount=initial_amount,
        robustness=robustness,
        train_start=train_start,
        trade_start=trade_start,
        trade_end=trade_end,
        benchmark_choice=benchmark_choice,
        scenario_mode=scenario_mode,
        rebalance_cadence_days=rebalance_cadence_days,
        investor_profile=investor_profile,
        policy_settings=policy_settings,
        engine=engine,
        risk_mode=risk_mode,
        max_position_weight=max_position_weight,
        min_cash_weight=min_cash_weight,
        max_turnover=max_turnover,
        explanation_depth=explanation_depth,
    )


def execute_trade(socket, payload: Dict[str, Any]) -> AdvisoryMandate:
    mandate = build_advisory_mandate(payload)
    run_portfolio_advisory_session(socket=socket, mandate=mandate)
    return mandate