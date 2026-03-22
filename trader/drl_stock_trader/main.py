from __future__ import annotations

import json
import re
import uuid
import warnings
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from trader.drl_stock_trader import models
from trader.drl_stock_trader.config import paths
from trader.services.trading_service import build_advisory_mandate

warnings.filterwarnings("ignore")


def _socket_send(socket, message_type: str, message: Any) -> None:
    socket.send(
        text_data=json.dumps(
            {
                "type": message_type,
                "message": message,
            },
            ensure_ascii=False,
        )
    )


def _normalize_robustness(value: Any) -> int:
    raw = str(value).strip()
    if "_" in raw:
        raw = raw.split("_")[-1]
    return int(raw)


def _normalize_date(value: Any) -> str:
    raw = str(value).strip()
    if not raw:
        raise ValueError("Date value is empty.")
    raw = raw.replace("/", "-")

    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
        return raw.replace("-", "")
    if re.fullmatch(r"\d{8}", raw):
        return raw
    raise ValueError(f"Unsupported date format: {value}")


def _extract_trade_window(period_trade: Any) -> Tuple[str, str]:
    text = str(period_trade or "").strip()
    if not text:
        raise ValueError("period_trade is required when trade_start/trade_end are not provided.")

    matches = re.findall(r"\d{4}-\d{2}-\d{2}|\d{8}", text)
    if len(matches) != 2:
        raise ValueError(
            "period_trade must contain exactly two dates, e.g. "
            "'20160101-20190101' or '2016-01-01 to 2019-01-01'."
        )
    return _normalize_date(matches[0]), _normalize_date(matches[1])


def normalize_etf30_input(
    initial_amount: Any,
    robustness: Any,
    train_start: Any,
    period_trade: Optional[Any] = None,
    *,
    trade_start: Optional[Any] = None,
    trade_end: Optional[Any] = None,
) -> Tuple[float, int, str, str, str]:
    initial_amount = float(initial_amount)
    robustness = _normalize_robustness(robustness)
    train_start = _normalize_date(train_start)

    if trade_start is not None and trade_end is not None:
        trade_start = _normalize_date(trade_start)
        trade_end = _normalize_date(trade_end)
    else:
        trade_start, trade_end = _extract_trade_window(period_trade)

    return initial_amount, robustness, train_start, trade_start, trade_end


def _build_legacy_payload(
    *,
    session_id: str,
    initial_amount: float,
    robustness: int,
    train_start: str,
    trade_start: str,
    trade_end: str,
    benchmark_choice: Optional[str] = None,
    scenario_mode: bool = False,
    investor_profile: Optional[Dict[str, Any]] = None,
    policy_settings: Optional[Dict[str, Any]] = None,
    rebalance_cadence_days: Optional[int] = None,
    engine: Optional[str] = None,
):
    payload: Dict[str, Any] = {
        "market": "etf30",
        "session_id": session_id,
        "initial_amount": initial_amount,
        "robustness": robustness,
        "date_train": train_start,
        "date_trade_1": trade_start,
        "date_trade_2": trade_end,
        "scenario_mode": bool(scenario_mode),
        "investor_profile": investor_profile or {},
        "policy_settings": policy_settings or {},
    }
    if benchmark_choice is not None:
        payload["benchmark_choice"] = benchmark_choice
    if rebalance_cadence_days is not None:
        payload["rebalance_cadence_days"] = int(rebalance_cadence_days)
    if engine is not None:
        payload["engine"] = str(engine)
    return payload


def execute_etf30_session(
    socket,
    initial_amount: Any,
    robustness: Any,
    train_start: Any,
    period_trade: Optional[Any] = None,
    *,
    trade_start: Optional[Any] = None,
    trade_end: Optional[Any] = None,
    benchmark_choice: Optional[str] = None,
    scenario_mode: bool = False,
    investor_profile: Optional[Dict[str, Any]] = None,
    policy_settings: Optional[Dict[str, Any]] = None,
    rebalance_cadence_days: Optional[int] = None,
    session_id: Optional[str] = None,
    engine: Optional[str] = None,
):
    pd.set_option("display.max_rows", None, "display.max_columns", None)

    initial_amount, robustness, train_start, trade_start, trade_end = normalize_etf30_input(
        initial_amount=initial_amount,
        robustness=robustness,
        train_start=train_start,
        period_trade=period_trade,
        trade_start=trade_start,
        trade_end=trade_end,
    )

    session_id = session_id or f"etf30_{uuid.uuid4().hex[:8]}"
    payload = _build_legacy_payload(
        session_id=session_id,
        initial_amount=initial_amount,
        robustness=robustness,
        train_start=train_start,
        trade_start=trade_start,
        trade_end=trade_end,
        benchmark_choice=benchmark_choice,
        scenario_mode=scenario_mode,
        investor_profile=investor_profile,
        policy_settings=policy_settings,
        rebalance_cadence_days=rebalance_cadence_days,
        engine=engine,
    )
    mandate = build_advisory_mandate(payload)

    _socket_send(socket, "terminal", f"[SESSION] session_id={mandate.session_id}")
    _socket_send(socket, "terminal", "[PIPELINE] ETF30 advisory execution started.")
    _socket_send(
        socket,
        "terminal",
        (
            f"[MANDATE] engine={mandate.engine} benchmark={mandate.benchmark_choice} "
            f"scenario_mode={mandate.scenario_mode} rebalance_days={mandate.rebalance_cadence_days}"
        ),
    )

    initial_balance_path = paths.initial_balance_file_path()
    paths.ensure_parent_dir(initial_balance_path)
    with open(initial_balance_path, "w", encoding="utf-8") as f:
        f.write(str(initial_amount))

    models.run_portfolio_advisory_session(socket=socket, mandate=mandate)
    return mandate


def run_model_offline(
    socket,
    market: str,
    initial_amount: Any,
    robustness: Any,
    train_start: Any,
    period_trade: Optional[Any] = None,
    *,
    trade_start: Optional[Any] = None,
    trade_end: Optional[Any] = None,
    benchmark_choice: Optional[str] = None,
    scenario_mode: bool = False,
    investor_profile: Optional[Dict[str, Any]] = None,
    policy_settings: Optional[Dict[str, Any]] = None,
    rebalance_cadence_days: Optional[int] = None,
    engine: Optional[str] = None,
):
    market = str(market).strip().lower()
    if market != "etf30":
        raise ValueError("Only ETF30 is supported in the current advisory pipeline.")

    return execute_etf30_session(
        socket=socket,
        initial_amount=initial_amount,
        robustness=robustness,
        train_start=train_start,
        period_trade=period_trade,
        trade_start=trade_start,
        trade_end=trade_end,
        benchmark_choice=benchmark_choice,
        scenario_mode=scenario_mode,
        investor_profile=investor_profile,
        policy_settings=policy_settings,
        rebalance_cadence_days=rebalance_cadence_days,
        engine=engine,
    )
