from __future__ import annotations

import json
import time
from typing import List

from trader.domain.session_models import ExplanationBundle, InvestorProfile, PolicyCheckResult, PortfolioPolicy, RuleExplanation
from trader.drl_stock_trader.config.app_config import APP_CONFIG
from trader.drl_stock_trader.engines.engine_registry import run_engine
from trader.drl_stock_trader.pipeline.data_stage import prepare_execution_dataset, prepare_iteration_bundle
from trader.services import artifact_store


def _send(socket, message_type: str, message):
    socket.send(text_data=json.dumps({"type": message_type, "message": message}, ensure_ascii=False))


def _build_scenario_result(allocation, risk_snapshot, policy_settings, benchmark_choice: str):
    target_weights = (
        allocation.target_weights if hasattr(allocation, "target_weights") else dict(allocation.get("target_weights", {}))
    )
    cash_weight = float(target_weights.get("CASH", 0.0))
    stricter_cap = min(float(policy_settings.get("max_single_position_cap", 0.12)), 0.08)
    stricter_cash = max(float(policy_settings.get("min_cash_weight", 0.05)), 0.08)
    top_risky = sorted(
        [(k, float(v)) for k, v in target_weights.items() if k != "CASH"],
        key=lambda kv: kv[1],
        reverse=True,
    )[:5]
    turnover = max(0.0, stricter_cash - cash_weight) + sum(max(0.0, w - stricter_cap) for _, w in top_risky)
    return {
        "request": {
            "scenario_name": "stricter_policy",
            "shock_type": "policy_tightening",
            "shock_magnitude": turnover,
            "horizon_days": int(policy_settings.get("rebalance_cadence_days", APP_CONFIG.policy.rebalance_cadence_days)),
            "benchmark_name": benchmark_choice,
        },
        "projected_return": max(-0.25, float(getattr(risk_snapshot, "realized_volatility", 0.0)) * -0.10),
        "projected_volatility": float(getattr(risk_snapshot, "realized_volatility", 0.0)) * 0.95,
        "projected_drawdown": float(getattr(risk_snapshot, "max_drawdown", 0.0)) * 0.90,
        "narrative": (
            f"A stricter policy would likely require about {turnover:.2%} of turnover, "
            f"mostly by trimming positions above {stricter_cap:.0%} and raising cash toward {stricter_cash:.0%}."
        ),
    }


def _build_policy_check(trade_result, mandate) -> PolicyCheckResult:
    allocation = getattr(trade_result, "allocation", None)
    target_weights = getattr(allocation, "target_weights", {}) if allocation is not None else {}

    breaches: List[str] = []
    applied_clips = []
    max_cap = float(getattr(mandate, "max_position_weight", mandate.policy_settings.get("max_single_position_cap", 0.12)))
    min_cash = float(getattr(mandate, "min_cash_weight", mandate.policy_settings.get("min_cash_weight", 0.05)))
    max_turnover = float(getattr(mandate, "max_turnover", mandate.policy_settings.get("turnover_budget", 0.35)))

    cash_weight = float(target_weights.get("CASH", 0.0) or 0.0)
    if cash_weight < min_cash:
        breaches.append(f"Cash weight {cash_weight:.2%} is below required floor {min_cash:.2%}.")
        applied_clips.append({"type": "cash_floor", "observed": cash_weight, "limit": min_cash})

    for ticker, raw_weight in target_weights.items():
        if ticker == "CASH":
            continue
        weight = float(raw_weight or 0.0)
        if weight > max_cap:
            breaches.append(f"{ticker} weight {weight:.2%} exceeds max position cap {max_cap:.2%}.")
            applied_clips.append({"type": "position_cap", "ticker": ticker, "observed": weight, "limit": max_cap})

    turnover = float(getattr(getattr(trade_result, "risk_snapshot", None), "turnover", 0.0) or 0.0)
    if turnover > max_turnover:
        breaches.append(f"Turnover {turnover:.2%} exceeds turnover budget {max_turnover:.2%}.")
        applied_clips.append({"type": "turnover", "observed": turnover, "limit": max_turnover})

    severity = "none"
    if breaches:
        severity = "high" if len(breaches) >= 2 else "medium"

    return PolicyCheckResult(
        passed=not breaches,
        breaches=breaches,
        severity=severity,
        applied_clips=applied_clips,
        human_summary=(
            "Portfolio satisfies current policy constraints."
            if not breaches
            else "Policy review found one or more allocation constraints that may need advisor attention."
        ),
    )


def _build_rule_summary(explain_result, policy_check: PolicyCheckResult) -> RuleExplanation:
    risk_flags = []
    allocation_flags = []
    benchmark_flags = []
    rules_triggered = []

    risk_text = str(getattr(explain_result, "risk_text", "") or "").strip()
    policy_text = str(getattr(explain_result, "policy_text", "") or "").strip()
    benchmark_text = str(getattr(explain_result, "benchmark_text", "") or "").strip()

    if risk_text:
        risk_flags.append(risk_text)
        rules_triggered.append("risk_commentary_available")
    if benchmark_text:
        benchmark_flags.append(benchmark_text)
        rules_triggered.append("benchmark_commentary_available")
    if policy_text:
        allocation_flags.append(policy_text)
        rules_triggered.append("policy_commentary_available")
    if policy_check.breaches:
        allocation_flags.extend(policy_check.breaches)
        rules_triggered.append("policy_breach_detected")

    summary_parts = []
    if risk_text:
        summary_parts.append(risk_text)
    if benchmark_text:
        summary_parts.append(benchmark_text)
    if policy_check.human_summary:
        summary_parts.append(policy_check.human_summary)

    return RuleExplanation(
        rules_triggered=rules_triggered,
        risk_flags=risk_flags,
        allocation_flags=allocation_flags,
        benchmark_flags=benchmark_flags,
        summary_text=" ".join(part for part in summary_parts if part).strip(),
    )


def _build_explanation_bundle(explain_result, rule_summary: RuleExplanation) -> ExplanationBundle:
    shap_payload = getattr(explain_result, "shap_payload", None) or {}
    lime_payload = getattr(explain_result, "lime_payload", None) or {}
    advisor_summary = str(getattr(explain_result, "summary_text", "") or "").strip()
    technical_parts = [
        str(getattr(explain_result, "benchmark_text", "") or "").strip(),
        str(getattr(explain_result, "risk_text", "") or "").strip(),
        str(getattr(explain_result, "policy_text", "") or "").strip(),
    ]
    technical_summary = " ".join(part for part in technical_parts if part).strip()

    consensus_points = []
    if advisor_summary:
        consensus_points.append("Allocation explanation summary is available.")
    if rule_summary.summary_text:
        consensus_points.append("Rule-based explanation summary is available.")

    return ExplanationBundle(
        shap=shap_payload,
        lime=lime_payload,
        rule_summary=rule_summary.to_dict(),
        consensus_points=consensus_points,
        disagreement_points=[],
        advisor_summary=advisor_summary,
        technical_summary=technical_summary,
    )


def run_portfolio_advisory_session(socket, mandate) -> None:
    start = time.time()
    market = "etf30"
    investor_profile = InvestorProfile.from_dict(mandate.investor_profile)
    portfolio_policy = PortfolioPolicy.from_dict(mandate.policy_settings)

    _send(socket, "engine_status", f"[ENGINE] selected engine={getattr(mandate, 'engine', 'legacy_rl')}")
    _send(socket, "risk_status", f"[RISK] mode={getattr(mandate, 'risk_mode', 'standard')}")
    _send(socket, "explain_status", f"[XAI] depth={getattr(mandate, 'explanation_depth', 'standard')}")

    execution = prepare_execution_dataset(
        train_start=int(mandate.train_start),
        trade_start=int(mandate.trade_start),
        trade_end=int(mandate.trade_end),
    )
    prepared = execution.prepared
    unique_trade_date = execution.unique_trade_date
    prepared.metadata["timesteps"] = APP_CONFIG.rl.timesteps_for_robustness(mandate.robustness)

    rebalance_window = int(mandate.rebalance_cadence_days)
    validation_window = int(APP_CONFIG.rl.validation_window)

    _send(socket, "terminal", f"[DATA] trade dates available = {len(unique_trade_date)}")
    if unique_trade_date:
        _send(socket, "terminal", f"[DATA] first trade date = {unique_trade_date[0]}")
        _send(socket, "terminal", f"[DATA] last trade date = {unique_trade_date[-1]}")
    _send(socket, "terminal", f"[WINDOWS] rebalance={rebalance_window}, validation={validation_window}")

    if len(unique_trade_date) == 0:
        raise ValueError(
            "No trade dates found in the requested ETF30 window. "
            "Check the cached dataset or widen the trade date range."
        )
    if len(unique_trade_date) <= rebalance_window + validation_window:
        raise ValueError(
            f"Not enough trade dates for this setup. available={len(unique_trade_date)}, "
            f"required_more_than={rebalance_window + validation_window}. "
            "Try a wider trade window, a smaller rebalance cadence, or rebuild the ETF30 dataset cache."
        )

    previous_weights_vector = None

    for iteration in range(rebalance_window + validation_window, len(unique_trade_date), rebalance_window):
        bundle = prepare_iteration_bundle(
            prepared=prepared,
            unique_trade_date=unique_trade_date,
            train_start=int(mandate.train_start),
            iteration=iteration,
            investor_profile=investor_profile,
            policy=portfolio_policy,
            benchmark_name=mandate.benchmark_choice,
            rebalance_window=rebalance_window,
            validation_window=validation_window,
        )

        _send(socket, "terminal", f"Training window: {mandate.train_start} -> {bundle.validation_start}")
        _send(socket, "terminal", f"Validation window: {bundle.validation_start} -> {bundle.validation_end}")

        engine_result = run_engine(
            getattr(mandate, "engine", APP_CONFIG.engine.default_engine),
            socket=socket,
            mandate=mandate,
            market=market,
            prepared=prepared,
            bundle=bundle,
            iteration=iteration,
            previous_weights_vector=previous_weights_vector,
        )

        trade_result = engine_result.trade_result
        explain_result = engine_result.explain_result
        previous_weights_vector = list(trade_result.target_weights.values())

        _send(socket, "terminal", f"Selected advisor model: {engine_result.selected_model}")
        _send(socket, "terminal", f"Trade window: {bundle.trade_start} -> {bundle.trade_end}")

        scenario_result = (
            _build_scenario_result(
                trade_result.allocation,
                trade_result.risk_snapshot,
                mandate.policy_settings,
                mandate.benchmark_choice,
            )
            if mandate.scenario_mode
            else None
        )

        policy_check = _build_policy_check(trade_result, mandate)
        rule_summary = _build_rule_summary(explain_result, policy_check)
        explanation_bundle = _build_explanation_bundle(explain_result, rule_summary)

        run_id = f"{mandate.session_id}_{iteration}"
        manifest = artifact_store.persist_portfolio_advisory_bundle(
            session_id=mandate.session_id,
            run_id=run_id,
            market=market,
            iteration=iteration,
            trade_window={"start": bundle.trade_start, "end": bundle.trade_end},
            selected_model=engine_result.selected_model,
            run_level_metrics=engine_result.run_level_metrics,
            allocation_recommendation=trade_result.allocation,
            benchmark_comparison=trade_result.benchmark_comparison,
            risk_snapshot=trade_result.risk_snapshot,
            investor_profile=investor_profile,
            portfolio_policy=portfolio_policy,
            explanation_summary=explain_result.summary_text,
            xai_payload=explain_result.xai_payload,
            advisory_summary_text=trade_result.allocation.rationale_summary,
            advisory_summary_payload={
                "run_id": run_id,
                "summary": trade_result.allocation.rationale_summary,
                "benchmark_commentary": explain_result.benchmark_text,
                "risk_commentary": explain_result.risk_text,
                "policy_commentary": explain_result.policy_text,
            },
            scenario_result=scenario_result,
            validation_csv_path=(
                engine_result.selected_candidate.validation_summary.validation_csv_path
                if engine_result.selected_candidate.validation_summary
                else ""
            ),
            engine_info=engine_result.engine_info,
            policy_check=policy_check,
            shap_payload=explain_result.shap_payload if APP_CONFIG.xai.enable_shap else None,
            lime_payload=explain_result.lime_payload if APP_CONFIG.xai.enable_lime else None,
            rule_summary_payload=rule_summary,
            explanation_bundle_payload=explanation_bundle,
            explanation_lab_payload=explain_result.explanation_lab_payload if APP_CONFIG.xai.enable_explanation_lab else None,
        )

        _send(socket, "allocation", trade_result.allocation.to_dict())
        _send(socket, "benchmark", trade_result.benchmark_comparison.to_dict())
        _send(socket, "risk", trade_result.risk_snapshot.to_dict())
        _send(socket, "advisor_summary", trade_result.allocation.rationale_summary)
        _send(socket, "explain", explain_result.summary_text)
        _send(socket, "manifest", manifest)
        _send(socket, "engine_status", engine_result.engine_info.to_dict())

        if policy_check.breaches:
            _send(socket, "risk_status", policy_check.to_dict())
        if rule_summary.summary_text:
            _send(socket, "explain_status", rule_summary.to_dict())

    elapsed = time.time() - start
    _send(socket, "terminal", f"[SESSION] completed in {elapsed:.2f}s")
    socket.send(text_data=json.dumps({"type": "done_session", "session_id": mandate.session_id}, ensure_ascii=False))
