from __future__ import annotations

from typing import Any, Dict, List


def _as_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "to_dict"):
        try:
            return obj.to_dict()
        except Exception:
            pass
    return dict(obj)


def build_rule_summary(
    *,
    allocation_recommendation: Any,
    benchmark_comparison: Any,
    risk_snapshot: Any,
    portfolio_policy: Any = None,
    policy_check: Any = None,
) -> Dict[str, Any]:
    allocation = _as_dict(allocation_recommendation)
    benchmark = _as_dict(benchmark_comparison)
    risk = _as_dict(risk_snapshot)
    policy = _as_dict(portfolio_policy)
    checks = _as_dict(policy_check)

    target_weights = allocation.get("target_weights") or {}
    rules_triggered: List[str] = []
    risk_flags: List[str] = []
    allocation_flags: List[str] = []
    benchmark_flags: List[str] = []
    evidence: Dict[str, Any] = {}

    cash_weight = float(target_weights.get("CASH", 0.0) or 0.0)
    min_cash = float(policy.get("min_cash_weight", policy.get("target_cash_floor", 0.05)) or 0.05)
    if cash_weight >= min_cash + 0.02:
        rules_triggered.append("cash_buffer_elevated")
        allocation_flags.append(f"Cash allocation is elevated at {cash_weight:.2%}, which suggests a more defensive posture.")
    elif cash_weight < min_cash:
        rules_triggered.append("cash_floor_breach")
        allocation_flags.append(f"Cash allocation at {cash_weight:.2%} is below the policy floor of {min_cash:.2%}.")

    turnover = float(risk.get("turnover", 0.0) or 0.0)
    turnover_budget = float(policy.get("turnover_budget", 0.35) or 0.35)
    if turnover > turnover_budget:
        rules_triggered.append("turnover_pressure")
        risk_flags.append(f"Turnover at {turnover:.2%} exceeds the configured budget of {turnover_budget:.2%}.")

    concentration = float(risk.get("concentration_hhi", 0.0) or 0.0)
    if concentration >= 0.10:
        rules_triggered.append("concentration_elevated")
        risk_flags.append(f"Concentration appears elevated with HHI near {concentration:.3f}.")

    downside_vol = float(risk.get("downside_volatility", 0.0) or 0.0)
    realized_vol = float(risk.get("realized_volatility", 0.0) or 0.0)
    if downside_vol > 0 and downside_vol >= realized_vol * 0.9:
        rules_triggered.append("downside_pressure")
        risk_flags.append("Downside volatility is close to total volatility, which suggests losses are not well-contained.")

    active_return = float(benchmark.get("active_return", 0.0) or 0.0)
    tracking_error = float(benchmark.get("tracking_error", 0.0) or 0.0)
    if active_return >= 0:
        rules_triggered.append("benchmark_outperformance")
        benchmark_flags.append("The current allocation is outperforming its benchmark on an active-return basis.")
    else:
        rules_triggered.append("benchmark_underperformance")
        benchmark_flags.append("The current allocation is lagging its benchmark on an active-return basis.")
    if tracking_error >= 0.05:
        benchmark_flags.append(f"Tracking error is material at about {tracking_error:.2%}, so the portfolio is meaningfully diverging from the benchmark.")

    for breach in checks.get("breaches", []) or []:
        rules_triggered.append("policy_breach_detected")
        allocation_flags.append(str(breach))

    evidence.update(
        {
            "cash_weight": cash_weight,
            "min_cash_weight": min_cash,
            "turnover": turnover,
            "turnover_budget": turnover_budget,
            "concentration_hhi": concentration,
            "active_return": active_return,
            "tracking_error": tracking_error,
        }
    )

    summary_parts = risk_flags[:2] + benchmark_flags[:2] + allocation_flags[:2]
    summary_text = " ".join(summary_parts).strip() or "Rule summary found no material issues beyond the current allocation and risk snapshot."

    return {
        "rules_triggered": list(dict.fromkeys(rules_triggered)),
        "risk_flags": risk_flags,
        "allocation_flags": allocation_flags,
        "benchmark_flags": benchmark_flags,
        "summary_text": summary_text,
        "evidence": evidence,
    }
