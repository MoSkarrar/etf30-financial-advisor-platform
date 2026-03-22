from __future__ import annotations

import pytest


risk_metrics = pytest.importorskip('trader.drl_stock_trader.risk.risk_metrics')
policy_checks = pytest.importorskip('trader.drl_stock_trader.risk.policy_checks')
risk_overlay = pytest.importorskip('trader.drl_stock_trader.risk.risk_overlay')


def test_risk_metrics_compute_expected_shapes():
    vol = risk_metrics.compute_volatility([0.01, -0.02, 0.03])
    dd = risk_metrics.compute_max_drawdown([100, 110, 90, 95])
    turnover = risk_metrics.compute_turnover({'AAA': 0.5, 'CASH': 0.5}, {'AAA': 0.7, 'CASH': 0.3})
    assert vol > 0
    assert dd > 0
    assert turnover > 0


def test_policy_check_result_flags_breaches():
    from trader.domain.session_models import PortfolioPolicy

    result = policy_checks.build_policy_check_result(
        weights={'AAA': 0.25, 'CASH': 0.01},
        turnover=0.60,
        concentration=0.20,
        policy=PortfolioPolicy(max_single_position_cap=0.12, min_cash_weight=0.05, turnover_budget=0.35),
    )
    assert result['passed'] is False
    assert result['severity'] in {'medium', 'high'}
    assert result['breaches']


def test_risk_overlay_caps_positions_and_respects_cash_floor():
    from trader.domain.session_models import PortfolioPolicy

    report = risk_overlay.apply_risk_overlay(
        {'AAA': 0.80, 'BBB': 0.15, 'CASH': 0.05},
        PortfolioPolicy(max_single_position_cap=0.50, min_cash_weight=0.10, turnover_budget=1.0),
        market_context={'previous_weights': {'AAA': 0.6, 'BBB': 0.3, 'CASH': 0.1}},
    )
    adjusted = report['adjusted_weights']
    assert adjusted['CASH'] >= 0.10 - 1e-9
    assert adjusted['AAA'] <= 0.50 + 1e-9
    assert report['applied_clips']
