from __future__ import annotations


def test_investor_profile_roundtrip():
    from trader.domain import session_models as sm

    profile = sm.InvestorProfile.from_dict(
        {
            'profile_name': 'balanced',
            'target_style': 'balanced',
            'risk_tolerance': 0.6,
            'target_volatility': 0.15,
        }
    )
    payload = profile.to_dict()
    assert payload['profile_name'] == 'balanced'
    assert abs(payload['risk_tolerance'] - 0.6) < 1e-9


def test_portfolio_policy_defaults_and_to_dict():
    from trader.domain import session_models as sm

    policy = sm.PortfolioPolicy.from_dict({'max_single_position_cap': 0.1, 'min_cash_weight': 0.07})
    payload = policy.to_dict()
    assert payload['max_single_position_cap'] == 0.1
    assert payload['min_cash_weight'] == 0.07
    assert 'rebalance_cadence_days' in payload


def test_run_manifest_record_deserializes_nested_blocks():
    from trader.domain import session_models as sm

    manifest = sm.RunManifestRecord.from_dict(
        {
            'session_id': 's1',
            'run_id': 'r1',
            'trade_window': {'start': '20200101', 'end': '20200201'},
            'allocation_recommendation': {'target_weights': {'AAA': 0.6, 'CASH': 0.4}},
            'benchmark_comparison': {'benchmark_name': 'equal_weight'},
            'risk_snapshot': {'cash_weight': 0.4},
            'artifacts': {'xai_text': 'demo.txt'},
        }
    )

    payload = manifest.to_dict()
    assert payload['session_id'] == 's1'
    assert payload['trade_window']['start'] == '20200101'
    assert payload['allocation_recommendation']['target_weights']['AAA'] == 0.6
    assert payload['artifacts']['xai_text'] == 'demo.txt'
