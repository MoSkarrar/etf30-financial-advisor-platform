from __future__ import annotations

from types import SimpleNamespace


def test_resolve_algorithm_order_filters_invalid_names():
    from trader.drl_stock_trader.pipeline.train_stage import resolve_algorithm_order

    out = resolve_algorithm_order(['ppo', 'bad', 'a2c'])
    assert out == ['PPO', 'A2C']


def test_train_candidate_models_uses_algorithms_module(monkeypatch):
    from trader.drl_stock_trader.pipeline import train_stage
    from trader.drl_stock_trader import algorithms
    from trader.drl_stock_trader.config.app_config import TimestepsProfile

    calls = {'train': [], 'eval': []}

    def fake_train_algorithm(**kwargs):
        calls['train'].append(kwargs['algorithm_name'])
        return SimpleNamespace(name=kwargs['algorithm_name'])

    def fake_evaluate_model(**kwargs):
        calls['eval'].append(kwargs['label'])
        return algorithms.ValidationSummary(
            label=kwargs['label'],
            sharpe=1.0,
            annualized_return=0.1,
            max_drawdown=0.05,
            turnover=0.1,
            concentration=0.1,
            benchmark_return=0.02,
            active_return=0.08,
            information_ratio=1.1,
            stability_score=0.9,
            policy_compliance_score=1.0,
            validation_csv_path='validation.csv',
        )

    monkeypatch.setattr(train_stage.algorithms, 'train_algorithm', fake_train_algorithm)
    monkeypatch.setattr(train_stage.algorithms, 'evaluate_model', fake_evaluate_model)
    monkeypatch.setattr(train_stage.algorithms, '_socket_log', lambda *a, **k: None)

    env = SimpleNamespace(envs=[SimpleNamespace(investor_profile=None, policy=None)])
    results = train_stage.train_candidate_models(
        socket=None,
        train_environment=env,
        validation_environment=env,
        iteration=1,
        market='etf30',
        timesteps=TimestepsProfile(a2c=1, ppo=1, ddpg=1),
        algorithm_choices=['A2C', 'PPO'],
    )

    assert [r.label for r in results] == ['A2C', 'PPO']
    assert calls['train'] == ['A2C', 'PPO']
    assert calls['eval'] == ['A2C', 'PPO']


def test_select_best_candidate_prefers_higher_combined_score():
    from trader.drl_stock_trader.pipeline.selection_stage import select_best_candidate

    def candidate(label, sharpe, active, dd, turnover, concentration, stability=0.9, compliance=1.0):
        summary = SimpleNamespace(
            sharpe=sharpe,
            active_return=active,
            max_drawdown=dd,
            turnover=turnover,
            concentration=concentration,
            stability_score=stability,
            policy_compliance_score=compliance,
        )
        return SimpleNamespace(label=label, validation_summary=summary)

    result = select_best_candidate(
        [
            candidate('A2C', 0.6, 0.05, 0.12, 0.20, 0.14),
            candidate('PPO', 0.7, 0.06, 0.08, 0.10, 0.10),
        ]
    )

    assert result.best_candidate.label == 'PPO'
    assert [c.label for c in result.ranked_candidates] == ['PPO', 'A2C']
