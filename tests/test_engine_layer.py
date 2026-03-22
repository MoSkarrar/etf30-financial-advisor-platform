from __future__ import annotations

from types import SimpleNamespace

import pytest


engine_registry = pytest.importorskip('trader.drl_stock_trader.engines.engine_registry')
legacy_module = pytest.importorskip('trader.drl_stock_trader.engines.legacy_rl_engine')
finrl_module = pytest.importorskip('trader.drl_stock_trader.engines.finrl_engine')


def test_get_engine_falls_back_to_legacy_when_finrl_disabled(monkeypatch):
    cfg = SimpleNamespace(default_engine='legacy_rl', enable_finrl=False, enable_engine_fallback=True)
    monkeypatch.setattr(engine_registry, 'APP_CONFIG', SimpleNamespace(engine=cfg))
    engine = engine_registry.get_engine('finrl')
    assert getattr(engine, 'name', '') == 'legacy_rl'


def test_run_engine_calls_train_validate_trade(monkeypatch):
    sentinel = object()

    class FakeEngine:
        def train_validate_trade(self, **kwargs):
            return sentinel

    monkeypatch.setattr(engine_registry, 'get_engine', lambda name: FakeEngine())
    assert engine_registry.run_engine('legacy_rl', bundle='x') is sentinel


def test_finrl_engine_delegates_and_relabels(monkeypatch):
    engine = finrl_module.FinRLEngine()
    fake_result = SimpleNamespace(
        engine_name='legacy_rl',
        selected_model='PPO',
        engine_info=None,
        backend_notes='',
    )
    monkeypatch.setattr(finrl_module.FinRLEngine, '_can_run_native_finrl', lambda self: False)
    monkeypatch.setattr(
        finrl_module.LegacyEngine,
        'train_validate_trade',
        lambda self, **kwargs: fake_result,
    )

    mandate = SimpleNamespace(robustness=3, rebalance_cadence_days=21, benchmark_choice='equal_weight', risk_mode='standard', explanation_depth='standard')
    result = engine.train_validate_trade(socket=None, mandate=mandate, market='etf30', prepared=SimpleNamespace(), bundle=SimpleNamespace(), iteration=1)
    assert result.engine_name == 'finrl'
    assert result.engine_info.engine_name == 'finrl'
