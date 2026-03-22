from __future__ import annotations

import pytest


def test_build_advisory_mandate_normalizes_core_fields():
    from trader.services.trading_service import build_advisory_mandate

    mandate = build_advisory_mandate(
        {
            'market': 'etf30',
            'initial_amount': '1000000',
            'robustness': 'robustness_3',
            'date_train': '2010-01-01',
            'date_trade_1': '2016-01-01',
            'date_trade_2': '2019-01-01',
            'benchmark_choice': 'equal_weight',
        }
    )

    assert mandate.initial_amount == 1_000_000.0
    assert mandate.robustness == 3
    assert mandate.train_start == '20100101'
    assert mandate.trade_start == '20160101'
    assert mandate.trade_end == '20190101'
    assert mandate.benchmark_choice == 'equal_weight'


def test_build_advisory_mandate_rejects_non_etf30_market():
    from trader.services.trading_service import TradingServiceError, build_advisory_mandate

    with pytest.raises(TradingServiceError):
        build_advisory_mandate(
            {
                'market': 'dow30',
                'initial_amount': '100000',
                'date_train': '20100101',
                'date_trade_1': '20160101',
                'date_trade_2': '20190101',
            }
        )


def test_execute_trade_delegates_to_models(monkeypatch):
    from trader.services import trading_service

    captured = {}

    def fake_runner(*, socket, mandate):
        captured['socket'] = socket
        captured['mandate'] = mandate

    monkeypatch.setattr(trading_service, 'run_portfolio_advisory_session', fake_runner)
    socket = object()
    mandate = trading_service.execute_trade(
        socket,
        {
            'market': 'etf30',
            'initial_amount': '1000000',
            'date_train': '20100101',
            'date_trade_1': '20160101',
            'date_trade_2': '20190101',
        },
    )

    assert captured['socket'] is socket
    assert captured['mandate'].session_id == mandate.session_id
