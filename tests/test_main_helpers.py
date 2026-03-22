from __future__ import annotations


def test_extract_trade_window_accepts_multiple_formats():
    from trader.drl_stock_trader.main import _extract_trade_window

    assert _extract_trade_window('20160101-20190101') == ('20160101', '20190101')
    assert _extract_trade_window('2016-01-01 to 2019-01-01') == ('20160101', '20190101')


def test_normalize_etf30_input_parses_dates_and_robustness():
    from trader.drl_stock_trader.main import normalize_etf30_input

    initial_amount, robustness, train_start, trade_start, trade_end = normalize_etf30_input(
        initial_amount='1000000',
        robustness='robustness_2',
        train_start='2010-01-01',
        trade_start='2016-01-01',
        trade_end='2019-01-01',
    )

    assert initial_amount == 1_000_000.0
    assert robustness == 2
    assert train_start == '20100101'
    assert trade_start == '20160101'
    assert trade_end == '20190101'
