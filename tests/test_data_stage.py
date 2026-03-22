from __future__ import annotations

import numpy as np
import pandas as pd


def _sample_long_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'datadate': [20200101, 20200101, 20200102, 20200102, 20200103, 20200103],
            'tic': ['AAA', 'BBB', 'AAA', 'BBB', 'AAA', 'BBB'],
            'adjcp': [100.0, 200.0, 101.0, 202.0, 103.0, 201.0],
            'volume': [10.0, 20.0, 10.0, 20.0, 11.0, 19.0],
            'macd': [0.1, 0.2, 0.15, 0.25, 0.2, 0.1],
        }
    )


def test_benchmark_frame_from_long_produces_expected_columns():
    from trader.drl_stock_trader.pipeline.data_stage import benchmark_frame_from_long

    bench = benchmark_frame_from_long(_sample_long_frame())
    assert 'benchmark_equal_weight_return' in bench.columns
    assert 'benchmark_spy_return' in bench.columns
    assert 'benchmark_60_40_return' in bench.columns
    assert len(bench.index) == 3


def test_attach_benchmark_columns_merges_without_dropping_rows():
    from trader.drl_stock_trader.pipeline.data_stage import attach_benchmark_columns

    long_frame, bench = attach_benchmark_columns(_sample_long_frame())
    assert len(long_frame) == 6
    assert len(bench) == 3
    assert 'benchmark_equal_weight_return' in long_frame.columns


def test_rolling_covariance_by_date_returns_square_matrices():
    from trader.drl_stock_trader.pipeline.data_stage import rolling_covariance_by_date

    returns_wide = pd.DataFrame(
        {'AAA': [0.0, 0.01, -0.02], 'BBB': [0.0, 0.02, -0.01]},
        index=[20200101, 20200102, 20200103],
    )
    cov = rolling_covariance_by_date(returns_wide, window=2)
    assert set(cov.keys()) == {20200101, 20200102, 20200103}
    assert all(isinstance(v, np.ndarray) and v.shape == (2, 2) for v in cov.values())
