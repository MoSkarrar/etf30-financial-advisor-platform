from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv

from trader.domain.session_models import InvestorProfile, PortfolioPolicy
from trader.drl_stock_trader.RL_envs.EnvMultipleStock_Validation import StockEnvValidation
from trader.drl_stock_trader.RL_envs.EnvMultipleStocks_Train import StockEnvTrain
from trader.drl_stock_trader.config import paths
from trader.drl_stock_trader.config.app_config import APP_CONFIG
from trader.drl_stock_trader.data.make_etf_dataset_yf import DEFAULT_ETF30, build_etf_dataset
from trader.drl_stock_trader.preprocess import clean_etf_frame, data_split, infer_feature_columns, pivot_features, pivot_prices

logger = logging.getLogger(__name__)


@dataclass
class PortfolioPreparedData:
    long_frame: pd.DataFrame
    prices: pd.DataFrame
    feature_panels: Dict[str, pd.DataFrame]
    covariance_by_date: Dict[int, np.ndarray]
    benchmarks: pd.DataFrame
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class IterationBundle:
    iteration: int
    first_run: bool
    train_data: pd.DataFrame
    validation_data: pd.DataFrame
    train_environment: DummyVecEnv
    validation_environment: DummyVecEnv
    validation_observation: object
    validation_start: int
    validation_end: int
    trade_start: int
    trade_end: int
    covariance_train: Dict[int, np.ndarray]
    covariance_validation: Dict[int, np.ndarray]
    feature_columns: List[str]


@dataclass
class ExecutionDataset:
    prepared: PortfolioPreparedData
    unique_trade_date: List[int]


def load_etf30_dataset(force_rebuild: bool = False) -> pd.DataFrame:
    dataset_path = paths.etf30_dataset_path()
    covariance_path = paths.covariance_cache_path()

    if pd.io.common.file_exists(dataset_path) and not (force_rebuild or getattr(APP_CONFIG.dataset, "force_rebuild", False)):
        return pd.read_csv(dataset_path, index_col=0)

    paths.ensure_parent_dir(dataset_path)
    return build_etf_dataset(
        start=APP_CONFIG.dataset.build_start,
        end=getattr(APP_CONFIG.dataset, "build_end", "2025-12-31"),
        cache_csv=dataset_path,
        tickers=list(DEFAULT_ETF30),
        min_common_days=getattr(APP_CONFIG.dataset, "min_common_days", 260),
        turbulence_window=getattr(APP_CONFIG.dataset, "turbulence_window", 252),
        cache_covariance_csv=covariance_path,
    )


def rolling_covariance_by_date(returns_wide: pd.DataFrame, window: int = 20) -> Dict[int, np.ndarray]:
    output: Dict[int, np.ndarray] = {}
    for i, date in enumerate(returns_wide.index):
        hist = returns_wide.iloc[max(0, i - window + 1): i + 1]
        if len(hist) < 2:
            output[int(date)] = np.zeros((returns_wide.shape[1], returns_wide.shape[1]), dtype=np.float32)
        else:
            output[int(date)] = hist.cov().fillna(0.0).to_numpy(dtype=np.float32)
    return output


def _build_equal_weight_return(returns: pd.DataFrame) -> pd.Series:
    return returns.mean(axis=1).fillna(0.0)


def _build_spy_return(prices: pd.DataFrame, fallback: pd.Series) -> pd.Series:
    if "SPY" in prices.columns:
        return prices["SPY"].pct_change().fillna(0.0)
    return fallback.copy()


def _build_60_40_return(prices: pd.DataFrame, equity_return: pd.Series) -> pd.Series:
    bond_col = next((c for c in ["IEF", "TLT", "SHY"] if c in prices.columns), None)
    bond_return = prices[bond_col].pct_change().fillna(0.0) if bond_col else pd.Series(0.0, index=prices.index)
    return (0.60 * equity_return + 0.40 * bond_return).fillna(0.0)


def _build_cap_weight_proxy_return(prices: pd.DataFrame, volume_panel: pd.DataFrame) -> pd.Series:
    returns = prices.pct_change().fillna(0.0)
    dollar_volume = (prices * volume_panel).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    lagged = dollar_volume.shift(1).fillna(0.0)
    weights = lagged.div(lagged.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    return (weights * returns).sum(axis=1).fillna(0.0)


def benchmark_frame_from_long(df: pd.DataFrame) -> pd.DataFrame:
    prices = pivot_prices(df)
    returns = prices.pct_change().fillna(0.0)
    if "volume" in df.columns:
        volume_panel = (
            df.pivot(index="datadate", columns="tic", values="volume")
            .sort_index()
            .reindex(prices.index)
            .ffill()
            .fillna(0.0)
        )
    else:
        volume_panel = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    equal_weight = _build_equal_weight_return(returns)
    spy = _build_spy_return(prices, equal_weight)
    sixty_forty = _build_60_40_return(prices, spy)
    cap_proxy = _build_cap_weight_proxy_return(prices, volume_panel)

    bench = pd.DataFrame(
        {
            "benchmark_equal_weight_return": equal_weight,
            "benchmark_spy_return": spy,
            "benchmark_60_40_return": sixty_forty,
            "benchmark_cap_weight_proxy_return": cap_proxy,
        }
    )
    bench.index.name = "datadate"
    return bench


def attach_benchmark_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    long_frame = df.copy()
    bench = benchmark_frame_from_long(long_frame)
    long_frame = long_frame.drop(columns=[c for c in bench.columns if c in long_frame.columns], errors="ignore")
    long_frame = long_frame.merge(bench.reset_index(), on="datadate", how="left")
    long_frame = long_frame.sort_values(["datadate", "tic"], ignore_index=True)
    return long_frame, bench


def prepare_portfolio_dataset(df: Optional[pd.DataFrame] = None) -> PortfolioPreparedData:
    df = clean_etf_frame(df if df is not None else load_etf30_dataset())
    df, benchmarks = attach_benchmark_columns(df)
    feature_columns = infer_feature_columns(df)
    prices = pivot_prices(df)
    feature_panels = pivot_features(df, feature_columns)
    returns = prices.pct_change().fillna(0.0)
    covariance = rolling_covariance_by_date(returns, window=getattr(APP_CONFIG.rl, "covariance_window", 20))

    metadata = {
        "tickers": prices.columns.tolist(),
        "feature_columns": feature_columns,
        "stock_dim": int(prices.shape[1]),
        "date_count": int(prices.shape[0]),
        "benchmark_options": ["equal_weight", "spy", "60_40", "cap_weight_proxy"],
        "benchmark_labels": {
            "equal_weight": "Equal-weight ETF30",
            "spy": "SPY proxy",
            "60_40": "60/40 proxy",
            "cap_weight_proxy": "Liquidity-weighted cap proxy",
        },
        "risk_parity_baseline": "placeholder_for_future_stage",
    }
    return PortfolioPreparedData(
        long_frame=df,
        prices=prices,
        feature_panels=feature_panels,
        covariance_by_date=covariance,
        benchmarks=benchmarks,
        metadata=metadata,
    )


def prepare_execution_dataset(train_start: int, trade_start: int, trade_end: int) -> ExecutionDataset:
    prepared = prepare_portfolio_dataset()
    unique_trade_date = prepared.long_frame[
        (prepared.long_frame.datadate > trade_start) & (prepared.long_frame.datadate <= trade_end)
    ].datadate.unique().tolist()
    unique_trade_date = [int(v) for v in unique_trade_date]
    return ExecutionDataset(prepared=prepared, unique_trade_date=unique_trade_date)


def _subset_covariance(covariance_by_date: Dict[int, np.ndarray], dates: Sequence[int]) -> Dict[int, np.ndarray]:
    return {int(d): covariance_by_date[int(d)] for d in dates if int(d) in covariance_by_date}


def _window_stats(df: pd.DataFrame) -> Dict[str, object]:
    if df.empty:
        return {
            "rows": 0,
            "unique_dates": 0,
            "first_date": None,
            "last_date": None,
            "min_tickers_per_date": 0,
            "max_tickers_per_date": 0,
        }

    counts = df.groupby("datadate")["tic"].nunique()
    return {
        "rows": int(len(df)),
        "unique_dates": int(df["datadate"].nunique()),
        "first_date": int(df["datadate"].min()),
        "last_date": int(df["datadate"].max()),
        "min_tickers_per_date": int(counts.min()),
        "max_tickers_per_date": int(counts.max()),
    }


def _log_window(name: str, stats: Dict[str, object]) -> None:
    logger.info(
        "[DATA][%s] rows=%s unique_dates=%s first_date=%s last_date=%s "
        "min_tickers_per_date=%s max_tickers_per_date=%s",
        name,
        stats["rows"],
        stats["unique_dates"],
        stats["first_date"],
        stats["last_date"],
        stats["min_tickers_per_date"],
        stats["max_tickers_per_date"],
    )


def prepare_iteration_bundle(
    prepared: PortfolioPreparedData,
    unique_trade_date: Sequence[int],
    train_start: int,
    iteration: int,
    investor_profile: Optional[InvestorProfile] = None,
    policy: Optional[PortfolioPolicy] = None,
    benchmark_name: str = "equal_weight",
    rebalance_window: Optional[int] = None,
    validation_window: Optional[int] = None,
) -> IterationBundle:
    rebalance_window = int(rebalance_window or APP_CONFIG.rl.rebalance_window)
    validation_window = int(validation_window or APP_CONFIG.rl.validation_window)

    if iteration >= len(unique_trade_date):
        raise IndexError(
            f"Iteration {iteration} out of bounds for unique_trade_date with length {len(unique_trade_date)}."
        )

    if iteration - rebalance_window - validation_window < 0:
        raise ValueError(
            "Not enough trade dates available for the selected rebalance and validation windows."
        )

    first_run = iteration - rebalance_window - validation_window == 0
    feature_columns = list(prepared.metadata.get("feature_columns") or [])

    train_cutoff = unique_trade_date[iteration - rebalance_window - validation_window]
    validation_start = unique_trade_date[iteration - rebalance_window - validation_window]
    validation_end = unique_trade_date[iteration - rebalance_window]
    trade_start = unique_trade_date[iteration - rebalance_window]
    trade_end = unique_trade_date[iteration]

    earliest_available_date = int(prepared.long_frame["datadate"].min())
    effective_train_start = int(train_start)
    if effective_train_start > int(train_cutoff):
        effective_train_start = earliest_available_date
    else:
        effective_train_start = max(earliest_available_date, effective_train_start)

    train_data = data_split(prepared.long_frame, start_date=effective_train_start, end_date=train_cutoff)
    validation_data = data_split(prepared.long_frame, start_date=validation_start, end_date=validation_end)

    train_stats = _window_stats(train_data)
    validation_stats = _window_stats(validation_data)
    _log_window("train", train_stats)
    _log_window("validation", validation_stats)

    if train_data.empty:
        raise RuntimeError(
            f"Train window is empty. requested_train_start={train_start}, "
            f"effective_train_start={effective_train_start}, "
            f"train_cutoff={train_cutoff}, iteration={iteration}"
        )

    if validation_data.empty:
        raise RuntimeError(
            f"Validation window is empty. validation_start={validation_start}, validation_end={validation_end}, iteration={iteration}"
        )

    train_dates = sorted(int(v) for v in train_data["datadate"].unique().tolist())
    validation_dates = sorted(int(v) for v in validation_data["datadate"].unique().tolist())

    if len(validation_dates) < 2:
        raise RuntimeError(
            f"Validation window too small: only {len(validation_dates)} unique date(s) "
            f"between {validation_start} and {validation_end}."
        )

    covariance_train = _subset_covariance(prepared.covariance_by_date, train_dates)
    covariance_validation = _subset_covariance(prepared.covariance_by_date, validation_dates)

    logger.info(
        "[DATA][covariance] train_dates=%s validation_dates=%s covariance_train=%s covariance_validation=%s",
        len(train_dates),
        len(validation_dates),
        len(covariance_train),
        len(covariance_validation),
    )

    if not covariance_train:
        raise RuntimeError("Train covariance window is empty after subsetting.")
    if not covariance_validation:
        raise RuntimeError("Validation covariance window is empty after subsetting.")

    train_environment = DummyVecEnv(
        [
            lambda: StockEnvTrain(
                df=train_data,
                covariance_by_date=covariance_train,
                feature_columns=feature_columns,
                benchmark_name=benchmark_name,
                investor_profile=investor_profile,
                policy=policy,
            )
        ]
    )
    validation_environment = DummyVecEnv(
        [
            lambda: StockEnvValidation(
                df=validation_data,
                covariance_by_date=covariance_validation,
                feature_columns=feature_columns,
                benchmark_name=benchmark_name,
                investor_profile=investor_profile,
                policy=policy,
                iteration=iteration,
            )
        ]
    )
    validation_observation = validation_environment.reset()

    return IterationBundle(
        iteration=iteration,
        first_run=first_run,
        train_data=train_data,
        validation_data=validation_data,
        train_environment=train_environment,
        validation_environment=validation_environment,
        validation_observation=validation_observation,
        validation_start=int(validation_start),
        validation_end=int(validation_end),
        trade_start=int(trade_start),
        trade_end=int(trade_end),
        covariance_train=covariance_train,
        covariance_validation=covariance_validation,
        feature_columns=feature_columns,
    )