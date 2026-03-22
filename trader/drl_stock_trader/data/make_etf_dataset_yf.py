from __future__ import annotations

import os
import time
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf


DEFAULT_ETF30: List[str] = [
    "SPY", "QQQ", "IWM", "DIA",
    "EFA", "EEM", "VEA", "VWO",
    "VGK", "VNQ", "IYR", "XLF",
    "XLK", "XLE", "XLV", "XLY",
    "XLI", "XLP", "XLB",
    "TLT", "IEF", "SHY",
    "LQD", "HYG",
    "GLD", "SLV",
    "DBC", "USO", "UNG",
    "MDY",
]


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _macd(close: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    return _ema(close, fast) - _ema(close, slow)


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.replace([np.inf, -np.inf], np.nan).fillna(50.0)


def _cci(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    tp = (high + low + close) / 3.0
    sma = tp.rolling(window=window, min_periods=window).mean()
    mad = (tp - sma).abs().rolling(window=window, min_periods=window).mean()
    cci = (tp - sma) / (0.015 * mad.replace(0.0, np.nan))
    return cci.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr_n = tr.rolling(window=window, min_periods=window).sum()
    plus_dm_n = pd.Series(plus_dm, index=high.index).rolling(window=window, min_periods=window).sum()
    minus_dm_n = pd.Series(minus_dm, index=high.index).rolling(window=window, min_periods=window).sum()

    plus_di = 100.0 * (plus_dm_n / tr_n.replace(0.0, np.nan))
    minus_di = 100.0 * (minus_dm_n / tr_n.replace(0.0, np.nan))
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx = dx.rolling(window=window, min_periods=window).mean()
    return adx.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _download_yf(
    tickers: Iterable[str],
    start: str,
    end: str,
    max_retries: int = 3,
    pause_seconds: float = 1.5,
) -> pd.DataFrame:
    tickers = list(dict.fromkeys(str(t).upper() for t in tickers))
    tickers_str = " ".join(tickers)
    last_err: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            raw = yf.download(
                tickers=tickers_str,
                start=start,
                end=end,
                progress=False,
                group_by="ticker",
                auto_adjust=False,
                threads=False,
            )
            if raw is not None and not raw.empty:
                return raw
        except Exception as exc:
            last_err = exc
        time.sleep(pause_seconds * attempt)

    if last_err:
        raise RuntimeError(f"yfinance download failed after {max_retries} attempts: {last_err}") from last_err
    raise RuntimeError("yfinance download returned empty data.")


def _extract_ticker_frame(raw: pd.DataFrame, tic: str) -> Optional[pd.DataFrame]:
    if raw is None or raw.empty:
        return None

    if isinstance(raw.columns, pd.MultiIndex):
        if tic in raw.columns.get_level_values(-1):
            df_t = raw.xs(tic, axis=1, level=-1, drop_level=True).copy()
        elif tic in raw.columns.get_level_values(0):
            df_t = raw[tic].copy()
        else:
            return None
    else:
        df_t = raw.copy()

    if "Adj Close" not in df_t.columns and "Close" in df_t.columns:
        df_t["Adj Close"] = df_t["Close"]

    needed = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    if not set(needed).issubset(df_t.columns):
        return None

    out = df_t[needed].rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adjcp",
            "Volume": "volume",
        }
    )
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="first")]
    out = out.dropna(subset=["adjcp"])
    out["tic"] = tic
    return out


def _resolve_stable_universe(frames: Dict[str, pd.DataFrame], min_common_days: int = 260) -> Dict[str, pd.DataFrame]:
    survivors = {tic: frame for tic, frame in frames.items() if frame is not None and len(frame) >= min_common_days}
    if len(survivors) < 5:
        raise RuntimeError("Too few ETFs survived the data quality filter.")
    return survivors


def _compute_factor_like_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["ret_1"] = out["adjcp"].pct_change(1).fillna(0.0)
    out["ret_5"] = out["adjcp"].pct_change(5).fillna(0.0)
    out["ret_21"] = out["adjcp"].pct_change(21).fillna(0.0)
    out["momentum_21"] = (out["adjcp"] / out["adjcp"].shift(21) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["momentum_63"] = (out["adjcp"] / out["adjcp"].shift(63) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    returns = out["adjcp"].pct_change().fillna(0.0)
    out["realized_vol_20"] = returns.rolling(20, min_periods=2).std().fillna(0.0)
    out["downside_vol_20"] = returns.where(returns < 0.0, 0.0).rolling(20, min_periods=2).std().fillna(0.0)
    return out


def _build_benchmark_frame(price_wide: pd.DataFrame, external_spy: Optional[pd.Series] = None) -> pd.DataFrame:
    returns = price_wide.pct_change().fillna(0.0)
    equal_weight = returns.mean(axis=1)

    if external_spy is not None and not external_spy.empty:
        spy_series = external_spy.reindex(price_wide.index).ffill().pct_change().fillna(0.0)
    elif "SPY" in price_wide.columns:
        spy_series = price_wide["SPY"].pct_change().fillna(0.0)
    else:
        spy_series = equal_weight.copy()

    bond_candidates = [c for c in ["IEF", "TLT", "SHY"] if c in price_wide.columns]
    if bond_candidates:
        bond_series = price_wide[bond_candidates[0]].pct_change().fillna(0.0)
    else:
        bond_series = pd.Series(0.0, index=price_wide.index)

    sixty_forty = 0.60 * spy_series + 0.40 * bond_series
    return pd.DataFrame(
        {
            "benchmark_equal_weight_return": equal_weight,
            "benchmark_spy_return": spy_series,
            "benchmark_60_40_return": sixty_forty,
        }
    )


def _corr_proxy_by_date(returns_wide: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    proxy = pd.DataFrame(index=returns_wide.index, columns=returns_wide.columns, data=0.0)
    for i in range(window, len(returns_wide) + 1):
        hist = returns_wide.iloc[i - window:i]
        corr = hist.corr().fillna(0.0)
        mean_corr = corr.abs().mean(axis=1)
        proxy.loc[returns_wide.index[i - 1], mean_corr.index] = mean_corr.values
    return proxy.fillna(0.0)


def _build_covariance_cache(returns_wide: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    tickers = list(returns_wide.columns)
    rows: List[Dict[str, float]] = []
    for i in range(window, len(returns_wide) + 1):
        hist = returns_wide.iloc[i - window:i]
        cov = hist.cov().fillna(0.0)
        row: Dict[str, float] = {"datadate": int(returns_wide.index[i - 1])}
        for a in tickers:
            for b in tickers:
                row[f"cov__{a}__{b}"] = float(cov.loc[a, b])
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["datadate"])
    return pd.DataFrame(rows)


def build_etf_dataset(
    start: str,
    end: str,
    cache_csv: Optional[str] = None,
    tickers: Optional[List[str]] = None,
    min_common_days: int = 260,
    turbulence_window: int = 252,
    cache_covariance_csv: Optional[str] = None,
    max_retries: int = 3,
    pause_seconds: float = 1.5,
) -> pd.DataFrame:
    tickers = tickers or DEFAULT_ETF30

    if cache_csv and os.path.exists(cache_csv):
        return pd.read_csv(cache_csv, index_col=0)

    raw = _download_yf(
        tickers=tickers,
        start=start,
        end=end,
        max_retries=max_retries,
        pause_seconds=pause_seconds,
    )

    frames: Dict[str, pd.DataFrame] = {}
    for tic in tickers:
        frame = _extract_ticker_frame(raw, tic)
        if frame is not None:
            frames[tic] = frame

    frames = _resolve_stable_universe(frames, min_common_days=min_common_days)

    enriched_frames: List[pd.DataFrame] = []
    for tic, frame in frames.items():
        out = frame.copy()
        out["macd"] = _macd(out["adjcp"]).fillna(0.0)
        out["rsi"] = _rsi(out["adjcp"]).fillna(50.0)
        out["cci"] = _cci(out["high"], out["low"], out["close"]).fillna(0.0)
        out["adx"] = _adx(out["high"], out["low"], out["close"]).fillna(0.0)
        out = _compute_factor_like_columns(out)
        enriched_frames.append(out.reset_index().rename(columns={"Date": "date", "index": "date"}))

    df = pd.concat(enriched_frames, ignore_index=True)
    if "date" not in df.columns:
        date_col = df.columns[0]
        df = df.rename(columns={date_col: "date"})

    df["date"] = pd.to_datetime(df["date"])
    df["datadate"] = df["date"].dt.strftime("%Y%m%d").astype(int)
    df = df.sort_values(["datadate", "tic"], ignore_index=True)

    price_wide = df.pivot(index="datadate", columns="tic", values="adjcp").sort_index().ffill().dropna()
    returns_wide = price_wide.pct_change().fillna(0.0)
    corr_proxy = _corr_proxy_by_date(returns_wide, window=20)
    benchmark_frame = _build_benchmark_frame(price_wide)

    corr_long = (
        corr_proxy.stack()
        .rename("corr_proxy_20")
        .reset_index()
        .rename(columns={"level_1": "tic"})
    )
    df = df.merge(corr_long, on=["datadate", "tic"], how="left")
    df = df.merge(benchmark_frame.reset_index(), on="datadate", how="left")
    df["universe_name"] = "ETF30"
    df["universe_size"] = int(len(price_wide.columns))

    if cache_covariance_csv:
        cov_df = _build_covariance_cache(returns_wide, window=20)
        os.makedirs(os.path.dirname(cache_covariance_csv), exist_ok=True)
        cov_df.to_csv(cache_covariance_csv, index=False)

    df = df.fillna(0.0)
    df.index = df["datadate"].factorize()[0]

    if cache_csv:
        os.makedirs(os.path.dirname(cache_csv), exist_ok=True)
        df.to_csv(cache_csv)

    return df