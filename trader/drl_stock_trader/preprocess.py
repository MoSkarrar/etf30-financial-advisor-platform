from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd


REQUIRED_BASE_COLUMNS = ["datadate", "tic", "adjcp"]


def data_split(df: pd.DataFrame, start_date, end_date) -> pd.DataFrame:
    data = df[(df["datadate"] >= start_date) & (df["datadate"] < end_date)].copy()
    data = data.sort_values(["datadate", "tic"], ignore_index=True)
    data.index = data["datadate"].factorize()[0]
    return data


def ensure_datetime_int(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if np.issubdtype(out["datadate"].dtype, np.datetime64):
        out["datadate"] = out["datadate"].dt.strftime("%Y%m%d").astype(int)
    else:
        out["datadate"] = out["datadate"].astype(int)
    return out


def clean_etf_frame(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_BASE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"ETF dataframe missing required columns: {missing}")

    out = ensure_datetime_int(df)
    out = out.sort_values(["datadate", "tic"], ignore_index=True)
    out = out.drop_duplicates(subset=["datadate", "tic"], keep="last")
    out["tic"] = out["tic"].astype(str)

    numeric_cols = [c for c in out.columns if c not in {"tic"}]
    for col in numeric_cols:
        if col == "datadate":
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["adjcp"])
    out = out.fillna(0.0)
    out.index = out["datadate"].factorize()[0]
    return out


def infer_feature_columns(df: pd.DataFrame, exclude: Iterable[str] | None = None) -> List[str]:
    exclude = set(exclude or [])
    reserved = {
        "datadate",
        "tic",
        "adjcp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "benchmark_equal_weight_return",
        "benchmark_spy_return",
        "benchmark_60_40_return",
    }
    reserved |= exclude
    feature_cols = [c for c in df.columns if c not in reserved]
    return sorted(feature_cols)


def pivot_prices(df: pd.DataFrame, price_col: str = "adjcp") -> pd.DataFrame:
    wide = df.pivot(index="datadate", columns="tic", values=price_col).sort_index()
    wide = wide.ffill().dropna(how="all")
    return wide


def pivot_features(df: pd.DataFrame, feature_columns: Sequence[str]) -> dict[str, pd.DataFrame]:
    panels: dict[str, pd.DataFrame] = {}
    for col in feature_columns:
        if col not in df.columns:
            continue
        panels[col] = df.pivot(index="datadate", columns="tic", values=col).sort_index().ffill().fillna(0.0)
    return panels