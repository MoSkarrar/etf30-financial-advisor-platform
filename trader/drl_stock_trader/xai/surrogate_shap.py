from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor

from trader.drl_stock_trader.xai.shap_service import run_shap_explanation

DEFAULT_EXCLUDE = {
    "datadate",
    "date",
    "tic",
    "universe_name",
    "universe_size",
    "benchmark_equal_weight_return",
    "benchmark_spy_return",
    "benchmark_60_40_return",
}


@dataclass
class SurrogateArtifacts:
    model: Any
    X: pd.DataFrame
    y: np.ndarray
    feature_columns: List[str]

    def predict_fn(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return np.asarray(self.model.predict(arr), dtype=float).reshape(-1)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _as_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "to_dict"):
        try:
            return obj.to_dict()
        except Exception:
            pass
    return dict(obj)


def infer_feature_columns(frame: pd.DataFrame, feature_columns: Optional[Sequence[str]] = None) -> List[str]:
    if feature_columns:
        return [c for c in feature_columns if c in frame.columns]
    cols = []
    for col in frame.columns:
        if col in DEFAULT_EXCLUDE:
            continue
        if pd.api.types.is_numeric_dtype(frame[col]):
            cols.append(col)
    return cols


def train_surrogate_for_portfolio(
    current_frame: pd.DataFrame,
    allocation_recommendation: Any,
    feature_columns: Optional[Sequence[str]] = None,
) -> SurrogateArtifacts:
    frame = current_frame.copy().sort_values("tic").reset_index(drop=True)
    if frame.empty:
        raise ValueError("current_frame is empty.")

    allocation = _as_dict(allocation_recommendation)
    target_weights = allocation.get("target_weights") or {}

    feature_cols = infer_feature_columns(frame, feature_columns)
    if not feature_cols:
        raise ValueError("No numeric feature columns available for surrogate training.")

    X = frame[feature_cols].astype(float).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    y = frame["tic"].map(lambda t: _safe_float(target_weights.get(t, 0.0))).to_numpy(dtype=float)

    if len(np.unique(y)) <= 1 or len(X.index) < 3:
        model: Any = DummyRegressor(strategy="constant", constant=float(y[0]) if len(y) else 0.0)
    else:
        model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X, y)

    return SurrogateArtifacts(model=model, X=X, y=y, feature_columns=feature_cols)


def explain_portfolio_decision(
    current_frame: pd.DataFrame,
    allocation_recommendation: Any,
    previous_weights: Optional[Dict[str, float]] = None,
    benchmark_name: str = "equal_weight",
    policy: Optional[Any] = None,
    risk_snapshot: Optional[Any] = None,
    feature_columns: Optional[Sequence[str]] = None,
    top_k_assets: int = 5,
    top_k_features: int = 6,
) -> Dict[str, Any]:
    surrogate = train_surrogate_for_portfolio(
        current_frame=current_frame,
        allocation_recommendation=allocation_recommendation,
        feature_columns=feature_columns,
    )
    sample = surrogate.X.iloc[[int(np.argmax(np.abs(surrogate.y))) if len(surrogate.y) else 0]]
    shap_payload = run_shap_explanation(
        model_or_surrogate=surrogate.model,
        background=surrogate.X,
        sample=sample,
        feature_names=surrogate.feature_columns,
        top_k=max(top_k_features, 1),
    )
    allocation = _as_dict(allocation_recommendation)
    target_weights = allocation.get("target_weights") or {}
    top_assets = sorted(
        [(str(k), float(v)) for k, v in target_weights.items() if str(k) != "CASH"],
        key=lambda kv: abs(kv[1]),
        reverse=True,
    )[:max(top_k_assets, 1)]
    risk_dict = _as_dict(risk_snapshot)
    explanation = {
        "risk_posture": {
            "mode": "defensive" if float(target_weights.get("CASH", 0.0) or 0.0) >= 0.10 else "balanced",
            "message": "Cash and concentration suggest a more defensive posture." if float(target_weights.get("CASH", 0.0) or 0.0) >= 0.10 else "The portfolio remains broadly invested without an extreme defensive tilt.",
        },
        "benchmark_relative_explanation": {
            "benchmark_name": benchmark_name,
            "message": "Benchmark-relative context should be read together with the allocation and risk snapshot.",
        },
        "policy_compliance_explanation": {
            "message": "Policy compliance should be confirmed against the persisted policy checks and risk overlay outputs.",
        },
        "concentration_risk_explanation": {
            "message": "Concentration appears elevated." if float(risk_dict.get("concentration_hhi", 0.0) or 0.0) >= 0.10 else "Concentration appears moderate.",
        },
        "correlation_risk_explanation": {
            "message": "Correlation risk should be interpreted from the current cross-asset feature structure rather than a single attribution method.",
        },
        "main_drivers_of_change": [
            {"feature": str(feature), "importance": float((shap_payload.get("attribution_values") or {}).get(feature, 0.0))}
            for feature in (shap_payload.get("top_features") or [])[:max(top_k_features, 1)]
        ],
        "top_weight_explanations": [
            {"tic": tic, "target_weight": weight}
            for tic, weight in top_assets
        ],
    }
    return {
        "as_of_date": str(current_frame["datadate"].max()) if "datadate" in current_frame.columns else "",
        "asset_count": int(len(current_frame.index)),
        "top_assets_explained": int(len(top_assets)),
        "explanation": explanation,
        "note": "Legacy-compatible surrogate explanation payload.",
    }


def to_user_friendly_text(explain_dict: Dict[str, Any]) -> str:
    ex = explain_dict.get("explanation", explain_dict)
    lines: List[str] = ["Portfolio advisory explanation", ""]
    posture = ex.get("risk_posture", {})
    if posture:
        lines.append(f"Risk posture: {posture.get('mode', 'balanced')}")
        lines.append(f"- {posture.get('message', '')}")
        lines.append("")
    lines.append("Main drivers of allocation change:")
    for item in ex.get("main_drivers_of_change", [])[:6]:
        lines.append(f"- {item['feature']}: importance={float(item['importance']):+.4f}")
    lines.append("")
    bench = ex.get("benchmark_relative_explanation", {})
    if bench:
        lines.append("Benchmark-relative view:")
        lines.append(f"- {bench.get('message', '')}")
        lines.append("")
    return "\n".join(lines).strip()
