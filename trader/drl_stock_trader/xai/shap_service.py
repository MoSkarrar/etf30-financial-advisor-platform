from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:  # pragma: no cover
    import shap  # type: ignore
except Exception:  # pragma: no cover
    shap = None  # type: ignore


def _to_frame(data: Any, feature_names: Optional[Sequence[str]] = None) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        frame = data.copy()
    elif isinstance(data, pd.Series):
        frame = data.to_frame().T
    else:
        arr = np.asarray(data)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        cols = list(feature_names or [f"feature_{i}" for i in range(arr.shape[1])])
        frame = pd.DataFrame(arr, columns=cols)
    if feature_names:
        keep = [c for c in feature_names if c in frame.columns]
        if keep:
            frame = frame[keep]
    return frame.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def _top_k_from_map(values: Dict[str, float], top_k: int) -> List[str]:
    return [
        str(k)
        for k, _ in sorted(values.items(), key=lambda kv: abs(float(kv[1])), reverse=True)[:max(int(top_k), 1)]
    ]


def _fallback_attributions(model: Any, background: pd.DataFrame, sample: pd.DataFrame) -> Dict[str, float]:
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        importances = np.ones(sample.shape[1], dtype=float) / max(sample.shape[1], 1)
    else:
        importances = np.asarray(importances, dtype=float)
        if importances.size != sample.shape[1]:
            importances = np.resize(importances, sample.shape[1])
    baseline = background.mean(axis=0).to_numpy(dtype=float)
    row = sample.iloc[0].to_numpy(dtype=float)
    contrib = (row - baseline) * importances
    return {str(col): float(val) for col, val in zip(sample.columns, contrib)}


def _select_shap_method(model: Any) -> str:
    model_name = type(model).__name__.lower()
    if "forest" in model_name or "tree" in model_name or "xgb" in model_name or "gbm" in model_name:
        return "tree"
    return "generic"


def run_shap_explanation(
    model_or_surrogate: Any,
    background: Any,
    sample: Any,
    feature_names: Optional[Sequence[str]] = None,
    mode: str = "regression",
    top_k: int = 8,
) -> Dict[str, Any]:
    background_df = _to_frame(background, feature_names)
    sample_df = _to_frame(sample, feature_names)
    if sample_df.empty:
        return normalize_shap_output({}, sample_df, background_df, method="empty", top_k=top_k)

    method = _select_shap_method(model_or_surrogate)
    attribution_values: Dict[str, float]

    if shap is not None:
        try:
            if method == "tree":
                explainer = shap.TreeExplainer(model_or_surrogate)
                values = explainer.shap_values(sample_df)
            else:
                explainer = shap.Explainer(model_or_surrogate.predict, background_df)
                values = explainer(sample_df).values
            if isinstance(values, list):
                values = values[0]
            arr = np.asarray(values)
            if arr.ndim == 2:
                arr = arr[0]
            attribution_values = {str(col): float(v) for col, v in zip(sample_df.columns, arr.reshape(-1))}
        except Exception:
            method = "fallback"
            attribution_values = _fallback_attributions(model_or_surrogate, background_df, sample_df)
    else:
        method = "fallback"
        attribution_values = _fallback_attributions(model_or_surrogate, background_df, sample_df)

    return normalize_shap_output(
        attribution_values,
        sample_df,
        background_df,
        method=method,
        top_k=top_k,
        mode=mode,
    )


def normalize_shap_output(
    attribution_values: Dict[str, float],
    sample_df: pd.DataFrame,
    background_df: pd.DataFrame,
    *,
    method: str,
    top_k: int,
    mode: str = "regression",
) -> Dict[str, Any]:
    if sample_df.empty:
        feature_values: Dict[str, Any] = {}
    else:
        feature_values = {str(k): float(v) for k, v in sample_df.iloc[0].to_dict().items()}

    attribution_values = {str(k): float(v) for k, v in (attribution_values or {}).items()}
    top_features = _top_k_from_map(attribution_values, top_k)
    payload = {
        "method": method,
        "mode": mode,
        "background_size": int(len(background_df.index)),
        "top_features": top_features,
        "feature_values": feature_values,
        "attribution_values": attribution_values,
    }
    payload["summary_text"] = build_shap_summary_text(payload)
    return payload


def build_shap_summary_text(shap_payload: Dict[str, Any]) -> str:
    top_features = list(shap_payload.get("top_features") or [])
    if not top_features:
        return "No SHAP feature attributions were available for this allocation."

    parts: List[str] = []
    for feature in top_features[:5]:
        value = float((shap_payload.get("attribution_values") or {}).get(feature, 0.0))
        direction = "supports" if value >= 0 else "pushes against"
        parts.append(f"{feature} {direction} the current allocation ({value:+.4f})")
    return "SHAP view: " + "; ".join(parts) + "."
