from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

try:  # pragma: no cover
    from lime.lime_tabular import LimeTabularExplainer  # type: ignore
except Exception:  # pragma: no cover
    LimeTabularExplainer = None  # type: ignore


PredictFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class SimpleLimeExplainer:
    training_data: np.ndarray
    feature_names: List[str]
    mode: str = "regression"

    def explain_instance(
        self,
        instance: np.ndarray,
        predict_fn: PredictFn,
        num_features: int,
        num_samples: int,
    ) -> Dict[str, float]:
        train = np.asarray(self.training_data, dtype=float)
        x0 = np.asarray(instance, dtype=float).reshape(1, -1)
        std = np.nanstd(train, axis=0)
        std[~np.isfinite(std)] = 0.0
        std = np.where(std <= 1e-8, 0.05, std)

        noise = np.random.normal(loc=0.0, scale=1.0, size=(max(int(num_samples), 32), x0.shape[1]))
        samples = x0 + noise * std
        preds = np.asarray(predict_fn(samples), dtype=float).reshape(-1)

        weights = np.exp(-np.sum(((samples - x0) / std) ** 2, axis=1) / (2.0 * x0.shape[1]))
        model = Ridge(alpha=1.0, fit_intercept=True)
        model.fit(samples, preds, sample_weight=weights)
        coefs = np.asarray(model.coef_, dtype=float).reshape(-1)
        ranked_idx = np.argsort(np.abs(coefs))[::-1][:max(int(num_features), 1)]
        return {str(self.feature_names[i]): float(coefs[i]) for i in ranked_idx}


def build_lime_explainer(training_data: Any, feature_names: Sequence[str], mode: str = "regression") -> Any:
    training_arr = np.asarray(training_data, dtype=float)
    if LimeTabularExplainer is not None:
        try:
            return LimeTabularExplainer(
                training_data=training_arr,
                feature_names=list(feature_names),
                mode=mode,
                discretize_continuous=False,
            )
        except Exception:
            pass
    return SimpleLimeExplainer(training_data=training_arr, feature_names=list(feature_names), mode=mode)


def explain_allocation_instance(
    instance: Any,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    num_features: int,
    num_samples: int,
    *,
    explainer: Any,
    feature_names: Sequence[str],
) -> Dict[str, Any]:
    instance_arr = np.asarray(instance, dtype=float).reshape(-1)
    local_weights: Dict[str, float]

    if LimeTabularExplainer is not None and hasattr(explainer, "explain_instance") and not isinstance(explainer, SimpleLimeExplainer):
        try:
            exp = explainer.explain_instance(instance_arr, predict_fn, num_features=max(int(num_features), 1), num_samples=max(int(num_samples), 32))
            local_weights = {str(name): float(weight) for name, weight in exp.as_list()}
        except Exception:
            local_weights = SimpleLimeExplainer(np.atleast_2d(instance_arr), list(feature_names)).explain_instance(
                instance_arr, predict_fn, num_features, num_samples
            )
    else:
        local_weights = explainer.explain_instance(instance_arr, predict_fn, num_features, num_samples)

    payload = normalize_lime_output(
        local_weights=local_weights,
        instance=instance_arr,
        feature_names=feature_names,
    )
    payload["summary_text"] = build_lime_summary_text(payload)
    return payload


def normalize_lime_output(
    *,
    local_weights: Dict[str, float],
    instance: np.ndarray,
    feature_names: Sequence[str],
) -> Dict[str, Any]:
    context = {str(name): float(val) for name, val in zip(feature_names, instance.reshape(-1))}
    ranked = sorted(local_weights.items(), key=lambda kv: abs(float(kv[1])), reverse=True)
    return {
        "instance_id": "allocation_instance_0",
        "top_local_features": [str(k) for k, _ in ranked],
        "local_weights": {str(k): float(v) for k, v in ranked},
        "local_prediction_context": context,
    }


def build_lime_summary_text(lime_payload: Dict[str, Any]) -> str:
    ranked = list((lime_payload.get("local_weights") or {}).items())[:5]
    if not ranked:
        return "No local LIME explanation was available for the selected allocation instance."
    parts: List[str] = []
    for feature, weight in ranked:
        direction = "supports" if float(weight) >= 0 else "pushes against"
        parts.append(f"{feature} {direction} the local allocation decision ({float(weight):+.4f})")
    return "LIME view: " + "; ".join(parts) + "."
