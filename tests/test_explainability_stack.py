from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


shap_service = pytest.importorskip('trader.drl_stock_trader.xai.shap_service')
lime_service = pytest.importorskip('trader.drl_stock_trader.xai.lime_service')
rule_summary_mod = pytest.importorskip('trader.drl_stock_trader.xai.rule_summary')
explanation_bundle_mod = pytest.importorskip('trader.drl_stock_trader.xai.explanation_bundle')
explanation_lab_mod = pytest.importorskip('trader.drl_stock_trader.xai.explanation_lab')


def test_shap_service_returns_summary_with_fallback_model():
    class Model:
        feature_importances_ = np.array([0.6, 0.4])

        def predict(self, x):
            return np.sum(x, axis=1)

    background = pd.DataFrame({'f1': [0.0, 1.0], 'f2': [1.0, 2.0]})
    sample = pd.DataFrame({'f1': [2.0], 'f2': [0.5]})

    payload = shap_service.run_shap_explanation(Model(), background, sample, feature_names=['f1', 'f2'], top_k=2)
    assert payload['top_features']
    assert 'summary_text' in payload


def test_lime_service_returns_local_weights():
    training = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 1.0]])
    explainer = lime_service.build_lime_explainer(training, ['f1', 'f2'])
    payload = lime_service.explain_allocation_instance(
        np.array([1.5, 1.2]),
        predict_fn=lambda arr: np.sum(arr, axis=1),
        num_features=2,
        num_samples=50,
        explainer=explainer,
        feature_names=['f1', 'f2'],
    )
    assert payload['top_local_features']
    assert 'summary_text' in payload


def test_rule_summary_bundle_and_lab_report_connect():
    rule_summary = rule_summary_mod.build_rule_summary(
        allocation_recommendation={'target_weights': {'AAA': 0.7, 'CASH': 0.02}},
        benchmark_comparison={'active_return': -0.02, 'tracking_error': 0.08},
        risk_snapshot={'turnover': 0.4, 'concentration_hhi': 0.12, 'realized_volatility': 0.2, 'downside_volatility': 0.19},
        portfolio_policy={'min_cash_weight': 0.05, 'turnover_budget': 0.35},
        policy_check={'breaches': ['Cash below floor']},
    )
    bundle = explanation_bundle_mod.build_explanation_bundle(
        {'top_features': ['f1'], 'summary_text': 'shap summary'},
        {'top_local_features': ['f1'], 'summary_text': 'lime summary'},
        rule_summary,
        {'turnover': 0.4},
        {'breaches': ['Cash below floor']},
    )
    lab = explanation_lab_mod.build_explanation_lab_report(bundle, {'concentration_hhi': 0.12}, {'breaches': ['Cash below floor']})
    assert bundle['advisor_summary']
    assert 'confidence_score' in lab
    assert isinstance(lab['hypotheses'], list)
