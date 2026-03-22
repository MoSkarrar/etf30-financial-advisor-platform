from __future__ import annotations

import inspect
from pathlib import Path


def _patch_artifact_paths(monkeypatch, artifact_store_module, tmp_path):
    p = artifact_store_module.paths
    monkeypatch.setattr(p, 'session_manifest_path', lambda session_id: str(tmp_path / f'session_manifest_{session_id}.json'))
    monkeypatch.setattr(p, 'run_manifest_path', lambda run_id: str(tmp_path / f'run_manifest_{run_id}.json'))
    monkeypatch.setattr(p, 'allocation_json_path', lambda run_id: str(tmp_path / f'allocation_{run_id}.json'))
    monkeypatch.setattr(p, 'benchmark_json_path', lambda run_id: str(tmp_path / f'benchmark_{run_id}.json'))
    monkeypatch.setattr(p, 'risk_report_path', lambda run_id: str(tmp_path / f'risk_{run_id}.json'))
    monkeypatch.setattr(p, 'xai_text_path', lambda run_id: str(tmp_path / f'xai_text_{run_id}.txt'))
    monkeypatch.setattr(p, 'advisory_summary_json_path', lambda run_id: str(tmp_path / f'advisory_summary_{run_id}.json'))
    monkeypatch.setattr(p, 'advisory_summary_text_path', lambda run_id: str(tmp_path / f'advisory_summary_{run_id}.txt'))
    monkeypatch.setattr(p, 'investor_profile_json_path', lambda run_id: str(tmp_path / f'investor_profile_{run_id}.json'), raising=False)
    monkeypatch.setattr(p, 'scenario_result_path', lambda run_id, scenario_name='base': str(tmp_path / f'scenario_{scenario_name}_{run_id}.json'))
    monkeypatch.setattr(p, 'benchmark_series_path', lambda run_id: str(tmp_path / f'benchmark_series_{run_id}.json'), raising=False)
    monkeypatch.setattr(p, 'active_return_metrics_path', lambda run_id: str(tmp_path / f'active_return_metrics_{run_id}.json'), raising=False)
    monkeypatch.setattr(p, 'get_engine_info_path', lambda run_id: str(tmp_path / f'engine_info_{run_id}.json'), raising=False)
    monkeypatch.setattr(p, 'policy_check_path', lambda run_id: str(tmp_path / f'policy_check_{run_id}.json'), raising=False)
    monkeypatch.setattr(p, 'get_shap_path', lambda run_id: str(tmp_path / f'shap_{run_id}.json'), raising=False)
    monkeypatch.setattr(p, 'get_lime_path', lambda run_id: str(tmp_path / f'lime_{run_id}.json'), raising=False)
    monkeypatch.setattr(p, 'get_rule_summary_path', lambda run_id: str(tmp_path / f'rule_summary_{run_id}.json'), raising=False)
    monkeypatch.setattr(p, 'get_explanation_bundle_path', lambda run_id: str(tmp_path / f'explanation_bundle_{run_id}.json'), raising=False)
    monkeypatch.setattr(p, 'get_explanation_lab_path', lambda run_id: str(tmp_path / f'explanation_lab_{run_id}.json'), raising=False)
    monkeypatch.setattr(p, 'get_narration_cache_path', lambda run_id: str(tmp_path / f'narration_cache_{run_id}.json'), raising=False)
    monkeypatch.setattr(p, 'ensure_parent_dir', lambda file_path: str(Path(file_path).parent.mkdir(parents=True, exist_ok=True) or file_path))
    monkeypatch.setattr(p, 'advisory_outputs_dir', lambda: str(tmp_path))



def test_append_run_to_session_and_available_runs(monkeypatch, tmp_path):
    from trader.services import artifact_store

    _patch_artifact_paths(monkeypatch, artifact_store, tmp_path)
    artifact_store.append_run_to_session('s1', 'etf30', 'r1', investor_profile={'profile_name': 'balanced'})
    artifact_store.append_run_to_session('s1', 'etf30', 'r2', investor_profile={'profile_name': 'balanced'})

    session = artifact_store.load_session_manifest('s1')
    assert session['latest_run_id'] == 'r2'
    assert artifact_store.available_runs_for_session('s1') == ['r1', 'r2']


def test_persist_and_load_advisory_bundle(monkeypatch, tmp_path):
    from trader.services import artifact_store

    _patch_artifact_paths(monkeypatch, artifact_store, tmp_path)

    kwargs = {
        'session_id': 's1',
        'run_id': 'r1',
        'market': 'etf30',
        'iteration': 1,
        'trade_window': {'start': '20200101', 'end': '20200201'},
        'selected_model': 'PPO',
        'run_level_metrics': {'selected_model': 'PPO'},
        'allocation_recommendation': {'run_id': 'r1', 'target_weights': {'AAA': 0.6, 'CASH': 0.4}},
        'benchmark_comparison': {'benchmark_name': 'equal_weight', 'active_return': 0.03},
        'risk_snapshot': {'cash_weight': 0.4, 'turnover': 0.1},
        'investor_profile': {'profile_name': 'balanced'},
        'portfolio_policy': {'max_single_position_cap': 0.12, 'min_cash_weight': 0.05},
        'explanation_summary': 'summary text',
        'xai_payload': {'explanation': {}},
        'advisory_summary_text': 'advisor summary',
    }

    sig = inspect.signature(artifact_store.persist_portfolio_advisory_bundle)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    artifact_store.persist_portfolio_advisory_bundle(**filtered)

    bundle = artifact_store.load_advisory_bundle('r1')
    assert bundle is not None
    assert bundle['manifest']['run_id'] == 'r1'
    assert 'allocation_recommendation' in bundle
    assert artifact_store.resolve_xai_text_for_run('r1')
