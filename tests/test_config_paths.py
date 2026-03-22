from __future__ import annotations

from pathlib import Path


def test_app_config_sections_exist():
    from trader.drl_stock_trader.config.app_config import APP_CONFIG

    assert APP_CONFIG.identity.app_title
    assert APP_CONFIG.dataset.etf30_filename.endswith('.csv')
    assert APP_CONFIG.benchmarks.default_primary
    assert APP_CONFIG.policy.rebalance_cadence_days > 0
    assert APP_CONFIG.rl.default_initial_cash > 0


def test_timesteps_profile_lookup_falls_back():
    from trader.drl_stock_trader.config.app_config import APP_CONFIG

    profile = APP_CONFIG.rl.timesteps_for_robustness('not-a-level')
    assert hasattr(profile, 'a2c')
    assert hasattr(profile, 'ppo')
    assert hasattr(profile, 'ddpg')


def test_paths_generate_expected_artifact_names(monkeypatch, tmp_path):
    from trader.drl_stock_trader.config import paths

    monkeypatch.setattr(paths, 'RESULTS_DIR', str(tmp_path / 'results'))
    monkeypatch.setattr(paths, 'MANIFESTS_DIR', str(tmp_path / 'results' / 'manifests'))
    monkeypatch.setattr(paths, 'ADVISORY_OUTPUTS_DIR', str(tmp_path / 'results' / 'advisory'))
    monkeypatch.setattr(paths, 'PORTFOLIO_OUTPUTS_DIR', str(tmp_path / 'results' / 'portfolio'))
    monkeypatch.setattr(paths, 'BENCHMARK_OUTPUTS_DIR', str(tmp_path / 'results' / 'benchmarks'))
    monkeypatch.setattr(paths, 'RISK_OUTPUTS_DIR', str(tmp_path / 'results' / 'risk'))
    monkeypatch.setattr(paths, 'SNAPSHOT_OUTPUTS_DIR', str(tmp_path / 'results' / 'snapshots'))

    session_path = Path(paths.session_manifest_path('s1'))
    run_path = Path(paths.run_manifest_path('r1'))
    xai_path = Path(paths.xai_text_path('r1'))
    alloc_path = Path(paths.allocation_json_path('r1'))
    risk_path = Path(paths.risk_report_path('r1'))

    assert session_path.name.startswith('session_manifest_')
    assert run_path.name.startswith('run_manifest_')
    assert xai_path.suffix == '.txt'
    assert alloc_path.suffix == '.json'
    assert risk_path.suffix == '.json'
    assert session_path.parent.exists()
