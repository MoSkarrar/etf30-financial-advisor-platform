from __future__ import annotations

import os
from typing import Optional

from trader.drl_stock_trader.config.app_config import APP_CONFIG


def _persistence_attr(name: str, default: str) -> str:
    return str(getattr(APP_CONFIG.persistence, name, default))


PACKAGE_ROOT = os.path.join("trader", "drl_stock_trader")
CONFIG_DIR = os.path.join(PACKAGE_ROOT, "config")
DATASETS_DIR = os.path.join(PACKAGE_ROOT, "datasets")

RESULTS_DIR = os.path.join(PACKAGE_ROOT, _persistence_attr("results_dirname", "results"))
MANIFESTS_DIR = os.path.join(RESULTS_DIR, _persistence_attr("manifests_dirname", "run_manifests"))
TRAINED_MODELS_DIR = os.path.join(PACKAGE_ROOT, _persistence_attr("trained_models_dirname", "trained_models"))

PORTFOLIO_OUTPUTS_DIR = os.path.join(RESULTS_DIR, _persistence_attr("portfolio_dirname", "portfolio_allocation"))
BENCHMARK_OUTPUTS_DIR = os.path.join(RESULTS_DIR, _persistence_attr("benchmark_dirname", "benchmarks"))
ADVISORY_OUTPUTS_DIR = os.path.join(RESULTS_DIR, _persistence_attr("advisory_dirname", "advisory"))
SCENARIO_OUTPUTS_DIR = os.path.join(RESULTS_DIR, _persistence_attr("scenario_dirname", "scenarios"))
RISK_OUTPUTS_DIR = os.path.join(RESULTS_DIR, _persistence_attr("risk_dirname", "risk"))
SNAPSHOT_OUTPUTS_DIR = os.path.join(RESULTS_DIR, _persistence_attr("snapshots_dirname", "snapshots"))
ENGINE_OUTPUTS_DIR = os.path.join(RESULTS_DIR, _persistence_attr("engines_dirname", "engines"))
EXPLAINABILITY_OUTPUTS_DIR = os.path.join(RESULTS_DIR, _persistence_attr("explainability_dirname", "explainability"))
NARRATION_OUTPUTS_DIR = os.path.join(RESULTS_DIR, _persistence_attr("narration_dirname", "narration"))


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def ensure_parent_dir(file_path: str) -> str:
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return file_path


def package_root() -> str:
    return PACKAGE_ROOT


def config_dir() -> str:
    return CONFIG_DIR


def datasets_dir() -> str:
    return DATASETS_DIR


def results_dir() -> str:
    return ensure_dir(RESULTS_DIR)


def manifests_dir() -> str:
    return ensure_dir(MANIFESTS_DIR)


def trained_models_dir() -> str:
    return ensure_dir(TRAINED_MODELS_DIR)


def portfolio_outputs_dir() -> str:
    return ensure_dir(PORTFOLIO_OUTPUTS_DIR)


def benchmark_outputs_dir() -> str:
    return ensure_dir(BENCHMARK_OUTPUTS_DIR)


def advisory_outputs_dir() -> str:
    return ensure_dir(ADVISORY_OUTPUTS_DIR)


def scenario_outputs_dir() -> str:
    return ensure_dir(SCENARIO_OUTPUTS_DIR)


def risk_outputs_dir() -> str:
    return ensure_dir(RISK_OUTPUTS_DIR)


def snapshot_outputs_dir() -> str:
    return ensure_dir(SNAPSHOT_OUTPUTS_DIR)


def engine_outputs_dir() -> str:
    return ensure_dir(ENGINE_OUTPUTS_DIR)


def explainability_outputs_dir() -> str:
    return ensure_dir(EXPLAINABILITY_OUTPUTS_DIR)


def narration_outputs_dir() -> str:
    return ensure_dir(NARRATION_OUTPUTS_DIR)


def initial_balance_file_path() -> str:
    return os.path.join(config_dir(), "initial_balance.txt")


def etf30_dataset_path() -> str:
    return os.path.join(datasets_dir(), APP_CONFIG.dataset.etf30_filename)


def covariance_cache_path() -> str:
    return os.path.join(datasets_dir(), APP_CONFIG.dataset.covariance_cache_filename)


def benchmark_cache_path() -> str:
    return os.path.join(datasets_dir(), APP_CONFIG.dataset.benchmark_cache_filename)


def session_manifest_path(session_id: str) -> str:
    filename = f"{_persistence_attr('session_manifest_prefix', 'session_manifest')}_{session_id}.json"
    return os.path.join(manifests_dir(), filename)


def run_manifest_path(run_id: str) -> str:
    filename = f"{_persistence_attr('run_manifest_prefix', 'run_manifest')}_{run_id}.json"
    return os.path.join(manifests_dir(), filename)


def _family_path(base_dir: str, stem: str, run_id: str, extension: str = "json") -> str:
    filename = f"{stem}_{run_id}.{extension}"
    return os.path.join(base_dir, filename)


def get_engine_artifact_dir(session_id: str) -> str:
    return ensure_dir(os.path.join(engine_outputs_dir(), session_id))


def get_finrl_run_dir(run_id: str) -> str:
    return ensure_dir(os.path.join(engine_outputs_dir(), "finrl", run_id))


def xai_text_path(run_id: str) -> str:
    filename = f"{_persistence_attr('xai_text_prefix', 'xai_text')}_{run_id}.txt"
    return os.path.join(advisory_outputs_dir(), filename)


def allocation_json_path(run_id: str) -> str:
    return _family_path(portfolio_outputs_dir(), _persistence_attr('allocation_prefix', 'allocation'), run_id)


def benchmark_json_path(run_id: str) -> str:
    return _family_path(benchmark_outputs_dir(), _persistence_attr('benchmark_prefix', 'benchmark'), run_id)


def advisory_summary_json_path(run_id: str) -> str:
    return _family_path(advisory_outputs_dir(), _persistence_attr('advisory_summary_prefix', 'advisory_summary'), run_id)


def advisory_summary_text_path(run_id: str) -> str:
    return _family_path(
        advisory_outputs_dir(),
        _persistence_attr('advisory_summary_prefix', 'advisory_summary'),
        run_id,
        _persistence_attr('advisory_summary_text_suffix', 'txt'),
    )


def investor_profile_json_path(run_id: str) -> str:
    return _family_path(advisory_outputs_dir(), _persistence_attr('investor_profile_prefix', 'investor_profile'), run_id)


def allocation_snapshot_path(run_id: str, extension: str = "csv") -> str:
    return _family_path(snapshot_outputs_dir(), _persistence_attr('allocation_snapshot_prefix', 'allocation_snapshot'), run_id, extension)


def risk_report_path(run_id: str) -> str:
    return _family_path(risk_outputs_dir(), _persistence_attr('risk_report_prefix', 'risk_report'), run_id)


def policy_check_path(run_id: str) -> str:
    return _family_path(risk_outputs_dir(), _persistence_attr('policy_check_prefix', 'policy_check'), run_id)


def scenario_result_path(run_id: str, scenario_name: str = "base") -> str:
    filename = f"{_persistence_attr('scenario_prefix', 'scenario')}_{scenario_name}_{run_id}.json"
    return os.path.join(scenario_outputs_dir(), filename)


def get_shap_path(run_id: str) -> str:
    return _family_path(explainability_outputs_dir(), _persistence_attr('shap_prefix', 'shap'), run_id)


def get_lime_path(run_id: str) -> str:
    return _family_path(explainability_outputs_dir(), _persistence_attr('lime_prefix', 'lime'), run_id)


def get_rule_summary_path(run_id: str) -> str:
    return _family_path(explainability_outputs_dir(), _persistence_attr('rule_summary_prefix', 'rule_summary'), run_id)


def get_explanation_bundle_path(run_id: str) -> str:
    return _family_path(explainability_outputs_dir(), _persistence_attr('explanation_bundle_prefix', 'explanation_bundle'), run_id)


def get_explanation_lab_path(run_id: str) -> str:
    return _family_path(explainability_outputs_dir(), _persistence_attr('explanation_lab_prefix', 'explanation_lab'), run_id)


def get_advisory_summary_path(run_id: str) -> str:
    return advisory_summary_json_path(run_id)


def get_benchmark_report_path(run_id: str) -> str:
    return benchmark_json_path(run_id)


def get_narration_cache_path(run_id: str) -> str:
    return _family_path(narration_outputs_dir(), _persistence_attr('narration_cache_prefix', 'narration_cache'), run_id)


def get_engine_info_path(run_id: str) -> str:
    return _family_path(engine_outputs_dir(), _persistence_attr('engine_info_prefix', 'engine_info'), run_id)


def benchmark_series_path(run_id: str) -> str:
    return _family_path(advisory_outputs_dir(), _persistence_attr('benchmark_series_prefix', 'benchmark_series'), run_id)


def active_return_metrics_path(run_id: str) -> str:
    return _family_path(advisory_outputs_dir(), _persistence_attr('active_return_metrics_prefix', 'active_return_metrics'), run_id)


def validation_csv_path(run_id: str, extension: str = "csv", label: Optional[str] = None) -> str:
    safe_extension = str(extension or "csv").strip().lstrip(".") or "csv"
    safe_label = str(label or "").strip().lower().replace(" ", "_")

    if safe_label:
        filename = f"{_persistence_attr('validation_prefix', 'validation')}_{safe_label}_{run_id}.{safe_extension}"
    else:
        filename = f"{_persistence_attr('validation_prefix', 'validation')}_{run_id}.{safe_extension}"

    return os.path.join(advisory_outputs_dir(), filename)


def existing_path(path: Optional[str]) -> str:
    if path and os.path.exists(path):
        return path
    return ""


def first_existing_path(*paths_to_try: Optional[str]) -> str:
    for path in paths_to_try:
        if path and os.path.exists(path):
            return path
    return ""
