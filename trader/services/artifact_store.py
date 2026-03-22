from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from trader.domain.session_models import (
    AllocationRecommendation,
    BenchmarkComparison,
    ExecutionEngineInfo,
    ExplanationBundle,
    ExplanationLabReport,
    InvestorProfile,
    PolicyCheckResult,
    PortfolioPolicy,
    RiskSnapshot,
    RunArtifacts,
    RunManifestRecord,
    SessionManifestRecord,
    TradeWindow,
)
from trader.drl_stock_trader.config import paths


def _utc_now_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _serialize(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    if hasattr(obj, "to_dict"):
        try:
            return _serialize(obj.to_dict())
        except Exception:
            pass
    if is_dataclass(obj):
        return _serialize(asdict(obj))
    return str(obj)


def _write_json(path: str, payload: Any) -> str:
    paths.ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_serialize(payload), fh, ensure_ascii=False, indent=2)
    return path


def _write_text(path: str, text: str) -> str:
    paths.ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write((text or "").strip())
    return path


def _read_json(path: str) -> Optional[dict]:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _read_text(path: str) -> str:
    if not path or not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read().strip()


def _advisory_family_path(run_id: str, stem: str, extension: str = "json") -> str:
    return os.path.join(paths.advisory_outputs_dir(), f"{stem}_{run_id}.{extension}")


def _first_existing_json(*candidate_paths: str) -> Optional[dict]:
    for path in candidate_paths:
        payload = _read_json(path)
        if payload is not None:
            return payload
    return None


def _first_existing_text(*candidate_paths: str) -> str:
    for path in candidate_paths:
        payload = _read_text(path)
        if payload:
            return payload
    return ""


def write_allocation_recommendation(run_id: str, payload: Any) -> str:
    return _write_json(paths.allocation_json_path(run_id), payload)


def write_benchmark_comparison(run_id: str, payload: Any) -> str:
    return _write_json(paths.benchmark_json_path(run_id), payload)


def write_risk_snapshot(run_id: str, payload: Any) -> str:
    return _write_json(paths.risk_report_path(run_id), payload)


def write_investor_profile(run_id: str, payload: Any) -> str:
    if hasattr(paths, "investor_profile_json_path"):
        return _write_json(paths.investor_profile_json_path(run_id), payload)
    return _write_json(_advisory_family_path(run_id, "investor_profile"), payload)


def write_scenario_result(run_id: str, payload: Any, scenario_name: str = "base") -> str:
    return _write_json(paths.scenario_result_path(run_id, scenario_name), payload)


def write_xai_json(run_id: str, payload: Any) -> str:
    return _write_json(_advisory_family_path(run_id, "xai"), payload)


def write_xai_text(run_id: str, text: str) -> str:
    return _write_text(paths.xai_text_path(run_id), text)


def write_advisory_summary_json(run_id: str, payload: Any) -> str:
    return _write_json(paths.advisory_summary_json_path(run_id), payload)


def write_advisory_summary_text(run_id: str, text: str) -> str:
    if hasattr(paths, "advisory_summary_text_path"):
        return _write_text(paths.advisory_summary_text_path(run_id), text)
    return _write_text(_advisory_family_path(run_id, "advisory_summary", "txt"), text)


def write_benchmark_series(run_id: str, payload: Any) -> str:
    if hasattr(paths, "benchmark_series_path"):
        return _write_json(paths.benchmark_series_path(run_id), payload)
    return _write_json(_advisory_family_path(run_id, "benchmark_series"), payload)


def write_active_return_metrics(run_id: str, payload: Any) -> str:
    if hasattr(paths, "active_return_metrics_path"):
        return _write_json(paths.active_return_metrics_path(run_id), payload)
    return _write_json(_advisory_family_path(run_id, "active_return_metrics"), payload)


def write_engine_info(run_id: str, payload: Any) -> str:
    if hasattr(paths, "get_engine_info_path"):
        return _write_json(paths.get_engine_info_path(run_id), payload)
    return _write_json(_advisory_family_path(run_id, "engine_info"), payload)


def write_policy_check(run_id: str, payload: Any) -> str:
    if hasattr(paths, "policy_check_path"):
        return _write_json(paths.policy_check_path(run_id), payload)
    return _write_json(_advisory_family_path(run_id, "policy_check"), payload)


def write_shap_explanation(run_id: str, payload: Any) -> str:
    if hasattr(paths, "get_shap_path"):
        return _write_json(paths.get_shap_path(run_id), payload)
    return _write_json(_advisory_family_path(run_id, "shap"), payload)


def write_lime_explanation(run_id: str, payload: Any) -> str:
    if hasattr(paths, "get_lime_path"):
        return _write_json(paths.get_lime_path(run_id), payload)
    return _write_json(_advisory_family_path(run_id, "lime"), payload)


def write_rule_summary(run_id: str, payload: Any) -> str:
    if hasattr(paths, "get_rule_summary_path"):
        return _write_json(paths.get_rule_summary_path(run_id), payload)
    return _write_json(_advisory_family_path(run_id, "rule_summary"), payload)


def write_explanation_bundle(run_id: str, payload: Any) -> str:
    if hasattr(paths, "get_explanation_bundle_path"):
        return _write_json(paths.get_explanation_bundle_path(run_id), payload)
    return _write_json(_advisory_family_path(run_id, "explanation_bundle"), payload)


def write_explanation_lab(run_id: str, payload: Any) -> str:
    if hasattr(paths, "get_explanation_lab_path"):
        return _write_json(paths.get_explanation_lab_path(run_id), payload)
    return _write_json(_advisory_family_path(run_id, "explanation_lab"), payload)


def write_narration_cache(run_id: str, payload: Any) -> str:
    if hasattr(paths, "get_narration_cache_path"):
        return _write_json(paths.get_narration_cache_path(run_id), payload)
    return _write_json(_advisory_family_path(run_id, "narration_cache"), payload)


def write_run_manifest(record: Dict[str, Any]) -> str:
    run_id = str(record.get("run_id", "")).strip()
    if not run_id:
        raise ValueError("run_id is required to write run manifest.")
    return _write_json(paths.run_manifest_path(run_id), record)


def load_run_manifest(run_id: str) -> Optional[Dict[str, Any]]:
    return _read_json(paths.run_manifest_path(run_id))


def write_session_manifest(record: Dict[str, Any]) -> str:
    session_id = str(record.get("session_id", "")).strip()
    if not session_id:
        raise ValueError("session_id is required to write session manifest.")
    return _write_json(paths.session_manifest_path(session_id), record)


def load_session_manifest(session_id: str) -> Optional[Dict[str, Any]]:
    return _read_json(paths.session_manifest_path(session_id))


def append_run_to_session(
    session_id: str,
    market: str,
    run_id: str,
    investor_profile: Optional[Any] = None,
) -> Dict[str, Any]:
    current = SessionManifestRecord.from_dict(load_session_manifest(session_id) or {})
    if not current.session_id:
        current = SessionManifestRecord(
            session_id=session_id,
            market=market or "etf30",
            created_at=_utc_now_z(),
            runs=[],
            latest_run_id="",
            investor_profile=InvestorProfile.from_dict(_serialize(investor_profile or {})),
        )

    if run_id not in current.runs:
        current.runs.append(run_id)
    current.latest_run_id = run_id

    if investor_profile is not None:
        current.investor_profile = InvestorProfile.from_dict(_serialize(investor_profile))

    payload = current.to_dict()
    write_session_manifest(payload)
    return payload


def available_runs_for_session(session_id: str) -> List[str]:
    session = SessionManifestRecord.from_dict(load_session_manifest(session_id) or {})
    return [str(v) for v in session.runs]


def get_session_run_records(session_id: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for run_id in available_runs_for_session(session_id):
        record = load_run_manifest(run_id)
        if record:
            out.append(record)
    return out


def resolve_xai_text_for_run(run_id: str) -> str:
    manifest = load_run_manifest(run_id) or {}
    artifacts = manifest.get("artifacts", {})
    return _first_existing_text(
        artifacts.get("xai_text", ""),
        paths.xai_text_path(run_id),
    )


def _legacy_manifest_to_current(manifest: Dict[str, Any], run_id: str) -> Dict[str, Any]:
    if not manifest:
        return {}

    if "allocation_recommendation" in manifest or "run_level_metrics" in manifest:
        return manifest

    artifacts = manifest.get("artifacts") or {}
    selected_model = str(manifest.get("selected_model", ""))
    sharpes = dict(manifest.get("sharpes") or {})

    return {
        "session_id": str(manifest.get("session_id", "")),
        "run_id": str(manifest.get("run_id", run_id)),
        "created_at": str(manifest.get("created_at", "")),
        "market": str(manifest.get("market", "etf30")),
        "iteration": int(manifest.get("iteration", 0)),
        "trade_window": dict(manifest.get("trade_window") or {}),
        "selected_model": selected_model,
        "engine_info": {
            "engine_name": "legacy_rl",
            "backend_type": "sb3",
            "algorithm_name": selected_model,
            "train_config": {},
            "engine_notes": "Hydrated from legacy manifest format.",
        },
        "run_level_metrics": {
            "legacy_sharpes": sharpes,
            "selected_model": selected_model,
        },
        "allocation_recommendation": {},
        "benchmark_comparison": {},
        "risk_snapshot": {},
        "investor_profile": {},
        "portfolio_policy": {},
        "policy_check": {},
        "explanation_summary": "",
        "explanation_bundle": {},
        "explanation_lab": {},
        "advisory_narrative": {},
        "artifacts": {
            "allocation_json": "",
            "benchmark_json": "",
            "risk_snapshot_json": "",
            "investor_profile_json": "",
            "scenario_result_json": "",
            "advisory_summary_json": "",
            "advisory_summary_text": "",
            "xai_json": str(artifacts.get("xai_json", "")),
            "xai_text": str(artifacts.get("xai_text", "")),
            "validation_csv": str(artifacts.get("validation_csv", "")),
            "benchmark_series_json": "",
            "active_return_metrics_json": "",
            "engine_info_json": "",
            "policy_check_json": "",
            "shap_json": "",
            "lime_json": "",
            "rule_summary_json": "",
            "explanation_bundle_json": "",
            "explanation_lab_json": "",
            "narration_cache_json": "",
        },
    }


def load_advisory_bundle(run_id: str) -> Optional[Dict[str, Any]]:
    raw_manifest = load_run_manifest(run_id)
    if not raw_manifest:
        return None

    manifest = _legacy_manifest_to_current(raw_manifest, run_id)
    artifacts = manifest.get("artifacts", {}) or {}

    investor_profile_fallback = paths.investor_profile_json_path(run_id) if hasattr(paths, "investor_profile_json_path") else _advisory_family_path(run_id, "investor_profile")
    advisory_summary_text_fallback = paths.advisory_summary_text_path(run_id) if hasattr(paths, "advisory_summary_text_path") else _advisory_family_path(run_id, "advisory_summary", "txt")
    benchmark_series_fallback = paths.benchmark_series_path(run_id) if hasattr(paths, "benchmark_series_path") else _advisory_family_path(run_id, "benchmark_series")
    active_return_fallback = paths.active_return_metrics_path(run_id) if hasattr(paths, "active_return_metrics_path") else _advisory_family_path(run_id, "active_return_metrics")

    return {
        "manifest": manifest,
        "allocation_recommendation": _first_existing_json(
            artifacts.get("allocation_json", ""),
            paths.allocation_json_path(run_id),
        ),
        "benchmark_comparison": _first_existing_json(
            artifacts.get("benchmark_json", ""),
            paths.benchmark_json_path(run_id),
        ),
        "risk_snapshot": _first_existing_json(
            artifacts.get("risk_snapshot_json", ""),
            paths.risk_report_path(run_id),
        ),
        "investor_profile": _first_existing_json(
            artifacts.get("investor_profile_json", ""),
            investor_profile_fallback,
        ),
        "scenario_result": _read_json(artifacts.get("scenario_result_json", "")),
        "advisory_summary": _first_existing_json(
            artifacts.get("advisory_summary_json", ""),
            paths.advisory_summary_json_path(run_id),
        ),
        "advisory_summary_text": _first_existing_text(
            artifacts.get("advisory_summary_text", ""),
            advisory_summary_text_fallback,
        ),
        "xai_json": _read_json(artifacts.get("xai_json", "")),
        "xai_text": _first_existing_text(
            artifacts.get("xai_text", ""),
            paths.xai_text_path(run_id),
        ),
        "benchmark_series": _first_existing_json(
            artifacts.get("benchmark_series_json", ""),
            benchmark_series_fallback,
        ),
        "active_return_metrics": _first_existing_json(
            artifacts.get("active_return_metrics_json", ""),
            active_return_fallback,
        ),
        "engine_info": _first_existing_json(
            artifacts.get("engine_info_json", ""),
            paths.get_engine_info_path(run_id) if hasattr(paths, "get_engine_info_path") else _advisory_family_path(run_id, "engine_info"),
        ),
        "policy_check": _first_existing_json(
            artifacts.get("policy_check_json", ""),
            paths.policy_check_path(run_id) if hasattr(paths, "policy_check_path") else _advisory_family_path(run_id, "policy_check"),
        ),
        "shap": _first_existing_json(
            artifacts.get("shap_json", ""),
            paths.get_shap_path(run_id) if hasattr(paths, "get_shap_path") else _advisory_family_path(run_id, "shap"),
        ),
        "lime": _first_existing_json(
            artifacts.get("lime_json", ""),
            paths.get_lime_path(run_id) if hasattr(paths, "get_lime_path") else _advisory_family_path(run_id, "lime"),
        ),
        "rule_summary": _first_existing_json(
            artifacts.get("rule_summary_json", ""),
            paths.get_rule_summary_path(run_id) if hasattr(paths, "get_rule_summary_path") else _advisory_family_path(run_id, "rule_summary"),
        ),
        "explanation_bundle": _first_existing_json(
            artifacts.get("explanation_bundle_json", ""),
            paths.get_explanation_bundle_path(run_id) if hasattr(paths, "get_explanation_bundle_path") else _advisory_family_path(run_id, "explanation_bundle"),
        ),
        "explanation_lab": _first_existing_json(
            artifacts.get("explanation_lab_json", ""),
            paths.get_explanation_lab_path(run_id) if hasattr(paths, "get_explanation_lab_path") else _advisory_family_path(run_id, "explanation_lab"),
        ),
        "narration_cache": _first_existing_json(
            artifacts.get("narration_cache_json", ""),
            paths.get_narration_cache_path(run_id) if hasattr(paths, "get_narration_cache_path") else _advisory_family_path(run_id, "narration_cache"),
        ),
    }


def load_full_run_bundle(run_id: str) -> Optional[Dict[str, Any]]:
    return load_advisory_bundle(run_id)


def persist_portfolio_advisory_bundle(
    *,
    session_id: str,
    run_id: str,
    market: str,
    iteration: int,
    trade_window: Dict[str, Any],
    selected_model: str,
    run_level_metrics: Dict[str, Any],
    allocation_recommendation: Any,
    benchmark_comparison: Any,
    risk_snapshot: Any,
    investor_profile: Any,
    portfolio_policy: Any,
    explanation_summary: str,
    xai_payload: Optional[Any] = None,
    advisory_summary_text: str = "",
    advisory_summary_payload: Optional[Any] = None,
    scenario_result: Optional[Any] = None,
    benchmark_series: Optional[Any] = None,
    active_return_metrics: Optional[Any] = None,
    validation_csv_path: str = "",
    created_at: Optional[str] = None,
    engine_info: Optional[Any] = None,
    policy_check: Optional[Any] = None,
    shap_payload: Optional[Any] = None,
    lime_payload: Optional[Any] = None,
    rule_summary_payload: Optional[Any] = None,
    explanation_bundle_payload: Optional[Any] = None,
    explanation_lab_payload: Optional[Any] = None,
    narration_cache_payload: Optional[Any] = None,
) -> Dict[str, Any]:
    allocation_json = write_allocation_recommendation(run_id, allocation_recommendation)
    benchmark_json = write_benchmark_comparison(run_id, benchmark_comparison)
    risk_json = write_risk_snapshot(run_id, risk_snapshot)
    investor_profile_json = write_investor_profile(run_id, investor_profile)
    xai_json = write_xai_json(run_id, xai_payload or {})
    xai_text = write_xai_text(run_id, explanation_summary)

    advisory_summary_json = write_advisory_summary_json(
        run_id,
        advisory_summary_payload or {"run_id": run_id, "summary": advisory_summary_text or explanation_summary},
    )
    advisory_summary_text_path = write_advisory_summary_text(run_id, advisory_summary_text or explanation_summary)

    benchmark_series_json = ""
    active_return_metrics_json = ""
    scenario_json = ""
    engine_info_json = ""
    policy_check_json = ""
    shap_json = ""
    lime_json = ""
    rule_summary_json = ""
    explanation_bundle_json = ""
    explanation_lab_json = ""
    narration_cache_json = ""

    if benchmark_series is not None:
        benchmark_series_json = write_benchmark_series(run_id, benchmark_series)

    if active_return_metrics is not None:
        active_return_metrics_json = write_active_return_metrics(run_id, active_return_metrics)

    if scenario_result is not None:
        serialized_scenario = _serialize(scenario_result) or {}
        scenario_name = str(serialized_scenario.get("request", {}).get("scenario_name", "base"))
        scenario_json = write_scenario_result(run_id, scenario_result, scenario_name=scenario_name)

    if engine_info is not None:
        engine_info_json = write_engine_info(run_id, engine_info)

    if policy_check is not None:
        policy_check_json = write_policy_check(run_id, policy_check)

    if shap_payload is not None:
        shap_json = write_shap_explanation(run_id, shap_payload)

    if lime_payload is not None:
        lime_json = write_lime_explanation(run_id, lime_payload)

    if rule_summary_payload is not None:
        rule_summary_json = write_rule_summary(run_id, rule_summary_payload)

    if explanation_bundle_payload is not None:
        explanation_bundle_json = write_explanation_bundle(run_id, explanation_bundle_payload)

    if explanation_lab_payload is not None:
        explanation_lab_json = write_explanation_lab(run_id, explanation_lab_payload)

    if narration_cache_payload is not None:
        narration_cache_json = write_narration_cache(run_id, narration_cache_payload)

    engine_info_obj = ExecutionEngineInfo.from_dict(
        _serialize(engine_info)
        or {
            "engine_name": "legacy_rl",
            "backend_type": "sb3",
            "algorithm_name": selected_model,
            "train_config": {},
            "engine_notes": "",
        }
    )
    policy_check_obj = PolicyCheckResult.from_dict(_serialize(policy_check) or {})
    explanation_bundle_obj = ExplanationBundle.from_dict(_serialize(explanation_bundle_payload) or {})
    explanation_lab_obj = ExplanationLabReport.from_dict(_serialize(explanation_lab_payload) or {})

    artifacts = RunArtifacts(
        allocation_json=allocation_json,
        benchmark_json=benchmark_json,
        risk_snapshot_json=risk_json,
        investor_profile_json=investor_profile_json,
        scenario_result_json=scenario_json,
        advisory_summary_json=advisory_summary_json,
        advisory_summary_text=advisory_summary_text_path,
        xai_json=xai_json,
        xai_text=xai_text,
        validation_csv=validation_csv_path or "",
        benchmark_series_json=benchmark_series_json,
        active_return_metrics_json=active_return_metrics_json,
        engine_info_json=engine_info_json,
        policy_check_json=policy_check_json,
        shap_json=shap_json,
        lime_json=lime_json,
        rule_summary_json=rule_summary_json,
        explanation_bundle_json=explanation_bundle_json,
        explanation_lab_json=explanation_lab_json,
        narration_cache_json=narration_cache_json,
    )

    record = RunManifestRecord(
        session_id=session_id,
        run_id=run_id,
        created_at=created_at or _utc_now_z(),
        market=market or "etf30",
        iteration=int(iteration),
        trade_window=TradeWindow.from_dict(trade_window),
        selected_model=selected_model,
        engine_info=engine_info_obj,
        run_level_metrics=dict(run_level_metrics or {}),
        allocation_recommendation=AllocationRecommendation.from_dict(_serialize(allocation_recommendation)),
        benchmark_comparison=BenchmarkComparison.from_dict(_serialize(benchmark_comparison)),
        risk_snapshot=RiskSnapshot.from_dict(_serialize(risk_snapshot)),
        investor_profile=InvestorProfile.from_dict(_serialize(investor_profile)),
        portfolio_policy=PortfolioPolicy.from_dict(_serialize(portfolio_policy)),
        policy_check=policy_check_obj,
        explanation_summary=(explanation_summary or "").strip(),
        explanation_bundle=explanation_bundle_obj,
        explanation_lab=explanation_lab_obj,
        artifacts=artifacts,
    )

    manifest = record.to_dict()
    write_run_manifest(manifest)
    append_run_to_session(
        session_id=session_id,
        market=market,
        run_id=run_id,
        investor_profile=investor_profile,
    )
    return manifest
