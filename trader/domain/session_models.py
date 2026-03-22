from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, List, Optional, Tuple


class SerializableDataclass:
    def to_dict(self) -> Dict[str, Any]:
        def _convert(value: Any) -> Any:
            if is_dataclass(value):
                return {k: _convert(v) for k, v in asdict(value).items()}
            if isinstance(value, list):
                return [_convert(v) for v in value]
            if isinstance(value, tuple):
                return [_convert(v) for v in value]
            if isinstance(value, dict):
                return {str(k): _convert(v) for k, v in value.items()}
            return value

        return _convert(self)

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)


@dataclass
class TradeWindow(SerializableDataclass):
    start: str = ""
    end: str = ""

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "TradeWindow":
        data = data or {}
        return cls(start=str(data.get("start", "")), end=str(data.get("end", "")))


@dataclass
class InvestorProfile(SerializableDataclass):
    profile_name: str = "moderate"
    target_style: str = "balanced"
    risk_tolerance: float = 0.55
    target_volatility: float = 0.16
    max_drawdown_preference: float = 0.20
    turnover_aversion: float = 0.60
    min_cash_preference: float = 0.05
    benchmark_preference: str = "equal_weight"
    advisor_mode: str = "advisor"

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "InvestorProfile":
        data = data or {}
        return cls(
            profile_name=str(data.get("profile_name", "moderate")),
            target_style=str(data.get("target_style", "balanced")),
            risk_tolerance=float(data.get("risk_tolerance", 0.55)),
            target_volatility=float(data.get("target_volatility", 0.16)),
            max_drawdown_preference=float(data.get("max_drawdown_preference", 0.20)),
            turnover_aversion=float(data.get("turnover_aversion", 0.60)),
            min_cash_preference=float(data.get("min_cash_preference", 0.05)),
            benchmark_preference=str(data.get("benchmark_preference", "equal_weight")),
            advisor_mode=str(data.get("advisor_mode", "advisor")),
        )


@dataclass
class PortfolioPolicy(SerializableDataclass):
    max_single_position_cap: float = 0.12
    min_cash_weight: float = 0.05
    turnover_budget: float = 0.35
    rebalance_cadence_days: int = 21
    long_only: bool = True
    allow_cash_sleeve: bool = True
    sector_caps: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "PortfolioPolicy":
        data = data or {}
        sector_caps = data.get("sector_caps") if isinstance(data.get("sector_caps"), dict) else {}
        return cls(
            max_single_position_cap=float(data.get("max_single_position_cap", 0.12)),
            min_cash_weight=float(data.get("min_cash_weight", 0.05)),
            turnover_budget=float(data.get("turnover_budget", 0.35)),
            rebalance_cadence_days=int(data.get("rebalance_cadence_days", 21)),
            long_only=bool(data.get("long_only", True)),
            allow_cash_sleeve=bool(data.get("allow_cash_sleeve", True)),
            sector_caps={str(k): float(v) for k, v in sector_caps.items()},
        )


@dataclass
class RiskSnapshot(SerializableDataclass):
    realized_volatility: float = 0.0
    downside_volatility: float = 0.0
    max_drawdown: float = 0.0
    concentration_hhi: float = 0.0
    turnover: float = 0.0
    tracking_error: float = 0.0
    cash_weight: float = 0.0

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "RiskSnapshot":
        data = data or {}
        return cls(
            realized_volatility=float(data.get("realized_volatility", 0.0)),
            downside_volatility=float(data.get("downside_volatility", 0.0)),
            max_drawdown=float(data.get("max_drawdown", 0.0)),
            concentration_hhi=float(data.get("concentration_hhi", 0.0)),
            turnover=float(data.get("turnover", 0.0)),
            tracking_error=float(data.get("tracking_error", 0.0)),
            cash_weight=float(data.get("cash_weight", 0.0)),
        )


@dataclass
class BenchmarkComparison(SerializableDataclass):
    benchmark_name: str = "equal_weight"
    benchmark_return: float = 0.0
    portfolio_return: float = 0.0
    active_return: float = 0.0
    tracking_error: float = 0.0
    information_ratio: float = 0.0
    benchmark_series_path: str = ""
    active_return_metrics_path: str = ""

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "BenchmarkComparison":
        data = data or {}
        return cls(
            benchmark_name=str(data.get("benchmark_name", "equal_weight")),
            benchmark_return=float(data.get("benchmark_return", 0.0)),
            portfolio_return=float(data.get("portfolio_return", 0.0)),
            active_return=float(data.get("active_return", 0.0)),
            tracking_error=float(data.get("tracking_error", 0.0)),
            information_ratio=float(data.get("information_ratio", 0.0)),
            benchmark_series_path=str(data.get("benchmark_series_path", "")),
            active_return_metrics_path=str(data.get("active_return_metrics_path", "")),
        )


@dataclass
class AllocationRecommendation(SerializableDataclass):
    run_id: str = ""
    as_of_date: str = ""
    target_weights: Dict[str, float] = field(default_factory=dict)
    previous_weights: Dict[str, float] = field(default_factory=dict)
    rebalance_deltas: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    rationale_summary: str = ""
    policy_breaches: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "AllocationRecommendation":
        data = data or {}
        return cls(
            run_id=str(data.get("run_id", "")),
            as_of_date=str(data.get("as_of_date", "")),
            target_weights={str(k): float(v) for k, v in (data.get("target_weights") or {}).items()},
            previous_weights={str(k): float(v) for k, v in (data.get("previous_weights") or {}).items()},
            rebalance_deltas={str(k): float(v) for k, v in (data.get("rebalance_deltas") or {}).items()},
            confidence=float(data.get("confidence", 0.0)),
            rationale_summary=str(data.get("rationale_summary", "")),
            policy_breaches=[str(v) for v in (data.get("policy_breaches") or [])],
        )


@dataclass
class ScenarioRequest(SerializableDataclass):
    scenario_name: str = "base"
    shock_type: str = "none"
    shock_magnitude: float = 0.0
    horizon_days: int = 21
    benchmark_name: str = "equal_weight"
    max_single_position_cap: Optional[float] = None
    min_cash_weight: Optional[float] = None
    profile_name: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ScenarioRequest":
        data = data or {}
        return cls(
            scenario_name=str(data.get("scenario_name", "base")),
            shock_type=str(data.get("shock_type", "none")),
            shock_magnitude=float(data.get("shock_magnitude", 0.0)),
            horizon_days=int(data.get("horizon_days", 21)),
            benchmark_name=str(data.get("benchmark_name", "equal_weight")),
            max_single_position_cap=float(data["max_single_position_cap"]) if data.get("max_single_position_cap") is not None else None,
            min_cash_weight=float(data["min_cash_weight"]) if data.get("min_cash_weight") is not None else None,
            profile_name=str(data["profile_name"]) if data.get("profile_name") is not None else None,
        )


@dataclass
class ScenarioResult(SerializableDataclass):
    request: ScenarioRequest = field(default_factory=ScenarioRequest)
    projected_return: float = 0.0
    projected_volatility: float = 0.0
    projected_drawdown: float = 0.0
    narrative: str = ""

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ScenarioResult":
        data = data or {}
        return cls(
            request=ScenarioRequest.from_dict(data.get("request")),
            projected_return=float(data.get("projected_return", 0.0)),
            projected_volatility=float(data.get("projected_volatility", 0.0)),
            projected_drawdown=float(data.get("projected_drawdown", 0.0)),
            narrative=str(data.get("narrative", "")),
        )


@dataclass
class ExecutionEngineInfo(SerializableDataclass):
    engine_name: str = "legacy_rl"
    backend_type: str = "sb3"
    algorithm_name: str = ""
    train_config: Dict[str, Any] = field(default_factory=dict)
    engine_notes: str = ""

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ExecutionEngineInfo":
        data = data or {}
        return cls(
            engine_name=str(data.get("engine_name", "legacy_rl")),
            backend_type=str(data.get("backend_type", "sb3")),
            algorithm_name=str(data.get("algorithm_name", "")),
            train_config=dict(data.get("train_config") or {}),
            engine_notes=str(data.get("engine_notes", "")),
        )


@dataclass
class PolicyCheckResult(SerializableDataclass):
    passed: bool = True
    breaches: List[str] = field(default_factory=list)
    severity: str = "none"
    applied_clips: List[Dict[str, Any]] = field(default_factory=list)
    human_summary: str = ""

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "PolicyCheckResult":
        data = data or {}
        return cls(
            passed=bool(data.get("passed", True)),
            breaches=[str(v) for v in (data.get("breaches") or [])],
            severity=str(data.get("severity", "none")),
            applied_clips=[dict(v) for v in (data.get("applied_clips") or [])],
            human_summary=str(data.get("human_summary", "")),
        )


@dataclass
class ShapExplanation(SerializableDataclass):
    method: str = ""
    background_size: int = 0
    top_features: List[str] = field(default_factory=list)
    feature_values: Dict[str, Any] = field(default_factory=dict)
    attribution_values: Dict[str, float] = field(default_factory=dict)
    summary_text: str = ""

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ShapExplanation":
        data = data or {}
        return cls(
            method=str(data.get("method", "")),
            background_size=int(data.get("background_size", 0)),
            top_features=[str(v) for v in (data.get("top_features") or [])],
            feature_values=dict(data.get("feature_values") or {}),
            attribution_values={str(k): float(v) for k, v in (data.get("attribution_values") or {}).items()},
            summary_text=str(data.get("summary_text", "")),
        )


@dataclass
class LimeExplanation(SerializableDataclass):
    instance_id: str = ""
    top_local_features: List[str] = field(default_factory=list)
    local_weights: Dict[str, float] = field(default_factory=dict)
    local_prediction_context: Dict[str, Any] = field(default_factory=dict)
    summary_text: str = ""

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "LimeExplanation":
        data = data or {}
        return cls(
            instance_id=str(data.get("instance_id", "")),
            top_local_features=[str(v) for v in (data.get("top_local_features") or [])],
            local_weights={str(k): float(v) for k, v in (data.get("local_weights") or {}).items()},
            local_prediction_context=dict(data.get("local_prediction_context") or {}),
            summary_text=str(data.get("summary_text", "")),
        )


@dataclass
class RuleExplanation(SerializableDataclass):
    rules_triggered: List[str] = field(default_factory=list)
    risk_flags: List[str] = field(default_factory=list)
    allocation_flags: List[str] = field(default_factory=list)
    benchmark_flags: List[str] = field(default_factory=list)
    summary_text: str = ""

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "RuleExplanation":
        data = data or {}
        return cls(
            rules_triggered=[str(v) for v in (data.get("rules_triggered") or [])],
            risk_flags=[str(v) for v in (data.get("risk_flags") or [])],
            allocation_flags=[str(v) for v in (data.get("allocation_flags") or [])],
            benchmark_flags=[str(v) for v in (data.get("benchmark_flags") or [])],
            summary_text=str(data.get("summary_text", "")),
        )


@dataclass
class ExplanationBundle(SerializableDataclass):
    shap: Dict[str, Any] = field(default_factory=dict)
    lime: Dict[str, Any] = field(default_factory=dict)
    rule_summary: Dict[str, Any] = field(default_factory=dict)
    consensus_points: List[str] = field(default_factory=list)
    disagreement_points: List[str] = field(default_factory=list)
    advisor_summary: str = ""
    technical_summary: str = ""

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ExplanationBundle":
        data = data or {}
        return cls(
            shap=dict(data.get("shap") or {}),
            lime=dict(data.get("lime") or {}),
            rule_summary=dict(data.get("rule_summary") or {}),
            consensus_points=[str(v) for v in (data.get("consensus_points") or [])],
            disagreement_points=[str(v) for v in (data.get("disagreement_points") or [])],
            advisor_summary=str(data.get("advisor_summary", "")),
            technical_summary=str(data.get("technical_summary", "")),
        )


@dataclass
class ExplanationLabReport(SerializableDataclass):
    hypotheses: List[str] = field(default_factory=list)
    cross_method_agreement: float = 0.0
    confidence_score: float = 0.0
    contradictions: List[str] = field(default_factory=list)
    final_interpretation: str = ""
    open_questions: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ExplanationLabReport":
        data = data or {}
        return cls(
            hypotheses=[str(v) for v in (data.get("hypotheses") or [])],
            cross_method_agreement=float(data.get("cross_method_agreement", 0.0)),
            confidence_score=float(data.get("confidence_score", 0.0)),
            contradictions=[str(v) for v in (data.get("contradictions") or [])],
            final_interpretation=str(data.get("final_interpretation", "")),
            open_questions=[str(v) for v in (data.get("open_questions") or [])],
        )


@dataclass
class AdvisorNarrativeBundle(SerializableDataclass):
    client_friendly: str = ""
    advisor_level: str = ""
    technical_level: str = ""
    risk_committee_level: str = ""

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "AdvisorNarrativeBundle":
        data = data or {}
        return cls(
            client_friendly=str(data.get("client_friendly", "")),
            advisor_level=str(data.get("advisor_level", "")),
            technical_level=str(data.get("technical_level", "")),
            risk_committee_level=str(data.get("risk_committee_level", "")),
        )


@dataclass
class NarrationContextRecord(SerializableDataclass):
    session_id: str = ""
    run_id: str = ""
    facts: Dict[str, Any] = field(default_factory=dict)
    xai_excerpt: str = ""
    history: List[Tuple[str, str]] = field(default_factory=list)

    def to_prompt(self) -> str:
        return json.dumps(self.facts, ensure_ascii=False) + "\n\nXAI:\n" + self.xai_excerpt


@dataclass
class RunArtifacts(SerializableDataclass):
    allocation_json: str = ""
    benchmark_json: str = ""
    risk_snapshot_json: str = ""
    investor_profile_json: str = ""
    scenario_result_json: str = ""
    advisory_summary_json: str = ""
    advisory_summary_text: str = ""
    xai_json: str = ""
    xai_text: str = ""
    validation_csv: str = ""
    benchmark_series_json: str = ""
    active_return_metrics_json: str = ""
    engine_info_json: str = ""
    policy_check_json: str = ""
    shap_json: str = ""
    lime_json: str = ""
    rule_summary_json: str = ""
    explanation_bundle_json: str = ""
    explanation_lab_json: str = ""
    narration_cache_json: str = ""

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "RunArtifacts":
        data = data or {}
        return cls(
            allocation_json=str(data.get("allocation_json", "")),
            benchmark_json=str(data.get("benchmark_json", "")),
            risk_snapshot_json=str(data.get("risk_snapshot_json", "")),
            investor_profile_json=str(data.get("investor_profile_json", "")),
            scenario_result_json=str(data.get("scenario_result_json", "")),
            advisory_summary_json=str(data.get("advisory_summary_json", "")),
            advisory_summary_text=str(data.get("advisory_summary_text", "")),
            xai_json=str(data.get("xai_json", "")),
            xai_text=str(data.get("xai_text", "")),
            validation_csv=str(data.get("validation_csv", "")),
            benchmark_series_json=str(data.get("benchmark_series_json", "")),
            active_return_metrics_json=str(data.get("active_return_metrics_json", "")),
            engine_info_json=str(data.get("engine_info_json", "")),
            policy_check_json=str(data.get("policy_check_json", "")),
            shap_json=str(data.get("shap_json", "")),
            lime_json=str(data.get("lime_json", "")),
            rule_summary_json=str(data.get("rule_summary_json", "")),
            explanation_bundle_json=str(data.get("explanation_bundle_json", "")),
            explanation_lab_json=str(data.get("explanation_lab_json", "")),
            narration_cache_json=str(data.get("narration_cache_json", "")),
        )


@dataclass
class RunManifestRecord(SerializableDataclass):
    session_id: str = ""
    run_id: str = ""
    created_at: str = ""
    market: str = "etf30"
    iteration: int = 0
    trade_window: TradeWindow = field(default_factory=TradeWindow)
    selected_model: str = ""
    engine_info: ExecutionEngineInfo = field(default_factory=ExecutionEngineInfo)
    run_level_metrics: Dict[str, Any] = field(default_factory=dict)
    allocation_recommendation: AllocationRecommendation = field(default_factory=AllocationRecommendation)
    benchmark_comparison: BenchmarkComparison = field(default_factory=BenchmarkComparison)
    risk_snapshot: RiskSnapshot = field(default_factory=RiskSnapshot)
    investor_profile: InvestorProfile = field(default_factory=InvestorProfile)
    portfolio_policy: PortfolioPolicy = field(default_factory=PortfolioPolicy)
    policy_check: PolicyCheckResult = field(default_factory=PolicyCheckResult)
    explanation_summary: str = ""
    explanation_bundle: ExplanationBundle = field(default_factory=ExplanationBundle)
    explanation_lab: ExplanationLabReport = field(default_factory=ExplanationLabReport)
    advisory_narrative: AdvisorNarrativeBundle = field(default_factory=AdvisorNarrativeBundle)
    artifacts: RunArtifacts = field(default_factory=RunArtifacts)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunManifestRecord":
        data = data or {}
        return cls(
            session_id=str(data.get("session_id", "")),
            run_id=str(data.get("run_id", "")),
            created_at=str(data.get("created_at", "")),
            market=str(data.get("market", "etf30")),
            iteration=int(data.get("iteration", 0)),
            trade_window=TradeWindow.from_dict(data.get("trade_window")),
            selected_model=str(data.get("selected_model", "")),
            engine_info=ExecutionEngineInfo.from_dict(data.get("engine_info")),
            run_level_metrics=dict(data.get("run_level_metrics") or {}),
            allocation_recommendation=AllocationRecommendation.from_dict(data.get("allocation_recommendation")),
            benchmark_comparison=BenchmarkComparison.from_dict(data.get("benchmark_comparison")),
            risk_snapshot=RiskSnapshot.from_dict(data.get("risk_snapshot")),
            investor_profile=InvestorProfile.from_dict(data.get("investor_profile")),
            portfolio_policy=PortfolioPolicy.from_dict(data.get("portfolio_policy")),
            policy_check=PolicyCheckResult.from_dict(data.get("policy_check")),
            explanation_summary=str(data.get("explanation_summary", "")),
            explanation_bundle=ExplanationBundle.from_dict(data.get("explanation_bundle")),
            explanation_lab=ExplanationLabReport.from_dict(data.get("explanation_lab")),
            advisory_narrative=AdvisorNarrativeBundle.from_dict(data.get("advisory_narrative")),
            artifacts=RunArtifacts.from_dict(data.get("artifacts")),
        )

    @property
    def xai_text_path(self) -> str:
        return self.artifacts.xai_text

    def compact_facts(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "run_id": self.run_id,
            "market": self.market,
            "iteration": self.iteration,
            "trade_window": self.trade_window.to_dict(),
            "selected_model": self.selected_model,
            "risk_snapshot": self.risk_snapshot.to_dict(),
            "benchmark_comparison": self.benchmark_comparison.to_dict(),
            "created_at": self.created_at,
        }


@dataclass
class SessionManifestRecord(SerializableDataclass):
    session_id: str = ""
    market: str = "etf30"
    created_at: str = ""
    runs: List[str] = field(default_factory=list)
    latest_run_id: str = ""
    investor_profile: InvestorProfile = field(default_factory=InvestorProfile)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "SessionManifestRecord":
        data = data or {}
        return cls(
            session_id=str(data.get("session_id", "")),
            market=str(data.get("market", "etf30")),
            created_at=str(data.get("created_at", "")),
            runs=[str(v) for v in (data.get("runs") or [])],
            latest_run_id=str(data.get("latest_run_id", "")),
            investor_profile=InvestorProfile.from_dict(data.get("investor_profile")),
        )

    def add_run(self, run_id: str) -> None:
        run_id = str(run_id)
        if run_id not in self.runs:
            self.runs.append(run_id)
        self.latest_run_id = run_id


# Backward-compatible alias for older code that imported SessionRecord.
SessionRecord = SessionManifestRecord
