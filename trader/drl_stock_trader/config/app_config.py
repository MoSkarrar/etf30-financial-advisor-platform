from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


DEFAULT_UNIVERSE = "ETF30"


@dataclass(frozen=True)
class AppIdentityConfig:
    app_title: str = "ETF30 Financial Advisor"
    market_key: str = "etf30"
    default_page_name: str = "home"
    default_universe: str = DEFAULT_UNIVERSE


@dataclass(frozen=True)
class DatasetConfig:
    etf30_filename: str = "done_data_etf30_yf.csv"
    build_start: str = "2014-01-01"
    build_end: str = "2025-12-31"
    force_rebuild: bool = False
    covariance_cache_filename: str = "done_data_etf30_covariance.csv"
    benchmark_cache_filename: str = "done_data_etf30_benchmarks.csv"
    min_common_days: int = 260
    turbulence_window: int = 252
    covariance_window: int = 20
    downside_window: int = 20
    stable_universe_only: bool = True


@dataclass(frozen=True)
class BenchmarkConfig:
    default_primary: str = "equal_weight"
    available: Tuple[str, ...] = ("equal_weight", "spy", "60_40")
    spy_ticker: str = "SPY"
    sixty_forty_equity_weight: float = 0.60
    sixty_forty_bond_proxy: str = "IEF"


@dataclass(frozen=True)
class InvestorProfileDefaults:
    profile_name: str = "moderate"
    risk_tolerance: float = 0.55
    target_volatility: float = 0.16
    max_drawdown_preference: float = 0.20
    turnover_aversion: float = 0.60
    min_cash_preference: float = 0.05
    benchmark_preference: str = "equal_weight"
    target_style: str = "balanced"
    advisor_mode: str = "advisor"


@dataclass(frozen=True)
class PortfolioPolicyConfig:
    max_single_position_cap: float = 0.12
    target_cash_floor: float = 0.05
    turnover_budget: float = 0.35
    rebalance_cadence_days: int = 21
    validation_window: int = 63
    training_window_step: int = 63
    allow_cash_sleeve: bool = True
    long_only: bool = True
    soft_sector_cap: float = 0.35


@dataclass(frozen=True)
class AdvisoryRewardWeights:
    portfolio_return_weight: float = 1.00
    benchmark_excess_weight: float = 0.35
    target_volatility_penalty: float = 0.30
    turnover_penalty: float = 0.20
    concentration_penalty: float = 0.35
    drawdown_penalty: float = 0.45
    cash_floor_penalty: float = 0.20
    policy_breach_penalty: float = 0.50
    confidence_bonus_weight: float = 0.05


@dataclass(frozen=True)
class TimestepsProfile:
    a2c: int
    ppo: int
    ddpg: int


@dataclass(frozen=True)
class RLExecutionConfig:
    supported_markets: Tuple[str, ...] = ("etf30",)
    stock_dim: int = 30
    action_dim_with_cash: int = 31
    rebalance_window: int = 63
    validation_window: int = 63
    transaction_cost_pct: float = 0.001
    reward_scale: float = 1e-4
    default_initial_cash: float = 1_000_000.0
    benchmark_relative_reward: bool = True
    covariance_window: int = 20
    semivariance_window: int = 20
    momentum_windows: Tuple[int, ...] = (5, 21, 63)
    feature_columns: Tuple[str, ...] = (
        "macd", "rsi", "cci", "adx", "ret_1", "ret_5", "ret_21", "momentum_21",
        "momentum_63", "realized_vol_20", "downside_vol_20", "corr_proxy_20",
    )
    enabled_algorithms: Tuple[str, ...] = ("A2C", "PPO", "DDPG")
    ddpg_candidate_buffers: Tuple[int, ...] = (200_000, 100_000, 50_000, 20_000)
    ddpg_batch_size: int = 128
    ddpg_sigma: float = 0.10
    ddpg_learning_starts_cap: int = 10_000
    ddpg_learning_starts_floor: int = 1_000
    robustness_profiles: Dict[int, TimestepsProfile] = field(
        default_factory=lambda: {
            1: TimestepsProfile(a2c=150, ppo=150, ddpg=70),
            2: TimestepsProfile(a2c=300, ppo=300, ddpg=150),
            3: TimestepsProfile(a2c=600, ppo=600, ddpg=250),
        }
    )

    def timesteps_for_robustness(self, robustness: int) -> TimestepsProfile:
        try:
            key = int(robustness)
        except Exception:
            key = 3
        return self.robustness_profiles.get(key, self.robustness_profiles[3])


@dataclass(frozen=True)
class EngineConfig:
    enable_finrl: bool = True
    default_engine: str = "legacy_rl"
    allowed_engines: Tuple[str, ...] = ("legacy_rl", "finrl")
    enable_engine_fallback: bool = True


@dataclass(frozen=True)
class RiskFeatureFlags:
    enable_risk_overlay: bool = False
    risk_penalty_vol: float = 0.0
    risk_penalty_dd: float = 0.0
    risk_penalty_turnover: float = 0.0
    risk_penalty_concentration: float = 0.0
    max_position_weight: float = 0.12
    min_cash_weight: float = 0.05
    max_turnover_per_rebalance: float = 0.35


@dataclass(frozen=True)
class XAIConfig:
    enable_shap: bool = False
    enable_lime: bool = False
    enable_rule_summary: bool = True
    enable_explanation_lab: bool = True
    shap_top_k: int = 8
    lime_num_features: int = 8
    lime_num_samples: int = 500
    explanation_text_max_chars: int = 8000


@dataclass(frozen=True)
class OllamaConfig:
    base_url: str = "http://127.0.0.1:11434"
    model: str = "gpt-oss:120b-cloud"
    connect_timeout_s: int = 3
    read_timeout_s: int = 180
    timeout_s: int = 180
    temperature: float = 0.0
    top_p: float = 0.95
    num_predict: int = 1400
    chat_num_predict: int = 1400
    repeat_penalty: float = 1.08


@dataclass(frozen=True)
class OllamaModeConfig:
    default_mode: str = "advisor_summary"
    technical_mode: str = "technical_xai"
    client_mode: str = "advisor_summary"
    risk_mode: str = "risk_committee"
    include_xai_raw: bool = True
    include_explanation_lab: bool = True
    include_policy_breaches: bool = True


@dataclass(frozen=True)
class Page2NarrationConfig:
    default_style: str = "advisor"
    include_benchmark_context: bool = True
    include_policy_context: bool = True
    max_context_chars: int = 50000
    max_history_turns: int = 8
    summary_sentence_limit: int = 50
    default_question: str = "Explain the allocation, risks, and benchmark trade-offs."


@dataclass(frozen=True)
class SelectionConfig:
    sharpe_weight: float = 0.30
    active_return_weight: float = 0.20
    drawdown_weight: float = 0.18
    turnover_weight: float = 0.10
    concentration_weight: float = 0.10
    stability_weight: float = 0.07
    policy_compliance_weight: float = 0.05


@dataclass(frozen=True)
class PersistenceConfig:
    results_dirname: str = "results"
    manifests_dirname: str = "run_manifests"
    trained_models_dirname: str = "trained_models"
    portfolio_dirname: str = "portfolio_allocation"
    benchmark_dirname: str = "benchmarks"
    advisory_dirname: str = "advisory"
    scenario_dirname: str = "scenarios"
    risk_dirname: str = "risk"
    snapshots_dirname: str = "snapshots"
    engines_dirname: str = "engines"
    explainability_dirname: str = "explainability"
    narration_dirname: str = "narration"

    session_manifest_prefix: str = "session_manifest"
    run_manifest_prefix: str = "run_manifest"
    xai_text_prefix: str = "xai_text"
    allocation_prefix: str = "allocation"
    benchmark_prefix: str = "benchmark"
    advisory_summary_prefix: str = "advisory_summary"
    allocation_snapshot_prefix: str = "allocation_snapshot"
    risk_report_prefix: str = "risk_report"
    scenario_prefix: str = "scenario"
    validation_prefix: str = "validation"

    engine_info_prefix: str = "engine_info"
    policy_check_prefix: str = "policy_check"
    shap_prefix: str = "shap"
    lime_prefix: str = "lime"
    rule_summary_prefix: str = "rule_summary"
    explanation_bundle_prefix: str = "explanation_bundle"
    explanation_lab_prefix: str = "explanation_lab"
    benchmark_series_prefix: str = "benchmark_series"
    active_return_metrics_prefix: str = "active_return_metrics"
    narration_cache_prefix: str = "narration_cache"
    investor_profile_prefix: str = "investor_profile"
    advisory_summary_text_suffix: str = "txt"


@dataclass(frozen=True)
class AppConfig:
    identity: AppIdentityConfig = field(default_factory=AppIdentityConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    benchmarks: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    investor_defaults: InvestorProfileDefaults = field(default_factory=InvestorProfileDefaults)
    policy: PortfolioPolicyConfig = field(default_factory=PortfolioPolicyConfig)
    advisory_reward: AdvisoryRewardWeights = field(default_factory=AdvisoryRewardWeights)
    rl: RLExecutionConfig = field(default_factory=RLExecutionConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    risk_flags: RiskFeatureFlags = field(default_factory=RiskFeatureFlags)
    xai: XAIConfig = field(default_factory=XAIConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    ollama_modes: OllamaModeConfig = field(default_factory=OllamaModeConfig)
    narration: Page2NarrationConfig = field(default_factory=Page2NarrationConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)


APP_CONFIG = AppConfig()
