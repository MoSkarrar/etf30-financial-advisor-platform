from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

from trader.domain.session_models import ExecutionEngineInfo
from trader.drl_stock_trader.pipeline.explain_stage import ExplainStageResult, generate_explanation
from trader.drl_stock_trader.pipeline.selection_stage import SelectionResult, select_best_candidate
from trader.drl_stock_trader.pipeline.trade_stage import TradeStageResult, execute_trade_stage
from trader.drl_stock_trader.pipeline.train_stage import CandidateModelResult, train_candidate_models
from trader.drl_stock_trader.preprocess import data_split


@dataclass
class EngineIterationResult:
    engine_name: str
    selected_model: str
    selected_candidate: CandidateModelResult
    selection: SelectionResult
    trade_result: TradeStageResult
    explain_result: ExplainStageResult
    run_level_metrics: Dict[str, Any]
    engine_info: ExecutionEngineInfo
    raw_trade_data_rows: int = 0
    backend_notes: str = ""


@dataclass
class LegacyEngine:
    name: str = "legacy_rl"

    def build_engine_metadata(self, mandate, winner: CandidateModelResult, notes: str = "") -> ExecutionEngineInfo:
        train_config = {
            "robustness": int(mandate.robustness),
            "rebalance_cadence_days": int(mandate.rebalance_cadence_days),
            "benchmark_choice": str(mandate.benchmark_choice),
            "risk_mode": str(getattr(mandate, "risk_mode", "standard")),
            "explanation_depth": str(getattr(mandate, "explanation_depth", "standard")),
        }
        return ExecutionEngineInfo(
            engine_name=self.name,
            backend_type="sb3",
            algorithm_name=str(getattr(winner, "label", "") or ""),
            train_config=train_config,
            engine_notes=notes or "Legacy SB3 ETF30 engine adapter.",
        )

    def train_validate_trade(
        self,
        *,
        socket,
        mandate,
        market: str,
        prepared,
        bundle,
        iteration: int,
        previous_weights_vector: Optional[Sequence[float]] = None,
        engine_notes: str = "",
    ) -> EngineIterationResult:
        candidates = train_candidate_models(
            socket=socket,
            train_environment=bundle.train_environment,
            validation_environment=bundle.validation_environment,
            iteration=iteration,
            market=market,
            timesteps=prepared.metadata.get("timesteps") or mandate.robustness and __import__("trader.drl_stock_trader.config.app_config", fromlist=["APP_CONFIG"]).APP_CONFIG.rl.timesteps_for_robustness(mandate.robustness),
            investor_profile=bundle.train_environment.envs[0].investor_profile if hasattr(bundle.train_environment, "envs") else None,
            policy=bundle.train_environment.envs[0].policy if hasattr(bundle.train_environment, "envs") else None,
        )
        selection = select_best_candidate(candidates)
        winner = selection.best_candidate

        trade_data = data_split(prepared.long_frame, start_date=bundle.trade_start, end_date=bundle.trade_end)
        trade_dates = sorted(int(v) for v in trade_data["datadate"].unique().tolist())
        covariance_trade = {int(d): prepared.covariance_by_date[int(d)] for d in trade_dates if int(d) in prepared.covariance_by_date}

        run_id = f"{mandate.session_id}_{iteration}"
        trade_result = execute_trade_stage(
            socket=socket,
            df=trade_data,
            covariance_by_date=covariance_trade,
            feature_columns=bundle.feature_columns,
            model=winner.model,
            run_id=run_id,
            benchmark_name=mandate.benchmark_choice,
            investor_profile=bundle.train_environment.envs[0].investor_profile if hasattr(bundle.train_environment, "envs") else None,
            policy=bundle.train_environment.envs[0].policy if hasattr(bundle.train_environment, "envs") else None,
            previous_weights=previous_weights_vector,
            strategy_name=winner.label.lower(),
        )

        current_frame = trade_data[trade_data["datadate"] == trade_data["datadate"].max()].copy()
        explain_result = generate_explanation(
            current_frame=current_frame,
            allocation_recommendation=trade_result.allocation,
            previous_weights=trade_result.previous_weights,
            benchmark_comparison=trade_result.benchmark_comparison,
            risk_snapshot=trade_result.risk_snapshot,
            portfolio_policy=getattr(mandate, "policy_settings", {}),
            feature_columns=bundle.feature_columns,
        )

        run_level_metrics = {
            "candidates": {
                c.label: c.validation_summary.__dict__ for c in candidates if c.validation_summary is not None
            },
            "selected_model": winner.label,
            "selected_validation": winner.validation_summary.__dict__ if winner.validation_summary else {},
            "engine_name": self.name,
            "ranked_candidates": [c.label for c in selection.ranked_candidates],
        }

        engine_info = self.build_engine_metadata(mandate, winner, notes=engine_notes)
        return EngineIterationResult(
            engine_name=self.name,
            selected_model=winner.label,
            selected_candidate=winner,
            selection=selection,
            trade_result=trade_result,
            explain_result=explain_result,
            run_level_metrics=run_level_metrics,
            engine_info=engine_info,
            raw_trade_data_rows=len(trade_data),
            backend_notes=engine_notes or "Legacy SB3 pipeline executed successfully.",
        )
