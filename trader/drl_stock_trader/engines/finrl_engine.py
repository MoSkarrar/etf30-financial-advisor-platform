from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from trader.domain.session_models import ExecutionEngineInfo
from trader.drl_stock_trader.engines.legacy_rl_engine import EngineIterationResult, LegacyEngine


@dataclass
class FinRLEngine:
    name: str = "finrl"

    def _can_run_native_finrl(self) -> bool:
        try:
            import finrl  # noqa: F401
            return True
        except Exception:
            return False

    def build_engine_metadata(self, mandate, winner_label: str, notes: str = "") -> ExecutionEngineInfo:
        return ExecutionEngineInfo(
            engine_name=self.name,
            backend_type="finrl",
            algorithm_name=str(winner_label or ""),
            train_config={
                "robustness": int(mandate.robustness),
                "rebalance_cadence_days": int(mandate.rebalance_cadence_days),
                "benchmark_choice": str(mandate.benchmark_choice),
                "risk_mode": str(getattr(mandate, "risk_mode", "standard")),
                "explanation_depth": str(getattr(mandate, "explanation_depth", "standard")),
            },
            engine_notes=notes or "FinRL adapter placeholder executed.",
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
    ) -> EngineIterationResult:
        native_finrl_available = self._can_run_native_finrl()

        # Safe first-pass implementation: normalize FinRL selection through the same output schema.
        # If native FinRL is unavailable, delegate to legacy engine while preserving engine metadata.
        delegated = LegacyEngine().train_validate_trade(
            socket=socket,
            mandate=mandate,
            market=market,
            prepared=prepared,
            bundle=bundle,
            iteration=iteration,
            previous_weights_vector=previous_weights_vector,
            engine_notes=(
                "FinRL requested but native FinRL backend is not available in this environment; "
                "delegated to legacy engine with normalized outputs."
                if not native_finrl_available
                else "FinRL native backend hook point reached; using normalized legacy-compatible first pass."
            ),
        )
        delegated.engine_name = self.name
        delegated.engine_info = self.build_engine_metadata(
            mandate,
            winner_label=delegated.selected_model,
            notes=(
                "First-pass FinRL adapter returned legacy-compatible normalized outputs. "
                "Swap in native FinRL train/eval/env logic later without changing downstream orchestration."
            ),
        )
        delegated.backend_notes = delegated.engine_info.engine_notes
        return delegated
