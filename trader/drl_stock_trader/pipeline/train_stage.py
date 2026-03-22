from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from trader.domain.session_models import InvestorProfile, PortfolioPolicy
from trader.drl_stock_trader import algorithms
from trader.drl_stock_trader.config.app_config import APP_CONFIG, TimestepsProfile

logger = logging.getLogger(__name__)


@dataclass
class CandidateModelResult:
    label: str
    model: object
    timesteps: int
    training_metadata: Dict[str, object] = field(default_factory=dict)
    validation_summary: algorithms.ValidationSummary | None = None


def _send_terminal(socket, message: str) -> None:
    algorithms._socket_log(socket, message)


def resolve_algorithm_order(algorithm_choices: Optional[Sequence[str]] = None) -> List[str]:
    choices = [str(v).upper() for v in (algorithm_choices or APP_CONFIG.rl.enabled_algorithms)]
    out = [v for v in choices if v in APP_CONFIG.rl.enabled_algorithms]
    return out or list(APP_CONFIG.rl.enabled_algorithms)


def train_candidate_models(
    socket,
    train_environment,
    validation_environment,
    iteration: int,
    market: str,
    timesteps: TimestepsProfile,
    algorithm_choices: Optional[Sequence[str]] = None,
    investor_profile: Optional[InvestorProfile] = None,
    policy: Optional[PortfolioPolicy] = None,
) -> List[CandidateModelResult]:
    investor_profile = investor_profile or InvestorProfile()
    policy = policy or PortfolioPolicy()
    candidates: List[CandidateModelResult] = []
    skipped_algorithms: Dict[str, str] = {}

    timestep_map = {
        "A2C": int(timesteps.a2c),
        "PPO": int(timesteps.ppo),
        "DDPG": int(timesteps.ddpg),
    }

    for label in resolve_algorithm_order(algorithm_choices):
        try:
            _send_terminal(socket, f"====== {label} Training ======")

            model = algorithms.train_algorithm(
                socket=socket,
                env_train=train_environment,
                algorithm_name=label,
                model_name=f"{label}_{market}_{iteration}",
                timesteps=timestep_map[label],
            )

            validation_summary = algorithms.evaluate_model(
                label=label,
                model=model,
                validation_environment=validation_environment,
                iteration=iteration,
                socket=socket,
            )

            _send_terminal(
                socket,
                (
                    f"{label}: Sharpe={validation_summary.sharpe:.3f}, "
                    f"ActiveReturn={validation_summary.active_return:.4f}, "
                    f"Drawdown={validation_summary.max_drawdown:.4f}, "
                    f"Turnover={validation_summary.turnover:.4f}"
                ),
            )

            candidates.append(
                CandidateModelResult(
                    label=label,
                    model=model,
                    timesteps=timestep_map[label],
                    training_metadata={
                        "iteration": iteration,
                        "market": market,
                        "investor_profile": investor_profile.to_dict(),
                        "portfolio_policy": policy.to_dict(),
                    },
                    validation_summary=validation_summary,
                )
            )

        except MemoryError as exc:
            skipped_algorithms[label] = f"memory pressure: {exc}"
            logger.exception("[TRAIN] %s skipped due to memory pressure", label)
            _send_terminal(socket, f"{label} skipped due to memory pressure: {exc}")

        except Exception as exc:
            skipped_algorithms[label] = str(exc)
            logger.exception("[TRAIN] %s skipped", label)
            _send_terminal(socket, f"{label} skipped: {exc}")

    if skipped_algorithms:
        _send_terminal(socket, f"Skipped algorithms: {skipped_algorithms}")

    if not candidates:
        raise RuntimeError(f"No candidate algorithms completed successfully. Skipped={skipped_algorithms}")

    return candidates