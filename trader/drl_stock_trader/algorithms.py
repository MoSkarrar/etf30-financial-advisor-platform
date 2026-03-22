from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from stable_baselines3 import A2C, DDPG, PPO
from stable_baselines3.common.noise import NormalActionNoise

from trader.drl_stock_trader.config import paths
from trader.drl_stock_trader.config.app_config import APP_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class ValidationSummary:
    label: str
    sharpe: float
    annualized_return: float
    max_drawdown: float
    turnover: float
    concentration: float
    benchmark_return: float
    active_return: float
    information_ratio: float
    stability_score: float
    policy_compliance_score: float
    validation_csv_path: str


ALGORITHM_MAP = {
    "A2C": A2C,
    "PPO": PPO,
    "DDPG": DDPG,
}


def _socket_log(socket, message: str) -> None:
    if socket is None:
        return
    try:
        socket.send(text_data=json.dumps({"type": "terminal", "message": message}, ensure_ascii=False))
    except Exception:
        pass


def _trained_model_path(model_name: str) -> str:
    model_path = os.path.join(paths.trained_models_dir(), model_name)
    paths.ensure_parent_dir(model_path)
    return model_path


def _unwrap_env(vec_env):
    env = vec_env
    try:
        if hasattr(env, "envs"):
            env = env.envs[0]
        if hasattr(env, "unwrapped"):
            env = env.unwrapped
    except Exception:
        pass
    return env


def _space_dim(space) -> int:
    shape = getattr(space, "shape", None)
    if shape:
        return int(np.prod(shape))
    n = getattr(space, "n", None)
    if n is not None:
        return int(n)
    return 1


def _estimate_ddpg_buffer_bytes(observation_dim: int, action_dim: int, buffer_size: int) -> int:
    bytes_per_transition = (observation_dim * 4 * 2) + (action_dim * 4) + 4 + 1
    return int(buffer_size * bytes_per_transition)


def _recommend_ddpg_buffer_size(env_train, timesteps: int) -> int:
    obs_dim = _space_dim(env_train.observation_space)
    action_dim = _space_dim(env_train.action_space)

    candidate_sizes = list(getattr(APP_CONFIG.rl, "ddpg_candidate_buffers", (200000, 100000, 50000, 20000)))
    target_from_timesteps = max(20_000, min(max(candidate_sizes), int(timesteps) * 4))
    max_memory_budget_bytes = 512 * 1024 * 1024

    valid_candidates = []
    for candidate in sorted(candidate_sizes):
        est_bytes = _estimate_ddpg_buffer_bytes(obs_dim, action_dim, candidate)
        if est_bytes <= max_memory_budget_bytes:
            valid_candidates.append(candidate)

    if not valid_candidates:
        return 20_000

    affordable = [c for c in valid_candidates if c <= target_from_timesteps]
    if affordable:
        return max(affordable)
    return min(valid_candidates)


def _safe_std(series: pd.Series) -> float:
    value = float(series.std(ddof=0)) if len(series) > 1 else 0.0
    return value if math.isfinite(value) else 0.0


def _ensure_numeric_column(hist: pd.DataFrame, column: str, default: float = 0.0) -> None:
    if column not in hist.columns:
        hist[column] = default
    hist[column] = pd.to_numeric(hist[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def _episode_rollout(model, env, deterministic: bool = True) -> Tuple[pd.DataFrame, np.ndarray, List[dict]]:
    """
    Run the raw underlying env directly.

    This avoids the DummyVecEnv auto-reset behavior that was collapsing
    validation/trade history down to a 1-row reset-state frame.
    """
    obs = env.reset()
    done = False
    infos: List[dict] = []

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done, info = env.step(action)

        if isinstance(info, dict):
            infos.append(info)
        elif isinstance(info, (list, tuple)) and info:
            infos.append(info[0])

    history = env.get_history_frame().copy()
    weights = env.get_current_weights().copy()
    return history, weights, infos


def _log_validation_debug(label: str, iteration: int, hist: pd.DataFrame) -> None:
    date_min = hist["datadate"].min() if "datadate" in hist.columns and not hist.empty else "n/a"
    date_max = hist["datadate"].max() if "datadate" in hist.columns and not hist.empty else "n/a"
    account_min = float(hist["account_value"].min()) if "account_value" in hist.columns and not hist.empty else 0.0
    account_max = float(hist["account_value"].max()) if "account_value" in hist.columns and not hist.empty else 0.0
    port_abs = float(hist["portfolio_return"].abs().sum()) if "portfolio_return" in hist.columns else 0.0
    bench_abs = float(hist["benchmark_return"].abs().sum()) if "benchmark_return" in hist.columns else 0.0
    active_abs = float(hist["active_return"].abs().sum()) if "active_return" in hist.columns else 0.0

    logger.info(
        "[VAL][%s][iter=%s] rows=%s, dates=%s→%s, account=%.2f→%.2f, "
        "|portfolio_return|=%.6f, |benchmark_return|=%.6f, |active_return|=%.6f",
        label,
        iteration,
        len(hist),
        date_min,
        date_max,
        account_min,
        account_max,
        port_abs,
        bench_abs,
        active_abs,
    )


def train_algorithm(
    socket,
    env_train,
    algorithm_name: str,
    model_name: str,
    timesteps: int,
):
    algorithm_name = str(algorithm_name).upper()
    algo_cls = ALGORITHM_MAP.get(algorithm_name)
    if algo_cls is None:
        raise ValueError(f"Unsupported algorithm: {algorithm_name}")

    kwargs: Dict[str, Any] = {"policy": "MlpPolicy", "env": env_train, "verbose": 0}

    if algorithm_name == "DDPG":
        n_actions = env_train.action_space.shape[-1]
        buffer_size = _recommend_ddpg_buffer_size(env_train, timesteps)
        learning_starts = min(
            getattr(APP_CONFIG.rl, "ddpg_learning_starts_cap", 10_000),
            max(getattr(APP_CONFIG.rl, "ddpg_learning_starts_floor", 1_000), buffer_size // 10),
        )
        batch_size = min(getattr(APP_CONFIG.rl, "ddpg_batch_size", 128), max(32, buffer_size // 20))

        kwargs["action_noise"] = NormalActionNoise(
            mean=np.zeros(n_actions, dtype=np.float32),
            sigma=getattr(APP_CONFIG.rl, "ddpg_sigma", 0.10) * np.ones(n_actions, dtype=np.float32),
        )
        kwargs["buffer_size"] = int(buffer_size)
        kwargs["batch_size"] = int(batch_size)
        kwargs["learning_starts"] = int(learning_starts)
        kwargs["optimize_memory_usage"] = True
        kwargs["train_freq"] = (1, "step")
        kwargs["gradient_steps"] = 1

        _socket_log(
            socket,
            f"DDPG memory-safe settings: buffer_size={buffer_size}, batch_size={batch_size}, learning_starts={learning_starts}",
        )

    start = time.time()
    model = algo_cls(**kwargs)
    model.learn(total_timesteps=int(timesteps))
    duration_min = (time.time() - start) / 60.0

    model_path = _trained_model_path(model_name)
    model.save(model_path)
    _socket_log(socket, f"{algorithm_name} training finished in {duration_min:.2f} minutes")
    return model


def rollout_model(model, vec_env, deterministic: bool = True):
    env = _unwrap_env(vec_env)
    history, weights, infos = _episode_rollout(model=model, env=env, deterministic=deterministic)
    return {
        "history": history,
        "weights": weights,
        "infos": infos,
    }


def evaluate_model(label: str, model, validation_environment, iteration: int, socket=None) -> ValidationSummary:
    env = _unwrap_env(validation_environment)
    history, _, _ = _episode_rollout(model=model, env=env, deterministic=True)
    hist = history.copy()

    if hist.empty:
        raise RuntimeError(f"Validation history empty for {label}.")

    _ensure_numeric_column(hist, "account_value", default=0.0)
    _ensure_numeric_column(hist, "portfolio_return", default=0.0)
    _ensure_numeric_column(hist, "turnover", default=0.0)
    _ensure_numeric_column(hist, "concentration_hhi", default=0.0)
    _ensure_numeric_column(hist, "benchmark_return", default=0.0)
    _ensure_numeric_column(hist, "active_return", default=0.0)

    hist["daily_return"] = hist["account_value"].pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)
    hist["benchmark_cum"] = (1.0 + hist["benchmark_return"]).cumprod() - 1.0
    hist["active_cum"] = (1.0 + hist["active_return"]).cumprod() - 1.0

    _log_validation_debug(label=label, iteration=iteration, hist=hist)

    if socket is not None:
        try:
            _socket_log(
                socket,
                (
                    f"[VAL][{label}] rows={len(hist)}, "
                    f"dates={hist['datadate'].min() if 'datadate' in hist.columns and not hist.empty else 'n/a'}→"
                    f"{hist['datadate'].max() if 'datadate' in hist.columns and not hist.empty else 'n/a'}, "
                    f"account={float(hist['account_value'].min()) if 'account_value' in hist.columns and not hist.empty else 0.0:.2f}→"
                    f"{float(hist['account_value'].max()) if 'account_value' in hist.columns and not hist.empty else 0.0:.2f}"
                ),
            )
        except Exception:
            pass

    annualized_return = float((1.0 + hist["daily_return"].mean()) ** 252 - 1.0)
    sharpe_std = _safe_std(hist["daily_return"])
    sharpe = float(np.sqrt(252.0) * hist["daily_return"].mean() / sharpe_std) if sharpe_std > 0 else 0.0

    peaks = hist["account_value"].cummax().replace(0.0, np.nan)
    drawdowns = 1.0 - hist["account_value"] / peaks
    max_drawdown = float(drawdowns.fillna(0.0).max())

    turnover = float(hist["turnover"].mean())
    concentration = float(hist["concentration_hhi"].mean())
    benchmark_return = float(hist["benchmark_return"].sum())
    active_return = float(hist["active_return"].sum())

    active_std = _safe_std(hist["active_return"])
    information_ratio = float(np.sqrt(252.0) * hist["active_return"].mean() / active_std) if active_std > 0 else 0.0

    rolling_active_std = hist["active_return"].rolling(21, min_periods=2).std().fillna(0.0).mean()
    stability_score = float(1.0 / (1.0 + float(rolling_active_std)))

    breach_count = sum(len(v) for v in getattr(env, "policy_breaches_memory", []))
    compliance_score = float(1.0 / (1.0 + breach_count))

    out_csv = paths.validation_csv_path(iteration, label=label)
    paths.ensure_parent_dir(out_csv)
    hist.to_csv(out_csv, index=False)

    return ValidationSummary(
        label=label,
        sharpe=sharpe,
        annualized_return=annualized_return,
        max_drawdown=max_drawdown,
        turnover=turnover,
        concentration=concentration,
        benchmark_return=benchmark_return,
        active_return=active_return,
        information_ratio=information_ratio,
        stability_score=stability_score,
        policy_compliance_score=compliance_score,
        validation_csv_path=out_csv,
    )

def predict_last_allocation(model, vec_env, deterministic: bool = True) -> Dict[str, Any]:
    env = _unwrap_env(vec_env)
    history, weights, infos = _episode_rollout(model=model, env=env, deterministic=deterministic)

    tickers = (["CASH"] if getattr(env, "allow_cash", False) else []) + list(getattr(env, "tickers", []))
    latest = infos[-1].get("diagnostics", {}) if infos and isinstance(infos[-1], dict) else {}
    if not latest and not history.empty:
        latest = history.iloc[-1].to_dict()

    return {
        "tickers": tickers,
        "weights": {str(t): float(w) for t, w in zip(tickers, weights.tolist())},
        "latest_diagnostics": latest,
        "history": history,
    }