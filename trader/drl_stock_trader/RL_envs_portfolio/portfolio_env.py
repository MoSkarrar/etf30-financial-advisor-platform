from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import gym
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding

from trader.domain.session_models import InvestorProfile, PortfolioPolicy
from trader.drl_stock_trader.config.app_config import APP_CONFIG


@dataclass
class PortfolioStepDiagnostics:
    date: int
    portfolio_value: float
    portfolio_return: float
    benchmark_return: float
    active_return: float
    turnover: float
    concentration_hhi: float
    cash_weight: float
    max_drawdown: float
    policy_breaches: List[str]


class PortfolioAllocationEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        covariance_by_date: Dict[int, np.ndarray],
        feature_columns: Sequence[str],
        benchmark_name: str = "equal_weight",
        investor_profile: Optional[InvestorProfile] = None,
        policy: Optional[PortfolioPolicy] = None,
        initial_cash: float = APP_CONFIG.rl.default_initial_cash,
        reward_scale: float = APP_CONFIG.rl.reward_scale,
        transaction_cost_pct: float = APP_CONFIG.rl.transaction_cost_pct,
        mode: str = "train",
        socket=None,
        model_name: str = "",
        iteration: str | int = "",
        previous_weights: Optional[Sequence[float]] = None,
    ):
        super().__init__()
        self.df = df.copy()
        self.covariance_by_date = covariance_by_date
        self.feature_columns = list(feature_columns)
        self.benchmark_name = str(benchmark_name)
        self.investor_profile = investor_profile or InvestorProfile()
        self.policy = policy or PortfolioPolicy()
        self.initial_cash = float(initial_cash)
        self.reward_scale = float(reward_scale)
        self.transaction_cost_pct = float(transaction_cost_pct)
        self.mode = str(mode)
        self.socket = socket
        self.model_name = str(model_name)
        self.iteration = str(iteration)

        self.df = self.df.sort_values(["datadate", "tic"], ignore_index=True)
        self.tickers = sorted(self.df["tic"].astype(str).unique().tolist())
        self.stock_dim = len(self.tickers)
        self.allow_cash = bool(self.policy.allow_cash_sleeve)
        self.cash_dim = 1 if self.allow_cash else 0
        self.action_dim = self.stock_dim + self.cash_dim

        self.dates = sorted(int(v) for v in self.df["datadate"].unique().tolist())
        self.price_panel = self.df.pivot(index="datadate", columns="tic", values="adjcp").reindex(self.dates).ffill().fillna(0.0)
        self.feature_panels: Dict[str, pd.DataFrame] = {
            col: self.df.pivot(index="datadate", columns="tic", values=col).reindex(self.dates).ffill().fillna(0.0)
            for col in self.feature_columns
            if col in self.df.columns
        }

        benchmark_col = f"benchmark_{self.benchmark_name}_return"
        if benchmark_col not in self.df.columns:
            benchmark_col = "benchmark_equal_weight_return"
        bench = self.df[["datadate", benchmark_col]].drop_duplicates("datadate").sort_values("datadate")
        self.benchmark_series = bench.set_index("datadate")[benchmark_col].reindex(self.dates).fillna(0.0)

        obs_dim = self.action_dim + self.stock_dim + (self.stock_dim * len(self.feature_panels)) + (self.stock_dim * self.stock_dim) + 2
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.previous_weights_input = previous_weights
        self._seed()
        self.reset()

    def _seed(self, seed: Optional[int] = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.day = 0
        base_weights = np.zeros(self.action_dim, dtype=np.float64)
        if self.allow_cash:
            base_weights[0] = 1.0
        elif self.stock_dim > 0:
            base_weights[:] = 1.0 / self.stock_dim

        if self.previous_weights_input is not None:
            prev = np.asarray(self.previous_weights_input, dtype=np.float64).reshape(-1)
            if prev.size == self.action_dim:
                prev = self._normalize_weights(prev)
                base_weights = prev

        self.current_weights = base_weights
        self.portfolio_value = self.initial_cash
        self.asset_memory: List[float] = [self.portfolio_value]
        self.return_memory: List[float] = []
        self.turnover_memory: List[float] = []
        self.concentration_memory: List[float] = []
        self.cash_memory: List[float] = [float(self.current_weights[0]) if self.allow_cash else 0.0]
        self.active_return_memory: List[float] = []
        self.benchmark_memory: List[float] = []
        self.policy_breaches_memory: List[List[str]] = []
        return self._get_observation()

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        target_weights = self._normalize_weights(action)
        target_weights, breaches = self._apply_policy(target_weights)

        current_weights = self.current_weights.copy()
        turnover = 0.5 * float(np.abs(target_weights - current_weights).sum())
        transaction_cost = turnover * self.transaction_cost_pct

        date_t = self.dates[self.day]
        date_tp1 = self.dates[min(self.day + 1, len(self.dates) - 1)]
        price_t = self.price_panel.loc[date_t, self.tickers].to_numpy(dtype=np.float64)
        price_tp1 = self.price_panel.loc[date_tp1, self.tickers].to_numpy(dtype=np.float64)
        asset_returns = np.divide(price_tp1, price_t, out=np.ones_like(price_tp1), where=price_t > 0) - 1.0

        cash_weight = float(target_weights[0]) if self.allow_cash else 0.0
        risky_weights = target_weights[self.cash_dim:]
        portfolio_return = float(np.dot(risky_weights, asset_returns))
        benchmark_return = float(self.benchmark_series.loc[date_tp1])
        active_return = portfolio_return - benchmark_return

        gross_value = self.portfolio_value * (1.0 + portfolio_return)
        cost_value = self.portfolio_value * transaction_cost
        new_value = max(1.0, gross_value - cost_value)

        effective_weights = self._post_return_weights(target_weights, asset_returns)
        concentration_hhi = float(np.square(effective_weights[self.cash_dim:]).sum())
        max_dd = self._max_drawdown(self.asset_memory + [new_value])
        reward = self._reward(
            portfolio_return=portfolio_return,
            active_return=active_return,
            turnover=turnover,
            concentration_hhi=concentration_hhi,
            max_drawdown=max_dd,
            cash_weight=cash_weight,
            breaches=breaches,
        )

        self.portfolio_value = new_value
        self.current_weights = effective_weights
        self.asset_memory.append(new_value)
        self.return_memory.append(portfolio_return)
        self.turnover_memory.append(turnover)
        self.concentration_memory.append(concentration_hhi)
        self.cash_memory.append(float(self.current_weights[0]) if self.allow_cash else 0.0)
        self.active_return_memory.append(active_return)
        self.benchmark_memory.append(benchmark_return)
        self.policy_breaches_memory.append(breaches)

        diagnostics = PortfolioStepDiagnostics(
            date=date_tp1,
            portfolio_value=new_value,
            portfolio_return=portfolio_return,
            benchmark_return=benchmark_return,
            active_return=active_return,
            turnover=turnover,
            concentration_hhi=concentration_hhi,
            cash_weight=float(self.current_weights[0]) if self.allow_cash else 0.0,
            max_drawdown=max_dd,
            policy_breaches=breaches,
        )

        done = self.day >= len(self.dates) - 2
        self.day = min(self.day + 1, len(self.dates) - 1)
        obs = self._get_observation()
        info = {"diagnostics": diagnostics.__dict__}

        if done and self.socket is not None:
            self._socket_log(f"{self.mode.upper()} completed: portfolio_value={new_value:.2f}, active_return={sum(self.active_return_memory):.4f}")

        return obs, reward, done, info

    def render(self, mode="human"):
        return {
            "date": self.dates[self.day],
            "portfolio_value": self.portfolio_value,
            "weights": self.current_weights.copy(),
        }

    def get_current_weights(self) -> np.ndarray:
        return self.current_weights.copy()

    def get_history_frame(self) -> pd.DataFrame:
        usable_dates = self.dates[: len(self.asset_memory)]
        return pd.DataFrame(
            {
                "datadate": usable_dates,
                "account_value": self.asset_memory,
                "portfolio_return": [0.0] + self.return_memory,
                "turnover": [0.0] + self.turnover_memory,
                "concentration_hhi": [0.0] + self.concentration_memory,
                "cash_weight": self.cash_memory,
                "benchmark_return": [0.0] + self.benchmark_memory,
                "active_return": [0.0] + self.active_return_memory,
            }
        )

    def _get_observation(self) -> np.ndarray:
        date = self.dates[self.day]
        prices = self.price_panel.loc[date, self.tickers].to_numpy(dtype=np.float64)
        norm_prices = prices / np.maximum(prices.mean(), 1e-8)

        feature_chunks: List[np.ndarray] = []
        for col in self.feature_columns:
            panel = self.feature_panels.get(col)
            if panel is None:
                feature_chunks.append(np.zeros(self.stock_dim, dtype=np.float64))
            else:
                feature_chunks.append(panel.loc[date, self.tickers].to_numpy(dtype=np.float64))

        covariance = self.covariance_by_date.get(date)
        if covariance is None:
            covariance = np.zeros((self.stock_dim, self.stock_dim), dtype=np.float64)

        benchmark_return = float(self.benchmark_series.loc[date])
        obs = np.concatenate(
            [
                self.current_weights.astype(np.float64),
                norm_prices.astype(np.float64),
                *feature_chunks,
                covariance.reshape(-1).astype(np.float64),
                np.array([benchmark_return, self.portfolio_value / max(self.initial_cash, 1.0)], dtype=np.float64),
            ]
        )
        return obs.astype(np.float32)

    def _normalize_weights(self, action: np.ndarray) -> np.ndarray:
        action = np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
        if action.size != self.action_dim:
            resized = np.zeros(self.action_dim, dtype=np.float64)
            resized[: min(action.size, self.action_dim)] = action[: min(action.size, self.action_dim)]
            action = resized

        clipped = np.clip(action, -1.0, 1.0)
        shifted = clipped - clipped.max()
        exp = np.exp(shifted)
        weights = exp / np.maximum(exp.sum(), 1e-8)
        return weights

    def _apply_policy(self, weights: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        breaches: List[str] = []
        out = weights.copy()

        cash_idx = 0 if self.allow_cash else None
        risky_slice = slice(self.cash_dim, None)
        risky = out[risky_slice]
        cap = float(self.policy.max_single_position_cap)

        if self.policy.long_only:
            risky = np.clip(risky, 0.0, None)

        if cap > 0:
            over = risky > cap
            if over.any():
                breaches.append("max_single_position_cap")
                excess = risky[over].sum() - (cap * over.sum())
                risky[over] = cap
                if excess > 0 and (~over).any():
                    redistribute_idx = np.where(~over)[0]
                    risky[redistribute_idx] += excess / len(redistribute_idx)

        min_cash = float(self.policy.min_cash_weight) if self.allow_cash else 0.0
        if self.allow_cash:
            if out[cash_idx] < min_cash:
                breaches.append("cash_floor")
            out[cash_idx] = max(out[cash_idx], min_cash)

        risky_sum = risky.sum()
        available_risky = max(1e-8, 1.0 - (out[cash_idx] if self.allow_cash else 0.0))
        if risky_sum <= 0:
            risky[:] = available_risky / max(len(risky), 1)
        else:
            risky[:] = risky / risky_sum * available_risky

        out[risky_slice] = risky
        if self.allow_cash:
            out[cash_idx] = max(min_cash, 1.0 - risky.sum())
        out = out / np.maximum(out.sum(), 1e-8)
        return out, breaches

    def _post_return_weights(self, target_weights: np.ndarray, asset_returns: np.ndarray) -> np.ndarray:
        if self.allow_cash:
            cash = target_weights[0]
            risky = target_weights[1:]
        else:
            cash = 0.0
            risky = target_weights

        grown_risky = risky * (1.0 + asset_returns)
        gross = cash + grown_risky.sum()
        if gross <= 0:
            return target_weights

        if self.allow_cash:
            out = np.concatenate([[cash], grown_risky]) / gross
        else:
            out = grown_risky / gross
        return out

    def _reward(
        self,
        portfolio_return: float,
        active_return: float,
        turnover: float,
        concentration_hhi: float,
        max_drawdown: float,
        cash_weight: float,
        breaches: Sequence[str],
    ) -> float:
        cfg = APP_CONFIG.advisory_reward
        profile = self.investor_profile

        target_vol_gap = max(0.0, self._realized_volatility(self.return_memory[-20:] + [portfolio_return]) - profile.target_volatility)
        cash_floor_gap = max(0.0, profile.min_cash_preference - cash_weight)
        breach_penalty = float(len(breaches))

        reward = (
            cfg.portfolio_return_weight * portfolio_return
            + cfg.benchmark_excess_weight * active_return
            - cfg.turnover_penalty * turnover * (1.0 + profile.turnover_aversion)
            - cfg.concentration_penalty * concentration_hhi
            - cfg.drawdown_penalty * max(0.0, max_drawdown - profile.max_drawdown_preference)
            - cfg.target_volatility_penalty * target_vol_gap
            - cfg.cash_floor_penalty * cash_floor_gap
            - cfg.policy_breach_penalty * breach_penalty
        )
        return float(reward / max(self.reward_scale, 1e-8))

    @staticmethod
    def _realized_volatility(returns: Sequence[float]) -> float:
        arr = np.asarray(list(returns), dtype=np.float64)
        if arr.size < 2:
            return 0.0
        return float(arr.std(ddof=0) * np.sqrt(252.0))

    @staticmethod
    def _max_drawdown(values: Sequence[float]) -> float:
        arr = np.asarray(list(values), dtype=np.float64)
        if arr.size == 0:
            return 0.0
        peaks = np.maximum.accumulate(arr)
        drawdowns = 1.0 - np.divide(arr, peaks, out=np.ones_like(arr), where=peaks > 0)
        return float(np.nanmax(drawdowns)) if drawdowns.size else 0.0

    def _socket_log(self, message: str) -> None:
        try:
            self.socket.send(text_data=json.dumps({"type": "terminal", "message": message}))
        except Exception:
            pass