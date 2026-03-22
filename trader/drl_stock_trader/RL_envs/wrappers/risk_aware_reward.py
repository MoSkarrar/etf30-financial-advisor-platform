import numpy as np
import gym


class RiskAwareRewardWrapper(gym.Wrapper):
    """
    Reward shaping wrapper (no changes to env dynamics).

    It can penalize:
      - rolling volatility of portfolio returns
      - rolling downside volatility
      - current drawdown from peak portfolio value
      - action turnover between consecutive allocations
      - action concentration (HHI-like)

    It can also add a benchmark-relative term when `info` exposes
    `active_return` or `benchmark_return`.

    New reward:
      shaped = base_reward
               - penalty(vol, downside, dd, turnover, concentration)
               + benchmark_adjustment
    """

    def __init__(
        self,
        env: gym.Env,
        stock_dim: int,
        vol_window: int = 20,
        lambda_vol: float = 0.10,
        lambda_dd: float = 0.10,
        reward_scale: float = 1e-4,
        clip_penalty=None,
        eps: float = 1e-12,
        lambda_turnover: float = 0.0,
        lambda_concentration: float = 0.0,
        lambda_benchmark: float = 0.0,
        lambda_downside: float = 0.0,
    ):
        super().__init__(env)
        self.stock_dim = int(stock_dim)
        self.vol_window = int(vol_window)
        self.lambda_vol = float(lambda_vol)
        self.lambda_dd = float(lambda_dd)
        self.reward_scale = float(reward_scale)
        self.clip_penalty = clip_penalty
        self.eps = float(eps)
        self.lambda_turnover = float(lambda_turnover)
        self.lambda_concentration = float(lambda_concentration)
        self.lambda_benchmark = float(lambda_benchmark)
        self.lambda_downside = float(lambda_downside)

        self._values = []
        self._rets = []
        self._prev_action = None

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        if isinstance(obs, tuple):
            obs_value = obs[0]
        else:
            obs_value = obs

        self._values = []
        self._rets = []
        self._prev_action = None

        v = self._portfolio_value(obs_value)
        self._values.append(v)
        return obs

    def step(self, action):
        out = self.env.step(action)
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            done = bool(terminated or truncated)
            new_api = True
        else:
            obs, reward, done, info = out
            terminated = bool(done)
            truncated = False
            new_api = False

        info = info or {}
        base_reward = float(reward)
        v = self._portfolio_value(obs)

        if self._values:
            prev = self._values[-1]
            r = (v / (prev + self.eps)) - 1.0
            self._rets.append(float(r))
        self._values.append(v)

        if len(self._rets) >= 2:
            window = self._rets[-self.vol_window :]
            vol = float(np.std(window, ddof=1)) if len(window) >= 2 else 0.0
            downside_window = [x for x in window if x < 0.0]
            downside = float(np.std(downside_window, ddof=1)) if len(downside_window) >= 2 else 0.0
        else:
            vol = 0.0
            downside = 0.0

        peak = max(self._values) if self._values else v
        dd = float(max(0.0, (peak - v) / (peak + self.eps)))

        action_weights = self._normalize_action(action)
        prev_action = self._prev_action if self._prev_action is not None else action_weights
        turnover = 0.5 * float(np.sum(np.abs(action_weights - prev_action)))
        concentration = float(np.sum(np.square(action_weights)))
        self._prev_action = action_weights

        pen_vol = self.lambda_vol * vol * v * self.reward_scale
        pen_downside = self.lambda_downside * downside * v * self.reward_scale
        pen_dd = self.lambda_dd * dd * v * self.reward_scale
        pen_turnover = self.lambda_turnover * turnover * v * self.reward_scale
        pen_concentration = self.lambda_concentration * concentration * v * self.reward_scale
        penalty = float(pen_vol + pen_downside + pen_dd + pen_turnover + pen_concentration)

        active_return = None
        if isinstance(info, dict):
            if "active_return" in info:
                active_return = float(info.get("active_return") or 0.0)
            elif "portfolio_return" in info and "benchmark_return" in info:
                active_return = float(info.get("portfolio_return") or 0.0) - float(info.get("benchmark_return") or 0.0)

        benchmark_adjustment = 0.0
        if active_return is not None and self.lambda_benchmark != 0.0:
            benchmark_adjustment = float(self.lambda_benchmark * active_return * v * self.reward_scale)

        if self.clip_penalty is not None:
            penalty = float(np.clip(penalty, 0.0, self.clip_penalty))

        shaped_reward = base_reward - penalty + benchmark_adjustment

        if isinstance(info, dict):
            info["base_reward"] = base_reward
            info["risk_penalty"] = penalty
            info["risk_vol"] = vol
            info["risk_downside_vol"] = downside
            info["risk_drawdown"] = dd
            info["risk_turnover"] = turnover
            info["risk_concentration"] = concentration
            info["benchmark_adjustment"] = benchmark_adjustment
            info["portfolio_value"] = v
            info["shaped_reward"] = shaped_reward

        if new_api:
            return obs, shaped_reward, terminated, truncated, info
        return obs, shaped_reward, done, info

    def _normalize_action(self, action):
        x = np.asarray(action, dtype=float).reshape(-1)
        if x.size == 0:
            return x
        x = np.where(np.isfinite(x), x, 0.0)
        x = np.maximum(x, 0.0)
        total = float(np.sum(x))
        if total <= self.eps:
            return np.ones_like(x) / max(len(x), 1)
        return x / total

    def _portfolio_value(self, obs) -> float:
        x = np.asarray(obs, dtype=float).reshape(-1)
        cash = x[0]
        prices = x[1 : 1 + self.stock_dim]
        shares = x[1 + self.stock_dim : 1 + 2 * self.stock_dim]
        pv = float(cash + np.dot(prices, shares))
        return pv
