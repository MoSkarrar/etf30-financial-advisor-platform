import numpy as np
import gym


class RiskAwareRewardWrapper(gym.Wrapper):
    """
    Gym wrapper that *modifies reward* by subtracting a risk penalty.

    It recomputes portfolio value directly from the observation:
      obs = [cash] + [prices x stock_dim] + [shares x stock_dim] + ...

    Penalty is computed on returns history (rolling window):
      - "vol"  : rolling std(returns)
      - "dd"   : rolling max drawdown (fraction)
      - "cvar" : rolling CVaR(alpha) of returns (as loss, positive)

    The env's reward is assumed to be in *scaled dollars* (FinRL-style).
    We compute penalty in dollars and scale it using reward_scaling.
    """

    def __init__(
        self,
        env,
        stock_dim: int,
        reward_scaling: float = 1e-4,
        risk_lambda: float = 0.0,
        risk_window: int = 20,
        risk_metric: str = "vol",   # "vol" | "dd" | "cvar"
        cvar_alpha: float = 0.05,
        min_history: int = 10,
    ):
        super().__init__(env)
        self.stock_dim = int(stock_dim)
        self.reward_scaling = float(reward_scaling)
        self.risk_lambda = float(risk_lambda)
        self.risk_window = int(risk_window)
        self.risk_metric = str(risk_metric).lower()
        self.cvar_alpha = float(cvar_alpha)
        self.min_history = int(min_history)

        self._asset_history = []
        self._ret_history = []
        self._prev_asset = None

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        asset = self._portfolio_value(obs)
        self._asset_history = [asset]
        self._ret_history = []
        self._prev_asset = asset
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # Compute current portfolio value from obs
        cur_asset = self._portfolio_value(obs)

        # Compute return vs previous step
        if self._prev_asset is None or self._prev_asset <= 0:
            step_ret = 0.0
        else:
            step_ret = (cur_asset / self._prev_asset) - 1.0

        self._asset_history.append(cur_asset)
        self._ret_history.append(step_ret)
        self._prev_asset = cur_asset

        # Apply risk penalty (optional)
        if self.risk_lambda > 0:
            risk_value = self._compute_risk_value()
            # Convert fraction risk -> dollar penalty using last asset
            # Then scale to match env reward units
            penalty_dollars = risk_value * cur_asset
            penalty_scaled = penalty_dollars * self.reward_scaling
            reward = reward - (self.risk_lambda * penalty_scaled)

            # Put diagnostics in info (helpful for debugging/logging)
            info = dict(info) if isinstance(info, dict) else {}
            info["risk_value"] = float(risk_value)
            info["risk_penalty_scaled"] = float(self.risk_lambda * penalty_scaled)
            info["asset_value"] = float(cur_asset)

        return obs, reward, done, info

    # -----------------------
    # helpers
    # -----------------------
    def _portfolio_value(self, obs):
        obs = np.asarray(obs, dtype=np.float64).flatten()

        cash = float(obs[0])
        p0 = 1
        p1 = 1 + self.stock_dim
        s0 = p1
        s1 = p1 + self.stock_dim

        prices = obs[p0:p1]
        shares = obs[s0:s1]

        return cash + float(np.dot(prices, shares))

    def _compute_risk_value(self):
        """
        Returns a *fractional* risk number (e.g. 0.02),
        so penalty_dollars = risk_value * asset_value.
        """
        if len(self._ret_history) < max(2, self.min_history):
            return 0.0

        rets = np.asarray(self._ret_history[-self.risk_window:], dtype=np.float64)
        assets = np.asarray(self._asset_history[-self.risk_window:], dtype=np.float64)

        if self.risk_metric == "vol":
            if len(rets) < 2:
                return 0.0
            return float(np.std(rets, ddof=1))

        if self.risk_metric == "dd":
            # max drawdown (fraction)
            peak = np.maximum.accumulate(assets)
            dd = (peak - assets) / np.maximum(peak, 1e-12)
            return float(np.max(dd))

        if self.risk_metric == "cvar":
            # CVaR(alpha) of returns as *loss* (positive)
            q = np.quantile(rets, self.cvar_alpha)
            tail = rets[rets <= q]
            if len(tail) == 0:
                return 0.0
            # negative mean of worst tail (so "loss" is positive)
            return float(-np.mean(tail))

        # unknown metric -> no penalty
        return 0.0
