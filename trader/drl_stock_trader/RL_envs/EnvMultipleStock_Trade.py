from __future__ import annotations

from trader.domain.session_models import InvestorProfile, PortfolioPolicy
from trader.drl_stock_trader.RL_envs_portfolio.portfolio_env import PortfolioAllocationEnv


class StockEnvTrade(PortfolioAllocationEnv):
    def __init__(
        self,
        socket,
        df,
        covariance_by_date,
        feature_columns,
        benchmark_name: str = "equal_weight",
        investor_profile: InvestorProfile | None = None,
        policy: PortfolioPolicy | None = None,
        initial: bool = True,
        previous_state=None,
        model_name: str = "",
        iteration: str | int = "",
        day: int = 0,
    ):
        previous_weights = None
        if previous_state is not None:
            try:
                prev = list(previous_state)
                previous_weights = prev if prev else None
            except Exception:
                previous_weights = None

        super().__init__(
            df=df,
            covariance_by_date=covariance_by_date,
            feature_columns=feature_columns,
            benchmark_name=benchmark_name,
            investor_profile=investor_profile,
            policy=policy,
            mode="trade",
            socket=socket,
            model_name=model_name,
            iteration=iteration,
            previous_weights=previous_weights,
        )