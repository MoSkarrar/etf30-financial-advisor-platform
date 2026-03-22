from __future__ import annotations

from trader.domain.session_models import InvestorProfile, PortfolioPolicy
from trader.drl_stock_trader.RL_envs_portfolio.portfolio_env import PortfolioAllocationEnv


class StockEnvTrain(PortfolioAllocationEnv):
    def __init__(
        self,
        df,
        covariance_by_date,
        feature_columns,
        benchmark_name: str = "equal_weight",
        investor_profile: InvestorProfile | None = None,
        policy: PortfolioPolicy | None = None,
        day: int = 0,
    ):
        super().__init__(
            df=df,
            covariance_by_date=covariance_by_date,
            feature_columns=feature_columns,
            benchmark_name=benchmark_name,
            investor_profile=investor_profile,
            policy=policy,
            mode="train",
        )