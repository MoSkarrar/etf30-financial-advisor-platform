from __future__ import annotations

from typing import Dict

from trader.drl_stock_trader.config.app_config import APP_CONFIG
from trader.drl_stock_trader.engines.finrl_engine import FinRLEngine
from trader.drl_stock_trader.engines.legacy_rl_engine import EngineIterationResult, LegacyEngine


_ENGINE_REGISTRY: Dict[str, object] = {
    "legacy_rl": LegacyEngine(),
    "finrl": FinRLEngine(),
}


def get_engine(engine_name: str):
    normalized = str(engine_name or APP_CONFIG.engine.default_engine).strip().lower()
    if normalized == "finrl" and not APP_CONFIG.engine.enable_finrl:
        if APP_CONFIG.engine.enable_engine_fallback:
            return _ENGINE_REGISTRY[APP_CONFIG.engine.default_engine]
        raise ValueError("FinRL engine is disabled in app configuration.")

    engine = _ENGINE_REGISTRY.get(normalized)
    if engine is not None:
        return engine

    if APP_CONFIG.engine.enable_engine_fallback:
        return _ENGINE_REGISTRY[APP_CONFIG.engine.default_engine]
    raise ValueError(f"Unsupported engine: {engine_name}")


def run_engine(engine_name: str, **kwargs) -> EngineIterationResult:
    engine = get_engine(engine_name)
    return engine.train_validate_trade(**kwargs)
