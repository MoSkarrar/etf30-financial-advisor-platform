from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import requests

from trader.drl_stock_trader.config.app_config import APP_CONFIG
from trader.services.ollama_prompt_builder import (
    build_advisor_prompt,
    build_explainer_compare_prompt,
    build_risk_committee_prompt,
    build_technical_xai_prompt,
)
from trader.services.ollama_response_postprocess import postprocess_response


@dataclass(frozen=True)
class OllamaNarrationConfig:
    base_url: str = APP_CONFIG.ollama.base_url
    model: str = APP_CONFIG.ollama.model
    connect_timeout_s: int = APP_CONFIG.ollama.connect_timeout_s
    read_timeout_s: int = APP_CONFIG.ollama.read_timeout_s
    timeout_s: int = APP_CONFIG.ollama.timeout_s
    temperature: float = APP_CONFIG.ollama.temperature
    num_predict: int = APP_CONFIG.ollama.num_predict
    top_p: float = APP_CONFIG.ollama.top_p
    chat_num_predict: int = APP_CONFIG.ollama.chat_num_predict
    repeat_penalty: float = APP_CONFIG.ollama.repeat_penalty

    @classmethod
    def from_app_config(cls) -> "OllamaNarrationConfig":
        return cls()


History = List[Tuple[str, str]]


def _post_generate(prompt: str, cfg: OllamaNarrationConfig, num_predict: int, repeat_penalty: Optional[float] = None) -> str:
    response = requests.post(
        f"{cfg.base_url.rstrip('/')}/api/generate",
        json={
            "model": cfg.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": cfg.temperature,
                "top_p": cfg.top_p,
                "num_predict": int(num_predict),
                "repeat_penalty": repeat_penalty if repeat_penalty is not None else cfg.repeat_penalty,
            },
        },
        timeout=(cfg.connect_timeout_s, cfg.read_timeout_s),
    )
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}: {(response.text or '')[:600]}")
    data = response.json()
    return (data.get("response") or "").strip()


def generate_mode_response(
    *,
    context: Union[str, Dict[str, Any]],
    prompt_mode: str,
    question: str = "",
    history: Optional[History] = None,
    cfg: Optional[OllamaNarrationConfig] = None,
) -> str:
    cfg = cfg or OllamaNarrationConfig.from_app_config()

    if not isinstance(context, dict):
        import json
        try:
            context = json.loads(context)
        except Exception:
            context = {"raw_context": str(context)}

    if prompt_mode in {"technical_xai"}:
        prompt = build_technical_xai_prompt(context=context, question=question, history=history)
        mode_name = "technical_xai"
        budget = max(int(cfg.num_predict), int(cfg.chat_num_predict))
    elif prompt_mode in {"risk_summary", "risk_committee"}:
        prompt = build_risk_committee_prompt(context=context, question=question, history=history)
        mode_name = "risk_committee"
        budget = max(int(cfg.num_predict), int(cfg.chat_num_predict))
    elif prompt_mode in {"explainer_compare"}:
        prompt = build_explainer_compare_prompt(context=context, question=question, history=history)
        mode_name = "explainer_compare"
        budget = max(int(cfg.num_predict), int(cfg.chat_num_predict))
    else:
        prompt = build_advisor_prompt(context=context, question=question, history=history)
        mode_name = "advisor_summary"
        budget = max(int(cfg.chat_num_predict), 900)

    raw = _post_generate(
        prompt=prompt,
        cfg=cfg,
        num_predict=budget,
    )
    return postprocess_response(
        raw,
        mode=mode_name,
        context=context,
        max_chars=None,
    )
