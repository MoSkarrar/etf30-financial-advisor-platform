from __future__ import annotations

import re
from typing import Dict, List, Optional


def _normalize_whitespace(text: str) -> str:
    text = (text or "").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def strip_repetition(text: str) -> str:
    lines = [line.rstrip() for line in _normalize_whitespace(text).split("\n")]
    deduped: List[str] = []
    seen = set()
    for line in lines:
        key = re.sub(r"\W+", "", line.lower())
        if key and key in seen and len(key) > 8:
            continue
        if key:
            seen.add(key)
        deduped.append(line)
    return "\n".join(deduped).strip()


def _uncertainty_note(context: Dict[str, object]) -> str:
    lab = context.get("explanation_lab") or {}
    contradictions = getattr(lab, "get", lambda *_: [])("contradictions", []) or []
    confidence = float(getattr(lab, "get", lambda *_: 0.0)("confidence_score", 0.0) or 0.0)
    if contradictions:
        return "Confidence is limited because the explanation methods do not fully agree."
    if 0.0 < confidence < 0.5:
        return "Confidence is limited because the explanation confidence score is low."
    if context.get("uncertainty_summary"):
        return str(context.get("uncertainty_summary")).strip()
    return ""


def _enforce_mode_tone(text: str, mode: str) -> str:
    text = text.strip()
    if not text:
        return text
    if mode == "advisor_summary":
        text = text.replace("SHAP values", "factor signals")
        text = text.replace("LIME", "local explanation")
    elif mode == "risk_committee" and not text.lower().startswith("risk view"):
        text = f"Risk view:\n{text}"
    elif mode == "technical_xai" and not text.lower().startswith("technical view"):
        text = f"Technical view:\n{text}"
    elif mode == "explainer_compare" and not text.lower().startswith("explanation audit"):
        text = f"Explanation audit:\n{text}"
    return text.strip()


def postprocess_response(
    text: str,
    *,
    mode: str,
    context: Dict[str, object],
    max_chars: Optional[int] = None,
) -> str:
    out = strip_repetition(text)
    out = _enforce_mode_tone(out, mode)
    note = _uncertainty_note(context)
    if note and note not in out:
        out = (out + "\n\n" + note).strip()
    if max_chars and max_chars > 0 and len(out) > max_chars:
        out = out[: max_chars - 3].rstrip() + "..."
    return out
