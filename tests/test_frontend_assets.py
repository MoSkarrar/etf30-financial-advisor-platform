from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    candidates = [
        ROOT / relative_path,
        ROOT / 'trader' / relative_path,
    ]
    for path in candidates:
        if path.exists():
            return path.read_text(encoding='utf-8')
    raise FileNotFoundError(str(candidates[0]))


def test_home_template_contains_wave6_controls():
    html = _read('templates/trader/home.html')
    assert 'benchmark' in html.lower()
    assert 'advisor' in html.lower() or 'risk' in html.lower()
    assert 'trading-page' in html or 'trading-form' in html


def test_narration_template_contains_run_and_chat_sections():
    html = _read('templates/trader/narration_session.html')
    lowered = html.lower()
    assert 'session' in lowered
    assert 'run' in lowered
    assert 'chat' in lowered or 'advisor' in lowered


def test_trading_page_js_serializes_payload_and_statuses():
    js = _read('static/trader/js/trading_page.js')
    lowered = js.lower()
    assert 'websocket' in lowered or 'ws' in lowered
    assert 'benchmark' in lowered
    assert 'engine' in lowered or 'explain' in lowered or 'risk' in lowered


def test_narration_page_js_handles_rich_message_types():
    js = _read('static/trader/js/narration_page.js')
    lowered = js.lower()
    assert 'load_run' in js
    assert 'advisor_answer' in js or 'advisor' in lowered
    assert 'risk' in lowered or 'explanation' in lowered
