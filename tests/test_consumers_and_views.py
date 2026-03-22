from __future__ import annotations

import json
from types import SimpleNamespace


def test_trade_consumer_handles_invalid_json(monkeypatch):
    from trader import consumers

    consumer = consumers.TradeConsumer()
    sent = []

    def fake_send(*, text_data=None, bytes_data=None):
        sent.append({"text_data": text_data, "bytes_data": bytes_data})

    monkeypatch.setattr(consumer, "send", fake_send)
    consumer.receive(text_data='{bad json')
    assert sent
    payload = json.loads(sent[-1]['text_data'])
    assert payload['type'] == 'terminal'


def test_trade_consumer_delegates_to_trading_service(monkeypatch):
    from trader import consumers

    called = {}

    def fake_execute_trade(socket, payload):
        called['payload'] = payload

    monkeypatch.setattr(consumers.trading_service, 'execute_trade', fake_execute_trade)
    consumer = consumers.TradeConsumer()
    monkeypatch.setattr(consumer, "send", lambda **kwargs: None)
    consumer.receive(text_data=json.dumps({'market': 'etf30'}))
    assert called['payload']['market'] == 'etf30'


def test_narration_consumer_routes_load_run(monkeypatch):
    from trader import consumers

    monkeypatch.setattr(consumers.narration_service, 'create_session_state', lambda session_id: {'session_id': session_id})
    monkeypatch.setattr(consumers.narration_service, 'connect_messages', lambda state: [])
    monkeypatch.setattr(consumers.narration_service, 'load_run', lambda state, run_id: [{'type': 'advisor_answer', 'message': 'loaded'}])

    consumer = consumers.NarrationConsumer()
    consumer.scope = {'url_route': {'kwargs': {'session_id': 's1'}}}
    sent = []
    monkeypatch.setattr(consumer, "accept", lambda: None)
    monkeypatch.setattr(consumer, "_push", lambda payload: sent.append(payload))

    consumer.connect()
    consumer.receive(text_data=json.dumps({'type': 'load_run', 'run_id': 'r1'}))
    assert sent
    assert sent[-1]['message'] == 'loaded'


def test_home_view_returns_context_for_authenticated_user(monkeypatch):
    from trader import views

    monkeypatch.setattr(views, 'render', lambda request, template, context: {'template': template, 'context': context})
    request = SimpleNamespace(user=SimpleNamespace(is_authenticated=True))
    response = views.home(request)
    assert response['template'].endswith('home.html')
    assert 'app_title' in response['context']


def test_narration_session_view_loads_bundle(monkeypatch):
    from trader import views

    monkeypatch.setattr(views, 'render', lambda request, template, context: {'template': template, 'context': context})
    monkeypatch.setattr(views.artifact_store, 'load_session_manifest', lambda session_id: {'runs': ['r1'], 'market': 'etf30', 'latest_run_id': 'r1'})
    monkeypatch.setattr(views.artifact_store, 'load_advisory_bundle', lambda run_id: {'advisory_summary_text': 'summary', 'benchmark_comparison': {}, 'risk_snapshot': {}, 'allocation_recommendation': {}, 'manifest': {'portfolio_policy': {}}})
    request = SimpleNamespace(user=SimpleNamespace(is_authenticated=True))
    response = views.narration_session(request, 's1')
    assert response['template'].endswith('narration_session.html')
    assert response['context']['session_id'] == 's1'
