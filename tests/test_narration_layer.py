from __future__ import annotations

from types import SimpleNamespace

import pytest


def _sample_context():
    from trader.services.narration_context import AdvisoryNarrationContext

    return AdvisoryNarrationContext(
        session_id='s1',
        run_id='r1',
        manifest={'selected_model': 'PPO', 'trade_window': {'start': '20200101', 'end': '20200201'}, 'run_level_metrics': {}},
        allocation_recommendation={'target_weights': {'AAA': 0.6, 'BBB': 0.3, 'CASH': 0.1}, 'rationale_summary': 'Base rationale'},
        benchmark_comparison={'benchmark_name': 'equal_weight', 'active_return': 0.03, 'tracking_error': 0.02, 'information_ratio': 1.1},
        risk_snapshot={'realized_volatility': 0.14, 'max_drawdown': 0.08, 'concentration_hhi': 0.10, 'cash_weight': 0.1},
        investor_profile={'profile_name': 'balanced'},
        portfolio_policy={'max_single_position_cap': 0.12, 'min_cash_weight': 0.05},
        advisory_summary_text='Advisor summary',
        xai_summary='XAI summary',
    )


def test_narration_context_builds_prompt_dict():
    from trader.services import narration_context

    ctx = narration_context.build_narration_context(
        {
            'manifest': {'session_id': 's1', 'run_id': 'r1', 'selected_model': 'PPO', 'trade_window': {'start': '20200101', 'end': '20200201'}},
            'allocation_recommendation': {'target_weights': {'AAA': 0.5, 'CASH': 0.5}},
            'benchmark_comparison': {'benchmark_name': 'equal_weight'},
            'risk_snapshot': {'cash_weight': 0.5},
            'investor_profile': {'profile_name': 'balanced'},
            'advisory_summary_text': 'advisor summary',
            'xai_text': 'xai text',
        }
    )
    prompt = ctx.to_prompt_dict()
    assert prompt['run_id'] == 'r1'
    assert 'allocation_recommendation' in prompt


def test_detect_intent_and_policy_override_parser():
    from trader.services import narration_chat_service as ncs

    assert ncs.detect_intent('How does this compare to the benchmark?').name in {'compare_benchmark', 'explain_allocation'}
    override = ncs.parse_policy_override_from_question('What if the client wants max weight 8% and keep 10% cash?')
    assert override['max_single_position_cap'] == 0.08
    assert override['min_cash_weight'] == 0.10


def test_fallback_answer_for_risk_question():
    from trader.services import narration_chat_service as ncs

    text = ncs._fallback_answer(_sample_context(), 'How risky is this?', ncs.detect_intent('How risky is this?'))
    assert 'volatility' in text.lower() or 'risk' in text.lower()


prompt_builder = pytest.importorskip('trader.services.ollama_prompt_builder')
response_post = pytest.importorskip('trader.services.ollama_response_postprocess')


def test_prompt_builder_and_postprocess_work_together():
    ctx = _sample_context().to_prompt_dict()
    prompt = prompt_builder.build_advisor_prompt(ctx, question='Explain the allocation', history=[('user', 'hello')])
    assert 'CONTEXT' in prompt
    processed = response_post.postprocess_response(
        'This is the answer. This is the answer. Another sentence.',
        mode='advisor_summary',
        context=ctx,
        max_chars=200,
    )
    assert processed


def test_narration_service_load_run_and_submit(monkeypatch):
    from trader.services import narration_service

    state = narration_service.create_session_state('s1')
    monkeypatch.setattr(narration_service.narration_context, 'load_context_for_run', lambda run_id, **kwargs: _sample_context())
    payloads = narration_service.load_run(state, 'r1')
    assert any(p['type'] == 'allocation' for p in payloads)

    pushed = []

    def fake_answer_question_async(**kwargs):
        kwargs['push']({'type': 'advisor_answer', 'message': 'ok'})
        if kwargs.get('on_success'):
            kwargs['on_success']('ok')

    monkeypatch.setattr(narration_service.narration_chat_service, 'answer_question_async', fake_answer_question_async)
    narration_service.handle_event_async(state, 'ask_advisor', {'message': 'Why?'}, pushed.append)
    assert pushed[-1]['message'] == 'ok'
