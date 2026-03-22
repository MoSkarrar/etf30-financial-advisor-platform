from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from trader.services import narration_chat_service, narration_context
from trader.services.narration_context import AdvisoryNarrationContext

PushFn = Callable[[dict], None]


@dataclass
class NarrationSessionState:
    session_id: str
    selected_run_id: str = ""
    context: Optional[AdvisoryNarrationContext] = None
    history: List[Tuple[str, str]] = field(default_factory=list)


MESSAGE_HINT = (
    "Run loaded. Ask why this allocation, what changed, how risky it is, how it compares to the benchmark, "
    "ask for technical XAI, compare explainers, or test stricter client rules like 8% max weight or 10% cash."
)


def create_session_state(session_id: str) -> NarrationSessionState:
    return NarrationSessionState(session_id=session_id)


def connect_messages(state: NarrationSessionState):
    return [
        {"type": "terminal", "message": f"[Narration] connected session_id={state.session_id}"},
        {"type": "advisor_answer", "message": "Load a run to begin portfolio-advisor Q&A."},
    ]


def _default_scenario_payload(ctx: AdvisoryNarrationContext) -> dict:
    impact = narration_context.stricter_policy_impact(ctx)
    return {
        "request": {
            "scenario_name": "stricter_policy",
            "shock_type": "policy_tightening",
        },
        "projected_return": 0.0,
        "projected_volatility": 0.0,
        "projected_drawdown": 0.0,
        "narrative": (
            f"A stricter client rule would imply roughly {float(impact.get('estimated_turnover', 0.0)):.2%} estimated turnover "
            f"and about {float(impact.get('additional_cash_needed', 0.0)):.2%} extra cash to reach the tighter policy."
        ),
        "derived_policy_impact": impact,
    }


def load_run(state: NarrationSessionState, run_id: str):
    run_id = (run_id or "").strip()
    if not run_id:
        return [{"type": "advisor_answer", "message": "run_id is required."}]

    ctx = narration_context.load_context_for_run(run_id)
    if not ctx:
        return [{"type": "advisor_answer", "message": "Run not found."}]

    state.selected_run_id = run_id
    state.context = ctx
    state.history = []
    payloads = [
        {"type": "terminal", "message": f"[Narration] loaded run_id={run_id}"},
        {"type": "advisor_answer", "message": MESSAGE_HINT},
        {"type": "allocation", "message": ctx.allocation_recommendation},
        {"type": "benchmark", "message": ctx.benchmark_comparison},
        {"type": "risk", "message": ctx.risk_snapshot},
        {"type": "explain", "message": ctx.xai_summary},
        {"type": "advisory_summary", "message": ctx.advisory_summary_text},
    ]
    if ctx.engine_info:
        payloads.append({"type": "engine_status", "message": ctx.engine_info})
    if ctx.policy_check:
        payloads.append({"type": "policy_check", "message": ctx.policy_check})
    if ctx.rule_summary:
        payloads.append({"type": "rule_summary", "message": ctx.rule_summary})
    if ctx.explanation_bundle:
        payloads.append({"type": "explanation_bundle", "message": ctx.explanation_bundle})
    if ctx.explanation_lab:
        payloads.append({"type": "explanation_lab", "message": ctx.explanation_lab})
    payloads.append({"type": "scenario", "message": ctx.scenario_result or _default_scenario_payload(ctx)})
    return payloads


def _submit_question(state: NarrationSessionState, question: str, push: PushFn) -> None:
    if not state.context:
        push({"type": "advisor_answer", "message": "Load a run first."})
        return

    question = (question or "").strip()
    if not question:
        push({"type": "advisor_answer", "message": "Question is required."})
        return

    state.history.append(("user", question))
    state.context.history = list(state.history)

    def _on_success(answer: str):
        state.history.append(("assistant", answer))
        if state.context:
            state.context.history = list(state.history)

    narration_chat_service.answer_question_async(
        context=state.context,
        question=question,
        history=list(state.history),
        push=push,
        on_success=_on_success,
    )


def handle_event_async(state: NarrationSessionState, event_type: str, payload: dict, push: PushFn) -> None:
    event_type = (event_type or "").strip()

    if event_type == "ask_advisor":
        _submit_question(state, payload.get("message", ""), push)
        return

    if event_type == "compare_benchmark":
        _submit_question(state, payload.get("message") or "How does this allocation compare to the benchmark?", push)
        return

    if event_type == "run_scenario":
        _submit_question(state, payload.get("message") or "What happens under a stricter client policy?", push)
        return

    if event_type == "ask_technical_xai":
        _submit_question(state, payload.get("message") or "Give the technical XAI explanation for this run.", push)
        return

    if event_type == "compare_explainers":
        _submit_question(state, payload.get("message") or "Compare SHAP, LIME, and the rule-based explanations for this run.", push)
        return

    if event_type == "show_risk_flags":
        _submit_question(state, payload.get("message") or "Summarize the main risk flags and policy issues for this run.", push)
        return

    if event_type == "explain_weight_change":
        asset = (payload.get("ticker") or "").strip()
        default_question = f"Why did the weight change for {asset}?" if asset else "What changed in the weights?"
        _submit_question(state, payload.get("message") or default_question, push)
        return

    push({"type": "advisor_answer", "message": f"Unsupported narration event: {event_type}"})
