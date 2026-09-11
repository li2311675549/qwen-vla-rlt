from dataclasses import replace

import numpy as np
import pytest

from qwen_vla_rlt.evaluation import EvaluationContext, evaluate_decisions
from qwen_vla_rlt.policies import ActionGuard, CallablePolicy
from qwen_vla_rlt.presets import build_multitask_graph, build_skill_registry
from qwen_vla_rlt.router import PolicyRouter
from qwen_vla_rlt.runtime import HierarchicalRuntime
from qwen_vla_rlt.schemas import BrainDecision, CompletionStatus, Observation
from qwen_vla_rlt.state_machine import TaskStateMachine
from qwen_vla_rlt.task_graph import TaskGraph, TaskNode


def decision(stage="align", version=0, completion=CompletionStatus.NOT_COMPLETE):
    return BrainDecision(
        "session",
        f"obs-{version}",
        "cable_tie",
        stage,
        completion,
        None,
        {"align": "align_hole", "insert": "insert_cable_tie"}[stage],
        decision_version=version,
    )


def observation(version=0, **facts):
    return Observation(
        "session",
        f"obs-{version}",
        0.0,
        np.zeros(2),
        ("front",),
        {"target_visible": True, "safety_ok": True, **facts},
    )


def runtime(policy, clock=lambda: 0):
    graph = build_multitask_graph("cable_tie:align")
    return HierarchicalRuntime(
        PolicyRouter(graph, build_skill_registry(), max_observation_age_s=0.5),
        TaskStateMachine(graph),
        policy,
        policy,
        ActionGuard(np.full(2, -1), np.full(2, 1), np.full(2, 0.1)),
        clock=clock,
    )


def test_one_confirmation_and_loop_require_fresh_completion():
    graph = TaskGraph(
        [
            TaskNode("t", "a", "s", "a", frozenset({"t:b"})),
            TaskNode("t", "b", "s", "b", frozenset({"t:a"})),
        ],
        "t:a",
    )
    state = TaskStateMachine(graph, confirmations=1, min_dwell=0)

    def tick(stage, version, complete=False):
        return state.update(
            BrainDecision(
                "s",
                str(version),
                "t",
                stage,
                CompletionStatus.COMPLETE if complete else CompletionStatus.NOT_COMPLETE,
                None,
                "s",
                decision_version=version,
            )
        )

    tick("a", 0, True)
    assert tick("b", 1).switched
    tick("b", 2, True)
    assert tick("a", 3).switched
    assert tick("b", 4).reason == "current_stage_not_complete"


def test_invalid_transition_does_not_consume_version_and_same_frame_cannot_vote():
    graph = build_multitask_graph("cable_tie:align")
    state = TaskStateMachine(graph)
    invalid = replace(decision(), stage_id="verify", skill_id="verify_insertion")
    with pytest.raises(ValueError, match="illegal task transition"):
        state.update(invalid)
    assert state.latest_version == -1
    state.update(decision())
    with pytest.raises(ValueError, match="repeated observation"):
        state.update(replace(decision(), decision_version=1))


def test_takeover_during_inference_cancels_returned_action():
    holder = {}

    def infer(_):
        holder["runtime"].halt()
        return np.ones((2, 2))

    runner = runtime(CallablePolicy(infer))
    holder["runtime"] = runner
    result = runner.step(decision().to_payload(), observation(), np.zeros(2), now_s=0)
    assert result.reason == "superseded" and result.action is None
    assert runner.take_action(now_s=0) is None


def test_expiry_during_inference_and_before_send():
    ticks = iter([0.0, 1.0])
    runner = runtime(CallablePolicy(lambda _: np.zeros((2, 2))), clock=lambda: next(ticks))
    result = runner.step(decision().to_payload(), observation(), np.zeros(2), now_s=0)
    assert result.reason == "expired_during_inference"
    runner = runtime(CallablePolicy(lambda _: np.zeros((2, 2))))
    runner.step(decision().to_payload(), observation(), np.zeros(2), now_s=0)
    assert runner.take_action(now_s=0.6) is None


def test_completion_and_manual_intervention_never_generate_an_action():
    def fail(_):
        raise AssertionError("completed/halted stage must not invoke policy")

    runner = runtime(CallablePolicy(fail))
    item = decision(completion=CompletionStatus.COMPLETE)
    assert runner.step(item.to_payload(), observation(), np.zeros(2), now_s=0).action is None
    runner.reset()
    assert (
        runner.step(
            decision().to_payload(), observation(human_intervention=True), np.zeros(2), now_s=0
        ).reason
        == "halted"
    )


def test_legality_uses_preconditions_and_not_just_registered_ids():
    graph = build_multitask_graph("cable_tie:align")
    item = decision("insert")
    context = EvaluationContext(observation(aligned=False), "cable_tie:align", True)
    metrics = evaluate_decisions([item], [item], graph, [context])
    assert metrics.stage_accuracy == 1
    assert metrics.legal_task_selection_rate == 0


def test_missing_confidence_cannot_bypass_threshold():
    graph = build_multitask_graph("cable_tie:align")
    router = PolicyRouter(graph, build_skill_registry(), min_confidence=0.8)
    with pytest.raises(ValueError, match="confidence is required"):
        router.route(decision(), observation())


def test_final_action_guard_rejects_nonfinite_bounds():
    with pytest.raises(ValueError):
        ActionGuard(np.array([np.nan]), np.ones(1), np.ones(1))
