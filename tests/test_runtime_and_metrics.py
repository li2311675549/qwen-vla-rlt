from __future__ import annotations

import numpy as np

from qwen_vla_rlt.adapters import RLTResidualPolicy
from qwen_vla_rlt.evaluation import EvaluationContext, evaluate_decisions
from qwen_vla_rlt.policies import ActionGuard, CallablePolicy
from qwen_vla_rlt.presets import build_multitask_graph, build_skill_registry
from qwen_vla_rlt.rlt.models import ResidualActor
from qwen_vla_rlt.router import PolicyRouter, RLTArtifact
from qwen_vla_rlt.runtime import HierarchicalRuntime
from qwen_vla_rlt.schemas import (
    BrainDecision,
    CompletionStatus,
    Observation,
    PolicyBackend,
)
from qwen_vla_rlt.state_machine import TaskStateMachine


def brain(backend: PolicyBackend = PolicyBackend.RLT) -> BrainDecision:
    return BrainDecision(
        "s",
        "o",
        "cable_tie",
        "align",
        CompletionStatus.NOT_COMPLETE,
        None,
        "align_hole",
        {"speed": 0.05},
        backend,
        0,
        0.95,
    )


def test_runtime_routes_fine_stage_through_reference_conditioned_rlt() -> None:
    graph = build_multitask_graph("cable_tie:align")
    router = PolicyRouter(
        graph,
        build_skill_registry(),
        rlt_artifacts={"align_hole": RLTArtifact("align_hole", "v1", "v1")},
    )
    reference = CallablePolicy(lambda _: np.full((2, 2), 0.1, dtype=np.float32))
    actor = ResidualActor(2, 1, (2, 2), np.array([0.05, 0.05]))
    rlt = RLTResidualPolicy(reference, lambda _: np.zeros(1, dtype=np.float32), actor)
    runtime = HierarchicalRuntime(
        router,
        TaskStateMachine(graph, confirmations=1, min_dwell=0),
        reference,
        rlt,
        ActionGuard(np.array([-1.0, -1.0]), np.array([1.0, 1.0]), np.array([0.2, 0.2])),
    )
    observation = Observation(
        "s", "o", 0.0, np.zeros(2), ("front",), {"target_visible": True, "safety_ok": True}
    )
    result = runtime.step(
        {
            "session_id": "s",
            "observation_id": "o",
            "task_id": "cable_tie",
            "stage_id": "align",
            "completion_status": "not_complete",
            "next_task_id": None,
            "skill_id": "align_hole",
            "skill_args": {"speed": 0.05},
            "policy_backend": "rlt",
            "decision_version": 0,
            "confidence": 0.95,
        },
        observation,
        np.zeros(2),
    )
    assert result.action is not None
    assert result.action.backend is PolicyBackend.RLT
    assert result.action.values.shape == (2, 2)


def test_metrics_keep_stage_completion_and_legality_separate() -> None:
    graph = build_multitask_graph("cable_tie:align")
    targets = [brain(), brain()]
    predictions = [
        brain(),
        BrainDecision(
            "s",
            "o",
            "cable_tie",
            "insert",
            CompletionStatus.COMPLETE,
            None,
            "insert_cable_tie",
            {"force_limit": 5.0},
            PolicyBackend.RLT,
            1,
            0.9,
        ),
    ]
    observation = Observation(
        "s",
        "o",
        0.0,
        np.zeros(2),
        ("front",),
        {"target_visible": True, "safety_ok": True, "aligned": True},
    )
    contexts = [EvaluationContext(observation, "cable_tie:align", True)] * 2
    metrics = evaluate_decisions(predictions, targets, graph, contexts)
    assert metrics.stage_accuracy == 0.5
    assert metrics.completion_accuracy == 0.5
    assert metrics.legal_task_selection_rate == 1.0
