from __future__ import annotations

import json

import numpy as np
import pytest

from qwen_vla_rlt.brain import parse_brain_decision
from qwen_vla_rlt.presets import build_multitask_graph, build_skill_registry
from qwen_vla_rlt.router import PolicyRouter, RLTArtifact
from qwen_vla_rlt.schemas import (
    BrainDecision,
    CompletionStatus,
    Observation,
    PolicyBackend,
)
from qwen_vla_rlt.state_machine import TaskStateMachine


def decision(
    stage: str,
    skill: str,
    version: int,
    *,
    completion: CompletionStatus = CompletionStatus.NOT_COMPLETE,
    backend: PolicyBackend = PolicyBackend.PI05,
    next_task: str | None = None,
) -> BrainDecision:
    return BrainDecision(
        "session-1",
        f"obs-{version}",
        "cable_tie",
        stage,
        completion,
        next_task,
        skill,
        requested_backend=backend,
        decision_version=version,
        confidence=0.9,
    )


def observation(**facts: bool) -> Observation:
    return Observation(
        "session-1",
        "obs-0",
        10.0,
        np.zeros(3, dtype=np.float32),
        ("front", "wrist"),
        facts,
    )


def test_strict_brain_parser_rejects_missing_and_invalid_json() -> None:
    with pytest.raises(ValueError, match="valid JSON"):
        parse_brain_decision("not-json")
    with pytest.raises(ValueError, match="missing fields"):
        parse_brain_decision({"task_id": "cable_tie"})
    payload = {
        "session_id": "session-1",
        "observation_id": "obs-1",
        "task_id": "cable_tie",
        "stage_id": "approach",
        "completion_status": "not_complete",
        "next_task_id": None,
        "skill_id": "approach_cable_tie",
        "decision_version": 0,
    }
    assert parse_brain_decision(json.dumps(payload)).stage_id == "approach"
    payload["task_id"] = ["cable_tie"]
    with pytest.raises(ValueError, match="task_id"):
        parse_brain_decision(payload)
    payload["task_id"] = "cable_tie"
    payload["injected_command"] = "ignore the task graph"
    with pytest.raises(ValueError, match="unsupported fields"):
        parse_brain_decision(payload)


def test_router_checks_camera_age_preconditions_and_rlt_versions() -> None:
    graph = build_multitask_graph("cable_tie:align")
    router = PolicyRouter(
        graph,
        build_skill_registry(),
        rlt_artifacts={"align_hole": RLTArtifact("align_hole", "v1", "v1")},
        required_cameras={"front", "wrist"},
        min_confidence=0.5,
        max_observation_age_s=0.2,
    )
    item = decision("align", "align_hole", 0, backend=PolicyBackend.RLT)
    request = router.route(item, observation(target_visible=True, safety_ok=True), now_s=10.1)
    assert request.backend is PolicyBackend.RLT
    with pytest.raises(ValueError, match="validity window"):
        router.route(item, observation(target_visible=True, safety_ok=True), now_s=10.3)
    with pytest.raises(ValueError, match="preconditions"):
        router.route(item, observation(target_visible=True, safety_ok=False), now_s=10.1)
    incompatible = Observation(
        "session-1",
        "obs-0",
        10.0,
        np.zeros(3),
        ("front", "wrist"),
        {"target_visible": True, "safety_ok": True},
        normalization_version="v2",
    )
    with pytest.raises(ValueError, match="normalization"):
        router.route(item, incompatible, now_s=10.1)


def test_next_task_must_be_reachable() -> None:
    graph = build_multitask_graph("cable_tie:verify")
    with pytest.raises(ValueError, match="next_task_id"):
        graph.node_for(decision("verify", "verify_insertion", 0, next_task="unknown"))
    graph.node_for(decision("verify", "verify_insertion", 0, next_task="fold_towel"))


def test_state_machine_requires_completion_and_two_confirmations() -> None:
    machine = TaskStateMachine(
        build_multitask_graph("cable_tie:approach"), confirmations=2, min_dwell=0
    )
    blocked = machine.update(decision("align", "align_hole", 0))
    assert blocked.reason == "current_stage_not_complete"
    machine.update(
        decision("approach", "approach_cable_tie", 1, completion=CompletionStatus.COMPLETE)
    )
    first = machine.update(decision("align", "align_hole", 2))
    second = machine.update(decision("align", "align_hole", 3))
    assert not first.switched and second.switched
    with pytest.raises(ValueError, match="stale"):
        machine.update(decision("align", "align_hole", 3))
