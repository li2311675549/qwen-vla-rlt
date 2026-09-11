from __future__ import annotations

import numpy as np

from qwen_vla_rlt.adapters import RLTResidualPolicy
from qwen_vla_rlt.policies import ActionGuard, CallablePolicy
from qwen_vla_rlt.presets import build_multitask_graph, build_skill_registry
from qwen_vla_rlt.rlt.models import ResidualActor
from qwen_vla_rlt.router import PolicyRouter, RLTArtifact
from qwen_vla_rlt.runtime import HierarchicalRuntime
from qwen_vla_rlt.schemas import Observation
from qwen_vla_rlt.state_machine import TaskStateMachine

graph = build_multitask_graph("cable_tie:align")
reference = CallablePolicy(lambda _: np.full((4, 2), 0.05, dtype=np.float32))
actor = ResidualActor(3, 2, (4, 2), np.array([0.02, 0.02]), seed=7)
rlt = RLTResidualPolicy(reference, lambda _: np.zeros(2, dtype=np.float32), actor)
runtime = HierarchicalRuntime(
    PolicyRouter(
        graph,
        build_skill_registry(),
        rlt_artifacts={"align_hole": RLTArtifact("align_hole", "v1", "v1")},
    ),
    TaskStateMachine(graph),
    reference,
    rlt,
    ActionGuard(np.full(2, -1.0), np.full(2, 1.0), np.full(2, 0.1)),
)
observation = Observation(
    "demo-session",
    "observation-0",
    0.0,
    np.zeros(3, dtype=np.float32),
    ("front", "left_wrist", "right_wrist"),
    {"target_visible": True, "safety_ok": True},
)
decision = {
    "session_id": "demo-session",
    "observation_id": "observation-0",
    "task_id": "cable_tie",
    "stage_id": "align",
    "completion_status": "not_complete",
    "next_task_id": None,
    "skill_id": "align_hole",
    "skill_args": {"speed": 0.05},
    "policy_backend": "rlt",
    "decision_version": 0,
    "confidence": 0.98,
}
result = runtime.step(decision, observation, np.zeros(2, dtype=np.float32))
assert result.action is not None
print(
    {
        "state": result.state.stable_key,
        "backend": result.action.backend,
        "action": result.action.values,
    }
)
