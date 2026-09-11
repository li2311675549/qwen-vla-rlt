from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from .schemas import BrainDecision, Observation, PolicyBackend


@dataclass(frozen=True, slots=True)
class TaskNode:
    task_id: str
    stage_id: str
    skill_id: str
    prompt: str
    allowed_next: frozenset[str] = field(default_factory=frozenset)
    required_facts: Mapping[str, bool | float | int | str] = field(default_factory=dict)
    supported_backends: frozenset[PolicyBackend] = field(
        default_factory=lambda: frozenset({PolicyBackend.PI05})
    )

    def __post_init__(self) -> None:
        if not all((self.task_id, self.stage_id, self.skill_id, self.prompt)):
            raise ValueError("task node identity, skill, and prompt cannot be empty")
        if not self.supported_backends:
            raise ValueError("task node must support at least one backend")

    @property
    def key(self) -> str:
        return f"{self.task_id}:{self.stage_id}"


class TaskGraph:
    def __init__(self, nodes: list[TaskNode], initial_key: str) -> None:
        self.nodes = {node.key: node for node in nodes}
        if len(self.nodes) != len(nodes):
            raise ValueError("duplicate task/stage node")
        if initial_key not in self.nodes:
            raise ValueError("initial_key is not registered")
        self.initial_key = initial_key
        for node in nodes:
            unknown = node.allowed_next - self.nodes.keys()
            if unknown:
                raise ValueError(f"node {node.key} references unknown transitions: {unknown}")

    def node_for(self, decision: BrainDecision) -> TaskNode:
        key = f"{decision.task_id}:{decision.stage_id}"
        if key not in self.nodes:
            raise ValueError(f"unknown task/stage: {key}")
        node = self.nodes[key]
        if node.skill_id != decision.skill_id:
            raise ValueError("decision skill does not match registered task node")
        if decision.next_task_id is not None:
            allowed_task_ids = {self.nodes[next_key].task_id for next_key in node.allowed_next}
            if decision.next_task_id not in allowed_task_ids:
                raise ValueError("next_task_id is not reachable from the current node")
        return node

    def validate_facts(self, node: TaskNode, observation: Observation) -> None:
        missing = {
            key: expected
            for key, expected in node.required_facts.items()
            if observation.facts.get(key) != expected
        }
        if missing:
            raise ValueError(f"task preconditions are not satisfied: {missing}")

    def validate_transition(self, current_key: str, proposed_key: str) -> None:
        if proposed_key == current_key:
            return
        if (
            current_key not in self.nodes
            or proposed_key not in self.nodes[current_key].allowed_next
        ):
            raise ValueError(f"illegal task transition: {current_key} -> {proposed_key}")
