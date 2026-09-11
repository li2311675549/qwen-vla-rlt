from __future__ import annotations

from dataclasses import dataclass

from .schemas import BrainDecision, CompletionStatus
from .task_graph import TaskGraph


@dataclass(frozen=True, slots=True)
class StateUpdate:
    stable_key: str
    switched: bool
    reason: str


class TaskStateMachine:
    """Debounced task state with dwell, completion, and monotonic version checks."""

    def __init__(self, graph: TaskGraph, *, confirmations: int = 2, min_dwell: int = 1) -> None:
        if confirmations < 1 or min_dwell < 0:
            raise ValueError("invalid stabilization settings")
        self.graph = graph
        self.confirmations = confirmations
        self.min_dwell = min_dwell
        self.stable_key = graph.initial_key
        self.candidate_key: str | None = None
        self.candidate_count = 0
        self.dwell = 0
        self.latest_version = -1
        self.completed: set[str] = set()
        self.latest_observation: str | None = None
        self.session_id: str | None = None

    def update(self, decision: BrainDecision) -> StateUpdate:
        if decision.decision_version <= self.latest_version:
            raise ValueError("stale or replayed decision version")
        if self.session_id is not None and decision.session_id != self.session_id:
            raise ValueError("state machine belongs to another session; reset required")
        if decision.observation_id == self.latest_observation:
            raise ValueError("repeated observation cannot confirm a transition twice")
        proposed = f"{decision.task_id}:{decision.stage_id}"
        self.graph.node_for(decision)
        self.graph.validate_transition(self.stable_key, proposed)
        self.latest_version = decision.decision_version
        self.latest_observation = decision.observation_id
        self.session_id = decision.session_id
        self.dwell += 1
        if proposed == self.stable_key:
            self.candidate_key = None
            self.candidate_count = 0
            if decision.completion_status is CompletionStatus.COMPLETE:
                self.completed.add(self.stable_key)
            else:
                self.completed.discard(self.stable_key)
            return StateUpdate(self.stable_key, False, "same_stage")
        if self.stable_key not in self.completed:
            return StateUpdate(self.stable_key, False, "current_stage_not_complete")
        if self.dwell <= self.min_dwell:
            return StateUpdate(self.stable_key, False, "minimum_dwell")
        if proposed != self.candidate_key:
            self.candidate_key = proposed
            self.candidate_count = 0
        self.candidate_count += 1
        if self.candidate_count < self.confirmations:
            return StateUpdate(self.stable_key, False, "awaiting_confirmation")
        self.stable_key = proposed
        # A new visit must earn completion again, including graph cycles.
        self.completed.discard(proposed)
        self.candidate_key = None
        self.candidate_count = 0
        self.dwell = 0
        return StateUpdate(self.stable_key, True, "confirmed_transition")
