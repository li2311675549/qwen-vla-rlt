from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from time import monotonic

from numpy.typing import NDArray

from .brain import parse_brain_decision
from .policies import ActionGuard, Policy, wrap_action
from .router import PolicyRouter
from .schemas import ActionChunk, CompletionStatus, Observation, PolicyBackend
from .state_machine import StateUpdate, TaskStateMachine


@dataclass(frozen=True, slots=True)
class RuntimeResult:
    state: StateUpdate
    action: ActionChunk | None
    reason: str = "action_ready"


class HierarchicalRuntime:
    """Validate before inference and again before releasing an action to the controller."""

    def __init__(
        self,
        router: PolicyRouter,
        state_machine: TaskStateMachine,
        pi05: Policy,
        rlt: Policy,
        guard: ActionGuard,
        clock=monotonic,
    ) -> None:
        if router.graph is not state_machine.graph:
            raise ValueError("router and state machine must share one task graph")
        self.router, self.state_machine, self.guard = router, state_machine, guard
        self.policies = {PolicyBackend.PI05: pi05, PolicyBackend.RLT: rlt}
        self.clock = clock
        self._generation = 0
        self._halted = False
        self._lock = RLock()
        self._pending: ActionChunk | None = None
        self._capture_time: float | None = None

    def halt(self) -> None:
        """Manual takeover/stop invalidates in-flight inference and pending action chunks."""
        with self._lock:
            self._generation += 1
            self._halted = True
            self._pending = None

    def reset(self) -> None:
        """Start a fresh episode, clearing completion memory and all command versions."""
        with self._lock:
            self._generation += 1
            self._halted = False
            self._pending = None
            old = self.state_machine
            self.state_machine = TaskStateMachine(
                old.graph,
                confirmations=old.confirmations,
                min_dwell=old.min_dwell,
            )

    def take_action(self, *, now_s: float) -> ActionChunk | None:
        """Sender must consume through this method; stale/expired actions never leave it."""
        with self._lock:
            action, self._pending = self._pending, None
            if action is None or self._halted:
                return None
            maximum_age = self.router.max_observation_age_s
            if maximum_age is not None:
                if self._capture_time is None or not 0 <= now_s - self._capture_time <= maximum_age:
                    return None
            if action.decision_version != self.state_machine.latest_version:
                return None
            return action

    def step(
        self,
        raw_decision: str | dict[str, object],
        observation: Observation,
        current_action: NDArray,
        *,
        now_s: float | None = None,
    ) -> RuntimeResult:
        start = self.clock()
        with self._lock:
            self._generation += 1
            generation = self._generation
            self._pending = None
            if self._halted or observation.facts.get("human_intervention") is True:
                self.halt()
                return RuntimeResult(
                    StateUpdate(self.state_machine.stable_key, False, "halted"), None, "halted"
                )
            decision = parse_brain_decision(raw_decision)
            request = self.router.route(decision, observation, now_s=now_s)
            state = self.state_machine.update(decision)
            if request.backend in {PolicyBackend.WAIT, PolicyBackend.STOP}:
                if request.backend is PolicyBackend.STOP:
                    self.halt()
                return RuntimeResult(state, None, request.backend.value)
            if f"{decision.task_id}:{decision.stage_id}" != state.stable_key:
                return RuntimeResult(state, None, "awaiting_transition")
            if decision.completion_status is CompletionStatus.COMPLETE:
                return RuntimeResult(state, None, "stage_complete")
        # Allow halt/reset/new decisions to cancel a slow policy call.
        try:
            values = self.policies[request.backend].infer(request)
        except Exception:
            self.halt()
            raise
        with self._lock:
            if generation != self._generation or self._halted:
                return RuntimeResult(state, None, "superseded")
            if now_s is not None:
                try:
                    self.router.route(decision, observation, now_s=now_s + self.clock() - start)
                except ValueError:
                    return RuntimeResult(state, None, "expired_during_inference")
            action = self.guard.validate(wrap_action(request, values), current_action)
            self._pending, self._capture_time = action, observation.capture_time_s
            return RuntimeResult(state, action)
