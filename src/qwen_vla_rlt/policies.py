from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .schemas import ActionChunk, PolicyBackend, PolicyRequest


class Policy(Protocol):
    def infer(self, request: PolicyRequest) -> NDArray[np.float32]: ...


class CallablePolicy:
    def __init__(self, fn: Callable[[PolicyRequest], NDArray[np.float32]]) -> None:
        self.fn = fn

    def infer(self, request: PolicyRequest) -> NDArray[np.float32]:
        return np.asarray(self.fn(request), dtype=np.float32)


class ActionGuard:
    """Final sender-side validation for joint/gripper action chunks."""

    def __init__(self, lower: NDArray, upper: NDArray, max_step_delta: NDArray) -> None:
        self.lower = np.asarray(lower, dtype=np.float32)
        self.upper = np.asarray(upper, dtype=np.float32)
        self.max_step_delta = np.asarray(max_step_delta, dtype=np.float32)
        if self.lower.ndim != 1 or self.lower.size == 0:
            raise ValueError("action bounds must be non-empty vectors")
        if any(not np.isfinite(x).all() for x in (self.lower, self.upper, self.max_step_delta)):
            raise ValueError("action bounds must be finite")
        if not (self.lower.shape == self.upper.shape == self.max_step_delta.shape):
            raise ValueError("action bounds must share shape")
        if np.any(self.lower > self.upper) or np.any(self.max_step_delta <= 0):
            raise ValueError("invalid action bounds")

    def validate(self, chunk: ActionChunk, current_action: NDArray) -> ActionChunk:
        values = chunk.values.copy()
        if values.shape[1] != len(self.lower):
            raise ValueError("action dimension does not match guard")
        previous = np.asarray(current_action, dtype=np.float32)
        if previous.shape != self.lower.shape or not np.all(np.isfinite(previous)):
            raise ValueError("invalid current action")
        if np.any(previous < self.lower) or np.any(previous > self.upper):
            raise ValueError("current action is outside configured bounds")
        for index in range(len(values)):
            values[index] = np.clip(values[index], self.lower, self.upper)
            delta = np.clip(values[index] - previous, -self.max_step_delta, self.max_step_delta)
            values[index] = previous + delta
            previous = values[index]
        return ActionChunk(values, chunk.observation_id, chunk.decision_version, chunk.backend)


def wrap_action(request: PolicyRequest, values: NDArray) -> ActionChunk:
    if request.backend not in {PolicyBackend.PI05, PolicyBackend.RLT}:
        raise ValueError("backend does not produce actions")
    return ActionChunk(
        np.asarray(values, dtype=np.float32),
        request.observation.observation_id,
        request.decision.decision_version,
        request.backend,
    )
