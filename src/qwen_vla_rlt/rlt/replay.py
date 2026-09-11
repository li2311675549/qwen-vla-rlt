from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray


class ReplaySource(StrEnum):
    POLICY = "policy"
    HUMAN_INTERVENTION = "human_intervention"


@dataclass(frozen=True, slots=True)
class Transition:
    state: NDArray[np.float32]
    token: NDArray[np.float32]
    action: NDArray[np.float32]
    reference: NDArray[np.float32]
    rewards: NDArray[np.float32]
    next_state: NDArray[np.float32]
    next_token: NDArray[np.float32]
    next_reference: NDArray[np.float32]
    done: bool
    source: ReplaySource = ReplaySource.POLICY
    executed_steps: int | None = None

    def __post_init__(self) -> None:
        names = (
            "state",
            "token",
            "action",
            "reference",
            "rewards",
            "next_state",
            "next_token",
            "next_reference",
        )
        arrays = {name: np.asarray(getattr(self, name), dtype=np.float32) for name in names}
        if any(not np.all(np.isfinite(value)) for value in arrays.values()):
            raise ValueError("transition arrays must be finite")
        if arrays["state"].ndim != 1 or arrays["state"].shape != arrays["next_state"].shape:
            raise ValueError("state and next_state must be equal-sized vectors")
        if arrays["token"].ndim != 1 or arrays["token"].shape != arrays["next_token"].shape:
            raise ValueError("token and next_token must be equal-sized vectors")
        if arrays["action"].ndim != 2 or 0 in arrays["action"].shape:
            raise ValueError("action must be a non-empty chunk matrix")
        if arrays["action"].shape != arrays["reference"].shape:
            raise ValueError("action and reference shapes differ")
        if arrays["action"].shape != arrays["next_reference"].shape:
            raise ValueError("next_reference must describe the next observation")
        if arrays["rewards"].shape != (arrays["action"].shape[0],):
            raise ValueError("rewards must contain one value per action-chunk step")
        steps = arrays["action"].shape[0] if self.executed_steps is None else self.executed_steps
        if type(steps) is not int or not 1 <= steps <= arrays["action"].shape[0]:
            raise ValueError("executed_steps must be within the action chunk")
        if np.any(arrays["rewards"][steps:] != 0):
            raise ValueError("unexecuted chunk steps must have zero reward")
        if type(self.done) is not bool:
            raise ValueError("done must be boolean")
        object.__setattr__(self, "source", ReplaySource(self.source))
        object.__setattr__(self, "executed_steps", steps)
        for name, value in arrays.items():
            immutable = value.copy()
            immutable.setflags(write=False)
            object.__setattr__(self, name, immutable)


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.items: list[Transition] = []
        self.position = 0
        self.rng = np.random.default_rng(seed)

    def add(self, transition: Transition) -> None:
        if len(self.items) < self.capacity:
            self.items.append(transition)
        else:
            self.items[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> list[Transition]:
        if batch_size < 1 or batch_size > len(self.items):
            raise ValueError("invalid replay batch size")
        indices = self.rng.choice(len(self.items), batch_size, replace=False)
        return [self.items[int(index)] for index in indices]

    def __len__(self) -> int:
        return len(self.items)
