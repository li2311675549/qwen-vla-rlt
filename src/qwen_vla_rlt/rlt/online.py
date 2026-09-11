"""Online collection contract records executed human/policy actions and true next reference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .replay import ReplayBuffer, ReplaySource, Transition


@dataclass(frozen=True)
class EncodedObservation:
    state: np.ndarray
    token: np.ndarray
    reference: np.ndarray


@dataclass(frozen=True)
class ExecutionFeedback:
    next_observation: object
    executed_action: np.ndarray
    rewards: np.ndarray
    done: bool
    executed_steps: int
    human_intervention: bool = False


class ChunkEnvironment(Protocol):
    def execute(self, proposed_action: np.ndarray) -> ExecutionFeedback: ...


class OnlineRLTLoop:
    """One step of observe -> act -> execute -> encode next -> replay -> update.

    encode must use a frozen VLA with the same prompt and normalization version. The
    environment adapter owns robot limits and returns the actual commanded action.
    """

    def __init__(
        self,
        environment: ChunkEnvironment,
        encode,
        act,
        replay: ReplayBuffer,
        trainer,
        batch_size: int = 32,
        warmup_chunks: int = 32,
    ) -> None:
        if batch_size < 1 or warmup_chunks < 0:
            raise ValueError("invalid online-learning schedule")
        self.environment, self.encode, self.act = environment, encode, act
        self.replay, self.trainer = replay, trainer
        self.batch_size, self.warmup_chunks = batch_size, warmup_chunks
        self.collected = 0

    def step(self, observation) -> tuple[ExecutionFeedback, object | None]:
        current = self.encode(observation)
        proposed = (
            current.reference.copy() if self.collected < self.warmup_chunks else self.act(current)
        )
        feedback = self.environment.execute(proposed)
        following = self.encode(feedback.next_observation)
        item = Transition(
            current.state,
            current.token,
            feedback.executed_action,
            current.reference,
            feedback.rewards,
            following.state,
            following.token,
            following.reference,
            feedback.done,
            ReplaySource.HUMAN_INTERVENTION if feedback.human_intervention else ReplaySource.POLICY,
            feedback.executed_steps,
        )
        self.replay.add(item)
        self.collected += 1
        stats = None
        if len(self.replay) >= self.batch_size and self.collected >= self.warmup_chunks:
            stats = self.trainer.train_step(self.replay.sample(self.batch_size))
        return feedback, stats
