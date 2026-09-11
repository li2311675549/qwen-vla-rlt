from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import ResidualActor, TwinLinearCritic
from .replay import ReplaySource, Transition


def discounted_chunk_return(rewards: np.ndarray, gamma: float) -> float:
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be within [0, 1]")
    values = np.asarray(rewards, dtype=np.float32)
    if values.ndim != 1 or len(values) < 1 or not np.all(np.isfinite(values)):
        raise ValueError("rewards must be a finite non-empty vector")
    return float(sum((gamma**index) * value for index, value in enumerate(values)))


@dataclass(frozen=True, slots=True)
class TrainingStats:
    critic_loss: float
    actor_loss: float
    target_mean: float


class RLTTrainer:
    """Chunk-level residual actor/Twin-Q update with a behavior-cloning anchor."""

    def __init__(
        self,
        actor: ResidualActor,
        critic: TwinLinearCritic,
        *,
        gamma: float = 0.99,
        critic_lr: float = 1e-3,
        actor_lr: float = 1e-4,
        bc_weight: float = 0.1,
        tau: float = 0.01,
    ) -> None:
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be within [0, 1]")
        if critic_lr <= 0 or actor_lr <= 0 or bc_weight < 0:
            raise ValueError("learning rates must be positive and BC weight non-negative")
        if not 0.0 < tau <= 1.0:
            raise ValueError("tau must be within (0, 1]")
        self.actor = actor
        self.critic = critic
        self.gamma = gamma
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr
        self.bc_weight = bc_weight
        self.tau = tau

    def td_target(self, transition: Transition) -> float:
        immediate = discounted_chunk_return(
            transition.rewards[: transition.executed_steps], self.gamma
        )
        if transition.done:
            return immediate
        next_action = self.actor.act(
            transition.next_state,
            transition.next_token,
            transition.next_reference,
        )
        next_q = min(
            self.critic.predict(
                transition.next_state,
                transition.next_token,
                next_action,
                transition.next_reference,
                target=True,
            )
        )
        return immediate + (self.gamma**transition.executed_steps) * next_q

    def train_step(self, batch: list[Transition]) -> TrainingStats:
        if not batch:
            raise ValueError("training batch cannot be empty")
        targets = [self.td_target(item) for item in batch]
        critic_batch = [
            (item.state, item.token, item.action, item.reference, target)
            for item, target in zip(batch, targets, strict=True)
        ]
        critic_loss = self.critic.update(critic_batch, self.critic_lr)
        actor_batch = []
        for item in batch:
            action = self.actor.act(item.state, item.token, item.reference)
            q_grad = self.critic.action_gradient(item.state, item.token, action, item.reference)
            anchor = (
                item.action if item.source is ReplaySource.HUMAN_INTERVENTION else item.reference
            )
            actor_batch.append((item.state, item.token, item.reference, q_grad, anchor))
        actor_loss = self.actor.update(
            actor_batch,
            learning_rate=self.actor_lr,
            bc_weight=self.bc_weight,
        )
        self.critic.soft_update(self.tau)
        return TrainingStats(critic_loss, actor_loss, float(np.mean(targets)))
