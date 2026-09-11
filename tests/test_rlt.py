from __future__ import annotations

import numpy as np
import pytest

from qwen_vla_rlt.rlt.models import ResidualActor, TwinLinearCritic
from qwen_vla_rlt.rlt.replay import ReplayBuffer, ReplaySource, Transition
from qwen_vla_rlt.rlt.token import LinearRLTokenAutoencoder
from qwen_vla_rlt.rlt.training import RLTTrainer, discounted_chunk_return


def transition(next_reference_value: float = 0.4) -> Transition:
    return Transition(
        state=np.array([0.1, 0.2], dtype=np.float32),
        token=np.array([0.3], dtype=np.float32),
        action=np.full((2, 2), 0.2, dtype=np.float32),
        reference=np.full((2, 2), 0.1, dtype=np.float32),
        rewards=np.array([0.2, 1.0], dtype=np.float32),
        next_state=np.array([0.2, 0.3], dtype=np.float32),
        next_token=np.array([0.4], dtype=np.float32),
        next_reference=np.full((2, 2), next_reference_value, dtype=np.float32),
        done=False,
        source=ReplaySource.HUMAN_INTERVENTION,
    )


def test_rl_token_autoencoder_fits_encodes_and_reconstructs() -> None:
    samples = np.arange(48, dtype=np.float32).reshape(4, 3, 4)
    encoder = LinearRLTokenAutoencoder(latent_dim=2).fit(samples)
    token = encoder.encode(samples[0])
    assert token.shape == (2,)
    assert encoder.reconstruct(token).shape == (3, 4)
    with pytest.raises(ValueError, match="shape"):
        encoder.encode(np.zeros((2, 4)))


def test_actor_residual_and_exploration_stay_bounded() -> None:
    actor = ResidualActor(
        2,
        1,
        (2, 3),
        np.array([0.1, 0.2, 0.3]),
        trainable_action_mask=np.array([True, True, False]),
    )
    reference = np.zeros((2, 3), dtype=np.float32)
    action = actor.act(
        np.zeros(2), np.zeros(1), reference, exploration_noise=np.full((2, 3), 100.0)
    )
    assert np.all(np.abs(action[:, :2]) <= np.array([0.1, 0.2]) + 1e-6)
    assert np.all(action[:, 2] == reference[:, 2])


def test_replay_keeps_true_next_reference_and_samples_interventions() -> None:
    buffer = ReplayBuffer(2, seed=1)
    item = transition(0.7)
    buffer.add(item)
    sampled = buffer.sample(1)[0]
    assert np.all(sampled.next_reference == 0.7)
    assert sampled.source is ReplaySource.HUMAN_INTERVENTION
    with pytest.raises(ValueError, match="read-only"):
        item.next_reference[:] = 99.0
    assert np.all(sampled.next_reference == 0.7)


def test_chunk_discount_and_training_step_are_executable() -> None:
    assert discounted_chunk_return(np.array([1.0, 2.0]), 0.5) == pytest.approx(2.0)
    actor = ResidualActor(2, 1, (2, 2), np.array([0.2, 0.2]), seed=3)
    critic = TwinLinearCritic(2, 1, (2, 2), seed=4)
    trainer = RLTTrainer(actor, critic)
    before = critic.w1.copy()
    stats = trainer.train_step([transition()])
    assert np.isfinite([stats.critic_loss, stats.actor_loss, stats.target_mean]).all()
    assert not np.array_equal(before, critic.w1)


def test_td_target_uses_next_observation_reference() -> None:
    actor = ResidualActor(2, 1, (2, 2), np.array([0.2, 0.2]), seed=5)
    critic = TwinLinearCritic(2, 1, (2, 2), seed=6)
    reference_start = 2 + 1 + 4
    critic.target_w1[reference_start:] = 1.0
    critic.target_w2[reference_start:] = 1.0
    trainer = RLTTrainer(actor, critic)
    assert trainer.td_target(transition(0.9)) != pytest.approx(trainer.td_target(transition(0.1)))
