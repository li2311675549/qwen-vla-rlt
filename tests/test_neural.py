from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from qwen_vla_rlt.rlt.neural import (  # noqa: E402
    FrozenVLAFeatures,
    NeuralResidualActor,
    NeuralRLTTrainer,
    NeuralTwinQ,
    RLTConfig,
    RLTokenAutoencoder,
)
from qwen_vla_rlt.rlt.replay import Transition  # noqa: E402

torch.set_num_threads(1)


def config():
    return RLTConfig(2, 8, 2, 3, hidden_dim=16, completion_index=2, target_noise=0)


def transitions():
    return [
        Transition(
            np.zeros(2),
            np.zeros(8),
            np.ones((2, 3)) * 0.2,
            np.ones((2, 3)) * 0.1,
            np.array([0, 1]),
            np.ones(2) * 0.2,
            np.ones(8) * 0.1,
            np.ones((2, 3)) * 0.3,
            False,
        )
    ]


def test_masked_token_learning_detaches_vla_and_ignores_padding():
    torch.manual_seed(3)
    model = RLTokenAutoencoder(8, 5, heads=2, layers=1)
    tokens = torch.randn(2, 4, 8, requires_grad=True)
    valid = torch.tensor([[True, True, False, False], [True, True, True, False]])
    loss, latent = model(tokens, valid)
    loss.backward()
    assert latent.shape == (2, 8)
    assert tokens.grad is None
    assert model.rl_query.grad is not None and model.rl_query.grad.abs().sum() > 0
    changed = tokens.detach().clone()
    changed[~valid] = float("nan")
    torch.testing.assert_close(model.encode(tokens, valid), model.encode(changed, valid))
    with pytest.raises(ValueError, match="finite valid"):
        model(tokens, torch.zeros_like(valid))


def test_zero_init_residual_and_completion_cannot_be_explored_or_exploit_q():
    cfg = config()
    actor, critic = NeuralResidualActor(cfg), NeuralTwinQ(cfg)
    state, token, reference = torch.zeros(2, 2), torch.zeros(2, 8), torch.ones(2, 2, 3)
    torch.testing.assert_close(actor(state, token, reference), reference)
    action = actor(state, token, reference, torch.full_like(reference, 100))
    assert (action - reference).abs().max() <= cfg.residual_limit + 1e-6
    torch.testing.assert_close(action[:, :, 2], reference[:, :, 2])
    q = critic(state, token, action, reference)
    modified = action.clone()
    modified[:, :, 2] = 999
    modified_ref = reference.clone()
    modified_ref[:, :, 2] = -999
    for a, b in zip(q, critic(state, token, modified, modified_ref), strict=True):
        torch.testing.assert_close(a, b)


def test_neural_training_delayed_actor_and_checkpoint(tmp_path):
    trainer = NeuralRLTTrainer(config())
    assert not trainer.train_step(transitions())["actor_updated"]
    assert trainer.train_step(transitions())["actor_updated"]
    path = tmp_path / "rlt.pt"
    trainer.save(path)
    restored = NeuralRLTTrainer.load(path)
    assert restored.steps == 2
    for key, value in trainer.actor.state_dict().items():
        torch.testing.assert_close(restored.actor.state_dict()[key], value)
    assert np.isfinite(restored.train_step(transitions())["critic_loss"])


def test_bootstrap_uses_next_reference_and_done_removes_bootstrap():
    trainer = NeuralRLTTrainer(config())
    # Force both target Q networks to depend monotonically on the next reference.
    with torch.no_grad():
        for p in trainer.target_critic.parameters():
            p.fill_(0.1)
    a = trainer.targets(trainer._batch(transitions()))
    changed = [replace(transitions()[0], next_reference=np.ones((2, 3)) * 0.9)]
    b = trainer.targets(trainer._batch(changed))
    assert not torch.allclose(a, b)
    done = [replace(changed[0], done=True)]
    torch.testing.assert_close(
        trainer.targets(trainer._batch(done)), torch.tensor([config().gamma])
    )


def test_frozen_vla_has_no_trainable_parameters_or_gradients():
    model = torch.nn.Linear(2, 8)

    def extract(backbone, state):
        return (
            backbone(state).unsqueeze(1),
            torch.ones(1, 1, dtype=torch.bool),
            torch.zeros(1, 2, 3),
        )

    frozen = FrozenVLAFeatures(model, extract)
    tokens, _, _ = frozen(torch.zeros(1, 2))
    assert not tokens.requires_grad and not any(p.requires_grad for p in model.parameters())
