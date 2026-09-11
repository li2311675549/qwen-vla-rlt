"""PyTorch RL Token and residual TD3+BC implementation (optional ``neural`` extra)."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from .replay import ReplaySource, Transition


class RLTokenAutoencoder(nn.Module):
    """Learn a single token bottleneck from detached, masked VLA embeddings."""

    def __init__(
        self, embedding_dim: int, max_tokens: int, heads: int = 4, layers: int = 2
    ) -> None:
        super().__init__()
        if min(embedding_dim, max_tokens, heads, layers) < 1 or embedding_dim % heads:
            raise ValueError("invalid RL Token dimensions")
        self.embedding_dim = embedding_dim
        self.max_tokens = max_tokens
        self.rl_query = nn.Parameter(torch.randn(1, 1, embedding_dim) * 0.02)
        self.positions = nn.Parameter(torch.randn(1, max_tokens, embedding_dim) * 0.02)
        encoder = nn.TransformerEncoderLayer(
            embedding_dim,
            heads,
            4 * embedding_dim,
            dropout=0,
            batch_first=True,
        )
        decoder = nn.TransformerDecoderLayer(
            embedding_dim,
            heads,
            4 * embedding_dim,
            dropout=0,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder, layers, enable_nested_tensor=False)
        self.decoder = nn.TransformerDecoder(decoder, layers)
        self.output = nn.Linear(embedding_dim, embedding_dim)

    def _validate(self, tokens: Tensor, valid: Tensor) -> None:
        if tokens.ndim != 3 or tokens.shape[-1] != self.embedding_dim:
            raise ValueError("tokens must have shape [batch, tokens, embedding_dim]")
        if not 1 <= tokens.shape[1] <= self.max_tokens:
            raise ValueError("token sequence exceeds configured capacity")
        if valid.dtype != torch.bool or valid.shape != tokens.shape[:2]:
            raise ValueError("valid mask must be boolean [batch, tokens]")
        if not valid.any(dim=1).all() or not torch.isfinite(tokens[valid]).all():
            raise ValueError("each sample needs finite valid tokens")

    def encode(self, tokens: Tensor, valid: Tensor) -> Tensor:
        self._validate(tokens, valid)
        batch, length, _ = tokens.shape
        clean = tokens.detach().masked_fill(~valid.unsqueeze(-1), 0)
        source = torch.cat(
            (clean + self.positions[:, :length], self.rl_query.expand(batch, -1, -1)), dim=1
        )
        ignore = torch.cat(
            (~valid, torch.zeros(batch, 1, dtype=torch.bool, device=valid.device)), dim=1
        )
        return self.encoder(source, src_key_padding_mask=ignore)[:, -1]

    def forward(self, tokens: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        latent = self.encode(tokens, valid)
        # Positional queries have no target-token shortcut around the bottleneck.
        queries = self.positions[:, : tokens.shape[1]].expand(tokens.shape[0], -1, -1)
        reconstructed = self.output(self.decoder(queries, latent.unsqueeze(1)))
        difference = reconstructed[valid] - tokens.detach()[valid]
        return difference.square().mean(), latent


@dataclass(frozen=True)
class RLTConfig:
    state_dim: int
    token_dim: int
    chunk_size: int
    action_dim: int
    hidden_dim: int = 256
    residual_limit: float = 0.1
    completion_index: int | None = None
    gamma: float = 0.99
    tau: float = 0.005
    learning_rate: float = 3e-4
    bc_weight: float = 0.1
    policy_delay: int = 2
    target_noise: float = 0.02

    def __post_init__(self) -> None:
        if (
            min(
                self.state_dim,
                self.token_dim,
                self.chunk_size,
                self.action_dim,
                self.hidden_dim,
                self.policy_delay,
            )
            < 1
        ):
            raise ValueError("model dimensions and policy_delay must be positive")
        if self.completion_index is not None and not 0 <= self.completion_index < self.action_dim:
            raise ValueError("completion_index is outside action vector")
        if not 0 <= self.gamma <= 1 or not 0 < self.tau <= 1:
            raise ValueError("invalid discount or Polyak coefficient")
        values = (self.residual_limit, self.learning_rate, self.bc_weight, self.target_noise)
        if not all(np.isfinite(x) for x in values) or min(values) < 0:
            raise ValueError("non-finite or negative optimization settings")
        if self.learning_rate == 0:
            raise ValueError("learning_rate must be positive")


def mlp(input_dim: int, output_dim: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, output_dim),
    )


class NeuralResidualActor(nn.Module):
    def __init__(self, config: RLTConfig) -> None:
        super().__init__()
        self.config = config
        width = config.chunk_size * config.action_dim
        self.net = mlp(config.state_dim + config.token_dim + width, width, config.hidden_dim)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        mask = torch.ones(config.chunk_size, config.action_dim)
        if config.completion_index is not None:
            mask[:, config.completion_index] = 0
        self.register_buffer("physical_mask", mask)

    def forward(
        self, state: Tensor, token: Tensor, reference: Tensor, noise: Tensor | None = None
    ) -> Tensor:
        cfg = self.config
        if state.ndim != 2 or state.shape[1] != cfg.state_dim:
            raise ValueError("invalid state shape")
        if token.shape != (len(state), cfg.token_dim):
            raise ValueError("invalid RL token shape")
        if reference.shape != (len(state), cfg.chunk_size, cfg.action_dim):
            raise ValueError("invalid reference chunk shape")
        features = torch.cat((state, token, (reference * self.physical_mask).flatten(1)), dim=1)
        residual = self.net(features).reshape_as(reference).tanh() * cfg.residual_limit
        if noise is not None:
            if noise.shape != reference.shape or not torch.isfinite(noise).all():
                raise ValueError("invalid exploration noise")
            residual = residual + noise
        residual = residual.clamp(-cfg.residual_limit, cfg.residual_limit) * self.physical_mask
        return reference + residual


class NeuralTwinQ(nn.Module):
    def __init__(self, config: RLTConfig) -> None:
        super().__init__()
        width = config.state_dim + config.token_dim + 2 * config.chunk_size * config.action_dim
        self.q1 = mlp(width, 1, config.hidden_dim)
        self.q2 = mlp(width, 1, config.hidden_dim)
        mask = torch.ones(config.chunk_size, config.action_dim)
        if config.completion_index is not None:
            mask[:, config.completion_index] = 0
        self.register_buffer("physical_mask", mask)

    def forward(
        self, state: Tensor, token: Tensor, action: Tensor, reference: Tensor
    ) -> tuple[Tensor, Tensor]:
        features = torch.cat(
            (
                state,
                token,
                (action * self.physical_mask).flatten(1),
                (reference * self.physical_mask).flatten(1),
            ),
            dim=1,
        )
        return self.q1(features).squeeze(-1), self.q2(features).squeeze(-1)


class NeuralRLTTrainer:
    """Delayed actor updates, target smoothing, Twin-Q and intervention-aware BC."""

    def __init__(self, config: RLTConfig, device: str = "cpu") -> None:
        self.config = config
        self.device = torch.device(device)
        self.actor = NeuralResidualActor(config).to(self.device)
        self.critic = NeuralTwinQ(config).to(self.device)
        self.target_actor = deepcopy(self.actor).eval().requires_grad_(False)
        self.target_critic = deepcopy(self.critic).eval().requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.learning_rate)
        self.steps = 0

    def _batch(self, transitions: list[Transition]) -> dict[str, Tensor]:
        if not transitions:
            raise ValueError("training batch cannot be empty")
        fields = (
            "state",
            "token",
            "action",
            "reference",
            "rewards",
            "next_state",
            "next_token",
            "next_reference",
        )
        batch = {
            key: torch.as_tensor(
                np.stack([getattr(t, key) for t in transitions]),
                dtype=torch.float32,
                device=self.device,
            )
            for key in fields
        }
        batch["done"] = torch.tensor([t.done for t in transitions], device=self.device)
        batch["steps"] = torch.tensor([t.executed_steps for t in transitions], device=self.device)
        batch["human"] = torch.tensor(
            [t.source is ReplaySource.HUMAN_INTERVENTION for t in transitions],
            device=self.device,
        )
        if batch["action"].shape[1:] != (self.config.chunk_size, self.config.action_dim):
            raise ValueError("replay chunk shape differs from model configuration")
        return batch

    @torch.no_grad()
    def targets(self, batch: dict[str, Tensor]) -> Tensor:
        cfg = self.config
        discount = cfg.gamma ** torch.arange(cfg.chunk_size, device=self.device)
        returns = (batch["rewards"] * discount).sum(dim=1)
        noise = torch.randn_like(batch["next_reference"]) * cfg.target_noise
        noise = noise.clamp(-2 * cfg.target_noise, 2 * cfg.target_noise)
        action = self.target_actor(
            batch["next_state"], batch["next_token"], batch["next_reference"], noise
        )
        q1, q2 = self.target_critic(
            batch["next_state"], batch["next_token"], action, batch["next_reference"]
        )
        return returns + cfg.gamma ** batch["steps"] * (~batch["done"]) * torch.minimum(q1, q2)

    def train_step(self, transitions: list[Transition]) -> dict[str, float | bool]:
        batch = self._batch(transitions)
        target = self.targets(batch)
        q1, q2 = self.critic(batch["state"], batch["token"], batch["action"], batch["reference"])
        critic_loss = (q1 - target).square().mean() + (q2 - target).square().mean()
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10, error_if_nonfinite=True)
        self.critic_optimizer.step()
        self.steps += 1
        updated = self.steps % self.config.policy_delay == 0
        actor_loss = torch.zeros((), device=self.device)
        if updated:
            self.critic.requires_grad_(False)
            try:
                action = self.actor(batch["state"], batch["token"], batch["reference"])
                qa, qb = self.critic(batch["state"], batch["token"], action, batch["reference"])
                anchor = torch.where(
                    batch["human"][:, None, None], batch["action"], batch["reference"]
                )
                valid_steps = (
                    torch.arange(self.config.chunk_size, device=self.device)[None, :]
                    < batch["steps"][:, None]
                )
                mask = valid_steps[:, :, None] * self.actor.physical_mask
                diff = (action - anchor) * mask
                bc = diff.square().sum() / mask.sum().clamp_min(1)
                actor_loss = -torch.minimum(qa, qb).mean() + self.config.bc_weight * bc
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 10, error_if_nonfinite=True)
                self.actor_optimizer.step()
            finally:
                self.critic.requires_grad_(True)
            with torch.no_grad():
                for online, target_net in (
                    (self.actor, self.target_actor),
                    (self.critic, self.target_critic),
                ):
                    for p, target_p in zip(
                        online.parameters(), target_net.parameters(), strict=True
                    ):
                        target_p.lerp_(p, self.config.tau)
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "actor_updated": updated,
            "target_mean": target.mean().item(),
        }

    def save(self, path: str | Path) -> None:
        payload = {"schema_version": 1, "config": asdict(self.config), "steps": self.steps}
        for name in (
            "actor",
            "critic",
            "target_actor",
            "target_critic",
            "actor_optimizer",
            "critic_optimizer",
        ):
            payload[name] = getattr(self, name).state_dict()
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> NeuralRLTTrainer:
        payload = torch.load(path, map_location=device, weights_only=True)
        if payload["schema_version"] != 1:
            raise ValueError("unsupported RLT checkpoint schema")
        trainer = cls(RLTConfig(**payload["config"]), device)
        for name in (
            "actor",
            "critic",
            "target_actor",
            "target_critic",
            "actor_optimizer",
            "critic_optimizer",
        ):
            getattr(trainer, name).load_state_dict(payload[name])
        trainer.steps = payload["steps"]
        return trainer


class FrozenVLAFeatures:
    """Freeze a VLA and extract detached tokens/reference from one observation call."""

    def __init__(self, model: nn.Module, extract) -> None:
        self.model = model.eval().requires_grad_(False)
        self.extract = extract

    @torch.no_grad()
    def __call__(self, observation) -> tuple[Tensor, Tensor, Tensor]:
        self.model.eval()
        tokens, valid, reference = self.extract(self.model, observation)
        return tokens.detach(), valid.detach(), reference.detach()
