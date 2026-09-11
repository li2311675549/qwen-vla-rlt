from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from .policies import Policy
from .rlt.models import ResidualActor
from .schemas import PolicyRequest


class RLTResidualPolicy:
    """Keep the frozen VLA reference in the RLT inference path."""

    def __init__(
        self,
        reference_policy: Policy,
        token_extractor: Callable[[PolicyRequest], NDArray[np.float32]],
        actor: ResidualActor,
    ) -> None:
        self.reference_policy = reference_policy
        self.token_extractor = token_extractor
        self.actor = actor

    def infer(self, request: PolicyRequest) -> NDArray[np.float32]:
        reference = self.reference_policy.infer(request)
        token = np.asarray(self.token_extractor(request), dtype=np.float32)
        return self.actor.act(request.observation.state, token, reference)


class OpenPIPolicy:
    """Wrap an OpenPI Policy or WebsocketClientPolicy with explicit camera/state mapping."""

    def __init__(
        self,
        client,
        camera_map: dict[str, str],
        *,
        state_key: str = "state",
        completion_index: int | None = None,
    ) -> None:
        if not camera_map or len(set(camera_map.values())) != len(camera_map):
            raise ValueError("camera_map must be non-empty and one-to-one")
        self.client, self.camera_map = client, dict(camera_map)
        self.state_key, self.completion_index = state_key, completion_index

    def infer(self, request: PolicyRequest) -> NDArray[np.float32]:
        missing = set(self.camera_map) - request.observation.images.keys()
        if missing:
            raise ValueError(f"OpenPI is missing image pixels: {sorted(missing)}")
        inputs = {
            self.state_key: request.observation.state.copy(),
            "prompt": request.canonical_prompt,
            "skill_args": dict(request.decision.skill_args),
        }
        for camera, key in self.camera_map.items():
            inputs[key] = request.observation.images[camera].copy()
        output = self.client.infer(inputs)
        actions = np.asarray(output["actions"], dtype=np.float32)
        if actions.ndim != 2 or not np.isfinite(actions).all():
            raise ValueError("OpenPI returned invalid action chunk")
        if self.completion_index is not None:
            if not 0 <= self.completion_index < actions.shape[1]:
                raise ValueError("completion_index outside model output")
            actions = np.delete(actions, self.completion_index, axis=1)
        return actions


class NeuralRLTPolicy:
    """Frozen VLA -> learned RL Token -> PyTorch residual actor -> physical actions."""

    def __init__(
        self, frozen_features, encoder, actor, *, state_normalizer=None, action_denormalizer=None
    ) -> None:
        self.features = frozen_features
        self.actor = actor.eval()
        self.encoder = encoder.to(next(actor.parameters()).device).eval().requires_grad_(False)
        self.state_normalizer = state_normalizer
        self.action_denormalizer = action_denormalizer

    def infer(self, request: PolicyRequest) -> NDArray[np.float32]:
        import torch

        device = next(self.actor.parameters()).device
        with torch.inference_mode():
            tokens, valid, reference = self.features(request)
            latent = self.encoder.encode(tokens.to(device), valid.to(device))
            state_values = request.observation.state.copy()
            if self.state_normalizer is not None:
                state_values = self.state_normalizer(state_values)
            state = torch.tensor(state_values[None], dtype=torch.float32, device=device)
            actions = self.actor(state, latent, reference.to(device))[0].cpu().numpy()
        if self.action_denormalizer is not None:
            actions = np.asarray(self.action_denormalizer(actions), dtype=np.float32)
        index = self.actor.config.completion_index
        return np.delete(actions, index, axis=1) if index is not None else actions
