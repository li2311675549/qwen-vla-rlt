from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _vector(value: NDArray, size: int, name: str) -> NDArray[np.float32]:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must have shape ({size},)")
    return result


class ResidualActor:
    """Reference-conditioned bounded residual actor for action chunks."""

    def __init__(
        self,
        state_dim: int,
        token_dim: int,
        action_shape: tuple[int, int],
        residual_limit: NDArray,
        *,
        trainable_action_mask: NDArray | None = None,
        seed: int = 0,
    ) -> None:
        self.state_dim = state_dim
        self.token_dim = token_dim
        self.action_shape = action_shape
        action_dim = action_shape[1]
        limit = _vector(residual_limit, action_dim, "residual_limit")
        if np.any(limit < 0):
            raise ValueError("residual limits must be non-negative")
        mask = np.ones(action_dim, dtype=np.float32)
        if trainable_action_mask is not None:
            supplied = np.asarray(trainable_action_mask, dtype=bool)
            if supplied.shape != (action_dim,):
                raise ValueError("trainable_action_mask has the wrong shape")
            mask = supplied.astype(np.float32)
        self.flat_limit = np.tile(limit * mask, action_shape[0]).astype(np.float32)
        feature_dim = state_dim + token_dim + action_shape[0] * action_dim
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(0.0, 0.01, (feature_dim, len(self.flat_limit))).astype(np.float32)
        self.bias = np.zeros(len(self.flat_limit), dtype=np.float32)

    def _features(self, state: NDArray, token: NDArray, reference: NDArray) -> NDArray[np.float32]:
        state_v = _vector(state, self.state_dim, "state")
        token_v = _vector(token, self.token_dim, "token")
        ref = np.asarray(reference, dtype=np.float32)
        if ref.shape != self.action_shape or not np.all(np.isfinite(ref)):
            raise ValueError(f"reference must have shape {self.action_shape}")
        return np.concatenate((state_v, token_v, ref.reshape(-1))).astype(np.float32)

    def act(
        self,
        state: NDArray,
        token: NDArray,
        reference: NDArray,
        *,
        exploration_noise: NDArray | None = None,
    ) -> NDArray[np.float32]:
        features = self._features(state, token, reference)
        residual = np.tanh(features @ self.weights + self.bias) * self.flat_limit
        if exploration_noise is not None:
            noise = np.asarray(exploration_noise, dtype=np.float32)
            if noise.shape != self.action_shape:
                raise ValueError("exploration noise has the wrong shape")
            residual += noise.reshape(-1)
        residual = np.clip(residual, -self.flat_limit, self.flat_limit)
        return (
            np.asarray(reference, dtype=np.float32) + residual.reshape(self.action_shape)
        ).astype(np.float32)

    def update(
        self,
        batch: list[tuple[NDArray, NDArray, NDArray, NDArray, NDArray]],
        *,
        learning_rate: float,
        bc_weight: float,
    ) -> float:
        """Update from (state, token, reference, dQ/da, behavior_action)."""
        if not batch or learning_rate <= 0 or bc_weight < 0:
            raise ValueError("invalid actor update configuration")
        grad_w = np.zeros_like(self.weights)
        grad_b = np.zeros_like(self.bias)
        loss = 0.0
        for state, token, reference, q_grad, behavior in batch:
            x = self._features(state, token, reference)
            z = x @ self.weights + self.bias
            tanh_z = np.tanh(z)
            action = np.asarray(reference, dtype=np.float32).reshape(-1) + tanh_z * self.flat_limit
            behavior_v = np.asarray(behavior, dtype=np.float32).reshape(-1)
            q_grad_v = np.asarray(q_grad, dtype=np.float32).reshape(-1)
            dloss_da = -q_grad_v + (2.0 * bc_weight * (action - behavior_v) / len(action))
            dloss_dz = dloss_da * self.flat_limit * (1.0 - tanh_z**2)
            grad_w += np.outer(x, dloss_dz)
            grad_b += dloss_dz
            loss += -float(q_grad_v @ action) + bc_weight * float(
                np.mean((action - behavior_v) ** 2)
            )
        scale = 1.0 / len(batch)
        self.weights -= learning_rate * grad_w * scale
        self.bias -= learning_rate * grad_b * scale
        return loss * scale


class TwinLinearCritic:
    """Twin-Q critic with independent online and Polyak target parameters."""

    def __init__(
        self, state_dim: int, token_dim: int, action_shape: tuple[int, int], seed: int = 0
    ) -> None:
        self.state_dim = state_dim
        self.token_dim = token_dim
        self.action_shape = action_shape
        self.action_size = action_shape[0] * action_shape[1]
        feature_dim = state_dim + token_dim + 2 * self.action_size
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0.0, 0.01, feature_dim).astype(np.float32)
        self.w2 = rng.normal(0.0, 0.01, feature_dim).astype(np.float32)
        self.b1 = 0.0
        self.b2 = 0.0
        self.target_w1 = self.w1.copy()
        self.target_w2 = self.w2.copy()
        self.target_b1 = self.b1
        self.target_b2 = self.b2

    def features(
        self, state: NDArray, token: NDArray, action: NDArray, reference: NDArray
    ) -> NDArray[np.float32]:
        state_v = _vector(state, self.state_dim, "state")
        token_v = _vector(token, self.token_dim, "token")
        action_v = np.asarray(action, dtype=np.float32)
        reference_v = np.asarray(reference, dtype=np.float32)
        if action_v.shape != self.action_shape or reference_v.shape != self.action_shape:
            raise ValueError("action/reference shape mismatch")
        if not np.all(np.isfinite(action_v)) or not np.all(np.isfinite(reference_v)):
            raise ValueError("action/reference values must be finite")
        return np.concatenate(
            (state_v, token_v, action_v.reshape(-1), reference_v.reshape(-1))
        ).astype(np.float32)

    def predict(
        self,
        state: NDArray,
        token: NDArray,
        action: NDArray,
        reference: NDArray,
        *,
        target: bool = False,
    ) -> tuple[float, float]:
        x = self.features(state, token, action, reference)
        if target:
            return float(x @ self.target_w1 + self.target_b1), float(
                x @ self.target_w2 + self.target_b2
            )
        return float(x @ self.w1 + self.b1), float(x @ self.w2 + self.b2)

    def update(
        self, batch: list[tuple[NDArray, NDArray, NDArray, NDArray, float]], learning_rate: float
    ) -> float:
        if not batch or learning_rate <= 0:
            raise ValueError("invalid critic update configuration")
        grad_w1 = np.zeros_like(self.w1)
        grad_w2 = np.zeros_like(self.w2)
        grad_b1 = 0.0
        grad_b2 = 0.0
        loss = 0.0
        for state, token, action, reference, target in batch:
            x = self.features(state, token, action, reference)
            e1 = float(x @ self.w1 + self.b1) - target
            e2 = float(x @ self.w2 + self.b2) - target
            grad_w1 += 2.0 * e1 * x
            grad_w2 += 2.0 * e2 * x
            grad_b1 += 2.0 * e1
            grad_b2 += 2.0 * e2
            loss += e1**2 + e2**2
        scale = 1.0 / len(batch)
        self.w1 -= learning_rate * grad_w1 * scale
        self.w2 -= learning_rate * grad_w2 * scale
        self.b1 -= learning_rate * grad_b1 * scale
        self.b2 -= learning_rate * grad_b2 * scale
        return loss * scale

    def action_gradient(
        self, state: NDArray, token: NDArray, action: NDArray, reference: NDArray
    ) -> NDArray[np.float32]:
        q1, q2 = self.predict(state, token, action, reference)
        weights = self.w1 if q1 <= q2 else self.w2
        start = self.state_dim + self.token_dim
        return weights[start : start + self.action_size].reshape(self.action_shape).copy()

    def soft_update(self, tau: float) -> None:
        if not 0.0 < tau <= 1.0:
            raise ValueError("tau must be in (0, 1]")
        self.target_w1 = (1 - tau) * self.target_w1 + tau * self.w1
        self.target_w2 = (1 - tau) * self.target_w2 + tau * self.w2
        self.target_b1 = (1 - tau) * self.target_b1 + tau * self.b1
        self.target_b2 = (1 - tau) * self.target_b2 + tau * self.b2
