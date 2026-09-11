from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class LinearRLTokenAutoencoder:
    """Small PCA autoencoder for fixed-shape frozen-VLA token tensors."""

    def __init__(self, latent_dim: int) -> None:
        if latent_dim < 1:
            raise ValueError("latent_dim must be positive")
        self.latent_dim = latent_dim
        self.token_shape: tuple[int, ...] | None = None
        self.mean: NDArray[np.float32] | None = None
        self.components: NDArray[np.float32] | None = None

    def fit(self, tokens: NDArray) -> LinearRLTokenAutoencoder:
        values = np.asarray(tokens, dtype=np.float32)
        if values.ndim < 3 or len(values) < 2 or not np.all(np.isfinite(values)):
            raise ValueError("tokens must be finite [samples, tokens, embedding...] data")
        flat = values.reshape(len(values), -1)
        if self.latent_dim > min(flat.shape):
            raise ValueError("latent_dim exceeds available rank")
        mean = flat.mean(axis=0)
        _, _, vh = np.linalg.svd(flat - mean, full_matrices=False)
        self.token_shape = values.shape[1:]
        self.mean = mean.astype(np.float32)
        self.components = vh[: self.latent_dim].astype(np.float32)
        return self

    def _flat(self, tokens: NDArray) -> NDArray[np.float32]:
        if self.token_shape is None or self.mean is None or self.components is None:
            raise RuntimeError("RL token encoder is not fitted")
        values = np.asarray(tokens, dtype=np.float32)
        if values.shape != self.token_shape or not np.all(np.isfinite(values)):
            raise ValueError(f"expected one token tensor with shape {self.token_shape}")
        return values.reshape(-1)

    def encode(self, tokens: NDArray) -> NDArray[np.float32]:
        flat = self._flat(tokens)
        if self.mean is None or self.components is None:
            raise RuntimeError("RL token encoder is not fitted")
        return ((flat - self.mean) @ self.components.T).astype(np.float32)

    def reconstruct(self, token: NDArray) -> NDArray[np.float32]:
        if self.token_shape is None or self.mean is None or self.components is None:
            raise RuntimeError("RL token encoder is not fitted")
        latent = np.asarray(token, dtype=np.float32)
        if latent.shape != (self.latent_dim,) or not np.all(np.isfinite(latent)):
            raise ValueError("invalid RL token")
        return (latent @ self.components + self.mean).reshape(self.token_shape).astype(np.float32)
