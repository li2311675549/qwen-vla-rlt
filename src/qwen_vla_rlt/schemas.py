from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray


class CompletionStatus(StrEnum):
    COMPLETE = "complete"
    NOT_COMPLETE = "not_complete"
    UNKNOWN = "unknown"


class PolicyBackend(StrEnum):
    PI05 = "pi05"
    RLT = "rlt"
    WAIT = "wait"
    STOP = "stop"


@dataclass(frozen=True, slots=True)
class BrainDecision:
    session_id: str
    observation_id: str
    task_id: str
    stage_id: str
    completion_status: CompletionStatus
    next_task_id: str | None
    skill_id: str
    skill_args: Mapping[str, float | int | str | bool] = field(default_factory=dict)
    requested_backend: PolicyBackend = PolicyBackend.PI05
    decision_version: int = 0
    confidence: float | None = None

    def __post_init__(self) -> None:
        for name in ("session_id", "observation_id", "task_id", "stage_id", "skill_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} cannot be empty")
        if type(self.decision_version) is not int or self.decision_version < 0:
            raise ValueError("decision_version must be non-negative")
        if self.confidence is not None and (
            type(self.confidence) not in (int, float) or not 0.0 <= self.confidence <= 1.0
        ):
            raise ValueError("confidence must be within [0, 1]")
        object.__setattr__(self, "completion_status", CompletionStatus(self.completion_status))
        object.__setattr__(self, "requested_backend", PolicyBackend(self.requested_backend))
        if self.next_task_id is not None and (
            not isinstance(self.next_task_id, str) or not self.next_task_id.strip()
        ):
            raise ValueError("next_task_id must be a non-empty string or null")
        for key, value in self.skill_args.items():
            if not isinstance(key, str) or type(value) not in (int, float, str, bool):
                raise ValueError("skill_args must contain scalar values")
            if isinstance(value, float) and not np.isfinite(value):
                raise ValueError("skill_args must contain finite values")
        object.__setattr__(self, "skill_args", MappingProxyType(dict(self.skill_args)))

    def to_payload(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "observation_id": self.observation_id,
            "task_id": self.task_id,
            "stage_id": self.stage_id,
            "completion_status": self.completion_status.value,
            "next_task_id": self.next_task_id,
            "skill_id": self.skill_id,
            "skill_args": dict(self.skill_args),
            "policy_backend": self.requested_backend.value,
            "decision_version": self.decision_version,
            "confidence": self.confidence,
        }


@dataclass(frozen=True, slots=True)
class Observation:
    session_id: str
    observation_id: str
    capture_time_s: float
    state: NDArray[np.float32]
    camera_names: tuple[str, ...]
    facts: Mapping[str, bool | float | int | str] = field(default_factory=dict)
    schema_version: str = "v1"
    normalization_version: str = "v1"
    images: Mapping[str, NDArray[np.uint8]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        state = np.asarray(self.state, dtype=np.float32)
        if state.ndim != 1 or not np.all(np.isfinite(state)):
            raise ValueError("state must be a finite one-dimensional array")
        if (
            not np.isfinite(self.capture_time_s)
            or self.capture_time_s < 0
            or not self.session_id
            or not self.observation_id
        ):
            raise ValueError("invalid observation identity or time")
        if len(set(self.camera_names)) != len(self.camera_names):
            raise ValueError("camera_names must be unique")
        if not self.schema_version or not self.normalization_version:
            raise ValueError("schema and normalization versions cannot be empty")
        state = state.copy()
        state.setflags(write=False)
        object.__setattr__(self, "state", state)
        images = {}
        for name, supplied in self.images.items():
            image = np.asarray(supplied)
            if name not in self.camera_names or image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError("image must have a registered camera name and HWC RGB shape")
            if image.dtype != np.uint8 or 0 in image.shape:
                raise ValueError("images must contain uint8 pixels")
            image = image.copy()
            image.setflags(write=False)
            images[name] = image
        object.__setattr__(self, "images", MappingProxyType(images))
        object.__setattr__(self, "facts", MappingProxyType(dict(self.facts)))


@dataclass(frozen=True, slots=True)
class PolicyRequest:
    decision: BrainDecision
    observation: Observation
    backend: PolicyBackend
    canonical_prompt: str


@dataclass(frozen=True, slots=True)
class ActionChunk:
    values: NDArray[np.float32]
    observation_id: str
    decision_version: int
    backend: PolicyBackend

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float32)
        if values.ndim != 2 or 0 in values.shape or not np.all(np.isfinite(values)):
            raise ValueError("action chunk must be a finite [chunk, action_dim] array")
        values = values.copy()
        values.setflags(write=False)
        object.__setattr__(self, "values", values)
