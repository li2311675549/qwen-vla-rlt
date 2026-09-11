from __future__ import annotations

import json
from dataclasses import dataclass
from math import isfinite

from .schemas import BrainDecision, Observation, PolicyBackend, PolicyRequest
from .skills import SkillRegistry
from .task_graph import TaskGraph


@dataclass(frozen=True, slots=True)
class RLTArtifact:
    skill_id: str
    schema_version: str
    normalization_version: str


class PolicyRouter:
    """Validate model suggestions before granting a policy backend."""

    def __init__(
        self,
        graph: TaskGraph,
        skills: SkillRegistry,
        *,
        rlt_artifacts: dict[str, RLTArtifact] | None = None,
        required_cameras: set[str] | None = None,
        min_confidence: float = 0.0,
        max_observation_age_s: float | None = None,
    ) -> None:
        self.graph = graph
        self.skills = skills
        self.rlt_artifacts = dict(rlt_artifacts or {})
        self.required_cameras = set(required_cameras or ())
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be within [0, 1]")
        if max_observation_age_s is not None and (
            not isfinite(max_observation_age_s) or max_observation_age_s < 0
        ):
            raise ValueError("max_observation_age_s must be non-negative")
        self.min_confidence = min_confidence
        self.max_observation_age_s = max_observation_age_s

    def route(
        self,
        decision: BrainDecision,
        observation: Observation,
        *,
        now_s: float | None = None,
    ) -> PolicyRequest:
        if decision.session_id != observation.session_id:
            raise ValueError("decision and observation sessions differ")
        if decision.observation_id != observation.observation_id:
            raise ValueError("decision references a stale observation")
        if self.max_observation_age_s is not None:
            if now_s is None or not isfinite(now_s):
                raise ValueError("now_s is required when observation expiry is enabled")
            age = now_s - observation.capture_time_s
            if age < 0 or age > self.max_observation_age_s:
                raise ValueError("observation is outside its validity window")
        missing_cameras = self.required_cameras - set(observation.camera_names)
        if missing_cameras:
            raise ValueError(f"missing cameras: {sorted(missing_cameras)}")
        if self.min_confidence > 0 and decision.confidence is None:
            raise ValueError("decision confidence is required by routing threshold")
        if decision.confidence is not None and decision.confidence < self.min_confidence:
            raise ValueError("decision confidence is below routing threshold")
        node = self.graph.node_for(decision)
        self.graph.validate_facts(node, observation)
        self.skills.validate(decision)
        backend = decision.requested_backend
        if backend not in node.supported_backends:
            raise ValueError("requested backend is unsupported for this task node")
        if backend is PolicyBackend.RLT:
            artifact = self.rlt_artifacts.get(decision.skill_id)
            if artifact is None:
                raise ValueError("RLT checkpoint is not registered for this skill")
            if artifact.skill_id != decision.skill_id:
                raise ValueError("RLT artifact belongs to another skill")
            if artifact.schema_version != observation.schema_version:
                raise ValueError("RLT input schema version mismatch")
            if artifact.normalization_version != observation.normalization_version:
                raise ValueError("RLT normalization version mismatch")
        prompt = node.prompt
        if decision.skill_args:
            prompt += "; parameters=" + json.dumps(dict(decision.skill_args), sort_keys=True)
        return PolicyRequest(decision, observation, backend, prompt)
