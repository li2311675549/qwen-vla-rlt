from __future__ import annotations

import json
from collections.abc import Mapping

from .schemas import BrainDecision, CompletionStatus, PolicyBackend

REQUIRED_FIELDS = {
    "session_id",
    "observation_id",
    "task_id",
    "stage_id",
    "completion_status",
    "skill_id",
    "decision_version",
}
OPTIONAL_FIELDS = {"next_task_id", "skill_args", "policy_backend", "confidence"}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _required_string(payload: dict[str, object], key: str) -> str:
    value = payload[key]
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def parse_brain_decision(raw: str | Mapping[str, object]) -> BrainDecision:
    """Strictly parse Qwen output without guessing skills or identifiers."""
    try:
        payload = (
            json.loads(raw, object_pairs_hook=_unique_object) if isinstance(raw, str) else dict(raw)
        )
    except json.JSONDecodeError as exc:
        raise ValueError("brain output is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("brain output must be an object")
    missing = REQUIRED_FIELDS - payload.keys()
    if missing:
        raise ValueError(f"brain output is missing fields: {sorted(missing)}")
    extra = payload.keys() - REQUIRED_FIELDS - OPTIONAL_FIELDS
    if extra:
        raise ValueError(f"brain output has unsupported fields: {sorted(extra)}")
    args = payload.get("skill_args", {})
    if not isinstance(args, dict):
        raise ValueError("skill_args must be an object")
    next_task = payload.get("next_task_id")
    if next_task is not None and (not isinstance(next_task, str) or not next_task):
        raise ValueError("next_task_id must be null or a non-empty string")
    version = payload["decision_version"]
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError("decision_version must be an integer")
    confidence = payload.get("confidence")
    if confidence is not None and type(confidence) not in (float, int):
        raise ValueError("confidence must be numeric or null")
    try:
        return BrainDecision(
            session_id=_required_string(payload, "session_id"),
            observation_id=_required_string(payload, "observation_id"),
            task_id=_required_string(payload, "task_id"),
            stage_id=_required_string(payload, "stage_id"),
            completion_status=CompletionStatus(_required_string(payload, "completion_status")),
            next_task_id=next_task,
            skill_id=_required_string(payload, "skill_id"),
            skill_args=args,
            requested_backend=PolicyBackend(payload.get("policy_backend", "pi05")),
            decision_version=version,
            confidence=confidence,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid brain decision: {exc}") from exc
