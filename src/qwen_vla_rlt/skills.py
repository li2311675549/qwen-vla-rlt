from __future__ import annotations

from dataclasses import dataclass, field

from .schemas import BrainDecision


@dataclass(frozen=True, slots=True)
class SkillSpec:
    skill_id: str
    allowed_args: frozenset[str] = field(default_factory=frozenset)
    numeric_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.skill_id:
            raise ValueError("skill_id cannot be empty")
        if not set(self.numeric_ranges).issubset(self.allowed_args):
            raise ValueError("numeric ranges must reference allowed arguments")
        if any(lower > upper for lower, upper in self.numeric_ranges.values()):
            raise ValueError("numeric argument range is inverted")

    def validate(self, decision: BrainDecision) -> None:
        unknown = set(decision.skill_args) - self.allowed_args
        if unknown:
            raise ValueError(f"unsupported skill arguments: {sorted(unknown)}")
        for key, bounds in self.numeric_ranges.items():
            if key not in decision.skill_args:
                continue
            value = decision.skill_args[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"skill argument {key} must be numeric")
            if not bounds[0] <= float(value) <= bounds[1]:
                raise ValueError(f"skill argument {key} outside range {bounds}")


class SkillRegistry:
    def __init__(self, skills: list[SkillSpec]) -> None:
        self.skills = {skill.skill_id: skill for skill in skills}
        if len(self.skills) != len(skills):
            raise ValueError("duplicate skill_id")

    def validate(self, decision: BrainDecision) -> None:
        if decision.skill_id not in self.skills:
            raise ValueError(f"unknown skill_id: {decision.skill_id}")
        self.skills[decision.skill_id].validate(decision)
