from __future__ import annotations

from .schemas import PolicyBackend
from .skills import SkillRegistry, SkillSpec
from .task_graph import TaskGraph, TaskNode


def build_skill_registry() -> SkillRegistry:
    return SkillRegistry(
        [
            SkillSpec("flatten_towel"),
            SkillSpec("grasp_corners", frozenset({"width"}), {"width": (0.1, 1.0)}),
            SkillSpec("fold_towel"),
            SkillSpec("place_towel"),
            SkillSpec("inspect_objects"),
            SkillSpec("pick_object"),
            SkillSpec("sort_object"),
            SkillSpec("place_object"),
            SkillSpec("approach_cable_tie"),
            SkillSpec("align_hole", frozenset({"speed"}), {"speed": (0.01, 0.2)}),
            SkillSpec("insert_cable_tie", frozenset({"force_limit"}), {"force_limit": (1.0, 20.0)}),
            SkillSpec("verify_insertion"),
        ]
    )


def build_multitask_graph(initial_key: str = "fold_towel:flatten") -> TaskGraph:
    pi = frozenset({PolicyBackend.PI05, PolicyBackend.WAIT, PolicyBackend.STOP})
    fine = frozenset(
        {PolicyBackend.PI05, PolicyBackend.RLT, PolicyBackend.WAIT, PolicyBackend.STOP}
    )
    nodes = [
        TaskNode(
            "fold_towel",
            "flatten",
            "flatten_towel",
            "flatten the towel",
            frozenset({"fold_towel:grasp"}),
            supported_backends=pi,
        ),
        TaskNode(
            "fold_towel",
            "grasp",
            "grasp_corners",
            "grasp both towel corners",
            frozenset({"fold_towel:fold"}),
            supported_backends=pi,
        ),
        TaskNode(
            "fold_towel",
            "fold",
            "fold_towel",
            "fold along the target line",
            frozenset({"fold_towel:place"}),
            supported_backends=pi,
        ),
        TaskNode(
            "fold_towel",
            "place",
            "place_towel",
            "place the folded towel",
            frozenset({"desk_tidy:inspect", "cable_tie:approach"}),
            supported_backends=pi,
        ),
        TaskNode(
            "desk_tidy",
            "inspect",
            "inspect_objects",
            "inspect objects on the desk",
            frozenset({"desk_tidy:pick"}),
            supported_backends=pi,
        ),
        TaskNode(
            "desk_tidy",
            "pick",
            "pick_object",
            "pick the selected object",
            frozenset({"desk_tidy:sort"}),
            required_facts={"target_visible": True},
            supported_backends=pi,
        ),
        TaskNode(
            "desk_tidy",
            "sort",
            "sort_object",
            "select the matching storage area",
            frozenset({"desk_tidy:place"}),
            supported_backends=pi,
        ),
        TaskNode(
            "desk_tidy",
            "place",
            "place_object",
            "place the object in its storage area",
            frozenset({"fold_towel:flatten", "cable_tie:approach"}),
            required_facts={"workspace_clear": True},
            supported_backends=pi,
        ),
        TaskNode(
            "cable_tie",
            "approach",
            "approach_cable_tie",
            "approach the cable tie and target hole",
            frozenset({"cable_tie:align"}),
            required_facts={"target_visible": True},
            supported_backends=pi,
        ),
        TaskNode(
            "cable_tie",
            "align",
            "align_hole",
            "align the cable tie with the hole",
            frozenset({"cable_tie:insert"}),
            required_facts={"target_visible": True, "safety_ok": True},
            supported_backends=fine,
        ),
        TaskNode(
            "cable_tie",
            "insert",
            "insert_cable_tie",
            "insert the cable tie through the hole",
            frozenset({"cable_tie:verify"}),
            required_facts={"aligned": True, "safety_ok": True},
            supported_backends=fine,
        ),
        TaskNode(
            "cable_tie",
            "verify",
            "verify_insertion",
            "verify cable-tie insertion",
            frozenset({"fold_towel:flatten", "desk_tidy:inspect"}),
            supported_backends=pi,
        ),
    ]
    return TaskGraph(nodes, initial_key)
