from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .schemas import BrainDecision


@dataclass(frozen=True, slots=True)
class MultiViewSFTSample:
    """One decision label with synchronized multi-view image references."""

    sample_id: str
    image_paths: dict[str, Path]
    instruction: str
    target: BrainDecision

    def __post_init__(self) -> None:
        if not self.sample_id or not self.instruction:
            raise ValueError("sample_id and instruction cannot be empty")
        if not self.image_paths or any(not name for name in self.image_paths):
            raise ValueError("at least one named camera view is required")

    def to_qwen_messages(self) -> list[dict[str, object]]:
        content: list[dict[str, object]] = []
        for camera, path in sorted(self.image_paths.items()):
            content.extend(
                [
                    {"type": "text", "text": f"Camera: {camera}"},
                    {"type": "image", "path": str(path)},
                ]
            )
        content.append({"type": "text", "text": self.instruction})
        target = self.target.to_payload()
        return [
            {"role": "user", "content": content},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": json.dumps(target, ensure_ascii=False)}],
            },
        ]
