from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from qwen_vla_rlt.adapters import OpenPIPolicy
from qwen_vla_rlt.brain import parse_brain_decision
from qwen_vla_rlt.presets import build_multitask_graph
from qwen_vla_rlt.qwen import QwenSFTCollator, QwenTaskBrain
from qwen_vla_rlt.schemas import (
    BrainDecision,
    CompletionStatus,
    Observation,
    PolicyBackend,
    PolicyRequest,
)
from qwen_vla_rlt.sft import MultiViewSFTSample


def decision():
    return BrainDecision(
        "s",
        "o",
        "cable_tie",
        "align",
        CompletionStatus.NOT_COMPLETE,
        None,
        "align_hole",
        decision_version=0,
    )


def test_sft_messages_are_valid_chat_content_and_roundtrip():
    sample = MultiViewSFTSample("example", {"front": Path("front.png")}, "Inspect", decision())
    messages = sample.to_qwen_messages()
    answer = messages[-1]["content"][0]["text"]
    assert parse_brain_decision(answer) == decision()
    assert messages[0]["content"][1] == {"type": "image", "path": "front.png"}


def test_collator_masks_prompt_and_padding_without_masking_answers():
    torch = pytest.importorskip("torch")

    class ProcessorFixture:
        tokenizer = SimpleNamespace(padding_side="left")

        def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):
            if add_generation_prompt:
                return {
                    "input_ids": torch.tensor([[1, 2, 3], [1, 2, 0]]),
                    "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
                }
            return {
                "input_ids": torch.tensor([[1, 2, 3, 4, 5], [1, 2, 6, 0, 0]]),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]]),
            }

    sample = MultiViewSFTSample("example", {"front": Path("front.png")}, "Inspect", decision())
    batch = QwenSFTCollator(ProcessorFixture())([sample, sample])
    assert batch["labels"].tolist() == [[-100, -100, -100, 4, 5], [-100, -100, 6, -100, -100]]


def test_qwen_inference_supplies_images_and_checks_decision_identity():
    torch = pytest.importorskip("torch")

    class ModelFixture:
        device = torch.device("cpu")

        def eval(self):
            return self

        def generate(self, **kwargs):
            assert not kwargs["do_sample"]
            return torch.tensor([[1, 2, 3]])

    class ProcessorFixture:
        def apply_chat_template(self, messages, **kwargs):
            assert any(x["type"] == "image" for x in messages[0]["content"])
            return {"input_ids": torch.tensor([[1, 2]])}

        def batch_decode(self, tokens, **kwargs):
            import json

            assert tokens.tolist() == [[3]]
            return [json.dumps(decision().to_payload())]

    observation = Observation(
        "s", "o", 0, np.zeros(2), ("front",), images={"front": np.zeros((8, 8, 3), dtype=np.uint8)}
    )
    brain = QwenTaskBrain(ModelFixture(), ProcessorFixture(), build_multitask_graph())
    assert brain.decide(observation, "cable_tie:align", set(), 0) == decision()
    with pytest.raises(ValueError, match="identity or version"):
        brain.decide(observation, "cable_tie:align", set(), 1)


def test_openpi_bridge_passes_pixels_and_separates_completion():
    class ClientFixture:
        def infer(self, inputs):
            assert inputs["images/front"].dtype == np.uint8
            assert inputs["prompt"] == "align"
            return {"actions": np.ones((2, 15))}

    observation = Observation(
        "s", "o", 0, np.zeros(14), ("front",), images={"front": np.zeros((8, 8, 3), dtype=np.uint8)}
    )
    policy = OpenPIPolicy(ClientFixture(), {"front": "images/front"}, completion_index=14)
    assert policy.infer(
        PolicyRequest(decision(), observation, PolicyBackend.PI05, "align")
    ).shape == (2, 14)
