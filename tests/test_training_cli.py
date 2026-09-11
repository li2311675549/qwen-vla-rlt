import sys

import numpy as np
import pytest


def test_token_and_residual_training_cli_write_checkpoints(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from qwen_vla_rlt.rlt.cli import main
    from qwen_vla_rlt.rlt.neural import NeuralRLTTrainer

    torch.set_num_threads(1)
    token_path = tmp_path / "tokens.npz"
    np.savez(token_path, tokens=np.ones((2, 3, 8)), valid=np.ones((2, 3), dtype=bool))
    token_out = tmp_path / "token.pt"
    monkeypatch.setattr(
        sys,
        "argv",
        ["train", "token", "--input", str(token_path), "--output", str(token_out), "--steps", "1"],
    )
    main()
    assert torch.load(token_out, weights_only=True)["embedding_dim"] == 8
    replay_path = tmp_path / "replay.npz"
    np.savez(
        replay_path,
        state=np.zeros((2, 2)),
        token=np.zeros((2, 8)),
        action=np.zeros((2, 2, 3)),
        reference=np.zeros((2, 2, 3)),
        rewards=np.ones((2, 2)),
        next_state=np.ones((2, 2)),
        next_token=np.ones((2, 8)),
        next_reference=np.ones((2, 2, 3)),
        done=np.array([False, True]),
        source=np.array(["policy", "human_intervention"]),
        executed_steps=np.array([2, 2]),
    )
    residual_out = tmp_path / "residual.pt"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "residual",
            "--input",
            str(replay_path),
            "--output",
            str(residual_out),
            "--steps",
            "2",
        ],
    )
    main()
    assert NeuralRLTTrainer.load(residual_out).steps == 2
