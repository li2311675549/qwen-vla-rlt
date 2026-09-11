"""Offline entry points for token pretraining and replay-based residual training."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .neural import NeuralRLTTrainer, RLTConfig, RLTokenAutoencoder
from .replay import ReplayBuffer, ReplaySource, Transition


def main() -> None:
    parser = argparse.ArgumentParser(description="RLT functional training; local NPZ inputs only")
    parser.add_argument("mode", choices=["token", "residual"])
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--completion-index", type=int)
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        parser.error("steps and batch size must be positive")
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with np.load(args.input, allow_pickle=False) as data:
        if args.mode == "token":
            tokens = torch.tensor(data["tokens"], dtype=torch.float32, device=args.device)
            valid = torch.tensor(data["valid"], dtype=torch.bool, device=args.device)
            model = RLTokenAutoencoder(tokens.shape[-1], tokens.shape[1], args.heads).to(
                args.device
            )
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            loss = None
            for _ in range(args.steps):
                index = rng.choice(len(tokens), min(args.batch_size, len(tokens)), replace=False)
                loss, _ = model(tokens[index], valid[index])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
                optimizer.step()
            torch.save(
                {
                    "schema_version": 1,
                    "embedding_dim": tokens.shape[-1],
                    "max_tokens": tokens.shape[1],
                    "heads": args.heads,
                    "layers": 2,
                    "weights": model.state_dict(),
                },
                output,
            )
            print(json.dumps({"steps": args.steps, "reconstruction_loss": loss.item()}))
        else:
            cfg = RLTConfig(
                data["state"].shape[-1],
                data["token"].shape[-1],
                data["action"].shape[1],
                data["action"].shape[2],
                completion_index=args.completion_index,
            )
            trainer = NeuralRLTTrainer(cfg, args.device)
            replay = ReplayBuffer(len(data["state"]), seed=0)
            fields = (
                "state",
                "token",
                "action",
                "reference",
                "rewards",
                "next_state",
                "next_token",
                "next_reference",
            )
            for index in range(len(data["state"])):
                replay.add(
                    Transition(
                        **{key: data[key][index] for key in fields},
                        done=bool(data["done"][index]),
                        source=ReplaySource(str(data["source"][index])),
                        executed_steps=int(data["executed_steps"][index]),
                    )
                )
            stats = {}
            for _ in range(args.steps):
                stats = trainer.train_step(replay.sample(min(args.batch_size, len(replay))))
            trainer.save(output)
            print(json.dumps({"steps": args.steps, "config": asdict(cfg), **stats}))


if __name__ == "__main__":
    main()
