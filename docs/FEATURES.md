# Feature-to-code mapping

| Functionality | Implementation | Executable evidence |
|---|---|---|
| Qwen3-VL multi-view SFT/LoRA and inference | `sft.MultiViewSFTSample`, `qwen.QwenSFTCollator`, `qwen.QwenTaskBrain`, `qwen.main` | Message schema, answer-only masking, image input and inference identity checks |
| Predict current task, stage, completion, and next task | `schemas.BrainDecision` | Required-field, type, enum, and unknown-field rejection |
| Enforce legal task selection | `task_graph.TaskGraph`, `presets.build_multitask_graph` | Reachability and precondition tests for three tasks |
| Prevent stage jitter and stale decisions | `state_machine.TaskStateMachine`, `router.PolicyRouter` | Completion gate, debounce, monotonic version, observation-expiry tests |
| Route common skills to π0.5 and fine skills to RLT | `router.PolicyRouter`, `runtime.HierarchicalRuntime` | Cable-tie alignment end-to-end route test |
| Freeze VLA and learn the RL Token bottleneck | `rlt.neural.RLTokenAutoencoder`, `FrozenVLAFeatures`, `openpi_features.OpenPIFrozenFeatures` | Detached input, encoder gradient, padded-token invariance and frozen parameters |
| Learn bounded residual Action Chunks | `rlt.neural.NeuralResidualActor` | Zero initialization, noise clipping and completion masking |
| Twin-Q, BC anchor, chunk TD target, online replay | `rlt.neural.NeuralRLTTrainer`, `rlt.online.OnlineRLTLoop`, `rlt.replay.ReplayBuffer` | Delayed actor update, intervention provenance, actual chunk duration and checkpoint restoration |
| Use the next observation's VLA reference for bootstrapping | `rlt.replay.Transition.next_reference`, `RLTTrainer.td_target` | Target changes when only `next_reference` changes |
| Keep robot commands inside final bounds | `policies.ActionGuard` | Joint bounds and per-step delta enforcement |
| Cancel obsolete commands | `runtime.halt`, `runtime.reset`, `runtime.take_action` | Takeover during inference, completion hold and expiry before send |
| Evaluate three high-level capabilities separately | `evaluation.evaluate_decisions` | Stage, completion, and legality remain independent metrics |

Tests use synthetic fixtures to validate software behavior. No demonstration datasets,
real-world evaluation sample counts, or robot success-rate measurements are distributed here.
