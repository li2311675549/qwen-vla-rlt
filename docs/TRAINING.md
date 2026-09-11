# 训练与接入

## Qwen 多视角 SFT

每行 JSONL 是一个同步观测，`images` 路径相对 manifest；三视角仍是一条决策样本。

```json
{"sample_id":"e1-t0","images":{"front":"images/front.png","left_wrist":"images/left.png","right_wrist":"images/right.png"},"instruction":"根据当前图像判断扎带阶段，输出规定的 JSON 决策。","target":{"session_id":"e1","observation_id":"e1-t0","task_id":"cable_tie","stage_id":"align","completion_status":"not_complete","next_task_id":null,"skill_id":"align_hole","skill_args":{"speed":0.05},"policy_backend":"rlt","decision_version":0,"confidence":0.9}}
```

`qwen-vla-sft` 加载 Qwen3-VL-4B，给 q/k/v/o 投影添加 LoRA；collator 检查 prompt 是完整序列的前缀，
将 prompt 和 padding 标签置为 -100，仅监督 assistant 回答。训练数据应按 episode 独立划分，
不能用测试帧构造任务记忆。CLI 只负责给定训练 manifest，不自动建立未经确认的评测集。

推理 API：`load_qwen(base_model, adapter_path=..., device="cuda")` → `QwenTaskBrain`。
`decide(observation, current_key, completed, decision_version)` 读取真实 RGB 数组、任务图与工作区事实，
解析完整决策并校验身份；返回值继续交给 `HierarchicalRuntime` 进行可执行性校验。
模型自报 confidence 仅用于门槛检查，不能视为已校准概率。

## RL Token 预训练

NPZ 字段：`tokens [N,M,D] float32`、`valid [N,M] bool`。token 来自冻结 VLA；`True` 表示有效位置。
编码器追加可学习 RL query，解码器仅从瓶颈与位置 query 重建。padding 不进入重建 loss；
VLA 输入 detach，反向传播只更新编码器和解码器。每个样本至少一个有效 token。

```bash
qwen-vla-rlt-train token --input data/tokens.npz --output checkpoints/token.pt --steps 100
```

## Residual Actor / Twin-Q

NPZ 字段：

| 字段 | 形状 | 含义 |
|---|---|---|
| state / next_state | [N,P] | 当前/下一本体状态 |
| token / next_token | [N,D] | 当前/下一 RL Token |
| action | [N,C,A] | 实际执行动作；接管时为人工动作 |
| reference / next_reference | [N,C,A] | 各自观测的 π0.5 参考动作 |
| rewards | [N,C] | chunk 内逐步奖励；未执行尾部补零 |
| done | [N] bool | 环境是否终止 |
| source | [N] unicode | policy 或 human_intervention |
| executed_steps | [N] int | 实际执行步数 1..C |

目标为 `Σ gamma^k r_k + gamma^executed_steps (1-done) min(Q1_target,Q2_target)`。
普通样本 BC 锚定 reference，人工样本锚定实际动作；BC 和 Q 不利用 completion 伪动作维度。
Actor 末层零初始化，初始动作等于参考策略，探索噪声添加后再次裁剪残差。

```bash
qwen-vla-rlt-train residual --input data/replay.npz --output checkpoints/residual.pt --steps 100
```

`NeuralRLTTrainer.load` 恢复配置、四套网络、优化器和更新次数。恢复后的随机采样序列不保证逐位一致。
CPU NumPy 版本保留用于检查接口；正式学习路径使用 `rlt.neural`。

## OpenPI 与在线采集

`OpenPIPolicy` 可包装 OpenPI Policy 或 WebsocketClientPolicy；camera_map 将相机名映射到训练时输入键。
`OpenPIFrozenFeatures` 基于 OpenPI PyTorch 的 prefix 接口提取视觉/语言表征，再对同一观测生成 reference。
它依赖 `_preprocess_observation` 等具体接口，替换 OpenPI 版本时需重新验证。
观测 builder 必须使用训练时 tokenizer、相机变换及归一化；不要混用不同 checkpoint 的统计量。

`NeuralRLTPolicy` 连接 VLA → encoder → Actor。可传入 `state_normalizer` 与 `action_denormalizer`，
将运行时物理本体状态映射到训练空间，并将模型输出恢复到物理单位。
`OnlineRLTLoop` 的环境适配器负责执行与硬件边界，必须返回实际动作、逐步奖励、实际执行时长和接管状态。
下一观测再次通过 encode 生成 next_reference，不能复用上一个 reference。

## 任务状态与控制器

`take_action(now_s=...)` 是控制器消费接口，`RuntimeResult.action` 仅用于诊断。
配置观测有效期并使用统一单调时钟。接管调用 `halt()`，新 episode 调用 `reset()`。
最终 guard 约束关节位置与逐步位移；加速度/力矩/碰撞约束应由具体机器人控制适配器实现。

## 参考接口

- [Qwen3-VL 官方 Transformers 文档](https://huggingface.co/docs/transformers/model_doc/qwen3_vl)
- [OpenPI 官方仓库](https://github.com/Physical-Intelligence/openpi)
- [RL Token 官方介绍](https://www.pi.website/research/rlt)

模型组件与软件链路可以用合成张量验证。完整 4B/π0.5 权重、传感器与硬件的集成测试需要对应外部环境。
