# Qwen VLA RLT

面向收纳、叠毛巾与扎带穿孔的 Qwen3-VL 任务决策、π0.5 技能执行和 RLT 精细动作增强。

```text
Multi-view images + task memory → Qwen3-VL structured decision
    → Task graph / preconditions → Stable stage → Policy router
    → π0.5 reference → Optional RLT residual → Action guard → Sender
    → Execution feedback → Next-state reference → Replay → Twin-Q / Actor
```

## 功能

- Qwen3-VL-4B 多视角 SFT/LoRA 训练入口，仅监督 assistant 回答；加载基座和 adapter 推理。
- 统一输出任务、阶段、完成状态、下一任务、技能参数与策略后端。
- 严格 JSON、重复字段拒绝、任务图前置条件、工作区约束、相机、观测时效与模型版本校验。
- 阶段完成后连续确认切换；同一观测不能重复投票，任务循环重新获得完成状态。
- 扎带粗接近/检查使用 π0.5，对孔/穿入允许启用已注册的 RLT。
- PyTorch Transformer RL Token 编码器/解码器，冻结 VLA 表征并使用 masked reconstruction loss。
- 零初始化残差 Actor、Twin-Q、Target Actor/Critic、延迟更新、目标平滑、BC 与残差限幅。
- 普通数据以参考动作作为 BC 锚点，人工接管数据以实际执行动作为锚点。
- 在线采集保存真实 `next_reference`、实际执行动作和 chunk 时长。
- completion 维度从物理 Q 输入和动作探索中屏蔽；机器人发送前去除该维度。
- 接管/停止/重置作废缓存与在途推理；推理结束及发送前均检查动作有效期。
- 阶段、完成和合法任务选择分别评估；合法性使用观测前置条件和当前任务状态。

## 安装与检查

Python 3.11+：

```bash
python -m pip install -e ".[dev,neural]"
pytest -q
ruff check .
ruff format --check .
python examples/synthetic_cable_tie.py
```

轻量 NumPy 组件和 PyTorch 学习组件均有测试。测试不下载大模型或数据集。

## Qwen3-VL 训练

```bash
python -m pip install -e ".[qwen]"
qwen-vla-sft --manifest data/train.jsonl --output runs/qwen-lora --device cuda
```

默认基座为 `Qwen/Qwen3-VL-4B-Instruct`，也可以传入本地模型目录。
JSONL 多视角格式及推理接入见 [训练与接入说明](docs/TRAINING.md)。

## RLT 两阶段训练

```bash
qwen-vla-rlt-train token --input data/tokens.npz --output checkpoints/token.pt
qwen-vla-rlt-train residual --input data/replay.npz --output checkpoints/residual.pt --completion-index 14
```

`--completion-index 14` 只适用于含第十五维 completion 的模型输出；
十四维物理动作应省略此参数。RL Token 的维度、动作 chunk 长度均由输入确定。

`OpenPIFrozenFeatures` 获取同一观测的 VLA prefix 和参考动作；
`NeuralRLTPolicy` 连接 encoder 与 Actor；`OnlineRLTLoop` 连接执行反馈、回放和训练。
OpenPI 需要其独立依赖环境和 checkpoint 配套观测变换，详见训练说明。

## 执行约定

控制器通过 `runtime.take_action(now_s=...)` 消费待执行 chunk，
不要直接发送 `RuntimeResult.action`：后者是供日志与诊断使用的快照。
观测时间与传入时间必须来自同一个单调时钟。设置明确的
`max_observation_age_s` 和实际硬件位置/步长界限。

[功能对应表](docs/FEATURES.md) · [训练说明](docs/TRAINING.md) · [公开方法与依赖来源](docs/SOURCES.md)

## 验证范围

测试覆盖小型神经网络反向传播、模型冻结、在线数据流、模型保存加载以及模型接口契约。
Qwen3-VL-4B 和 π0.5 完整权重推理、真实硬件执行和成功率实验依赖外部环境；
功能测试结果不代表机器人成功率。数据、权重、密钥文件默认忽略。
