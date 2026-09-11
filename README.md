# Qwen VLA RLT

面向收纳、叠毛巾与扎带穿孔的 Qwen3-VL 任务决策、π0.5 技能执行和 RLT 精细动作增强。

长程操作需要判断“当前做到哪一步、是否完成、下一步能否执行”，精细接触阶段还需要修正通用策略的动作。本项目将多模态任务决策、确定性执行校验和残差学习连接起来，提供可扩展的软件闭环。

合成示例可以直接运行；真实多模态推理、在线机器人学习需要自行接入模型权重、同步观测和环境适配器。

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

## 系统分层

| 层次 | 输入与输出 | 实现职责 |
|---|---|---|
| 多模态任务大脑 | 多视角图像、任务记忆、工作区事实 → JSON 决策 | 判断任务、阶段、完成状态、下一任务与技能 |
| 决策与状态校验 | 决策、任务图、当前观测 → 稳定且可执行的请求 | 身份、版本、前置条件、技能参数、连续完成确认 |
| 通用动作策略 | 观测、规范化技能指令 → 参考 action chunk | 通过 OpenPI 适配器调用 π0.5 |
| 精细残差策略 | RL Token、本体状态、参考 chunk → 修正 chunk | 在受限范围内调整参考动作 |
| 执行与学习 | 动作、执行反馈 → 新状态和 replay | 单次消费、失效处理、实际动作记录与 off-policy 更新 |

工作区事实由感知或环境适配器提供，不能仅由语言模型自行声明为真。任务图中的技能是接口定义；具体抓取、放置和力控能力仍来自已训练策略及机器人控制系统。

## 支持的任务模板

| 任务 | 阶段 | 默认策略范围 |
|---|---|---|
| `fold_towel` | flatten → grasp → fold → place | π0.5 |
| `desk_tidy` | inspect → pick → sort → place | π0.5 |
| `cable_tie` | approach → align → insert → verify | approach/verify 使用 π0.5，align/insert 可选 RLT |

`presets.py` 定义阶段依赖与跨任务转移。接入其他任务时扩展 `TaskNode` 和 `SkillSpec`，为 RLT 注册匹配的技能、schema 和归一化版本。示例参数范围是软件模板配置，应按实际机器人重新设定。

## 决策格式

下面是一条结构示例，所有身份、事实和版本都必须与当前运行状态匹配：

```json
{
  "session_id": "demo-session",
  "observation_id": "observation-0",
  "task_id": "cable_tie",
  "stage_id": "align",
  "completion_status": "not_complete",
  "next_task_id": null,
  "skill_id": "align_hole",
  "skill_args": {"speed": 0.05},
  "policy_backend": "rlt",
  "decision_version": 0,
  "confidence": 0.9
}
```

完成枚举为 `complete/not_complete/unknown`，后端为 `pi05/rlt/wait/stop`。模型自报 confidence 只用于阈值校验，不能视为经过校准的概率。重复 JSON 字段、未知字段和非法类型会被拒绝。

## Qwen 数据与训练目标

训练清单是 JSONL，每行包含唯一 `sample_id`、相机名称到图像路径的映射、任务 instruction 和目标决策。图像路径相对 manifest，多个相机属于同一决策样本。推理时 `Observation.images` 必须提供所有声明相机的实际 RGB 像素。

`QwenSFTCollator` 对完整对话构造 token，并验证输入 prompt 是训练序列的前缀；prompt 和 padding 的 label 置为 -100，只监督 assistant 回答。LoRA 训练更新 q/k/v/o 投影的低秩参数，输出 adapter 和 processor 文件。

```bash
qwen-vla-sft --manifest data/train.jsonl --output runs/qwen-lora --base-model Qwen/Qwen3-VL-4B-Instruct --device cuda --epochs 1 --rank 16
```

加载接口为 `load_qwen(base_model, adapter_path=..., device=...)`，返回模型和 processor；交给 `QwenTaskBrain` 后，通过 `decide(observation, current_key, completed, decision_version)` 得到结构化决策。真实数据、模型下载和显存配置见 [训练说明](docs/TRAINING.md)。

## RLT 学习机制

### 阶段一：表征压缩

输入冻结 VLA 的 token 序列和有效位置 mask。编码器追加可学习 query，将序列压缩为 RL Token；解码器只使用压缩表示与位置 query 重建输入。输入 detach，padding 不计入 reconstruction loss，目标 token 不通过旁路进入解码器。

### 阶段二：残差强化学习

```text
action = reference + bounded_residual(state, rl_token, reference)
target = discounted_executed_rewards
         + gamma^executed_steps * (1 - done) * min(target_Q1, target_Q2)
```

Actor 末层零初始化，使初始动作等于参考策略；探索和目标平滑之后再次限制残差。训练包含 Twin-Q、Target Actor/Critic、延迟 Actor 更新、软更新和 BC 正则。

普通策略数据的 BC 锚点是参考动作；人工接管数据的锚点是实际动作。`completion` 是语义标记时，需要从物理动作探索、相关 Q 输入和动作发送中分离。

### 回放数据

| NPZ 字段 | 形状 | 说明 |
|---|---|---|
| `state / next_state` | `[N,P]` | 当前与下一本体状态 |
| `token / next_token` | `[N,D]` | 对应观测的 RL Token |
| `action` | `[N,C,A]` | 实际执行动作，接管时为人工动作 |
| `reference / next_reference` | `[N,C,A]` | 分别从当前、下一观测生成的 VLA 动作 |
| `rewards` | `[N,C]` | 逐步奖励，未执行尾部补零 |
| `done / source / executed_steps` | `[N]` | 终止、来源、实际执行长度 |

RL Token 预训练 NPZ 使用 `tokens [N,M,D]` 和 `valid [N,M]`。`N` 为样本数，`C` 为 chunk 长度，`A` 为动作维度。输入格式和 checkpoint 恢复方式见 [训练说明](docs/TRAINING.md)。

`OnlineRLTLoop` 连接环境执行和回放更新。环境必须反馈实际动作及实际执行长度；下一状态的参考动作由下一观测重新生成。奖励来自环境适配器，本项目不根据任务名称自动生成真实成功信号。

## 动作生命周期

1. 新观测携带 session、observation、时间戳和模型接口版本。
2. 高层决策经过任务图、前置条件、技能参数和时效校验。
3. 状态机处理完成确认；同一观测不能重复计票。
4. 策略生成动作，返回时再次检查决策版本和有效期。
5. 控制器通过 `take_action(now_s=...)` 单次获取待执行 chunk。
6. 人工接管调用 `halt()`，新 episode 调用 `reset()`；旧缓存和在途推理结果随之失效。

位置与逐步位移 guard 在软件边界执行。技能参数中的 `force_limit` 只是经过校验后传递给策略的参数，不等于已经实现硬件力矩闭环；加速度、力矩、碰撞和急停必须由机器人适配器负责。

## 代码组织

```text
src/qwen_vla_rlt/
├── qwen.py / sft.py          # 多视角训练、推理及回答 mask
├── schemas.py / brain.py     # 决策契约和严格解析
├── task_graph.py / presets.py# 任务图与任务模板
├── skills.py / router.py     # 技能参数及后端选择
├── state_machine.py          # 完成确认与阶段状态
├── runtime.py / policies.py  # 动作生命周期与 guard
├── adapters.py              # OpenPI、残差策略接入
├── openpi_features.py        # 冻结 VLA 特征与参考动作
├── evaluation.py            # 阶段、完成、合法任务选择评估
└── rlt/
    ├── neural.py            # PyTorch RL Token、Actor、Twin-Q
    ├── replay.py / online.py# 回放与执行反馈
    ├── cli.py               # token / residual 训练入口
    └── models.py / token.py / training.py # NumPy 参考实现
```

## 验证与扩展

当前包含 30 项功能和回归测试，覆盖决策解析、任务转移、重复观测、接管与过期动作、小型神经网络训练、冻结基座、checkpoint、在线数据流和训练 CLI。CI 配置 Python 3.11/3.12。可补充执行：

```bash
ruff check --select S src
qwen-vla-sft --help
qwen-vla-rlt-train --help
```

接入顺序建议：先运行合成示例，再验证自己的观测/动作格式和任务图，然后接入基座与归一化，最后连接真实执行反馈。阶段准确率、完成准确率与任务选择合法性分别计算，不能直接平均成机器人成功率。

OpenPI 的 prefix 适配使用具体版本的私有接口，升级依赖时应重新核对。NumPy 示例用于理解数据流，完整学习路径在 `rlt/neural.py`。软件测试未加载完整 Qwen3-VL-4B/π0.5，也没有验证真实机器人性能。

## 许可证与方法来源

本仓库采用 [Apache-2.0](LICENSE)，版权声明见 [NOTICE](NOTICE)。Qwen、OpenPI、RL Token、TD3 和训练依赖的公开来源见 [SOURCES](docs/SOURCES.md)；模型权重和第三方组件分别适用其自身许可。
