# 公开方法与依赖来源

本项目实现多模态任务决策、技能路由与残差学习接口。下列公开工作提供方法和接口背景；本项目不声称提出 Qwen、π0.5、RL Token 或 TD3，也不代表上游项目。

- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) 与 [Transformers 接口](https://huggingface.co/docs/transformers/model_doc/qwen3_vl)：多模态基座、processor、训练和推理。
- [Physical Intelligence / OpenPI](https://github.com/Physical-Intelligence/openpi)：VLA 参考动作与模型表征接口。
- [Precise Manipulation with Efficient Online RL](https://www.pi.website/research/rlt)：RL Token 与高精度在线操作的公开参考。
- [TD3 原论文](https://proceedings.mlr.press/v80/fujimoto18a.html)：双 Critic、延迟更新和目标策略平滑的公开参考。
- [PyTorch](https://pytorch.org/)、[PEFT](https://huggingface.co/docs/peft/index) 与 [NumPy](https://numpy.org/)：训练和计算依赖。

第三方包由依赖管理器安装，模型权重由使用者单独获取；各自适用其原许可证和模型条款。本仓库许可证不替代第三方许可，也不授予第三方数据、模型或商标的权利。
