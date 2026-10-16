# 简历表述参考

## 当前版本：轨迹SFT与在线GRPO

**小模型 Agent 能力增强：轨迹 SFT 与在线 GRPO**

- 基于 Qwen2.5-1.5B-Instruct 与 PyTorch / Transformers / PEFT，在单张32GB显卡上搭建 LoRA SFT＋在线GRPO流程；利用600条经环境执行验证的规则专家轨迹构造10,461个决策样本，实现在线采样、动态筛组、任务类别配额与断点续训。
- 覆盖ALFWorld六类任务，完整任务成功率由Base的 **6.25%** 提升至SFT的 **68.75%**，GRPO进一步提升至 **89.58%**，较SFT提高 **20.83个百分点**；完成采样规模对比、逐题执行重放及失败原因分析。

**Small-Model Agent Post-Training: Trajectory SFT and Online GRPO**

Built a LoRA SFT and online GRPO pipeline for Qwen2.5-1.5B-Instruct using PyTorch, Transformers and PEFT on one 32GB GPU. Converted 600 environment-verified rule-based expert trajectories into 10,461 decision samples, with online rollouts, dynamic group filtering, task-type quotas and checkpoint recovery. Across six ALFWorld task types, complete-task success improved from **6.25%** for Base to **68.75%** after SFT and **89.58%** after GRPO (**+20.83 pp** over SFT).

面试展开时解释环境隔离、assistant loss mask、动态采样与任务分布控制、LoRA/AdamW一致续训、按类别评测与训练成本。三阶段数字来自相同协议的开发评测，未将多个不同集合的最高分拼接；详见[版本报告](../experiments/alfworld/releases/v0.3.0/README.md)。
