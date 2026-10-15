# 简历表述参考

## 当前版本：轨迹SFT与在线GRPO

**小模型 Agent 能力增强：轨迹 SFT 与在线 GRPO**

- 基于 Qwen2.5-1.5B-Instruct 与 PyTorch / Transformers / PEFT，在单张32GB显卡上搭建 LoRA SFT＋在线GRPO流程；利用600条经环境执行验证的规则专家轨迹构造10,461个决策样本，实现在线采样、动态筛组、任务类别配额与断点续训。
- 覆盖ALFWorld六类任务，完整任务成功率由Base的 **6.25%** 提升至SFT的 **68.75%**，GRPO进一步提升至 **89.58%**，较SFT提高 **20.83个百分点**；完成采样规模对比、逐题执行重放及失败原因分析。

**Small-Model Agent Post-Training: Trajectory SFT and Online GRPO**

Built a LoRA SFT and online GRPO pipeline for Qwen2.5-1.5B-Instruct using PyTorch, Transformers and PEFT on one 32GB GPU. Converted 600 environment-verified rule-based expert trajectories into 10,461 decision samples, with online rollouts, dynamic group filtering, task-type quotas and checkpoint recovery. Across six ALFWorld task types, complete-task success improved from **6.25%** for Base to **68.75%** after SFT and **89.58%** after GRPO (**+20.83 pp** over SFT).

面试展开时解释环境隔离、assistant loss mask、动态采样与任务分布控制、LoRA/AdamW一致续训、按类别评测与训练成本。三阶段数字来自相同协议的开发评测，未将多个不同集合的最高分拼接；详见[版本报告](../experiments/alfworld/releases/v0.3.0/README.md)。

## 历史版本：BFCL轨迹SFT

## 中文

**小模型Agent轨迹微调与闭环评测｜Qwen3、LoRA、PyTorch、BFCL**

- 构建参考轨迹执行重放、完整历史保留、下一步动作监督与停止策略训练流程，将120条多轮轨迹转化为1,129个可核查训练样本。
- 在单张RTX 5090上完成Qwen3-1.7B的LoRA微调，训练约70分钟、峰值CUDA分配显存12.41 GiB；在固定BFCL域内留出测试集上，将完整任务成功率从2.5%提升到32.5%（+30个百分点）。
- 建立基座预评测、运行预检、任务家族隔离、真实模拟工具闭环执行与配对统计分析流程，公开训练数据、adapter和逐任务结果，并用原版检查器复核160条模型轨迹。
- 补充通用能力保持与域外评测：ARC-Challenge 35%→38%、HellaSwag 65%→64%、ToolACE 53.75%→56.25%；当前小样本未观察到明显退化，正向变化尚未达到统计显著。

## English

**Trajectory SFT for Small Tool-Using Agents | Qwen3, LoRA, PyTorch, BFCL**

- Built an execution-replayed demonstration pipeline with complete histories, next-action supervision and explicit turn-ending targets; converted 120 multi-turn trajectories into 1,129 auditable training examples.
- Fine-tuned Qwen3-1.7B with LoRA on one RTX 5090 in approximately 70 minutes, improving whole-task success from 2.5% to 32.5% (+30 pp) on a fixed in-domain BFCL held-out split.
- Released data, adapter weights and task-level outcomes, with baseline/preflight gates, task-family split isolation, autonomous simulated-tool evaluation and CPU replay verification of 160 saved rollouts.
- Added paired retention/transfer checks: ARC-Challenge 35%→38%, HellaSwag 65%→64%, and ToolACE 53.75%→56.25%; observed broadly stable small-sample scores, with positive changes not statistically established.

项目链接：https://github.com/ttxttx1111/trajectory-sft-agents

面试展开时可解释：为什么保留完整历史；为什么加入停止目标；逐步模仿与自主闭环评测的区别；如何保证评测器正常工作；为什么公开内部划分结果要明确域内范围。
