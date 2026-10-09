# 简历表述参考

## 当前版本：轨迹SFT与在线GRPO

**小模型 Agent 能力增强：轨迹 SFT 与在线 GRPO**

基于 Qwen2.5-1.5B，搭建 ALFWorld Agent 的 SFT＋GRPO 后训练流程。利用600条经环境执行验证的专家轨迹进行 LoRA SFT，并通过在线交互采样、组内奖励差异筛选和任务配额控制开展强化学习。评测中，完整任务成功率由 Base 的 **6.25%** 提升至 SFT 的 **68.75%**，RL进一步提升至 **77.08%**；结合独立评测、逐题失败分析与能力退化监测验证训练收益。

**Small-Model Agent Post-Training: Trajectory SFT and Online GRPO**

Built an ALFWorld SFT-to-GRPO pipeline for Qwen2.5-1.5B using 600 environment-verified expert trajectories, online rollouts, mixed-reward group filtering and task quotas. Evaluated complete-task success improved from **6.25%** for Base to **68.75%** after SFT and **77.08%** after RL; used independent evaluations, per-task failure analysis and capability-regression monitoring to audit training outcomes.

面试展开时说明：三阶段数字来自同协议开发评测；如何独立reset环境、屏蔽观察的loss、过滤零方差组、保持来源配额，以及独立复核与开发结果的区别。详见[ALFWorld实验说明](../experiments/alfworld/README.md)。

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
