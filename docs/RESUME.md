# 简历表述参考

## 中文

**小模型Agent轨迹微调与闭环评测｜Qwen3、LoRA、PyTorch、BFCL**

- 构建参考轨迹执行重放、完整历史保留、下一步动作监督与停止策略训练流程，将120条多轮轨迹转化为1,129个可核查训练样本。
- 在单张RTX 5090上完成Qwen3-1.7B的LoRA微调，训练约70分钟、峰值CUDA分配显存12.41 GiB；在固定BFCL域内留出测试集上，将完整任务成功率从2.5%提升到32.5%（+30个百分点）。
- 建立基座预评测、运行预检、任务家族隔离、真实模拟工具闭环执行与配对统计分析流程，公开训练数据、adapter和逐任务结果，并用原版检查器复核160条模型轨迹。

## English

**Trajectory SFT for Small Tool-Using Agents | Qwen3, LoRA, PyTorch, BFCL**

- Built an execution-replayed demonstration pipeline with complete histories, next-action supervision and explicit turn-ending targets; converted 120 multi-turn trajectories into 1,129 auditable training examples.
- Fine-tuned Qwen3-1.7B with LoRA on one RTX 5090 in approximately 70 minutes, improving whole-task success from 2.5% to 32.5% (+30 pp) on a fixed in-domain BFCL held-out split.
- Released data, adapter weights and task-level outcomes, with baseline/preflight gates, task-family split isolation, autonomous simulated-tool evaluation and CPU replay verification of 160 saved rollouts.

项目链接：https://github.com/ttxttx1111/trajectory-sft-agents

面试展开时可解释：为什么保留完整历史；为什么加入停止目标；逐步模仿与自主闭环评测的区别；如何保证评测器正常工作；为什么公开内部划分结果要明确域内范围。
