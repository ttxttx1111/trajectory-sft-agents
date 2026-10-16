# ALFWorld SFT＋GRPO：阶段成果 v0.3.0

基于Qwen2.5-1.5B-Instruct，在单张RTX 5090D 32GB上完成专家轨迹SFT与在线GRPO。**完整任务成功率：6.25% → 68.75% → 89.58%，较SFT提高20.83个百分点。** 本页提供方法、逐题结果、数据与模型下载，历史E068快照保持不变。

## 结果

| 模型阶段 | 完整任务成功率 |
|---|---:|
| Base | 3/48 · 6.25% |
| LoRA SFT | 33/48 · 68.75% |
| SFT＋在线GRPO | **43/48 · 89.58%** |

同题配对，SFT→当前GRPO模型有 **11题改善、1题退化**；早期GRPO模型为37/48。本版模型身份为E111-A；其前一个多任务模型E085已经达到43/48，新模型在主开发集总分相同，存在类别间交换，不把最后一批更新当作全部RL收益。

| 任务类别 | Base | SFT | SFT＋GRPO |
|---|---:|---:|---:|
| 简单放置 | 2/8 | 7/8 | 7/8 |
| 照明检查 | 0/8 | 4/8 | 8/8 |
| 清洗后放置 | 1/8 | 7/8 | 7/8 |
| 加热后放置 | 0/8 | 8/8 | 8/8 |
| 冷却后放置 | 0/8 | 7/8 | 7/8 |
| 双物体放置 | 0/8 | 0/8 | 6/8 |
| 总计 | 3/48 | 33/48 | **43/48** |

每题在独立环境reset后执行一次完整任务，由环境`won`判分。所有阶段使用相同48题、六类各8题，来源为官方valid_seen/valid_unseen各24题；确定性解码，最多50个动作，history_length=2，输入上限2048 token，每步输出上限512 token。48题已经用于开发和模型选择，因此据实列为开发集；没有以改名方式声明官方完整测试或独立最终测试成绩。

另两批已使用的开发数据保持分开：官方valid来源98题，SFT70/98、早期GRPO73/98、当前87/98；官方train留出48题，规模对照A/B/C为46/48、46/48、45/48。不能将46/48的95.83%接到上面的同48题提升曲线。原始数据和执行记录分别位于[data](data/)与[results](results/)，不合并为一个新的benchmark分数。

## 技术方案

1. **可验证的监督轨迹。** 环境内规则专家生成并实际执行600条成功轨迹，六类各100条，得到10,461个逐步决策样本；输入包含当前任务、观察、可用动作和两步历史，只有assistant动作参与loss。数据复用[此前公开的600条轨迹](../../data/)，不依赖付费教师API。
2. **LoRA SFT。** Qwen2.5-1.5B-Instruct，BF16底座，r16/alpha32，学习率1e-4，1 epoch，有效batch8。SFT先建立动作格式、环境交互和任务执行能力。
3. **在线GRPO。** 学生在独立环境中采样完整episode，通常同题8条；成功奖励10，非法格式决策扣0.1。只保留终局成功/失败兼有、没有输出截断的完整组，按任务类别配额收集，达到采样上限仍不满足规则则停止或保留为未完成，不能无限过采样。
4. **稳定续训。** 固定合并SFT底座与reference，FP32 LoRA持续更新并恢复AdamW状态；调用固定版verl-agent的组内标准化优势与policy loss。LR1e-5，clip下界0.2/上界0.28，KL系数0.01，global-token loss，每256个决策一次更新，整批单遍。没有使用此前讨论过的无标准差归一化、无KL的MiMo变体。
5. **工程验收。** 训练前重算采样动作概率与原log-prob核对，检查策略身份、token边界、数据排除与checkpoint；保存LoRA、优化器、缓存和RNG，支持完成optimizer step后的断点恢复。按题和类别分析成功/退化，并对变化路径执行环境重放。

最终更新A使用16组×8轨迹，共128轨迹、3,919决策、111,241个assistant token，AdamW从134至150步。它复用相同起点策略已采集并核验的轨迹，**不是最终模型重新在线采样**；这些数据的采样费用保留在来源记录中。

## 训练成本与规模对照

在相同起点E085上，分别调整每批任务组数B和每题轨迹数G。B32与A16使用嵌套任务，G16与G8保留相同的前8条轨迹；各臂独立更新，不是串行续训。

| 规模 | 轨迹数 | 决策数 | optimizer更新 | 开发集48题 | 官方train留出开发48题 | 本臂新增作业耗时 |
|---|---:|---:|---:|---:|---:|---:|
| B16×G8（当前） | 128 | 3,919 | 16 | 43/48 | 46/48 | 22.6分钟 |
| B32×G8 | 256 | 7,870 | 31 | 43/48 | 46/48 | 37.2分钟 |
| B16×G16 | 256 | 7,558 | 30 | 43/48 | 45/48 | 54.9分钟 |

这次局部扩量没有额外总分收益，不能由此推出规模已饱和。A/B大量复用旧轨迹，C另有新采样；本臂耗时含评测与审核，不能当作从零等预算算法对比。整个规模探索累计217.3分钟，包含准备、采样失败和续作、更新、评测与审核。

SFT本身训练26.8分钟；早期三轮RL采样＋更新118.4分钟。到前一个43/48模型E085，已选训练路径累计SFT＋RL采样/更新约412.4分钟，排除评测、准备和历史失败方案探索；它也不是本版全部探索的总费用。详见[原始成本记录](results/cost_through_reference.json)、[补记](results/cost_corrections.json)与[summary.json](summary.json)，不同范围不得直接相加。尚未完成严格等预算SFT-only / RL-only / SFT→RL三路上限比较。

## 模型、数据和代码

- [v0.3.0模型下载](https://github.com/ttxttx1111/trajectory-sft-agents/releases/tag/v0.3.0)：`alfworld-inference-adapters.zip`含SFT与当前RL的原始adapter张量；`alfworld-update-input.zip`含最后一批更新的起点adapter及AdamW134。
- [models.lock.json](models.lock.json)固定原Qwen revision、SFT合并模型、各adapter与资产SHA256。adapter配置仅转换为可移植来源名，张量保持原样。
- CPU与原GPU合并adapter会有少量舍入差异；推理资产另含这些位置的原始SFT权重值。复现入口自动应用并检查完整模型SHA256，从而精确重建报告使用的SFT底座，不另行训练。
- [data/rollouts_A.json.gz](data/rollouts_A.json.gz)、B、C包含各臂实际使用的prompt/completion token、旧策略log-prob、动作及反馈。训练与评测任务不交叠。
- [source_snapshot](source_snapshot/)保存原更新器、checkpoint与审计/控制器源码；[scripts](scripts/)只对运行时/数据路径作可移植适配，变更由[PROVENANCE.json](PROVENANCE.json)记录。原控制器含机器路径，是来源快照；公开运行使用下述入口。
- 旧E068的[在线采样/SFT入口](../../README.md#复现入口linux或wsl2)仍可运行；它复现早期三轮流程，不声称三轮就能得到本版89.58%。

## 复现

CPU核验：在仓库根目录运行，无需模型或GPU。

```bash
python scripts/verify_alfworld_release.py
```

预期`latest_release.passed=true`、Base/SFT/GRPO为3/33/43；检查的是保存的结果、完整轨迹维度、源码与文件哈希，不是重新生成模型输出。

GPU训练/评测使用Linux或WSL2；先按[原环境说明](../../README.md#复现入口linux或wsl2)安装PyTorch及依赖并下载ALFWorld，版本锁见[environment.json](../../configs/environment.json)。从仓库根目录开始：

```bash
gh release download v0.3.0 --repo ttxttx1111/trajectory-sft-agents \
  --pattern 'alfworld-*.zip' --dir runs/alfworld-assets
unzip runs/alfworld-assets/alfworld-inference-adapters.zip -d runs/alfworld-assets
unzip runs/alfworld-assets/alfworld-update-input.zip -d runs/alfworld-assets
export ALFWORLD_DATA="$HOME/.cache/alfworld"
export CKPT=experiments/alfworld/releases/v0.3.0/reproduce_checkpoint.py
# BASE指向models.lock.json中固定revision的原Qwen模型
python "$CKPT" merge-sft --base "$BASE" --assets runs/alfworld-assets --run runs/alfworld-model
python "$CKPT" prepare --run runs/alfworld-eval
python "$CKPT" evaluate --run runs/alfworld-eval --policy rl --split seen \
  --model runs/alfworld-model/merged_sft --assets runs/alfworld-assets
python "$CKPT" evaluate --run runs/alfworld-eval --policy rl --split unseen \
  --model runs/alfworld-model/merged_sft --assets runs/alfworld-assets
```

**加载顺序：原Qwen底座 → 合并SFT adapter → 载入RL adapter。** 直接把RL adapter挂到原Qwen上不对应报告模型。对比SFT时使用`--policy sft`和同一合并模型；对比Base使用`--policy base --model "$BASE"`。可先加`--limit 2`检查链路，用新的run目录做正式评测；`--dry-run`仅打印命令。

重跑最终一批GRPO更新：

```bash
python "$CKPT" prepare --run runs/alfworld-update-A --arm A --assets runs/alfworld-assets
python "$CKPT" update-from-snapshot --run runs/alfworld-update-A --arm A \
  --model runs/alfworld-model/merged_sft --assets runs/alfworld-assets
```

准备阶段物化可移植gamefile与起点adapter路径；执行前核对发布的模型/优化器哈希与完整批次。B/C用新的run目录、分别`--arm B/C`，均从同一发布起点独立更新。保存轨迹重放更新可以复核目标函数和checkpoint流程；新在线采样受随机性影响，不保证逐位或逐题复制原成绩。本次发布只做CPU材料/入口/资产检查，不在暂停研究期间重新训练。

## 来源与适用范围

环境、专家与任务来自ALFWorld/ALFRED/TextWorld，环境接口及GRPO核心函数来自固定版GiGPO/verl-agent；代码和上游许可见仓库[NOTICE](../../../../NOTICE)及原vendor目录。本项目验证小模型在ALFWorld领域内的任务完成能力增强，未把它写成通用跨领域能力提升。

当前阶段结束，保留全部历史版本、负结果与原始产物，不自动继续训练。
