# ALFWorld：小模型轨迹SFT与在线GRPO

本目录发布E068实验快照。学生是Qwen2.5-1.5B-Instruct，使用环境内规则专家成功轨迹建立交互能力，再由学生在独立环境中在线探索，用终局奖励继续训练。不是外部教师模型API蒸馏，也不是完整GiGPO或MiMo系统复现。

## 主要结果

| 模型阶段 | Seen成功率 | Unseen成功率 | 总体任务成功率 |
|---|---:|---:|---:|
| Base | 2/24 | 1/24 | **6.25%（3/48）** |
| SFT | 16/24 | 17/24 | **68.75%（33/48）** |
| SFT＋RL第1轮 | 17/24 | 18/24 | 72.92%（35/48） |
| SFT＋RL第3轮 | 19/24 | 18/24 | **77.08%（37/48）** |

固定48个不同开发游戏，六类各8题，seen/unseen各24题。每游戏独立reset，执行一次完整episode；所有模型采用相同确定性解码、history_length=2、最多50个动作、每步最多512个输出token。终局`won`是主指标；这组数据用于开发选点，不能称作官方完整测试集成绩。

SFT→RL在开发集改善4题、退化0题。另有已冻结模型后的独立复核：final36为Base 1、SFT 24、RL 24；事前冻结的扩展62题为SFT 46、RL 49。本版本提供这些逐题结果，便于区分开发收益与独立验证。当前最明确的结果是专家轨迹SFT建立了环境内执行能力；RL的总体额外收益仍需更多独立证据。

## 数据与训练

- SFT：600个不同官方训练游戏，六类各100；947次专家环境运行得到600条成功轨迹，展开为10,461个决策样本。输入是任务、观察、两步历史和可用动作，监督目标是下一动作；环境观察不计算输出loss。规则专家可使用游戏内部元数据，学生只接收保存的文本prompt。
- SFT配方：BF16 LoRA r16/alpha32，1epoch，LR=1e-4，有效batch8，约26.8分钟。
- RL：3轮fresh在线采样，每题8条完整轨迹。每轮接受双物体8题、灯下检查4题、其余四类合计4题；同组有成功也有失败、没有响应截断才入训。配额不足则不更新。
- RL配方：固定SFT底座/reference，持续恢复未合并adapter和AdamW；LR=1e-5，clip=0.2/0.28，KL=0.01，vendor组内标准化优势，global_token，256个决策/mini-batch，每批一遍。奖励为终局成功10、失败0，非法动作格式在对应决策扣0.1。
- 资源：109个不同候选游戏、872条候选轨迹，包含48个入训游戏/384条保留轨迹、13,358个决策、53个optimizer更新；采样和更新合计约118.4分钟，不含准备与评测。

## 发布内容

| 文件 | 内容 |
|---|---|
| `data/expert_trajectories.jsonl.gz` | 600条经过执行验证的专家轨迹 |
| `data/expert_decisions.jsonl.gz` | 10,461条可读SFT决策记录 |
| `data/expert_decisions_h2.parquet` | 对应训练输入与响应，保留原token长度 |
| `data/rl_round*_rollouts.json.gz` | 三轮实际使用的完整采样记录，含token及旧策略log-prob |
| `data/train_pool.json` | 冻结的2,789题候选训练池，不等于实际入训量 |
| `data/dev_games.json` / `final_games.json` | 不同的固定开发48题与独立36题 |
| `results/` | Base/SFT/RL逐题执行、原训练报告及独立复核 |
| `configs/` | 实际软件版本和固定训练配置 |
| `scripts/` | E068训练/采样/评测源码；公开入口只适配机器路径 |
| `vendor/verl-agent/` | 本轮使用的固定版运行代码，保留上游许可和逐文件哈希 |
| `PROVENANCE.json` | 原产物哈希、导出哈希及路径转换说明 |

数据中的gamefile改为相对`ALFWORLD_DATA`的ID；训练prompt、response和实际动作未改写。原始主机模型路径改为来源标记。CPU核验检查保存数据和结果的一致性，不执行新的模型或环境；重跑rollout需要GPU及ALFWorld环境。

## 复现入口（Linux或WSL2）

```bash
cd experiments/alfworld
python3.10 -m venv .venv
source .venv/bin/activate
# 按显卡安装对应PyTorch；原运行torch=2.8.0，软件锁记录于configs/environment.json
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
export ALFWORLD_DATA="$HOME/.cache/alfworld"
alfworld-download
python reproduce.py prepare --run runs/reproduction
```

下载`Qwen/Qwen2.5-1.5B-Instruct`的固定revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`到本地模型目录。依次运行：

```bash
python reproduce.py train-sft --run runs/reproduction --model /path/to/pinned-qwen2.5-1.5b
python reproduce.py train-rl --run runs/reproduction --model runs/reproduction/sft/merged_hf
python reproduce.py evaluate --run runs/reproduction --model runs/reproduction/sft/merged_hf --split seen
# 使用另一个新run目录对比RL候选；先prepare，再传入未合并adapter
python reproduce.py prepare --run runs/rl-evaluation
python reproduce.py evaluate --run runs/rl-evaluation --model runs/reproduction/sft/merged_hf --adapter runs/reproduction/rl/round_03/update/adapter --split seen
```

`--split unseen`评测另一半。`--dry-run`显示训练/评测命令，不运行模型。输出和checkpoint写入新run目录，发布结果不会被覆盖；随机rollout重跑不承诺逐位复制原成绩。本次没有发布ALFWorld模型权重；旧BFCL release中的adapter仅用于E017。

## 参考

- [ALFWorld官方环境](https://github.com/alfworld/alfworld)：训练任务、TextWorld交互与规则专家。
- [GiGPO / verl-agent](https://github.com/langfengQ/verl-agent)：环境接口和GRPO目标函数；上游commit及本地兼容改动记录在`vendor/verl-agent/UPSTREAM.json`。
- [GiGPO论文](https://arxiv.org/abs/2505.10978)：同型号学生的普通GRPO/GiGPO公开参照，评测协议与本地不同。

项目代码采用Apache-2.0；vendored源代码遵守随附上游许可。轨迹来源、专家先验与导出过程如上，保留ALFWorld/ALFRED与TextWorld来源归属。
