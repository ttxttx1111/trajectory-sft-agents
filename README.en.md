# Small-Model Agent Post-Training: Trajectory SFT and Online GRPO

**A single-GPU pipeline from executable expert demonstrations to supervised learning and online reinforcement learning.**

Built an ALFWorld SFT-to-GRPO pipeline with **Qwen2.5-1.5B-Instruct**. **600 environment-verified expert trajectories** provide 10,461 assistant-action targets for LoRA SFT. Online G8 rollouts, mixed-reward group filtering, and task-source quotas drive subsequent GRPO updates. Matched development evaluation improves complete-task success from **6.25% → 68.75% → 77.08%**.

[中文](README.md) · [ALFWorld recipe, data and evidence](experiments/alfworld/README.md) · [Training data](experiments/alfworld/data/) · [Resume summary](docs/RESUME.md)

![ALFWorld Base, SFT and GRPO development success](assets/alfworld-sft-grpo.png)

| Stage | Complete-task success |
|---|---:|
| Base | 6.25% (3/48) |
| SFT | **68.75% (33/48)** |
| SFT + GRPO | **77.08% (37/48)** |

Each fixed development game is executed once; the environment determines terminal task completion. Independent evaluations and protocol details are documented alongside the results. Training takes approximately 27 minutes for SFT and 118 minutes for three RL collection/update rounds on one RTX 5090D, excluding data generation, evaluation and earlier debugging.

The release includes expert data, retained RL trajectories, the training/evaluation source snapshot, a pinned ALFWorld runtime, and per-task outcomes. Check saved evidence without downloading a model:

```bash
python scripts/verify_alfworld_release.py
```

The earlier BFCL release remains available and unchanged under its original tag.

## BFCL trajectory-SFT baseline (E017)

**A reproducible, single-GPU trajectory-SFT baseline for multi-turn tool use.**

[中文](README.md) · [Dataset](data/README.md) · [Reproduction](docs/REPRODUCING.md) · [Weights](https://github.com/ttxttx1111/trajectory-sft-agents/releases/tag/v0.1.0-e017) · [Results](docs/RESULTS.md)

Fine-tuning **Qwen3-1.7B** on **120 reference trajectories** improves whole-task success from **2.5% to 32.5% (+30 percentage points)** on a fixed, in-domain held-out split of BFCL `multi_turn_base`. The E017 run used BF16 LoRA on one RTX 5090, taking approximately **70 minutes** with **12.41 GiB peak allocated CUDA memory**. No reinforcement learning or paid teacher API is needed.

![E017 development and test results](assets/e017-results.png)

| Evaluation | Base | E017 LoRA | Gain |
|---|---:|---:|---:|
| Development: 40 tasks | 5/40 (12.5%) | **18/40 (45.0%)** | +32.5 pp |
| Test: 40 tasks | 1/40 (2.5%) | **13/40 (32.5%)** | +30.0 pp |
| Audited test subset: 30 tasks | 1/30 (3.3%) | **10/30 (33.3%)** | +30.0 pp |

The test comparison has **12 paired wins and no paired losses**, with a component-cluster bootstrap 95% gain interval of **+16.3 to +45.0 pp**. The audited subset reuses the same saved predictions; it is a post-hoc sensitivity check.

### General-capability retention and out-of-domain evaluation

The same Base and E017 checkpoints were also compared on two general-capability datasets and an out-of-domain tool-call dataset:

| Evaluation | Base | E017 LoRA | Change | Paired 95% gain interval |
|---|---:|---:|---:|---:|
| ARC-Challenge, 100 questions | 35/100 (35%) | **38/100 (38%)** | **+3.0 pp** | −2.0 to +8.0 pp |
| HellaSwag, 100 questions | 65/100 (65%) | 64/100 (64%) | −1.0 pp | −5.0 to +2.0 pp |
| ToolACE complete next-call match, 80 decisions | 43/80 (53.75%) | **45/80 (56.25%)** | **+2.5 pp** | −7.5 to +12.5 pp |

**The strong in-domain agent-task gain is accompanied by broadly stable observed scores on the sampled general-capability tests, with small positive changes on ARC-Challenge and out-of-domain ToolACE.** All three paired intervals include zero: these are preliminary retention observations, not established transfer gains or proof of zero degradation. ToolACE measures a complete next-call match, not multi-turn task completion.

These are the existing E017 paired evaluations, with no new training or sample selection for presentation. See the [original auxiliary statistics](results/reference/ood_retention/analysis.json) and [results report](docs/RESULTS.md).

### What is released

- **120 readable trajectories and 1,129 next-action examples:** 685 tool-call targets plus 444 per-turn stop targets, complete history, no truncation, target-only loss.
- **The original E017 LoRA weights**, released separately from Git history, with SHA-256 verification and a pinned base-model revision.
- **The original core training/evaluation code**, frozen task-family splits, training logs and all 160 saved Base/LoRA development/test rollouts.
- **An offline BFCL runtime subset and CPU verification:** replay saved tool calls with the original checker and recompute paired statistics, without a model or API key.
- **An isolated reproduction entry point:** new outputs go to `runs/<name>/`, preserving reference results.

### Quick verification

```bash
git clone https://github.com/ttxttx1111/trajectory-sft-agents.git
cd trajectory-sft-agents
python -m pip install -r requirements-check.txt
python -m unittest discover -s tests -v
python scripts/verify_release.py --replay
```

Expected: `passed: true`, `official_saved_rollouts_replayed: 160`. This replays saved actions; it does not regenerate model outputs. See [reproduction instructions](docs/REPRODUCING.md) for GPU inference and training, including a two-task smoke test before a full evaluation.

### Method and scope

Replay public BFCL reference calls in the simulated tool environment → construct complete history/tool-schema/next-action examples → optimize assistant-target cross-entropy with LoRA → evaluate autonomous tool execution with the upstream whole-task checker. At inference, the model sees messages and visible tools, never reference actions or hidden task state.

This release establishes a useful **in-domain tool-task baseline** on a **120/40/40 task-family split**, not an official blind BFCL leaderboard result. Small ToolACE transfer tests show no reliable gain; small ARC/HellaSwag checks show no obvious large decline. E017 retains the original reference targets; passing execution replay does not certify each action's semantic correctness. Single-seed scope, later quality audits and prior task exposure are documented in [the results report](docs/RESULTS.md).

Code and the distributed BFCL-derived data are Apache-2.0. Credits: [Qwen3](https://huggingface.co/Qwen/Qwen3-1.7B) and [Gorilla/BFCL](https://github.com/ShishirPatil/gorilla). See [NOTICE](NOTICE), [dataset card](data/README.md), [model card](docs/MODEL_CARD.md) and [CITATION.cff](CITATION.cff).
