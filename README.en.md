# Small-Model Agent Post-Training: Trajectory SFT and Online GRPO

**A single-GPU pipeline from executable demonstrations to supervised learning, online reinforcement learning and complete-task evaluation.**

Using **Qwen2.5-1.5B-Instruct**, LoRA SFT on **600 environment-verified rule-based expert trajectories** builds the initial policy. Online interaction, mixed-reward group filtering and task-type quotas support subsequent GRPO training. ALFWorld complete-task success improves from **6.25% → 68.75% → 89.58%**, a **20.83 percentage-point** gain over SFT.

[中文](README.md) · [Latest results and reproduction](experiments/alfworld/releases/v0.3.0/README.md) · [Model assets](https://github.com/ttxttx1111/trajectory-sft-agents/releases/tag/v0.3.0) · [Training data](experiments/alfworld/data/) · [Resume summary](docs/RESUME.md)

![ALFWorld Base, SFT and GRPO complete-task success](assets/alfworld-sft-grpo.png)

| Stage | Complete-task success |
|---|---:|
| Base | 6.25% (3/48) |
| SFT | **68.75% (33/48)** |
| SFT + GRPO | **89.58% (43/48)** |

Each game is executed once under the same protocol, with success determined by the environment. These matched results cover all six task types and were used for development and model selection; the release report provides task-level outcomes and split details.

- **Executable supervision:** 600 successful expert trajectories yield 10,461 decisions; only assistant actions contribute to SFT loss.
- **Online RL:** independent environments, terminal rewards, dynamic group filtering, task-type quotas, and persistent LoRA/AdamW state.
- **Auditable results:** per-task outcomes, six-type breakdowns, execution replay evidence, weight hashes, and 16×8 / 32×8 / 16×16 scaling comparisons.
- **Usable release artifacts:** SFT/RL adapters, the preceding training checkpoint and optimizer, original and portable updater code, training batches and CPU verification.

```bash
python scripts/verify_alfworld_release.py
```

SFT takes approximately 26.8 minutes on one RTX 5090D. The final policy results from further online training; the early three-round RL timing is not its total cost. See the [version report](experiments/alfworld/releases/v0.3.0/README.md) for stage-specific costs and reproduction commands. This research phase is now concluded.


## Quick verification

```bash
python -m unittest discover -s tests -v
python scripts/verify_release.py
python scripts/verify_alfworld_release.py
```

These checks use only the Python standard library. Current training data, results, model assets and reproduction commands are in [experiments/alfworld](experiments/alfworld/releases/v0.3.0/README.md).

Project code is [Apache-2.0](LICENSE). Credits: [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), [ALFWorld](https://github.com/alfworld/alfworld), and [verl-agent](https://github.com/langfengQ/verl-agent). See [NOTICE](NOTICE) and [CITATION.cff](CITATION.cff).
