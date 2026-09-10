# v0.1.0-e017 — Reproducible trajectory SFT for a 1.7B tool-use agent

The first public release packages E017: 120 reference trajectories, single-GPU LoRA training, and autonomous BFCL tool-environment evaluation.

**Primary result:** Qwen3-1.7B test whole-task success improves from **1/40 to 13/40 (+30 pp)** on the fixed in-domain split; development improves from **5/40 to 18/40**. The original training took about **70 minutes on one RTX 5090**.

Included in the repository:

- 120 readable full trajectories and 1,129 readable/tokenized supervision examples.
- Original core training/evaluation code, frozen split and source hashes.
- All 160 saved Base/LoRA development/test rollouts, training logs and paired statistics.
- An offline upstream BFCL subset, CPU verification, tests and isolated reproduction commands.
- Dataset/model cards, Chinese/English project summaries and a detailed results report.

The attached `e017-qwen3-1.7b-lora.zip` contains the original adapter tensors, a portable adapter configuration and license. Base weights are downloaded separately from the pinned Qwen3-1.7B revision. File checksums are in `configs/adapter.lock.json`.

**General-capability retention and out-of-domain checks:**

| Evaluation | Base | E017 | Change | Paired 95% gain interval |
|---|---:|---:|---:|---:|
| ARC-Challenge | 35/100 | 38/100 | +3.0 pp | −2.0 to +8.0 pp |
| HellaSwag | 65/100 | 64/100 | −1.0 pp | −5.0 to +2.0 pp |
| ToolACE complete next-call match | 43/80 | 45/80 | +2.5 pp | −7.5 to +12.5 pp |

These existing E017 checks show broadly stable observed general-capability scores and small positive changes on ARC and ToolACE. All intervals include zero; this is a preliminary retention observation, not proof of zero degradation or established transfer gains. The [current README](https://github.com/ttxttx1111/trajectory-sft-agents#通用能力保持与域外评测) highlights these results. This documentation clarification leaves the original release tag, model weights and evaluation records unchanged.

This is an in-domain internal-split research result; see `docs/RESULTS.md` for the evaluation protocol and the concise scope disclosure. Future experiments are separate from this versioned snapshot.
