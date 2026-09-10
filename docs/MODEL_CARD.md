# E017 Qwen3-1.7B LoRA

**Multi-turn tool-use adaptation of Qwen3-1.7B through supervised reference trajectories.**

| Property | Value |
|---|---|
| Base | `Qwen/Qwen3-1.7B` — not Qwen3.8-27B |
| Base revision | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| Training | BF16 LoRA, target-only causal cross-entropy |
| LoRA | rank 16, alpha 32, dropout 0; all attention and MLP projection modules |
| Data | 120 BFCL reference trajectories → 1,129 decision targets |
| Schedule | 3 epochs, learning rate 1e-4, effective batch 8, cosine, 5% warmup |
| Optimizer | AdamW, weight decay 0.01, gradient clipping 1.0 |
| Compute | One RTX 5090; 4,201 seconds; 12.41 GiB peak allocated CUDA memory |
| Inference | BF16, Qwen JSON tool-call template, greedy, `enable_thinking=False` |
| Distribution | LoRA adapter only; base weights downloaded separately |
| License | Apache-2.0; retain base and upstream notices |

The complete target history is preserved. Training processes one example at a time with gradient accumulation to effective batch eight. It performs 424 optimizer updates and 3,387 example exposures. Peak allocated memory does not include every driver/reserved-memory overhead and is not a guaranteed minimum GPU size.

## Result

On the frozen in-domain test split, whole-task success increases from **1/40 to 13/40**. The development result is **5/40 to 18/40**. See the [results report](RESULTS.md) for paired statistics and scope, and the [reproduction guide](REPRODUCING.md) for loading and evaluation.

Auxiliary paired checks show **ARC-Challenge 35/100 → 38/100**, **HellaSwag 65/100 → 64/100**, and **ToolACE complete next-call match 43/80 → 45/80**. General-capability scores remain broadly stable on these small samples, with positive point estimates for ARC and ToolACE. Their paired 95% change intervals all include zero; they establish neither transfer gains nor zero degradation.

## Download integrity

The [v0.1.0-e017 release](https://github.com/ttxttx1111/trajectory-sft-agents/releases/tag/v0.1.0-e017) contains `e017-qwen3-1.7b-lora.zip` with `adapter_model.safetensors`, `adapter_config.json` and `LICENSE`. Checksums are in [`configs/adapter.lock.json`](../configs/adapter.lock.json).

The weight bytes match the original E017 checkpoint. Only adapter configuration metadata was made portable: the private local base-model path was replaced by the public Qwen model ID and its pinned revision. No merging, quantization or extra fine-tuning was applied for release.

## Intended use

Local research, reproducible trajectory-SFT experiments and simulated tool-use evaluation. The published result concerns the shared BFCL tool domain and its internal held-out task families. The adapter has not established broad cross-domain agent generalization. It has not been evaluated for unsupervised operation of real external services.
