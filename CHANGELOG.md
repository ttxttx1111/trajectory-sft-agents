# Changelog

## 0.2.0-e068 — 2026-10-09

- Add the single-GPU ALFWorld expert-SFT and online GRPO experiment, with matched development success of 6.25% → 68.75% → 77.08%.
- Release 600 verified expert trajectories, 10,461 SFT decisions, three retained RL rollout batches, pinned source/runtime, configurations and per-task evaluation records.
- Include independent validation outcomes and a CPU-only evidence verifier; provide portable Linux/WSL entry points without changing ongoing research runs.
- Update Chinese/English project introductions and resume summaries. Preserve the E017 snapshot, its original tag, weights and evidence.

## Documentation update — 2026-09-10

- Surface the existing ARC-Challenge, HellaSwag and ToolACE paired evaluations in both READMEs, the model card, release notes and resume examples.
- Report point estimates and paired intervals together, distinguishing preliminary capability retention from a proven absence of degradation.
- Preserve E017's original tag, adapter weights and evaluation records.

## 0.1.0-e017

- Publish the E017 reference-trajectory SFT baseline and unchanged LoRA weight tensors.
- Export full readable training trajectories and verify every tokenized training record against execution replay.
- Bundle the pinned offline BFCL runtime needed for this evaluation, with upstream attribution.
- Preserve original core scripts and reference results; add isolated reproduction commands and CPU checks.
- Document primary results, small transfer/retention checks, data provenance and reproduction steps.
