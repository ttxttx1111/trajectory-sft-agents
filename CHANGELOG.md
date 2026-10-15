# Changelog

## 0.3.0 — 2026-10-16

- Update the ALFWorld headline to Base 6.25% → SFT 68.75% → SFT+GRPO 89.58%, using the same 48-task development protocol.
- Add current task-level outcomes, six-type results, execution/weight audits and the 16×8, 32×8, 16×16 training-scale comparison; preserve all earlier results.
- Release SFT/RL inference adapters and the preceding adapter/AdamW checkpoint as separate model assets; include saved training batches, original updater source and portable checkpoint-reproduction commands.
- Extend CPU verification and update Chinese/English documentation, figures and resume summaries. Distinguish stage-specific costs and keep ALFWorld results separate from the earlier BFCL retention checks.
- Conclude this research phase without launching further training or changing historical release tags.

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
