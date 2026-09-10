# Changelog

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
