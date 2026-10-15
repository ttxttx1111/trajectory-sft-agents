# Maintaining the public snapshot

`v0.1.0-e017` is a fixed, outward-facing release. It contains E017 evidence and the later clean-ID sensitivity check; it does not mirror every ongoing research experiment. There is no automatic synchronization with another local research directory.

The current ALFWorld phase release is `v0.3.0`, in `experiments/alfworld/releases/v0.3.0/`. Its headline is Base 6.25% → SFT 68.75% → GRPO 89.58% under the same development protocol. Preserve the earlier `v0.2.0-e068` files and tag; the root README and figure point to the current version. Research is paused; publishing documentation or evidence does not authorize further experiments.

Keep new experimental outputs under a new `runs/<name>/`. Never overwrite `results/reference/` or reuse E017's frozen manifest to label a changed method. Publish a future experiment with its own dataset/model identifiers, results and version tag; leave the original tag available for citation.

Before publishing an intentional update:

```bash
git add .
python scripts/update_checksums.py
git add checksums.json
python -m unittest discover -s tests -v
python scripts/verify_release.py --replay
python scripts/verify_alfworld_release.py
```

`checksums.json` seals the staged release contents. The updater refuses unstaged byte changes, including unintended newline conversion. The original core source hashes are separately fixed in E017's manifest; change a new experiment's code rather than rewriting that historical evidence.

Large weights belong in versioned GitHub Release assets, with their hashes and base-model revision in `configs/`. Keep downloaded models, environments and new runs out of Git. The [release-validation report](../results/release_validation.json) records the checks performed when packaging E017.
