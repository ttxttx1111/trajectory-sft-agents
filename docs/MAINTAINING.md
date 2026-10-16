# Maintaining the ALFWorld release

Current results, model locks, training batches and reproduction commands are in `experiments/alfworld/releases/v0.3.0/`. Adjacent expert data, pinned runtime and earlier ALFWorld outputs are dependencies and evidence for those results; preserve their hashes.

Use fresh `runs/<name>/` directories for new outputs. Do not overwrite published data or source snapshots. Large weights belong in versioned GitHub Release assets.

```bash
git add .
python scripts/update_checksums.py
git add checksums.json
python -m unittest discover -s tests -v
python scripts/verify_release.py
python scripts/verify_alfworld_release.py
```

The checksum updater verifies staged bytes. Original sources and runtime have separate provenance hashes. Research is paused; repository maintenance does not restart experiments. Historical commits and tags remain available without occupying the current project presentation.
