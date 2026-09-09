# Reproduce E017

Use the public entry point for new runs. It writes into `runs/<name>/` and refuses to mix an existing arm/split with new output. The original result files under `results/reference/` remain a fixed reference.

## 1. CPU verification, no model or API key

Python 3.10 or 3.11:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-check.txt
python -m unittest discover -s tests -v
python scripts/verify_release.py --replay
```

This validates frozen source hashes, complete training records, disjoint task-family splits, dataset-export alignment, saved task outcomes, paired statistics and all 160 saved rollouts with the pinned BFCL checker. It executes local simulated tools, not real booking/trading/messaging services. It does not invoke an LLM or download weights.

## 2. GPU environment and downloads

Install a CUDA-enabled PyTorch build compatible with your GPU, then:

```bash
python -m pip install -r requirements.txt
python scripts/reproduce.py download
```

The recorded training/release-check environment was Windows, Python 3.10, RTX 5090, PyTorch `2.12.0.dev20260331+cu128`, Transformers `5.4.0`, PEFT `0.19.1`, Accelerate `1.13.0`. The historical PyTorch nightly is recorded for provenance; availability of that exact wheel is not guaranteed. Choose an appropriate CUDA PyTorch build from the [official installer](https://pytorch.org/get-started/locally/). GPU inference/training is tested on the recorded environment; CI tests the CPU runtime on Linux. Different kernels/library versions may change greedy generations or trained weights.

Download uses the pinned Qwen3-1.7B revision and the checksummed E017 release archive. It needs network access once, several GB of disk for base weights and approximately 65 MB for the compressed adapter. Subsequent evaluation is local. Optional commands:

```bash
python scripts/reproduce.py download --only model
python scripts/reproduce.py download --only adapter
```

An existing base-model directory can be supplied via `--model /path/to/Qwen3-1.7B` to evaluation, training and dataset export. Config/tokenizer hashes must match the frozen snapshot. A custom adapter may be supplied via `--adapter /path/to/adapter`.

## 3. Cheap inference check first

```bash
python scripts/reproduce.py evaluate --run smoke --arm base --split dev --limit 2
python scripts/reproduce.py evaluate --run smoke --arm tuned --split dev --limit 2
```

These are the first two frozen development tasks, not a selected success showcase. Partial runs are explicitly marked and should not be reported as the 40-task result. A failure to finish writes an infrastructure error; fix it and use a fresh run name. A repeated command refuses to overwrite partial or complete output.

## 4. Re-evaluate the released model

```bash
python scripts/reproduce.py evaluate --run released-e017 --arm base --split dev
python scripts/reproduce.py evaluate --run released-e017 --arm tuned --split dev
python scripts/reproduce.py evaluate --run released-e017 --arm base --split test
python scripts/reproduce.py evaluate --run released-e017 --arm tuned --split test
```

Each command runs 40 tasks. The original limits are preserved: 32,768 input tokens, 512 new tokens per decision, 21 decisions per user turn, greedy decoding and non-thinking Qwen chat serialization. Context overflow is recorded as failure, not silently truncated. Full evaluation takes materially longer than a two-task check and depends on model-generated trajectory lengths.

Per-task JSONL, per-arm summaries and run provenance (model shard hashes, adapter hash, source manifest, limits) are stored under `runs/released-e017/`. Evaluation with the released adapter is a fresh reproduction of a public result; it does not create a new blind test.

## 5. Train the frozen recipe

Measure the complete Base development set before spending training compute:

```bash
python scripts/reproduce.py evaluate --run retrain-e017 --arm base --split dev
python scripts/reproduce.py train --run retrain-e017 --preflight
python scripts/reproduce.py train --run retrain-e017
python scripts/reproduce.py evaluate --run retrain-e017 --arm tuned --split dev
```

The training entry point invokes the original E017 implementation. Before training it checks all 200 reference paths, the parser, all 40 Base development records and frozen data/code hashes. `--preflight` performs forward/backward and one optimizer update on the longest two training examples; these disposable probe weights are not used to initialize the actual training run. The real training run starts from the base checkpoint and frozen seed, verifies masked-loss alignment, then saves `runs/retrain-e017/adapter/`.

When a run has its own trained adapter, tuned evaluation uses that adapter by default. Otherwise it uses `models/e017/`, the downloaded release. Never silently substitute an adapter: inspect each run's recorded adapter hash.

For a matching dev/test workflow, compare development outcomes before invoking the final test. E017's original go criterion is documented in the [results report](RESULTS.md). The public runner permits directly repeating the already-published test and does not claim to re-create test blindness.

## 6. Reconstruct readable data

```bash
python scripts/export_dataset.py --verify-only
```

This locally replays the 120 training reference paths, re-encodes all 1,129 decisions with the pinned tokenizer and requires exact equality with the published original token records. Omit `--verify-only` only when intentionally rebuilding the readable exports in your own checkout. The encoded file is distributed as gzip; training decompresses it only after checksum validation.

## Original sources and portable wrapper

Seven core files listed in the frozen manifest are distributed byte-for-byte, including their original line endings. `analyze_e017_results.py` is also preserved. The public `reproduce.py`, `export_dataset.py`, `verify_release.py` and tests add portability and isolation without rewriting the original objective or rollout code. `prepare_e017_split.py` is historical source: its old inventory inputs are not bundled, so reuse the published split manifest instead of rerunning split selection. Legacy convenience helpers in `common.py` refer to an older pilot configuration; the E017 entry points do not use them.

The vendored BFCL subset is pinned in [`UPSTREAM.json`](../vendor/gorilla/UPSTREAM.json), includes its upstream license, and is sufficient for `multi_turn_base`. It does not promise to run every other BFCL category. `.gitattributes` preserves the exact source/data bytes needed by the historical hashes on fresh Linux and Windows clones.

To regenerate the result figure, install `matplotlib` and run `python scripts/plot_results.py`; the chart reads the saved primary analysis and exports both PNG and SVG.
