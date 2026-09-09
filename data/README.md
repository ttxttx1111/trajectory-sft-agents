# E017 Dataset Card

## Source and construction

This release derives from public **Gorilla / BFCL v4 `multi_turn_base` reference tasks and reference calls**, at the commit pinned in [`sources.lock.json`](../configs/sources.lock.json). It contains **no newly generated proprietary teacher-model responses**. Calling this reference-trajectory SFT precisely describes E017; the broader research direction is learning agent behavior from demonstrations.

The 200 task families are split into **120 train / 40 development / 40 test**. Near-duplicate connected components are kept together by the frozen [`manifest.json`](E017_bfcl_indomain/manifest.json). Only the 120 training families supply SFT targets. The upstream benchmark files include all 200 tasks for reproducible evaluation, and evaluation answers remain separate from model-visible prompts.

Construction steps:

1. Initialize the official simulated tool state for a training task.
2. Follow its public reference calls sequentially, actually execute each call, and append the returned observation.
3. Before each action, save the full preceding `history`, the visible `tools`, and the next `candidate_action`.
4. Add one assistant stop reply at the end of each user turn. It teaches ending that turn; it is not an additional executable tool.
5. Verify the full reference path with the BFCL checker, then encode each decision with the pinned Qwen3 tokenizer and non-thinking chat template.

All 120 reconstructed training paths pass this execution checker. Export verification rebuilds all 1,129 encoded rows and compares every field with the actual E017 training records.

## Files and scale

| File | Rows | Purpose |
|---|---:|---|
| [`train_trajectories.jsonl.gz`](train_trajectories.jsonl.gz) | 120 | Complete readable trajectories, including tool definitions |
| [`train_decisions.jsonl.gz`](train_decisions.jsonl.gz) | 1,129 | Readable history → next-action samples |
| [`E017_bfcl_indomain/train_R.jsonl.gz`](E017_bfcl_indomain/train_R.jsonl.gz) | 1,129 | Exact tokenized E017 training records |
| [`example.json`](example.json) | 1 | Actual readable decision with an earlier tool observation |
| [`export_verification.json`](export_verification.json) | — | Export/re-encoding verification summary |
| [`quality_clean.json`](quality_clean.json) | — | Later audit's retained evaluation IDs, for sensitivity analysis |

There are **685 tool-call targets and 444 stop targets**. Complete encoded examples span **2,226–5,745 tokens**, including tool schemas and repeated history. Their total input length is **4,515,009 tokens per epoch**, with repeated histories counted. Three training epochs expose the optimizer to 3,387 examples. This is neither 1,129 independent task families nor 4.5 million unique tokens.

## Schema

Readable decision fields:

```text
id                  stable task/turn/action ID
task_id             task-family ID; belongs to the train split
turn, action_index  position in the interaction
target_type         tool_call or terminate
history             all preceding system/user/assistant/tool messages
tools               visible function schemas
candidate_action    assistant message to predict
source_gold_call    upstream reference call, if applicable
parameter_sources   heuristic provenance hints, if applicable
```

Read an actual sample without extra libraries:

```python
import gzip, json
with gzip.open("data/train_decisions.jsonl.gz", "rt", encoding="utf-8") as f:
    sample = json.loads(next(f))
print(sample["history"])
print(sample["candidate_action"])
```

The tokenized representation provides `input_ids`, `target_start`, `target_tokens` and `full_tokens`. Context labels are ignored; labels from `target_start` onward contain the assistant target tokens. The original trainer computes exactly this shifted causal cross-entropy using only the required logits. It checks that loss against ordinary masked-label loss before training.

The complete-trajectory export also includes `assistant_loss_mask`, a Boolean per message, indicating which messages are eligible assistant targets. Training E017 used separate decision examples, **not** a single averaged loss over every assistant token in a full trajectory. Switching to full-trajectory packing or token-weighted losses changes the training recipe.

## History, errors and quality

E017 uses **full history without truncation or summarization**. Each history includes earlier assistant calls, tool returns and stop replies. All original reference action targets are retained; the name `train_R` is the frozen internal identifier, not a claim that quality filtering was performed. Later audits found questionable actions in some reference trajectories. Checker success is evidence of a reference path's accepted execution outcome, not proof that every intermediate action or parameter is semantically ideal.

Tool errors occurring along reference paths are preserved. E017 does not deliberately generate counterfactual errors, adjacent repairs, missing-tool requests or a separately balanced recovery/no-tool curriculum. Stop replies provide 444 no-call targets, but should not be equated with a diverse tool-irrelevance dataset. `parameter_sources` is a heuristic diagnostic, not a correctness label.

The later clean-ID evaluation mask does not alter the E017 training dataset. See the [results report](../docs/RESULTS.md) for the matching sensitivity results.

## License and responsible reuse

BFCL-derived data and conversion additions are distributed under **Apache-2.0**, with the upstream license in [`vendor/gorilla/LICENSE`](../vendor/gorilla/LICENSE) and attribution in [`NOTICE`](../NOTICE). Task messages, schemas and tool states are public benchmark fixtures; names, accounts and credential-like strings found in them belong to those fixtures, not this developer's service credentials. This dataset is an internal-split research resource: training on it changes the meaning of later scores on the corresponding public BFCL tasks.
