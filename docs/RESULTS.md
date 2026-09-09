# E017 Experimental Results

E017 provides a positive, reproducible baseline for **in-domain multi-turn tool use with trajectory SFT**: the adapted 1.7B model completes more held-out tasks through autonomous interaction with a simulated environment.

## Primary evaluation

| Split | Base | E017 | Gain | Paired 95% gain interval |
|---|---:|---:|---:|---:|
| Development, 40 families | 5/40 (12.5%) | 18/40 (45.0%) | +32.5 pp | +17.1 to +48.7 pp |
| Test, 40 families | 1/40 (2.5%) | 13/40 (32.5%) | +30.0 pp | +16.3 to +45.0 pp |

The test set has 12 E017-only successes and 0 Base-only successes. Exact two-sided McNemar p = 0.00048828125. Intervals use the frozen implementation's **paired connected-component cluster bootstrap**, 20,000 draws, preserving within-component dependence. These intervals measure uncertainty across the sampled task components; they do not estimate variation across training seeds.

The metric requires both the upstream BFCL whole-task checker to return valid and the rollout to complete without the harness hitting a step/context limit. It is not next-token accuracy, a teacher-forced next-action score or a subjective LLM judge. The rollout model receives only user/history messages and visible tool schemas; reference actions and hidden environment state are used by execution/scoring, never inserted into its prompt.

The implementation and raw evidence are available in [`analysis.json`](../results/reference/analysis.json), the [Base rollouts](../results/reference/base/) and [adapted rollouts](../results/reference/tuned/). The CPU verifier replays **all 160 saved outputs** with the pinned upstream checker and recomputes the paired statistics.

## Additional behavioral evidence

On the 40 test tasks:

| Diagnostic | Base | E017 |
|---|---:|---:|
| First predicted tool name matches reference | 21/40 | 33/40 |
| Mean reference tool-name multiset recall | 61.8% | 90.5% |
| Tool-call decode-error events | 4 | 0 |

These diagnostics help describe behavior; the main claim rests on complete task outcomes. Tool-name recall alone does not guarantee correct parameters or successful execution.

## Quality-audit sensitivity check

A subsequent reference-quality audit defined clean evaluation subsets. Applying their IDs to the **same saved predictions** gives:

| Subset | Base | E017 | Gain |
|---|---:|---:|---:|
| Clean development, 22 tasks | 3/22 | 14/22 | +50.0 pp |
| Clean test, 30 tasks | 1/30 | 10/30 | +30.0 pp |

The positive test gain remains after this exclusion. This is a post-hoc sensitivity analysis, not a second independently held-out test or a quality-filtered training run. The mask is published as [`quality_clean.json`](../data/quality_clean.json).

## Small transfer and capability-retention checks

| Check | Base | E017 | Gain | Paired 95% interval |
|---|---:|---:|---:|---:|
| ToolACE complete next-call match | 43/80 | 45/80 | +2.5 pp | −7.5 to +12.5 pp |
| ARC-Challenge | 35/100 | 38/100 | +3.0 pp | −2.0 to +8.0 pp |
| HellaSwag | 65/100 | 64/100 | −1.0 pp | −5.0 to +2.0 pp |

These checks show no reliable ToolACE transfer gain and no obvious large decline on the two sampled general-capability tests. They do not establish unchanged general capability. The [original auxiliary summary](../results/reference/ood_retention/analysis.json) is included for context; this release's runnable end-to-end reproduction target is BFCL, and it does not bundle the separate auxiliary evaluation pipelines or datasets.

## Training and protocol

The original recipe uses rank-16 BF16 LoRA on Qwen3-1.7B, three epochs, an effective batch of eight, learning rate 1e-4 and a cosine schedule. It supervises the next reference call or end-of-turn reply conditioned on the full history and tool schemas. All 1,129 examples are kept intact. See the [training report](../results/reference/training_report.json), [full log](../results/reference/train_log.jsonl) and [dataset card](../data/README.md).

The 120/40/40 split was frozen before E017 model outputs. Connected near-duplicate task families remain in one split. Base measurement and an executable reference oracle gate precede training. The development go rule requires at least four adapter-only successes, more wins than losses, and no reduction in total successes. The saved [development decision](../results/reference/dev_go_decision.json) passed before the original final test evaluation.

## Interpretation

This is **one seed and one internal split of a public benchmark**, with shared tool domains across training and evaluation. It is not an official blind BFCL leaderboard score. The frozen manifest discloses prior project exposure for 9/40 development and 9/40 test task families; E017's own SFT data contains only its 120 training families, but the evaluation is not globally untouched across the whole research project. E017 also preserves the upstream reference targets without error-action masking; a later audit flagged issues in 33/120 training families, so successful reference replay should not be read as semantic certification of every target. These boundaries support presenting the result as a useful local, in-domain baseline while further scale and transfer studies continue.
