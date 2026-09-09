"""Freeze the E017 BFCL in-domain 120/40/40 family split before model output."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from common import BFCL_ROOT, ROOT, digest, file_hash, read_json, read_jsonl, write_json


OUT = ROOT / "data/E017_bfcl_indomain"
TASK_PATH = BFCL_ROOT / "bfcl_eval/data/BFCL_v4_multi_turn_base.json"
ANSWER_PATH = BFCL_ROOT / "bfcl_eval/data/possible_answer/BFCL_v4_multi_turn_base.json"
INVENTORY_PATH = ROOT / "data/E001_v2/family_inventory.json"
OLD_MANIFEST_PATH = ROOT / "data/E001_v2/eval_manifest.json"
PLAN_PATH = ROOT / "paper/E017_BFCL_INDOMAIN_PRACTICE_PLAN.md"
MODEL = ROOT / "models/Qwen3-1.7B"
SALT = "E017-20260907-bfcl-indomain-622"
SEED = 20260917


def suffix(task_id: str) -> int:
    return int(task_id.rsplit("_", 1)[1])


def components() -> list[list[int]]:
    inventory = read_json(INVENTORY_PATH)
    parent = list(range(200))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for edge in inventory["edges"]:
        a, b = edge["families"]
        ra, rb = find(a), find(b)
        parent[ra] = rb
    grouped: dict[int, list[int]] = defaultdict(list)
    for i in range(200):
        grouped[find(i)].append(i)
    result = [sorted(group) for group in grouped.values()]
    assert sorted(i for group in result for i in group) == list(range(200))
    assert sorted(group for group in result if len(group) > 1) == [[21, 22], [40, 43], [116, 117]]
    return result


def allocate(groups: list[list[int]]) -> dict[str, list[int]]:
    pairs = sorted((g for g in groups if len(g) == 2), key=lambda g: digest([SALT, "pair", g]))
    singles = sorted((g[0] for g in groups if len(g) == 1), key=lambda i: digest([SALT, "single", i]))
    # Each split receives one duplicate pair. The remaining 194 singleton
    # components then give exact nominal counts of 120/40/40.
    split = {
        "train": sorted(pairs[0] + singles[:118]),
        "dev": sorted(pairs[1] + singles[118:156]),
        "test": sorted(pairs[2] + singles[156:]),
    }
    assert {key: len(ids) for key, ids in split.items()} == {"train": 120, "dev": 40, "test": 40}
    assert sorted(i for ids in split.values() for i in ids) == list(range(200))
    for group in groups:
        assert sum(bool(set(group) & set(ids)) for ids in split.values()) == 1
    return split


def split_stats(ids: list[int], tasks: dict[int, dict], answers: dict[int, list[list[str]]]) -> dict:
    classes = Counter()
    paths = Counter()
    turns, calls = [], []
    for i in ids:
        task, gold = tasks[i], answers[i]
        classes.update(task["involved_classes"])
        paths.update(task["path"])
        turns.append(len(gold))
        calls.append(sum(len(turn) for turn in gold))
    return {
        "n": len(ids),
        "turns": {"total": sum(turns), "min": min(turns), "max": max(turns), "mean": sum(turns) / len(turns)},
        "reference_calls": {"total": sum(calls), "min": min(calls), "max": max(calls), "mean": sum(calls) / len(calls)},
        "involved_class_counts": dict(sorted(classes.items())),
        "path_function_counts": dict(sorted(paths.items())),
    }


def main() -> None:
    manifest_path = OUT / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError("E017 manifest already exists; refusing to redraw the frozen split")
    tasks_list = read_jsonl(TASK_PATH)
    answers_list = read_jsonl(ANSWER_PATH)
    assert len(tasks_list) == len(answers_list) == 200
    tasks = {suffix(row["id"]): row for row in tasks_list}
    answers = {suffix(row["id"]): row["ground_truth"] for row in answers_list}
    assert set(tasks) == set(answers) == set(range(200))

    groups = components()
    split = allocate(groups)
    stats = {name: split_stats(ids, tasks, answers) for name, ids in split.items()}
    expected_classes = sorted({c for task in tasks.values() for c in task["involved_classes"]})
    for name in split:
        if sorted(stats[name]["involved_class_counts"]) != expected_classes:
            raise ValueError(f"{name} does not cover every tool class")

    old_exposed = set(read_json(OLD_MANIFEST_PATH)["excluded_families"])
    scripts = ROOT / "scripts"
    required_scripts = [
        "prepare_e017_split.py", "evaluate_e017_bfcl.py", "gate_e017_base.py",
        "prepare_e017_bfcl_gold.py", "train_e017_qwen3_17b.py",
        "bfcl_research_adapter.py", "common.py",
    ]
    for name in required_scripts:
        if not (scripts / name).exists():
            raise FileNotFoundError(f"Freeze requires completed script: {name}")
    manifest = {
        "experiment": "E017_BFCL_indomain_gold_trajectory_SFT",
        "status": "frozen_before_any_E017_model_output",
        "scope": "BFCL benchmark-specific practice; internal held-out split, not an official blind benchmark result",
        "split_policy": {
            "salt": SALT,
            "counts": {"train": 120, "dev": 40, "test": 40},
            "unit": "numeric multi_turn_base task family",
            "historical_exposure_used_for_allocation": False,
            "component_rule": "same-path/near-duplicate edges from E001 inventory stay together; one size-2 component assigned to each split, then hash-ordered singletons",
        },
        "family_ids": split,
        "task_ids": {name: [f"multi_turn_base_{i}" for i in ids] for name, ids in split.items()},
        "component_groups": sorted(groups),
        "stats": stats,
        "historical_exposure_disclosure": {
            name: {"previously_exposed": sum(i in old_exposed for i in ids), "other": sum(i not in old_exposed for i in ids)}
            for name, ids in split.items()
        },
        "source": {
            "category": "multi_turn_base",
            "tasks_sha256": file_hash(TASK_PATH),
            "answers_sha256": file_hash(ANSWER_PATH),
            "family_inventory_sha256": file_hash(INVENTORY_PATH),
        },
        "model": {
            "name": "Qwen/Qwen3-1.7B local snapshot",
            "config_sha256": file_hash(MODEL / "config.json"),
            "tokenizer_sha256": file_hash(MODEL / "tokenizer.json"),
        },
        "evaluation": {
            "split": "dev before recipe freeze; test exactly once after dev go decision",
            "primary": "official BFCL whole-task valid",
            "limits": {"input_tokens": 32768, "new_tokens": 512, "steps_per_turn": 21},
            "decoding": "greedy, enable_thinking=False, Qwen JSON tool-call parser",
            "dev_go_rule": "adapter-only >= 4, adapter-only > base-only, and tuned successes >= base successes",
        },
        "training": {
            "targets": "one target per sequential gold call plus one no-call terminal target per user turn",
            "history": "complete preceding user/assistant/tool history replayed in official BFCL environment",
            "truncation": "none; stop if any sample exceeds model context or longest-sample preflight fails",
            "epochs": 3,
            "learning_rate": 1e-4,
            "effective_batch": 8,
            "warmup_ratio": 0.05,
            "scheduler": "cosine",
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.0,
            "seed": SEED,
        },
        "gates": {
            "base_score_blocks_training": False,
            "blocking_failures": ["source/hash mismatch", "oracle failure", "parser unit failure", "execution/checker infrastructure failure", "incomplete Base run"],
        },
        "plan_sha256": file_hash(PLAN_PATH),
        "script_hashes": {name: file_hash(scripts / name) for name in required_scripts},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, manifest)
    print(json.dumps({
        "status": manifest["status"],
        "manifest_sha256": file_hash(manifest_path),
        "counts": {k: len(v) for k, v in split.items()},
        "stats": stats,
        "historical_exposure_disclosure": manifest["historical_exposure_disclosure"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
