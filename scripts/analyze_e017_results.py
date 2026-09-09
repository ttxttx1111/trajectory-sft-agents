"""Analyze E017 BFCL Base/Tuned outcomes and write the preregistered dev decision."""
from __future__ import annotations

import ast
import json
import math
import random
from collections import Counter

from common import BFCL_ROOT, ROOT, file_hash, read_json, read_jsonl, write_json


DATA = ROOT / "data/E017_bfcl_indomain"
OUT = ROOT / "outputs/E017_bfcl_indomain"
TASK_PATH = BFCL_ROOT / "bfcl_eval/data/BFCL_v4_multi_turn_base.json"
ANSWER_PATH = BFCL_ROOT / "bfcl_eval/data/possible_answer/BFCL_v4_multi_turn_base.json"


def call_name(text: str) -> str:
    node = ast.parse(text, mode="eval").body
    return node.func.id


def task_metrics(row: dict, raw: dict, gold: list[list[str]]) -> dict:
    predicted = [call for event in row["trace"] for call in event.get("calls", [])]
    predicted_names = [text.split("(", 1)[0] for text in predicted]
    gold_names = [call_name(call) for turn in gold for call in turn]
    overlap = sum((Counter(predicted_names) & Counter(gold_names)).values())
    return {
        "id": row["id"], "success": bool(row["success"]), "status": row["status"],
        "classes": raw["involved_classes"], "gold_calls": len(gold_names),
        "predicted_calls": len(predicted_names),
        "first_name_match": bool(predicted_names and gold_names and predicted_names[0] == gold_names[0]),
        "name_multiset_precision": overlap / len(predicted_names) if predicted_names else 0.0,
        "name_multiset_recall": overlap / len(gold_names) if gold_names else float(not predicted_names),
        "decode_errors": sum("decode_error" in event for event in row["trace"]),
        "tool_error_events": sum(bool(event.get("tool_results")) and any("error" in str(x).lower() for x in event["tool_results"]) for event in row["trace"]),
        "no_call_events": sum(not event.get("calls") and "decode_error" not in event for event in row["trace"]),
    }


def summarize(metrics: list[dict]) -> dict:
    n = len(metrics)
    return {
        "n": n, "successes": sum(m["success"] for m in metrics),
        "first_name_matches": sum(m["first_name_match"] for m in metrics),
        "mean_name_multiset_precision": sum(m["name_multiset_precision"] for m in metrics) / n,
        "mean_name_multiset_recall": sum(m["name_multiset_recall"] for m in metrics) / n,
        "predicted_calls": sum(m["predicted_calls"] for m in metrics),
        "gold_calls": sum(m["gold_calls"] for m in metrics),
        "decode_errors": sum(m["decode_errors"] for m in metrics),
        "tool_error_events": sum(m["tool_error_events"] for m in metrics),
        "zero_call_tasks": sum(m["predicted_calls"] == 0 for m in metrics),
        "step_limit_tasks": sum(m["status"] == "step_limit" for m in metrics),
        "success_by_class": {
            cls: {"n": sum(cls in m["classes"] for m in metrics), "successes": sum(cls in m["classes"] and m["success"] for m in metrics)}
            for cls in sorted({cls for m in metrics for cls in m["classes"]})
        },
    }


def cluster_bootstrap(base: dict[str, bool], tuned: dict[str, bool], manifest: dict, split: str, draws=20000):
    selected = set(manifest["family_ids"][split])
    groups = [[i for i in group if i in selected] for group in manifest["component_groups"]]
    groups = [group for group in groups if group]
    deltas = []
    for group in groups:
        vals = [int(tuned[f"multi_turn_base_{i}"]) - int(base[f"multi_turn_base_{i}"]) for i in group]
        deltas.append((sum(vals), len(vals)))
    rng = random.Random(20260917)
    samples = []
    for _ in range(draws):
        chosen = [deltas[rng.randrange(len(deltas))] for _ in deltas]
        samples.append(100 * sum(x[0] for x in chosen) / sum(x[1] for x in chosen))
    samples.sort()
    return [samples[round(0.025 * (draws - 1))], samples[round(0.975 * (draws - 1))]]


def mcnemar_exact(a_only: int, b_only: int) -> float:
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


def main() -> None:
    manifest = read_json(DATA / "manifest.json")
    raw = {r["id"]: r for r in read_jsonl(TASK_PATH)}
    gold = {r["id"]: r["ground_truth"] for r in read_jsonl(ANSWER_PATH)}
    report = {"experiment": manifest["experiment"], "manifest_sha256": file_hash(DATA / "manifest.json"), "splits": {}}
    for split in ("dev", "test"):
        arms = {}
        for arm in ("base", "tuned"):
            path = OUT / arm / f"{split}.jsonl"
            if path.exists():
                rows = read_jsonl(path)
                metrics = [task_metrics(row, raw[row["id"]], gold[row["id"]]) for row in rows]
                arms[arm] = {"summary": summarize(metrics), "tasks": metrics}
        if arms:
            report["splits"][split] = arms
        if set(arms) == {"base", "tuned"}:
            b = {x["id"]: x["success"] for x in arms["base"]["tasks"]}
            t = {x["id"]: x["success"] for x in arms["tuned"]["tasks"]}
            assert list(b) == list(t) == manifest["task_ids"][split]
            tuned_only = [i for i in b if t[i] and not b[i]]
            base_only = [i for i in b if b[i] and not t[i]]
            paired = {
                "tuned_only": tuned_only, "base_only": base_only,
                "gain_percentage_points": 100 * (sum(t.values()) - sum(b.values())) / len(b),
                "component_cluster_bootstrap_95_ci_pp": cluster_bootstrap(b, t, manifest, split),
                "mcnemar_exact_two_sided_p": mcnemar_exact(len(tuned_only), len(base_only)),
            }
            report["splits"][split]["paired"] = paired
            if split == "dev":
                rule = manifest["evaluation"]["dev_go_rule"]
                checks = {
                    "adapter_only_at_least_4": len(tuned_only) >= 4,
                    "adapter_only_exceeds_base_only": len(tuned_only) > len(base_only),
                    "tuned_total_not_lower": sum(t.values()) >= sum(b.values()),
                }
                decision = {
                    "experiment": manifest["experiment"], "go": all(checks.values()),
                    "checks": checks, "registered_rule": rule, "paired": paired,
                    "base_successes": sum(b.values()), "tuned_successes": sum(t.values()),
                    "manifest_sha256": file_hash(DATA / "manifest.json"),
                }
                write_json(OUT / "dev_go_decision.json", decision)
    write_json(OUT / "analysis.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
