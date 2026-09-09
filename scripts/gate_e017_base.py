"""Validate E017 Base measurement infrastructure before any training data is written."""
from __future__ import annotations

import json

from common import ROOT, calls_to_python, decode_calls, file_hash, read_json, read_jsonl, write_json


DATA = ROOT / "data/E017_bfcl_indomain"
OUT = ROOT / "outputs/E017_bfcl_indomain"


def main() -> None:
    manifest = read_json(DATA / "manifest.json")
    oracle = read_json(OUT / "oracle_gate.json")
    summary = read_json(OUT / "base/dev_summary.json")
    rows = read_jsonl(OUT / "base/dev.jsonl")
    ids = manifest["task_ids"]["dev"]
    checks = {
        "manifest_frozen": manifest["status"] == "frozen_before_any_E017_model_output",
        "oracle_200_of_200": oracle["n"] == oracle["oracle_passes"] == 200,
        "parser_unit_gate": all(oracle["parser_unit_gate"].values()) and all(summary["parser_unit_gate"].values()),
        "base_dev_complete_in_frozen_order": len(rows) == 40 and [r["id"] for r in rows] == ids,
        "no_context_limit": summary["result"]["statuses"]["context_limit"] == 0,
        "no_infrastructure_error_artifact": not (OUT / "base/infrastructure_error.json").exists(),
    }

    candidates = []
    categories_seen = set()
    for row in rows:
        for event in row["trace"]:
            category = "decode_error" if "decode_error" in event else ("tool_call" if event.get("calls") else "no_call")
            if category in categories_seen:
                continue
            categories_seen.add(category)
            try:
                task_tools = None
                # The saved event stores the canonical Python calls. Reparse with
                # the visible schema reconstructed through the recorded prompt is
                # not possible here, so use the full task schema from prepare_task.
                from bfcl_research_adapter import prepare_task, turn_inputs
                from common import BFCL_ROOT, make_tools
                raw = {r["id"]: r for r in read_jsonl(BFCL_ROOT / "bfcl_eval/data/BFCL_v4_multi_turn_base.json")}[row["id"]]
                task = prepare_task(raw)
                _, task_tools = turn_inputs(task, event["turn"])
                parsed = calls_to_python(decode_calls(event["response"], task_tools))
                matches = parsed == event.get("calls", [])
            except Exception as exc:
                parsed, matches = f"{type(exc).__name__}: {exc}", "decode_error" in event
            candidates.append({
                "category": category, "task_id": row["id"], "turn": event["turn"], "step": event["step"],
                "raw_response": event["response"], "saved_calls": event.get("calls", []),
                "reparsed": parsed, "parser_matches_saved": bool(matches),
            })
            if len(categories_seen) == 3:
                break
        if len(categories_seen) == 3:
            break
    checks["raw_outputs_reparse"] = bool(candidates) and all(x["parser_matches_saved"] for x in candidates)
    report = {
        "experiment": manifest["experiment"], "passed": all(checks.values()), "checks": checks,
        "base_successes": summary["result"]["successes"],
        "base_score_is_not_a_training_gate": True,
        "base_behavior": summary["result"], "raw_output_audit": candidates,
        "manifest_sha256": file_hash(DATA / "manifest.json"),
    }
    write_json(OUT / "base_gate.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise RuntimeError("E017 Base measurement gate failed")


if __name__ == "__main__":
    main()
