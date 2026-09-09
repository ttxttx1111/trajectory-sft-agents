"""Evaluate one frozen E017 arm in the BFCL multi_turn_base environment."""
from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import random
import time
import uuid
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import BFCL_ROOT, ROOT, calls_to_python, decode_calls, file_hash, read_json, read_jsonl, write_json
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import multi_turn_checker
from bfcl_research_adapter import (
    Limits, TransformersGenerator, clear_run_instances, evaluate_task, prepare_task,
)


DATA = ROOT / "data/E017_bfcl_indomain"
OUT = ROOT / "outputs/E017_bfcl_indomain"
MODEL = ROOT / "models/Qwen3-1.7B"
ADAPTER = OUT / "adapter"
TASK_PATH = BFCL_ROOT / "bfcl_eval/data/BFCL_v4_multi_turn_base.json"
ANSWER_PATH = BFCL_ROOT / "bfcl_eval/data/possible_answer/BFCL_v4_multi_turn_base.json"


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def parser_unit_gate() -> dict:
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]
    good = decode_calls('<tool_call>{"name":"lookup","arguments":{"id":"A1"}}</tool_call>', tools)
    assert good[0]["function"]["name"] == "lookup"
    assert decode_calls("The request is complete.", tools) == []
    rejected = False
    try:
        decode_calls("<tool_call>{bad}</tool_call>", tools)
    except (ValueError, TypeError, json.JSONDecodeError):
        rejected = True
    assert rejected
    return {"valid_call_parsed": True, "plain_text_is_no_call": True, "malformed_rejected": True}


def load_manifest() -> dict:
    manifest = read_json(DATA / "manifest.json")
    assert manifest["status"] == "frozen_before_any_E017_model_output"
    assert file_hash(MODEL / "config.json") == manifest["model"]["config_sha256"]
    assert file_hash(MODEL / "tokenizer.json") == manifest["model"]["tokenizer_sha256"]
    assert file_hash(TASK_PATH) == manifest["source"]["tasks_sha256"]
    assert file_hash(ANSWER_PATH) == manifest["source"]["answers_sha256"]
    assert file_hash(Path(__file__)) == manifest["script_hashes"][Path(__file__).name]
    return manifest


def source_rows():
    raw = {row["id"]: row for row in read_jsonl(TASK_PATH)}
    gold = {row["id"]: row["ground_truth"] for row in read_jsonl(ANSWER_PATH)}
    return raw, gold


def oracle_gate(manifest: dict) -> dict:
    raw, gold = source_rows()
    rows = []
    for split in ("train", "dev", "test"):
        for task_id in manifest["task_ids"][split]:
            run_name = "e017_oracle_" + uuid.uuid4().hex
            random.seed(manifest["training"]["seed"] + int(task_id.rsplit("_", 1)[1]))
            try:
                task = prepare_task(raw[task_id])
                positive = multi_turn_checker([[turn] for turn in gold[task_id]], gold[task_id], copy.deepcopy(task), "multi_turn_base", run_name)
                negative = multi_turn_checker([[[]] for _ in gold[task_id]], gold[task_id], copy.deepcopy(task), "multi_turn_base", run_name + "_empty")
                rows.append({
                    "id": task_id, "split": split,
                    "oracle_pass": bool(positive["valid"]),
                    "empty_rejected": not bool(negative["valid"]),
                })
            except Exception as exc:
                rows.append({"id": task_id, "split": split, "oracle_pass": False, "empty_rejected": False, "error": f"{type(exc).__name__}: {exc}"})
            finally:
                clear_run_instances(run_name)
                clear_run_instances(run_name + "_empty")
    report = {
        "n": len(rows), "oracle_passes": sum(r["oracle_pass"] for r in rows),
        "empty_rejections": sum(r["empty_rejected"] for r in rows),
        "parser_unit_gate": parser_unit_gate(), "rows": rows,
    }
    report["passed"] = report["oracle_passes"] == report["n"] and all(report["parser_unit_gate"].values())
    write_json(OUT / "oracle_gate.json", report)
    if not report["passed"]:
        raise RuntimeError("E017 oracle/parser gate failed")
    return report


def load_model(arm: str):
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, device_map={"": 0}, dtype=torch.bfloat16,
        attn_implementation="sdpa", low_cpu_mem_usage=True,
    )
    if arm == "tuned":
        report = read_json(OUT / "training_report.json")
        assert report["status"] == "complete"
        assert file_hash(ADAPTER / "adapter_model.safetensors") == report["adapter_sha256"]
        model = PeftModel.from_pretrained(model, ADAPTER, is_trainable=False)
    model.eval()
    return model, tokenizer


def summarize(rows: list[dict]) -> dict:
    statuses = {name: sum(row["status"] == name for row in rows) for name in ("completed", "step_limit", "context_limit")}
    return {
        "n": len(rows), "successes": sum(bool(row["success"]) for row in rows),
        "accuracy": sum(bool(row["success"]) for row in rows) / len(rows),
        "statuses": statuses,
        "tool_call_events": sum(bool(event.get("calls")) for row in rows for event in row["trace"]),
        "decode_error_events": sum("decode_error" in event for row in rows for event in row["trace"]),
        "tool_error_events": sum(bool(event.get("tool_results")) and any("error" in str(x).lower() for x in event["tool_results"]) for row in rows for event in row["trace"]),
        "output_limit_events": sum(bool(event.get("usage", {}).get("hit_output_limit")) for row in rows for event in row["trace"]),
        "seconds": sum(float(row["seconds"]) for row in rows),
    }


def evaluate(arm: str, split: str, manifest: dict) -> dict:
    if split == "test":
        decision = OUT / "dev_go_decision.json"
        if not decision.exists() or not read_json(decision).get("go"):
            raise ValueError("Final test is blocked until the E017 dev go decision passes")
    if arm == "tuned":
        gate = OUT / "base_gate.json"
        if not gate.exists() or not read_json(gate).get("passed"):
            raise ValueError("Tuned evaluation is blocked until the complete Base gate passes")
    ids = manifest["task_ids"][split]
    raw, gold = source_rows()
    path = OUT / arm / f"{split}.jsonl"
    done = read_jsonl(path) if path.exists() else []
    assert [row["id"] for row in done] == ids[:len(done)]
    torch.cuda.set_per_process_memory_fraction(0.80, 0)
    model, tokenizer = load_model(arm)
    limits = Limits(**manifest["evaluation"]["limits"])
    generator = TransformersGenerator(model, tokenizer, limits)
    for task_id in ids[len(done):]:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                row = evaluate_task(raw[task_id], gold[task_id], generator, limits=limits, seed=manifest["training"]["seed"])
        except Exception as exc:
            write_json(OUT / arm / "infrastructure_error.json", {"split": split, "id": task_id, "type": type(exc).__name__, "error": str(exc)})
            raise
        row["arm"], row["split"] = arm, split
        append_jsonl(path, row)
        done.append(row)
        print(json.dumps({
            "arm": arm, "split": split, "completed": len(done), "n": len(ids),
            "successes": sum(r["success"] for r in done), "last_status": row["status"],
            "last_seconds": round(row["seconds"], 2),
        }), flush=True)
    summary = {
        "experiment": manifest["experiment"], "arm": arm, "split": split,
        "result": summarize(done), "parser_unit_gate": parser_unit_gate(),
        "manifest_sha256": file_hash(DATA / "manifest.json"),
        "script_sha256": file_hash(Path(__file__)),
    }
    write_json(OUT / arm / f"{split}_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("base", "tuned"))
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--oracle-only", action="store_true")
    args = parser.parse_args()
    manifest = load_manifest()
    if args.oracle_only:
        print(json.dumps(oracle_gate(manifest), ensure_ascii=False, indent=2))
        return
    if args.arm is None:
        parser.error("Specify --arm or --oracle-only")
    evaluate(args.arm, args.split, manifest)


if __name__ == "__main__":
    main()
