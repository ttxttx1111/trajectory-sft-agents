"""Replay and encode E017 gold trajectories after the Base measurement gate passes."""
from __future__ import annotations

import ast
import copy
import json
import random
import uuid
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from common import BFCL_ROOT, ROOT, digest, file_hash, make_tools, read_json, read_jsonl, write_json, write_jsonl
from bfcl_eval.eval_checker.multi_turn_eval import multi_turn_utils as backend
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import multi_turn_checker
from bfcl_research_adapter import SYSTEM_PROMPT, clear_run_instances, prepare_task


DATA = ROOT / "data/E017_bfcl_indomain"
OUT = DATA
RUN_OUT = ROOT / "outputs/E017_bfcl_indomain"
MODEL = ROOT / "models/Qwen3-1.7B"
TASK_PATH = BFCL_ROOT / "bfcl_eval/data/BFCL_v4_multi_turn_base.json"
ANSWER_PATH = BFCL_ROOT / "bfcl_eval/data/possible_answer/BFCL_v4_multi_turn_base.json"
TERMINAL_TEXT = "The requested tool operations for this turn are complete."


def parse_gold_call(text: str, functions: list[dict]) -> dict:
    node = ast.parse(text, mode="eval").body
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        raise ValueError(f"Unsupported gold call: {text}")
    by_name = {function["name"]: function for function in functions}
    if node.func.id not in by_name:
        raise ValueError(f"Gold function is not visible: {text}")
    properties = list(by_name[node.func.id]["parameters"].get("properties", {}))
    if len(node.args) > len(properties) or any(isinstance(arg, ast.Starred) for arg in node.args):
        raise ValueError(f"Unsupported positional arguments: {text}")
    args = {name: ast.literal_eval(value) for name, value in zip(properties, node.args)}
    for keyword in node.keywords:
        if keyword.arg is None or keyword.arg in args:
            raise ValueError(f"Unsupported or duplicate keyword: {text}")
        args[keyword.arg] = ast.literal_eval(keyword.value)
    return {"type": "function", "function": {"name": node.func.id, "arguments": args}}


def scalar_strings(value):
    if isinstance(value, dict):
        for child in value.values():
            yield from scalar_strings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from scalar_strings(child)
    elif value is not None:
        yield str(value).lower()


def parameter_sources(call: dict, history: list[dict], functions: list[dict]) -> dict:
    user_text = "\n".join(str(m.get("content", "")) for m in history if m["role"] == "user").lower()
    tool_text = "\n".join(str(m.get("content", "")) for m in history if m["role"] == "tool").lower()
    schema = next(f for f in functions if f["name"] == call["function"]["name"])
    properties = schema["parameters"].get("properties", {})
    audit = {}
    for name, value in call["function"]["arguments"].items():
        values = [item for item in scalar_strings(value) if item]
        prop = properties.get(name, {})
        if "default" in prop and value == prop["default"]:
            source = "schema_default"
        elif values and all(item in user_text for item in values):
            source = "user_history"
        elif values and all(item in tool_text for item in values):
            source = "tool_observation"
        else:
            source = "derived_or_unresolved"
        audit[name] = {"source": source, "value_hash": digest(value)}
    return audit


def encode_row(tokenizer, row: dict) -> dict:
    prompt = tokenizer.apply_chat_template(
        row["history"], tools=row["tools"], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    full = tokenizer.apply_chat_template(
        row["history"] + [row["candidate_action"]], tools=row["tools"],
        tokenize=False, add_generation_prompt=False, enable_thinking=False,
    )
    if not full.startswith(prompt):
        raise ValueError(f"Template boundary mismatch for {row['id']}")
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(full[len(prompt):], add_special_tokens=False)["input_ids"]
    if not target_ids:
        raise ValueError(f"Empty target for {row['id']}")
    return {
        "id": row["id"], "task_id": row["task_id"], "turn": row["turn"],
        "action_index": row["action_index"], "target_type": row["target_type"],
        "input_ids": prompt_ids + target_ids, "target_start": len(prompt_ids),
        "target_tokens": len(target_ids), "full_tokens": len(prompt_ids) + len(target_ids),
        "history_digest": digest(row["history"]), "tools_digest": digest(row["tools"]),
        "parameter_sources": row.get("parameter_sources", {}),
        "source_gold_call": row.get("source_gold_call"),
    }


def replay_task(raw_task: dict, gold: list[list[str]], tokenizer, seed: int) -> tuple[list[dict], dict]:
    task = prepare_task(raw_task)
    run_name = "e017_train_" + uuid.uuid4().hex
    task_seed = seed + int(task["id"].rsplit("_", 1)[1])
    random.seed(task_seed)
    functions, tools = task["function"], make_tools(task["function"])
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    rows, decoded, trace = [], [], []
    try:
        backend.execute_multi_turn_func_call(
            [], task["initial_config"], task["involved_classes"], run_name, task["id"],
            long_context=False, is_evaL_run=False,
        )
        for turn_index, (messages, turn_gold) in enumerate(zip(task["question"], gold, strict=True)):
            history.extend(copy.deepcopy(messages))
            decoded_steps = []
            for action_index, gold_text in enumerate(turn_gold):
                call = parse_gold_call(gold_text, functions)
                candidate = {"role": "assistant", "content": "", "tool_calls": [call]}
                rows.append(encode_row(tokenizer, {
                    "id": f"{task['id']}:t{turn_index}:a{action_index}", "task_id": task["id"],
                    "turn": turn_index, "action_index": action_index, "target_type": "tool_call",
                    "history": copy.deepcopy(history), "candidate_action": candidate,
                    "tools": tools, "parameter_sources": parameter_sources(call, history, functions),
                    "source_gold_call": gold_text,
                }))
                history.append(copy.deepcopy(candidate))
                results, _ = backend.execute_multi_turn_func_call(
                    [gold_text], task["initial_config"], task["involved_classes"], run_name,
                    task["id"], long_context=False, is_evaL_run=False,
                )
                history.extend({"role": "tool", "content": result} for result in results)
                decoded_steps.append([gold_text])
                trace.append({"turn": turn_index, "action": action_index, "call": gold_text, "result_digest": digest(results)})
            terminal = {"role": "assistant", "content": TERMINAL_TEXT}
            rows.append(encode_row(tokenizer, {
                "id": f"{task['id']}:t{turn_index}:stop", "task_id": task["id"],
                "turn": turn_index, "action_index": len(turn_gold), "target_type": "terminate",
                "history": copy.deepcopy(history), "candidate_action": terminal, "tools": tools,
            }))
            history.append(terminal)
            decoded_steps.append([])
            decoded.append(decoded_steps)
        random.seed(task_seed)
        score = multi_turn_checker(decoded, gold, copy.deepcopy(raw_task), "multi_turn_base", run_name)
        report = {
            "task_id": task["id"], "valid": bool(score["valid"]), "score": score,
            "rows": len(rows), "calls": sum(len(turn) for turn in gold), "turns": len(gold),
            "trace_digest": digest(trace), "max_full_tokens": max(r["full_tokens"] for r in rows),
        }
        return (rows if report["valid"] else []), report
    finally:
        clear_run_instances(run_name)


def main() -> None:
    gate = read_json(RUN_OUT / "base_gate.json")
    if not gate.get("passed"):
        raise RuntimeError("E017 training data is blocked until the Base measurement gate passes")
    manifest = read_json(DATA / "manifest.json")
    if file_hash(DATA / "manifest.json") != gate["manifest_sha256"]:
        raise ValueError("Frozen E017 manifest changed after Base gate")
    if file_hash(Path(__file__)) != manifest["script_hashes"][Path(__file__).name]:
        raise ValueError("Training-data script differs from frozen manifest")
    if (OUT / "train_R.jsonl").exists():
        raise FileExistsError("E017 training data already exists; refusing to overwrite")

    raw = {row["id"]: row for row in read_jsonl(TASK_PATH)}
    gold = {row["id"]: row["ground_truth"] for row in read_jsonl(ANSWER_PATH)}
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    rows, reports = [], []
    for index, task_id in enumerate(manifest["task_ids"]["train"], 1):
        task_rows, report = replay_task(raw[task_id], gold[task_id], tokenizer, manifest["training"]["seed"])
        rows.extend(task_rows)
        reports.append(report)
        print(json.dumps({"completed": index, "n": 120, "id": task_id, "valid": report["valid"], "rows": report["rows"], "max_tokens": report["max_full_tokens"]}), flush=True)
    if not all(report["valid"] for report in reports):
        raise RuntimeError("At least one training gold trajectory failed official replay; no file written")
    expected_calls = manifest["stats"]["train"]["reference_calls"]["total"]
    expected_turns = manifest["stats"]["train"]["turns"]["total"]
    counts = Counter(row["target_type"] for row in rows)
    assert counts == {"tool_call": expected_calls, "terminate": expected_turns}
    assert len({row["id"] for row in rows}) == len(rows)
    if max(row["full_tokens"] for row in rows) > 32768:
        raise RuntimeError("At least one E017 training sample exceeds the registered model context")

    train_path = OUT / "train_R.jsonl"
    write_jsonl(train_path, rows)
    provenance = Counter(
        detail["source"] for row in rows for detail in row["parameter_sources"].values()
    )
    data_manifest = {
        "experiment": manifest["experiment"], "status": "execution_verified_and_encoded",
        "parent_manifest_sha256": file_hash(DATA / "manifest.json"),
        "n": len(rows), "target_types": dict(counts),
        "tasks": len(reports), "oracle_replay_valid": sum(r["valid"] for r in reports),
        "length": {
            "min": min(r["full_tokens"] for r in rows), "max": max(r["full_tokens"] for r in rows),
            "over_6144": sum(r["full_tokens"] > 6144 for r in rows),
            "over_8192": sum(r["full_tokens"] > 8192 for r in rows),
            "over_12288": sum(r["full_tokens"] > 12288 for r in rows),
            "total_repeated_history_tokens": sum(r["full_tokens"] for r in rows),
        },
        "parameter_source_audit": dict(provenance), "replay_reports": reports,
        "train_sha256": file_hash(train_path), "script_sha256": file_hash(Path(__file__)),
    }
    write_json(OUT / "training_data_manifest.json", data_manifest)
    print(json.dumps(data_manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
