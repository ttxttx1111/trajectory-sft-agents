#!/usr/bin/env python3
"""Collect fresh, distinct ALFWorld G=8 mixed-reward groups under one policy.

Every candidate consumes eight complete environment rollouts. All-pass/all-fail
groups are counted and discarded; only mixed groups retain token records for GRPO.
"""

import argparse
import glob
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import ray
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from e066_probe_online_group import one_group


def sft_gamefiles(patterns):
    found = set()
    for pattern in patterns:
        for path in glob.glob(pattern):
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    if row.get("accepted") and row.get("gamefile"):
                        found.add(row["gamefile"])
    return found


def write_atomic(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path,
                        help="Unmerged LoRA adapter for the rollout policy")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--target-mixed", type=int, default=16)
    parser.add_argument("--max-candidates", type=int, default=48)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--exclude-sft-glob", action="append",
                        default=["/home/codex/agentic-rl/e066_data/*.accepted.jsonl"])
    parser.add_argument("--allowed-task-types", nargs="+",
                        help="Optional pre-rollout task type filter for boundary-task collection")
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(args.output.suffix + ".tmp").exists():
        raise FileExistsError(f"Refusing to overwrite collector output: {args.output}")
    if args.target_mixed < 1 or args.max_candidates < args.target_mixed:
        raise ValueError("Need at least one target mixed group and enough candidate slots")

    seen = sft_gamefiles(args.exclude_sft_glob)
    sft_exclusions = len(seen)
    for path in args.exclude_manifest:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        prior = manifest.get("results", manifest.get("selected", []))
        seen.update(row["gamefile"] for row in prior)

    ray.init(include_dashboard=False, logging_level="ERROR")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="sdpa", local_files_only=True,
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model.eval()
    adapter_sha256 = sha256(args.adapter / "adapter_model.safetensors") if args.adapter else None
    selected = []
    scanned = []
    started = time.perf_counter()
    try:
        for offset in range(args.max_candidates):
            if len(selected) >= args.target_mixed:
                break
            seed = args.seed_start + offset
            random.seed(seed)
            np.random.seed(seed % 2**32)
            torch.manual_seed(seed)
            row = one_group(model, tokenizer, seed, args.max_steps, capture_tokens=True,
                            skip_gamefiles=seen,
                            allowed_task_types=set(args.allowed_task_types or []))
            duplicate = row.get("skipped") == "excluded_gamefile"
            task_type_skip = row.get("skipped") == "task_type"
            skipped = duplicate or task_type_skip
            valid = skipped or not sum(row["response_cap_hits"])
            if not skipped:
                rewards = [10.0 * float(won) - 0.1 * invalid
                           for won, invalid in zip(row["won"], row["invalid_format"])]
                row["reward_mixed_after_penalty"] = len(set(rewards)) > 1
            accepted = not skipped and row["mixed"] and valid
            scanned.append({
                "seed": seed, "gamefile": row["gamefile"],
                "successes": row.get("successes"), "mixed": row.get("mixed"),
                "reward_mixed_after_penalty": row.get("reward_mixed_after_penalty"),
                "duplicate_or_sft": duplicate, "task_type_skip": task_type_skip,
                "invalid_actions": None if skipped else sum(row["invalid_format"]),
                "response_capped": not valid,
                "accepted": accepted,
            })
            if accepted:
                selected.append(row)
            if not skipped:
                seen.add(row["gamefile"])
            print(json.dumps({"candidate": len(scanned), "seed": seed,
                              "successes": row.get("successes"), "accepted": accepted,
                              "mixed_collected": len(selected)}), flush=True)
            if accepted or (len(scanned) % 8 == 0):
                args.output.parent.mkdir(parents=True, exist_ok=True)
                write_atomic(args.output, {
                    "status": "collecting", "groups": len(selected),
                    "rollouts": len(selected) * 8, "results": selected,
                    "candidate_groups": sum(not (x["duplicate_or_sft"] or x["task_type_skip"]) for x in scanned),
                    "candidate_rollouts": 8 * sum(not (x["duplicate_or_sft"] or x["task_type_skip"]) for x in scanned),
                    "candidate_resets": len(scanned),
                    "scanned": scanned, "sft_gamefiles_excluded": sft_exclusions,
                    "seed_start": args.seed_start, "next_seed": seed + 1,
                    "target_mixed": args.target_mixed, "max_candidates": args.max_candidates,
                    "rollout_model": str(args.model),
                    "rollout_adapter": str(args.adapter) if args.adapter else None,
                    "rollout_adapter_sha256": adapter_sha256,
                })
    finally:
        ray.shutdown()
    report = {
        "status": "complete" if len(selected) == args.target_mixed else "insufficient_mixed",
        "groups": len(selected), "rollouts": len(selected) * 8, "results": selected,
        "candidate_groups": sum(not (x["duplicate_or_sft"] or x["task_type_skip"]) for x in scanned),
        "candidate_rollouts": 8 * sum(not (x["duplicate_or_sft"] or x["task_type_skip"]) for x in scanned),
        "candidate_resets": len(scanned),
        "scanned": scanned, "sft_gamefiles_excluded": sft_exclusions,
        "seed_start": args.seed_start,
        "next_seed": args.seed_start + len(scanned),
        "target_mixed": args.target_mixed, "max_candidates": args.max_candidates,
        "rollout_model": str(args.model),
        "rollout_adapter": str(args.adapter) if args.adapter else None,
        "rollout_adapter_sha256": adapter_sha256,
        "seconds": round(time.perf_counter() - started, 1),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(args.output, report)
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("results", "scanned")}), flush=True)


if __name__ == "__main__":
    main()
