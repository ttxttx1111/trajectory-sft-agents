#!/usr/bin/env python3
"""Collect on-policy G8 groups with bounded, explicit source quotas.

Games are pinned from a policy-independent shuffled train pool. Every candidate
is sampled online; no difficulty prescreen labels are used to select the pool.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import time

import numpy as np
from peft import PeftModel
import ray
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from e066_collect_fresh_mixed import sha256, write_atomic
from e066_probe_online_group import one_group


def parse_counts(values):
    pairs = [value.split("=", 1) for value in values]
    counts = {name: int(count) for name, count in pairs}
    if len(counts) != len(pairs) or any(value < 1 for value in counts.values()):
        raise ValueError("Source counts must be distinct and positive")
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--previous-round", type=Path)
    parser.add_argument("--seed-start", type=int, default=680000)
    parser.add_argument("--targets", nargs="+", default=["double=8", "look=4", "maintenance=4"])
    parser.add_argument("--caps", nargs="+", default=["double=24", "look=24", "maintenance=48"])
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--max-seconds", type=float, default=5400)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    targets, caps = parse_counts(args.targets), parse_counts(args.caps)
    pool = json.loads(args.pool.read_text(encoding="utf-8"))
    if pool.get("status") != "frozen" or set(targets) != set(caps) or set(targets) != set(pool["pools"]):
        raise ValueError("Targets, caps, and frozen pool sources must match")
    if any(caps[name] < targets[name] for name in targets):
        raise ValueError("Candidate cap is below its source target")
    if args.max_seconds <= 0 or not 1 <= args.max_steps <= 50:
        raise ValueError("Invalid collection time or environment-step budget")
    games = [row["gamefile"] for rows in pool["pools"].values() for row in rows]
    if len(set(games)) != len(games) or any(
            not game.startswith(pool["train_root"] + "/") for game in games):
        raise ValueError("Duplicate or non-train pool game")
    adapter_hash = sha256(args.adapter / "adapter_model.safetensors") if args.adapter else None
    config = {
        "pool": str(args.pool), "pool_sha256": sha256(args.pool),
        "source_targets": targets, "source_caps": caps,
        "max_steps": args.max_steps, "max_seconds": args.max_seconds,
        "group_size": 8, "history_length": 2, "max_new_tokens": 512,
        "temperature": 1.0, "top_p": 1.0, "top_k": 0,
        "rollout_model": str(args.model),
        "rollout_model_sha256": sha256(args.model / "model.safetensors"),
        "rollout_adapter": str(args.adapter) if args.adapter else None,
        "rollout_adapter_sha256": adapter_hash,
        "previous_round": str(args.previous_round) if args.previous_round else None,
        "previous_round_sha256": sha256(args.previous_round) if args.previous_round else None,
        "collector_sha256": sha256(Path(__file__)),
        "probe_sha256": sha256(Path(__file__).with_name("e066_probe_online_group.py")),
    }
    offsets = dict.fromkeys(targets, 0)
    seen = set()
    seed_start = args.seed_start
    if args.previous_round:
        prior = json.loads(args.previous_round.read_text(encoding="utf-8"))
        if prior["status"] != "complete" or prior["pool_sha256"] != config["pool_sha256"]:
            raise ValueError("Previous round is incomplete or used a different pool")
        offsets = dict(prior["pool_offsets"])
        # pool offsets exclude all earlier candidates, even discarded zero-variance ones.
        for name, offset in offsets.items():
            seen.update(row["gamefile"] for row in pool["pools"][name][:offset])
        seed_start = int(prior["next_seed"])
    selected, scanned = [], []
    elapsed_before = 0.0
    if args.output.exists():
        if not args.resume:
            raise FileExistsError(args.output)
        saved = json.loads(args.output.read_text(encoding="utf-8"))
        if saved["status"] != "collecting" or any(saved.get(key) != value for key, value in config.items()):
            raise ValueError("Only an unchanged, interrupted collection can resume")
        if saved["seed_start"] != seed_start:
            raise ValueError("Resume seed mismatch")
        selected, scanned = saved["results"], saved["scanned"]
        offsets = dict(saved["pool_offsets"])
        elapsed_before = float(saved["seconds"])
        seen.update(row["gamefile"] for row in scanned)
        if len({row["gamefile"] for row in scanned}) != len(scanned):
            raise ValueError("Duplicate scanned game in checkpoint")
    elif args.resume:
        raise FileNotFoundError(args.output)
    attempted = Counter(row["source"] for row in scanned)
    accepted = Counter(row["source"] for row in selected)
    for name in targets:
        if attempted[name] > caps[name] or accepted[name] > targets[name]:
            raise ValueError("Resume exceeds source budget")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    def save(status, reason=None):
        report = {
            **config, "status": status, "stop_reason": reason,
            "groups": len(selected), "rollouts": len(selected) * 8,
            "results": selected, "scanned": scanned,
            "candidate_groups": len(scanned), "candidate_resets": len(scanned),
            "candidate_rollouts": len(scanned) * 8,
            "source_attempted": {name: attempted[name] for name in targets},
            "source_accepted": {name: accepted[name] for name in targets},
            "pool_offsets": dict(offsets), "seed_start": seed_start,
            "next_seed": seed_start + len(scanned),
            "target_mixed": sum(targets.values()), "max_candidates": sum(caps.values()),
            "seconds": round(elapsed_before + time.perf_counter() - started, 1),
        }
        write_atomic(args.output, report)
        return report

    save("collecting")
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
    stop_reason = None
    try:
        while any(accepted[name] < targets[name] for name in targets):
            if elapsed_before + time.perf_counter() - started >= args.max_seconds:
                stop_reason = "collection_wall_time_cap"
                break
            progressed = False
            for name in targets:
                if accepted[name] >= targets[name] or attempted[name] >= caps[name]:
                    continue
                if offsets[name] >= len(pool["pools"][name]):
                    continue
                if elapsed_before + time.perf_counter() - started >= args.max_seconds:
                    stop_reason = "collection_wall_time_cap"
                    break
                expected = pool["pools"][name][offsets[name]]
                if expected["gamefile"] in seen:
                    raise ValueError("Repeated game across coverage rounds")
                seed = seed_start + len(scanned)
                random.seed(seed)
                np.random.seed(seed % 2**32)
                torch.manual_seed(seed)
                candidate_started = time.perf_counter()
                row = one_group(model, tokenizer, seed, args.max_steps, capture_tokens=True,
                                fixed_gamefile=expected["gamefile"], eval_split="train")
                if row["gamefile"] != expected["gamefile"]:
                    raise ValueError("Pinned train game mismatch")
                row.update(source=name, task_type=expected["task_type"])
                capped = bool(sum(row["response_cap_hits"]))
                keep = row["mixed"] and not capped
                rewards = [10.0 * float(won) - 0.1 * invalid
                           for won, invalid in zip(row["won"], row["invalid_format"])]
                scanned.append({
                    "seed": seed, "gamefile": row["gamefile"],
                    "source": name, "task_type": row["task_type"],
                    "successes": row["successes"], "mixed": row["mixed"],
                    "reward_mixed_after_penalty": len(set(rewards)) > 1,
                    "response_capped": capped, "accepted": keep,
                    "duplicate_or_sft": False, "task_type_skip": False,
                    "invalid_actions": sum(row["invalid_format"]),
                    "steps": row["steps"],
                    "seconds": round(time.perf_counter() - candidate_started, 2),
                })
                if keep:
                    selected.append(row)
                    accepted[name] += 1
                attempted[name] += 1
                offsets[name] += 1
                seen.add(row["gamefile"])
                save("collecting")
                print(json.dumps({"candidate": len(scanned), "source": name,
                                  "successes": row["successes"], "accepted": keep,
                                  "source_accepted": dict(accepted),
                                  "source_attempted": dict(attempted)}), flush=True)
                progressed = True
            if stop_reason:
                break
            if not progressed:
                stop_reason = "source_candidate_cap_or_pool_exhausted"
                break
    finally:
        ray.shutdown()
    complete = all(accepted[name] == targets[name] for name in targets)
    report = save("complete" if complete else "insufficient_mixed", stop_reason)
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("results", "scanned")}), flush=True)


if __name__ == "__main__":
    main()
