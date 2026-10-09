#!/usr/bin/env python3
"""Run G=8 sampled evaluation on already frozen E067 ALFWorld games."""

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
from peft import PeftModel
import ray
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from e066_probe_online_group import one_group


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--games", type=Path, required=True)
    parser.add_argument("--task-type", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    games = [row for row in json.loads(args.games.read_text())["games"]
             if row["task_type"] == args.task_type]
    if args.limit:
        games = games[:args.limit]
    if not games or len({row["gamefile"] for row in games}) != len(games):
        raise ValueError("Empty or repeated fixed games")
    if any(row["split"] not in ("seen", "unseen") for row in games):
        raise ValueError("This evaluator only supports frozen validation games")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="sdpa", local_files_only=True,
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model.eval()
    ray.init(include_dashboard=False, logging_level="ERROR")
    started = time.perf_counter()
    rows = []
    try:
        for index, game in enumerate(games):
            seed = 670100 + index
            random.seed(seed)
            np.random.seed(seed % 2**32)
            torch.manual_seed(seed)
            row = one_group(model, tokenizer, seed, 50,
                            fixed_gamefile=game["gamefile"], eval_split=game["split"])
            row["split"] = game["split"]
            row["task_type"] = game["task_type"]
            rows.append(row)
            print(json.dumps({"seed": seed, "split": row["split"],
                              "gamefile": row["gamefile"],
                              "successes": row["successes"]}), flush=True)
    finally:
        ray.shutdown()
    report = {"model": str(args.model),
              "adapter": str(args.adapter) if args.adapter else None,
              "games": len(rows), "groups": len(rows), "rollouts": len(rows) * 8,
              "terminal_successes": sum(row["successes"] for row in rows),
              "mixed_groups": sum(row["mixed"] for row in rows),
              "seconds": round(time.perf_counter() - started, 1), "results": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}), flush=True)


if __name__ == "__main__":
    main()
