#!/usr/bin/env python3
"""Evaluate a model on explicitly frozen, unique ALFWorld game files.

The prompt manager, action parser, reward, and model decoding match the E066
direct-HF evaluator. Each worker is pinned to one eligible game file.
"""

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
from omegaconf import OmegaConf
from peft import PeftModel
import ray
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from agent_system.environments.env_manager import AlfWorldEnvironmentManager
from agent_system.environments.env_package.alfworld import build_alfworld_envs, alfworld_projection

from e066_eval_direct_hf import evaluate_batch


CONFIG = (Path(__file__).resolve().parents[1] / "vendor" / "verl-agent" /
          "agent_system" / "environments" / "env_package" / "alfworld" /
          "configs" / "config_tw.yaml")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path,
                        help="Optional unmerged PEFT adapter on top of --model")
    parser.add_argument("--games", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "seen", "unseen"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0,
                        help="Smoke-test only; omit for the frozen full split")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    selected = [row for row in json.loads(args.games.read_text())["games"]
                if row["split"] == args.split]
    if args.limit:
        selected = selected[:args.limit]
    if not selected or len({row["gamefile"] for row in selected}) != len(selected):
        raise ValueError("Empty or duplicate selected games")
    dataset = "eval_in_distribution" if args.split == "seen" else "eval_out_of_distribution"
    random.seed(1000)
    np.random.seed(1000)
    torch.manual_seed(1000)
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        with args.output.open("w") as handle:
            for offset in range(0, len(selected), 8):
                batch = selected[offset:offset + 8]
                gamefiles = [row["gamefile"] for row in batch]
                envs = build_alfworld_envs(
                    str(CONFIG), seed=1000 + offset, env_num=len(batch), group_n=1,
                    resources_per_worker={"num_cpus": 0.1}, is_train=args.split == "train",
                    env_kwargs={"eval_dataset": dataset, "gamefiles": gamefiles},
                )
                manager = AlfWorldEnvironmentManager(
                    envs, alfworld_projection, OmegaConf.create({"env": {"history_length": 2}})
                )
                try:
                    rows = evaluate_batch(manager, model, tokenizer, 50, 512, offset)
                finally:
                    envs.close()
                for row, expected in zip(rows, batch):
                    if row["gamefile"] != expected["gamefile"]:
                        raise ValueError(f"Pinned game mismatch at case {row['index']}: "
                                         f"actual={row['gamefile']}, expected={expected['gamefile']}")
                    row["task_type"] = expected["task_type"]
                    row["split"] = args.split
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    print("E067_EVAL_CASE=" + json.dumps(row, ensure_ascii=False), flush=True)
                handle.flush()
                print(json.dumps({"completed": min(offset + 8, len(selected)),
                                  "total": len(selected),
                                  "seconds": round(time.perf_counter() - started, 1)}), flush=True)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
