#!/usr/bin/env python3
"""Audit G=8 terminal-reward variance with the official ALFWorld manager.

The SFT policy is sampled online after each environment observation. No model
update happens here; this is a prerequisite for deciding whether GRPO has a
nonzero terminal-reward signal on a given training game.
"""

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
from omegaconf import OmegaConf
import ray
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from agent_system.environments.env_manager import AlfWorldEnvironmentManager
from agent_system.environments.env_package.alfworld import build_alfworld_envs, alfworld_projection


CONFIG = Path("agent_system/environments/env_package/alfworld/configs/config_tw.yaml")


def one_group(model, tokenizer, seed, max_steps, capture_tokens=False, capture_anchors=False,
              skip_gamefiles=None, allowed_task_types=None,
              fixed_gamefile=None, eval_split=None):
    if bool(fixed_gamefile) != bool(eval_split):
        raise ValueError("Fixed gamefile and eval split must be supplied together")
    if eval_split not in (None, "train", "seen", "unseen"):
        raise ValueError("Unknown fixed game split")
    env_kwargs = {}
    if fixed_gamefile:
        env_kwargs = {
            "eval_dataset": ("eval_in_distribution" if eval_split == "seen"
                             else "eval_out_of_distribution"),
            "gamefiles": [str(fixed_gamefile)] * 8,
        }
    envs = build_alfworld_envs(
        str(CONFIG), seed=seed, env_num=1, group_n=8,
        resources_per_worker={"num_cpus": 0.1}, is_train=eval_split in (None, "train"),
        env_kwargs=env_kwargs,
    )
    manager = AlfWorldEnvironmentManager(
        envs, alfworld_projection, OmegaConf.create({"env": {"history_length": 2}})
    )
    try:
        obs, infos = manager.reset(kwargs=None)
        gamefiles = [item["extra.gamefile"] for item in infos]
        if len(set(gamefiles)) != 1:
            raise ValueError("G=8 workers reset to different gamefiles")
        if fixed_gamefile and gamefiles[0] != str(fixed_gamefile):
            raise ValueError("Pinned gamefile mismatch")
        if skip_gamefiles and gamefiles[0] in skip_gamefiles:
            return {"seed": seed, "gamefile": gamefiles[0], "skipped": "excluded_gamefile"}
        task_type = Path(gamefiles[0]).parent.parent.name.split("-")[0]
        if allowed_task_types and task_type not in allowed_task_types:
            return {"seed": seed, "gamefile": gamefiles[0], "skipped": "task_type"}
        done = [False] * 8
        won = [False] * 8
        invalid = [0] * 8
        actions = [[] for _ in range(8)]
        records = [[] for _ in range(8)]
        caps = [0] * 8
        for step in range(max_steps):
            active = [i for i in range(8) if not done[i]]
            if not active:
                break
            inputs = tokenizer(
                [tokenizer.apply_chat_template(
                    [{"role": "user", "content": obs["text"][i]}],
                    add_generation_prompt=True, tokenize=False,
                ) for i in active],
                return_tensors="pt", padding=True, add_special_tokens=False,
            ).to("cuda")
            if inputs.input_ids.shape[1] > 2048:
                raise ValueError(f"Overlong online prompt at step {step}")
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs, do_sample=True, temperature=1.0, top_p=1.0,
                    top_k=0, repetition_penalty=1.0, max_new_tokens=512,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    return_dict_in_generate=capture_tokens,
                    output_scores=capture_tokens,
                )
            if capture_tokens:
                generated = outputs.sequences[:, inputs.input_ids.shape[1]:]
                # MiMo's actor uses rollout log-probabilities. Save the actual
                # sampling distribution, then compare it with updater recomputation.
                sampled_logps = torch.stack([
                    torch.log_softmax(scores.float(), dim=-1)
                    .gather(1, generated[:, token_index:token_index + 1]).squeeze(1)
                    for token_index, scores in enumerate(outputs.scores)
                ], dim=1).cpu()
                outputs = outputs.sequences
            responses = ["<think></think><action>look</action>"] * 8
            for row, index in enumerate(active):
                new = outputs[row, inputs.input_ids.shape[1]:]
                eos = (new == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
                effective_length = int(eos[0]) + 1 if len(eos) else len(new)
                caps[index] += int(effective_length == 512)
                responses[index] = tokenizer.decode(new, skip_special_tokens=True)
                if capture_tokens:
                    records[index].append({
                        "prompt_ids": inputs.input_ids[row][inputs.attention_mask[row].bool()].tolist(),
                        "completion_ids": new[:effective_length].tolist(),
                        "rollout_logps": sampled_logps[row, :effective_length].tolist(),
                    })
            anchors = [obs["anchor"][i] for i in active] if capture_anchors else None
            obs, rewards, dones, infos = manager.step(responses)
            for index in active:
                done[index] = bool(dones[index])
                won[index] = bool(infos[index]["won"])
                invalid[index] += int(not infos[index]["is_action_valid"])
                if capture_tokens:
                    records[index][-1]["is_action_valid"] = bool(infos[index]["is_action_valid"])
                actions[index].append(responses[index])
                if capture_anchors:
                    if not capture_tokens:
                        raise ValueError("Anchor capture requires token capture")
                    row = active.index(index)
                    records[index][-1]["anchor_observation"] = str(anchors[row])
                    records[index][-1]["step_reward"] = float(np.asarray(rewards[index]).item())
            print(json.dumps({"seed": seed, "step": step + 1,
                              "active": len(active), "won_so_far": sum(won)}), flush=True)
        result = {
            "seed": seed, "gamefile": gamefiles[0],
            "successes": sum(won), "mixed": 0 < sum(won) < 8,
            "won": won, "steps": [len(a) for a in actions],
            "invalid_format": invalid, "response_cap_hits": caps,
            "actions": actions,
        }
        if capture_tokens:
            result["records"] = records
        return result
    finally:
        envs.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path,
                        help="Optional unmerged PEFT adapter on the base model")
    parser.add_argument("--seed-start", type=int, default=20260922)
    parser.add_argument("--groups", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", help="Explicit training-game seeds")
    parser.add_argument("--seed-manifest", type=Path, help="Selection manifest with selected seed records")
    parser.add_argument("--capture-tokens", action="store_true")
    parser.add_argument("--capture-anchors", action="store_true")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ray.init(include_dashboard=False, logging_level="ERROR")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="sdpa", local_files_only=True,
    ).eval()
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False).eval()
    started = time.perf_counter()
    results = []
    try:
        if args.seeds and args.seed_manifest:
            raise ValueError("Provide explicit seeds or a manifest, not both")
        if args.seed_manifest:
            manifest = json.loads(args.seed_manifest.read_text())
            seeds = [int(item["seed"]) for item in manifest["selected"]]
            if len(set(seeds)) != len(seeds) or not seeds:
                raise ValueError("Invalid selected seeds")
        else:
            seeds = args.seeds if args.seeds else range(args.seed_start, args.seed_start + args.groups)
        for seed in seeds:
            random.seed(seed)
            np.random.seed(seed % 2**32)
            torch.manual_seed(seed)
            result = one_group(model, tokenizer, seed, args.max_steps, args.capture_tokens, args.capture_anchors)
            results.append(result)
            print(json.dumps({key: value for key, value in result.items() if key not in ("actions", "records")}), flush=True)
    finally:
        ray.shutdown()
    report = {"model": str(args.model),
              "adapter": str(args.adapter) if args.adapter else None,
              "groups": len(results), "rollouts": len(results) * 8,
              "terminal_successes": sum(item["successes"] for item in results),
              "mixed_groups": sum(item["mixed"] for item in results),
              "seconds": round(time.perf_counter() - started, 1), "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}), flush=True)


if __name__ == "__main__":
    main()
