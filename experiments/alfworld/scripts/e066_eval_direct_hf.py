#!/usr/bin/env python3
"""Optional fast ALFWorld evaluation; require agreement with frozen official 32 gates.

Uses the same official Ray environment and prompt manager as the GRPO sampler.
This is a separate HF/SDPA inference backend from the FSDP/vLLM evaluation.
Its 128-draw outputs are eligible for a paired analysis only if the first 32
gamefiles, actions, and outcomes match the frozen vLLM gates for both arms.
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


def evaluate_batch(manager, model, tokenizer, max_steps, max_tokens, offset, debug_responses=False):
    obs, infos = manager.reset(kwargs=None)
    gamefiles = [item["extra.gamefile"] for item in infos]
    n = len(gamefiles)
    done = [False] * n
    won = [False] * n
    actions = [[] for _ in range(n)]
    invalid = [0] * n
    caps = [0] * n
    max_response = [0] * n
    for step in range(max_steps):
        active = [i for i in range(n) if not done[i]]
        if not active:
            break
        prompts = [tokenizer.apply_chat_template(
            [{"role": "user", "content": obs["text"][i]}],
            add_generation_prompt=True, tokenize=False,
        ) for i in active]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                           add_special_tokens=False).to("cuda")
        prompt_lengths = inputs.attention_mask.sum(dim=1).tolist()
        if max(prompt_lengths) > 2048:
            raise ValueError(f"Prompt exceeds registered 2048-token cap at step {step + 1}")
        with torch.inference_mode():
            outputs = model.generate(
                **inputs, do_sample=False, max_new_tokens=max_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        responses = ["<think></think><action>look</action>"] * n
        for row, index in enumerate(active):
            generated = outputs[row, inputs.input_ids.shape[1]:]
            eos = (generated == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
            length = int(eos[0]) + 1 if len(eos) else len(generated)
            max_response[index] = max(max_response[index], length)
            caps[index] += int(length >= max_tokens)
            responses[index] = tokenizer.decode(generated[:length], skip_special_tokens=True)
        # alfworld_projection mutates its input list in place. Keep the raw
        # model responses for manager.step(), which performs its own parse.
        parsed_actions, _ = alfworld_projection(responses.copy(), manager.envs.get_admissible_commands)
        if debug_responses and offset == 0 and step < 2:
            print(json.dumps({"step": step + 1, "raw_response_first": responses[0][:800],
                              "parsed_action_first": str(parsed_actions[0])}), flush=True)
        obs, _, dones, infos = manager.step(responses)
        for index in active:
            done[index] = bool(dones[index])
            won[index] = bool(infos[index]["won"])
            invalid[index] += int(not infos[index]["is_action_valid"])
            actions[index].append(str(parsed_actions[index]))
    return [{"index": offset + index, "gamefile": gamefiles[index], "won": won[index],
             "steps": len(actions[index]), "actions": actions[index],
             "invalid_format": invalid[index], "response_cap_hits": caps[index],
             "max_response_tokens": max_response[index]}
            for index in range(n)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--split", choices=["eval_in_distribution", "eval_out_of_distribution"], required=True)
    parser.add_argument("--count", type=int, choices=[32, 128], required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--debug-responses", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    expected = [json.loads(line)["info"]["extra.gamefile"] for line in args.manifest.read_text().splitlines()]
    if len(expected) < args.count:
        raise ValueError("Frozen reset manifest is too short")
    random.seed(1000)
    np.random.seed(1000)
    torch.manual_seed(1000)
    ray.init(include_dashboard=False, logging_level="ERROR")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="sdpa", local_files_only=True,
    ).eval()
    envs = build_alfworld_envs(str(CONFIG), seed=1000, env_num=8, group_n=1,
                               resources_per_worker={"num_cpus": 0.1}, is_train=False,
                               env_kwargs={"eval_dataset": args.split})
    manager = AlfWorldEnvironmentManager(
        envs, alfworld_projection, OmegaConf.create({"env": {"history_length": 2}})
    )
    started = time.perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("w") as handle:
            for offset in range(0, args.count, 8):
                rows = evaluate_batch(manager, model, tokenizer, 50, 512, offset, args.debug_responses)
                for row in rows:
                    if row["gamefile"] != expected[row["index"]]:
                        raise ValueError(f"Reset mismatch at draw {row['index']}")
                    line = "E066_EVAL_CASE=" + json.dumps(row, ensure_ascii=False)
                    handle.write(line + "\n")
                    print(line, flush=True)
                handle.flush()
                print(json.dumps({"completed": offset + 8, "total": args.count,
                                  "seconds": round(time.perf_counter() - started, 1)}), flush=True)
    finally:
        envs.close()
        ray.shutdown()


if __name__ == "__main__":
    main()
