"""Bounded one-batch GRPO bootstrap from the E025 CodeGym SFT adapter."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import pathlib
import random
import time
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "E025_codegym_sft_rl"
E026_DATA = ROOT / "data" / "E026_codegym_grpo"
OUT = ROOT / "outputs" / "E026_codegym_grpo"
BASE = ROOT / "models" / "Qwen3-1.7B"
SFT = ROOT / "outputs" / "E025_codegym_sft_rl" / "adapter"
MANIFEST = E026_DATA / "manifest.json"
TASKS = E026_DATA / "train_tasks.jsonl"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def read_json(path: pathlib.Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: pathlib.Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: pathlib.Path, value: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: pathlib.Path, value: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def file_hash(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_gzip_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]):
    payload = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(payload, compresslevel=6, mtime=0))


def load_gzip_jsonl(path: pathlib.Path):
    return [json.loads(line) for line in gzip.decompress(path.read_bytes()).decode().splitlines() if line]


def verify_frozen():
    manifest = read_json(MANIFEST)
    if manifest["task_file_sha256"] != file_hash(TASKS):
        raise RuntimeError("E026 task file changed")
    if manifest["sft_adapter_sha256"] != file_hash(SFT / "adapter_model.safetensors"):
        raise RuntimeError("E025 SFT adapter changed")
    validation_path = manifest.get("validation_report")
    if validation_path:
        validation = ROOT / validation_path
        if manifest["validation_sha256"] != file_hash(validation):
            raise RuntimeError("RL task validation report changed")
        if not read_json(validation)["summary"]["all_pass"]:
            raise RuntimeError("RL tasks have not all passed replay validation")
    return manifest


def load_policy(*, trainable: bool = False):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    manifest = read_json(MANIFEST)
    seed = manifest["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE,
        local_files_only=True,
        device_map={"": 0},
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, SFT, is_trainable=trainable)
    model.eval()
    return model, tokenizer


def render(tokenizer, messages, tools):
    return tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def sample_step(model, tokenizer, messages, tools, manifest):
    import torch

    text = render(tokenizer, messages, tools)
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    prompt_tokens = int(inputs.input_ids.shape[1])
    if prompt_tokens > 12_288:
        raise OverflowError(f"history reached {prompt_tokens} tokens")
    sampling = manifest["sampling"]
    started = time.monotonic()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=sampling["max_new_tokens_per_step"],
            do_sample=True,
            temperature=sampling["temperature"],
            top_p=sampling["top_p"],
            top_k=0,
            repetition_penalty=1.0,
            use_cache=True,
            logits_to_keep=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    completion_ids = generated[0, prompt_tokens:].tolist()
    raw = tokenizer.decode(completion_ids, skip_special_tokens=False)
    if raw.endswith(tokenizer.eos_token):
        raw = raw[: -len(tokenizer.eos_token)]
    return raw, {
        "prompt_ids": inputs.input_ids[0].tolist(),
        "completion_ids": completion_ids,
        "seconds": time.monotonic() - started,
    }


def rollout(task, sample_index, model, tokenizer, manifest):
    import torch

    from codegym_native_protocol import (
        assistant_tool_message,
        initial_messages,
        load_environment,
        parse_tool_call,
        tools_from_prompt,
    )

    seed = manifest["seed"] + task["source_row"] * 10 + sample_index
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    env = load_environment(DATA, task)
    tools = tools_from_prompt(task["prompt"])
    messages = initial_messages(task["prompt"])
    records = []
    trace = []
    status = "max_steps"
    previous_transition = None
    transition_streak = 0
    started = time.monotonic()
    for step_index in range(manifest["rollout_max_steps"]):
        try:
            raw, record = sample_step(model, tokenizer, messages, tools, manifest)
        except OverflowError as exc:
            status = "context_limit"
            trace.append({"step": step_index, "error": str(exc)})
            break
        records.append({"prompt_ids": record["prompt_ids"], "completion_ids": record["completion_ids"]})
        call, protocol = parse_tool_call(raw)
        event = {
            "step": step_index,
            "raw": raw,
            "protocol": protocol,
            "input_tokens": len(record["prompt_ids"]),
            "output_tokens": len(record["completion_ids"]),
            "seconds": record["seconds"],
        }
        if call is None:
            status = "no_parseable_tool_call"
            trace.append(event)
            break
        event["call"] = call
        _, observation = env.step(json.dumps(call, ensure_ascii=False, separators=(",", ":")))
        event["observation"] = observation
        event["environment_error"] = isinstance(observation, str) and observation.startswith("Error:")
        trace.append(event)
        messages.append(assistant_tool_message(raw, call))
        messages.append({"role": "tool", "content": str(observation)})
        if env.finished:
            status = "completed"
            break
        transition = (
            json.dumps(call, ensure_ascii=False, sort_keys=True),
            str(observation),
        )
        if transition == previous_transition:
            transition_streak += 1
        else:
            previous_transition = transition
            transition_streak = 1
        stagnation_limit = manifest.get("stagnation_transition_repeats")
        if stagnation_limit and transition_streak >= stagnation_limit:
            status = "stagnation"
            break
    success = bool(env.finished and float(env.reward) == 1.0)
    return {
        "task_id": task["task_id"],
        "env_id": task["env_id"],
        "sample_index": sample_index,
        "seed": seed,
        "status": status,
        "success": success,
        "reward": int(success),
        "steps": len(trace),
        "environment_errors": sum(bool(event.get("environment_error")) for event in trace),
        "seconds": time.monotonic() - started,
        "trace": trace,
    }, records


def collect():
    manifest = verify_frozen()
    summary_path = OUT / "sampling_summary.json"
    if summary_path.exists():
        raise FileExistsError(summary_path)
    tasks = read_jsonl(TASKS)
    dynamic_target = manifest.get("target_eligible_groups")
    task_cap = min(len(tasks), manifest.get("max_sampled_tasks", len(tasks)))
    candidate_tasks = tasks[:task_cap]
    schedule = [
        (task, sample_index)
        for task in candidate_tasks
        for sample_index in range(manifest["rollouts_per_task"])
    ]
    result_path = OUT / "rollouts.jsonl"
    done = read_jsonl(result_path)
    if [(row["task_id"], row["sample_index"]) for row in done] != [
        (task["task_id"], sample_index) for task, sample_index in schedule[: len(done)]
    ]:
        raise RuntimeError("Existing E026 rollout schedule differs")
    model, tokenizer = load_policy()
    def completed_groups(rows):
        groups = []
        for task in candidate_tasks:
            task_rows = [row for row in rows if row["task_id"] == task["task_id"]]
            if not task_rows:
                continue
            rewards = [row["reward"] for row in task_rows]
            complete = len(task_rows) == manifest["rollouts_per_task"]
            groups.append(
                {
                    "task_id": task["task_id"],
                    "rewards": rewards,
                    "complete": complete,
                    "eligible": complete and len(set(rewards)) > 1,
                }
            )
        return groups

    for task, sample_index in schedule[len(done) :]:
        if sample_index == 0 and dynamic_target is not None:
            eligible_so_far = sum(group["eligible"] for group in completed_groups(done))
            if eligible_so_far >= dynamic_target:
                break
        result, records = rollout(task, sample_index, model, tokenizer, manifest)
        token_path = OUT / "tokens" / f"{task['task_id']}_{sample_index}.jsonl.gz"
        save_gzip_jsonl(token_path, records)
        result["token_path"] = str(token_path.relative_to(OUT))
        result["token_sha256"] = file_hash(token_path)
        append_jsonl(result_path, result)
        done.append(result)
        print(
            json.dumps(
                {
                    "completed": len(done),
                    "total": len(schedule),
                    "task_id": result["task_id"],
                    "sample": sample_index,
                    "reward": result["reward"],
                    "steps": result["steps"],
                    "seconds": round(result["seconds"], 2),
                }
            ),
            flush=True,
        )
    groups = completed_groups(done)
    eligible_groups = sum(group["eligible"] for group in groups)
    if dynamic_target is None:
        stop_reason = "fixed_schedule_complete"
    elif eligible_groups >= dynamic_target:
        stop_reason = "target_eligible_groups_reached"
    else:
        stop_reason = "max_sampled_tasks_reached"
    summary = {
        "rollouts": len(done),
        "successes": sum(row["success"] for row in done),
        "success_rate": sum(row["success"] for row in done) / len(done),
        "total_seconds": sum(row["seconds"] for row in done),
        "groups": groups,
        "eligible_groups": eligible_groups,
        "can_update": any(group["eligible"] for group in groups),
        "tasks_available": len(tasks),
        "tasks_sampled": len(groups),
        "target_eligible_groups": dynamic_target,
        "max_sampled_tasks": task_cap,
        "stop_reason": stop_reason,
        "rollouts_sha256": file_hash(result_path),
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def advantages(rewards):
    mean = sum(rewards) / len(rewards)
    std = math.sqrt(sum((reward - mean) ** 2 for reward in rewards) / len(rewards))
    return [(reward - mean) / (std + 1e-4) for reward in rewards]


def token_logps(model, record):
    import torch

    prompt = record["prompt_ids"]
    completion = record["completion_ids"]
    ids = torch.tensor([prompt + completion], device=model.device, dtype=torch.long)
    positions = torch.arange(len(prompt) - 1, len(prompt) + len(completion) - 1, device=model.device)
    logits = model(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        logits_to_keep=positions,
        use_cache=False,
    ).logits[0].float()
    targets = ids[0, len(prompt) :]
    return logits.gather(-1, targets[:, None]).squeeze(-1) - torch.logsumexp(logits, dim=-1)


def policy_terms(current, old, advantage, clip_epsilon, kl_beta):
    import torch

    ratio = torch.exp(current - old)
    pg = -torch.minimum(
        ratio * advantage,
        ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * advantage,
    )
    delta = old - current
    kl = torch.exp(delta) - delta - 1
    return pg + kl_beta * kl, ratio, kl


def update():
    import torch

    manifest = verify_frozen()
    summary = read_json(OUT / "sampling_summary.json")
    if not summary["can_update"]:
        raise RuntimeError("No reward-variance group; do not force a GRPO update")
    if (OUT / "adapter").exists() or (OUT / "update_log.jsonl").exists():
        raise FileExistsError("E026 update outputs already exist")
    rollout_rows = read_jsonl(OUT / "rollouts.jsonl")
    eligible = {group["task_id"] for group in summary["groups"] if group["eligible"]}
    records = []
    for task_id in eligible:
        group = sorted(
            (row for row in rollout_rows if row["task_id"] == task_id),
            key=lambda row: row["sample_index"],
        )
        group_advantages = advantages([row["reward"] for row in group])
        for rollout_row, advantage in zip(group, group_advantages, strict=True):
            token_path = OUT / rollout_row["token_path"]
            if file_hash(token_path) != rollout_row["token_sha256"]:
                raise RuntimeError("Rollout token file changed")
            for record in load_gzip_jsonl(token_path):
                record["advantage"] = advantage
                record["task_id"] = task_id
                records.append(record)

    model, tokenizer = load_policy(trainable=True)
    model.config.use_cache = False
    started = time.monotonic()
    with torch.no_grad():
        for index, record in enumerate(records):
            record["old_logps"] = token_logps(model, record).cpu().tolist()
            if (index + 1) % 20 == 0:
                print(json.dumps({"old_logps": index + 1, "segments": len(records)}), flush=True)
    save_gzip_jsonl(OUT / "update_records.jsonl.gz", records)

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=manifest["learning_rate"], weight_decay=0.0, fused=True)
    denominator = sum(len(record["completion_ids"]) for record in records)
    torch.cuda.reset_peak_memory_stats()

    for update_index in range(manifest["updates"]):
        optimizer.zero_grad(set_to_none=True)
        order = list(range(len(records)))
        random.Random(manifest["seed"] + update_index).shuffle(order)
        total_loss = total_kl = total_ratio = clipped = 0.0
        for position, record_index in enumerate(order):
            record = records[record_index]
            current = token_logps(model, record)
            old = torch.tensor(record["old_logps"], device=current.device)
            terms, ratio, kl = policy_terms(
                current,
                old,
                record["advantage"],
                manifest["clip_epsilon"],
                manifest["kl_beta"],
            )
            loss = terms.sum() / denominator
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite GRPO loss")
            loss.backward()
            total_loss += float(loss.detach())
            total_kl += float(kl.detach().sum()) / denominator
            total_ratio += float(ratio.detach().sum()) / denominator
            clipped += float(((ratio.detach() - 1).abs() > manifest["clip_epsilon"]).sum()) / denominator
            if (position + 1) % 20 == 0:
                print(json.dumps({"update": update_index + 1, "segments": position + 1, "total": len(records)}), flush=True)
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError("Non-finite GRPO gradient")
        optimizer.step()
        entry = {
            "update": update_index + 1,
            "loss": total_loss,
            "mean_kl": total_kl,
            "mean_ratio": total_ratio,
            "clipped_fraction": clipped,
            "gradient_norm": float(grad_norm),
            "elapsed_seconds": time.monotonic() - started,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        }
        append_jsonl(OUT / "update_log.jsonl", entry)
        print(json.dumps(entry), flush=True)

    adapter_dir = OUT / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    report = {
        "status": "complete",
        "eligible_groups": len(eligible),
        "rollouts_used": manifest["rollouts_per_task"] * len(eligible),
        "generation_segments": len(records),
        "sampled_tokens_per_update": denominator,
        "optimizer_updates": manifest["updates"],
        "seconds": time.monotonic() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "initial_sft_adapter_sha256": file_hash(SFT / "adapter_model.safetensors"),
        "rl_adapter_sha256": file_hash(adapter_dir / "adapter_model.safetensors"),
    }
    write_json(OUT / "training_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["collect", "update"])
    args = parser.parse_args()
    if args.stage == "collect":
        collect()
    else:
        update()
