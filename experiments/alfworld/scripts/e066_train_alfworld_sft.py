#!/usr/bin/env python3
"""One-pass LoRA SFT on verified, per-step ALFWorld expert decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

import pandas as pd
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(path: Path, tokenizer, max_prompt: int, max_response: int) -> list[dict]:
    frame = pd.read_parquet(path)
    rows = []
    for row in frame.itertuples(index=False):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": row.prompt}], add_generation_prompt=True, tokenize=True
        )
        response = tokenizer.encode(row.response + tokenizer.eos_token, add_special_tokens=False)
        if len(prompt) > max_prompt or len(response) > max_response:
            raise ValueError(f"Overlong example: {row.gamefile}, step {row.step}")
        if len(prompt) != row.prompt_tokens or len(response) != row.response_tokens:
            raise ValueError(f"Tokenizer mismatch: {row.gamefile}, step {row.step}")
        rows.append({
            "ids": prompt + response,
            "target_start": len(prompt),
            "target_tokens": len(response),
            "gamefile": row.gamefile,
            "step": int(row.step),
        })
    return rows


def token_loss(model, row: dict) -> tuple[torch.Tensor, int]:
    ids = torch.tensor([row["ids"]], dtype=torch.long, device=model.device)
    first = row["target_start"]
    positions = torch.arange(first - 1, ids.shape[1] - 1, device=model.device)
    logits = model(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        logits_to_keep=positions,
        use_cache=False,
    ).logits[0]
    target = ids[0, first:]
    return F.cross_entropy(logits.float(), target, reduction="sum"), len(target)


def load_model(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map={"": 0}, dtype=torch.bfloat16,
        attn_implementation="sdpa", low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0.0,
        task_type="CAUSAL_LM", bias="none", target_modules=TARGET_MODULES,
    ))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max-prompt", type=int, default=2048)
    parser.add_argument("--max-response", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--effective-batch", type=int, default=8)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rows = load_rows(args.data, tokenizer, args.max_prompt, args.max_response)
    if not rows:
        raise ValueError("Empty SFT dataset")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "data": str(args.data), "data_sha256": sha256(args.data),
        "script_sha256": sha256(Path(__file__)), "model": args.model,
        "examples": len(rows), "episodes": len({r["gamefile"] for r in rows}),
        "target_tokens": sum(r["target_tokens"] for r in rows),
        "max_tokens": max(len(r["ids"]) for r in rows),
        "seed": args.seed, "epochs": args.epochs, "effective_batch": args.effective_batch,
        "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha, "lr": args.lr,
    }
    model = load_model(args)
    model.train()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr, weight_decay=0.01, fused=True,
    )
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    if args.preflight:
        probes = sorted(rows, key=lambda item: len(item["ids"]), reverse=True)[:2]
        optimizer.zero_grad(set_to_none=True)
        target_total = sum(item["target_tokens"] for item in probes)
        losses = []
        for item in probes:
            loss, count = token_loss(model, item)
            losses.append(float(loss.detach()) / count)
            (loss / target_total).backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        report = {
            **manifest, "preflight": bool(torch.isfinite(norm)),
            "probe_lengths": [len(item["ids"]) for item in probes],
            "probe_losses": losses, "grad_norm": float(norm),
            "seconds": time.perf_counter() - started,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        }
        (args.output / "preflight.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        return

    prior = args.output / "preflight.json"
    if not prior.exists() or not json.loads(prior.read_text())["preflight"]:
        raise RuntimeError("Passing memory/loss preflight is required")
    if json.loads(prior.read_text())["data_sha256"] != manifest["data_sha256"]:
        raise RuntimeError("Preflight used different training data")
    if (args.output / "train_log.jsonl").exists():
        raise FileExistsError("Training output exists; refusing to overwrite")

    order = []
    for epoch in range(args.epochs):
        shuffled = list(range(len(rows)))
        random.Random(args.seed + epoch).shuffle(shuffled)
        order.extend(shuffled)
    updates = math.ceil(len(order) / args.effective_batch)
    warmup = max(1, round(updates * 0.05))
    for step in range(updates):
        indices = order[step * args.effective_batch:(step + 1) * args.effective_batch]
        target_total = sum(rows[index]["target_tokens"] for index in indices)
        progress = max(0.0, (step - warmup) / max(1, updates - warmup))
        factor = (step + 1) / warmup if step < warmup else 0.5 * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = args.lr * factor
        optimizer.zero_grad(set_to_none=True)
        loss_total = 0.0
        for index in indices:
            loss, _ = token_loss(model, rows[index])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {step + 1}")
            loss_total += float(loss.detach())
            (loss / target_total).backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(norm):
            raise FloatingPointError(f"Non-finite gradient at step {step + 1}")
        optimizer.step()
        entry = {
            "step": step + 1, "updates": updates, "examples": len(indices),
            "mean_target_loss": loss_total / target_total,
            "lr": args.lr * factor, "grad_norm": float(norm),
            "elapsed_seconds": round(time.perf_counter() - started, 1),
        }
        with (args.output / "train_log.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
        if step == 0 or (step + 1) % 25 == 0 or step + 1 == updates:
            print(json.dumps(entry), flush=True)
        if (step + 1) in {updates // 4, updates // 2, updates}:
            checkpoint = args.output / f"adapter_step_{step + 1}"
            model.save_pretrained(checkpoint)
            tokenizer.save_pretrained(checkpoint)

    model.eval()
    merged = model.merge_and_unload()
    merged.config.use_cache = True
    merged.save_pretrained(args.output / "merged_hf", safe_serialization=True)
    tokenizer.save_pretrained(args.output / "merged_hf")
    report = {
        **manifest, "updates": updates, "seconds": time.perf_counter() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "final_adapter_sha256": sha256(args.output / f"adapter_step_{updates}" / "adapter_model.safetensors"),
    }
    (args.output / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
