"""Train the frozen E017 Qwen3-1.7B BFCL trajectory-SFT candidate."""
from __future__ import annotations

import argparse
import gc
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import ROOT, file_hash, read_json, read_jsonl, write_json


DATA = ROOT / "data/E017_bfcl_indomain"
OUT = ROOT / "outputs/E017_bfcl_indomain"
MODEL = ROOT / "models/Qwen3-1.7B"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def example_loss(model, row):
    ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=model.device)
    start = row["target_start"]
    positions = torch.arange(start - 1, ids.shape[1] - 1, device=model.device)
    logits = model(
        input_ids=ids, attention_mask=torch.ones_like(ids), logits_to_keep=positions,
        use_cache=False,
    ).logits
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), ids[:, start:].reshape(-1))


def verify_inputs():
    manifest = read_json(DATA / "manifest.json")
    gate = read_json(OUT / "base_gate.json")
    data_manifest = read_json(DATA / "training_data_manifest.json")
    train_path = DATA / "train_R.jsonl"
    if not gate.get("passed") or gate["manifest_sha256"] != file_hash(DATA / "manifest.json"):
        raise RuntimeError("E017 Base gate missing, failed, or refers to a different manifest")
    if file_hash(Path(__file__)) != manifest["script_hashes"][Path(__file__).name]:
        raise ValueError("Training script differs from frozen manifest")
    if data_manifest["parent_manifest_sha256"] != file_hash(DATA / "manifest.json"):
        raise ValueError("Training data refers to a different frozen manifest")
    if data_manifest["train_sha256"] != file_hash(train_path):
        raise ValueError("Training file differs from its manifest")
    rows = read_jsonl(train_path)
    if len(rows) != data_manifest["n"]:
        raise ValueError("Training row count differs from its manifest")
    return manifest, data_manifest, rows, train_path


def load_trainable(manifest):
    torch.cuda.set_per_process_memory_fraction(0.80, 0)
    torch.backends.cuda.matmul.allow_tf32 = True
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, device_map={"": 0}, dtype=torch.bfloat16,
        attn_implementation="sdpa", low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    cfg = manifest["training"]
    model = get_peft_model(model, LoraConfig(
        r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
        bias="none", task_type="CAUSAL_LM", target_modules=TARGET_MODULES,
    ))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    return model, tokenizer


def preflight(manifest, rows) -> None:
    path = OUT / "training_preflight.json"
    if path.exists():
        raise FileExistsError("E017 training preflight already exists")
    seed = manifest["training"]["seed"]
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats()
    model, _ = load_trainable(manifest)
    probes = sorted(rows, key=lambda row: row["full_tokens"], reverse=True)[:2]
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=manifest["training"]["learning_rate"], fused=True)
    model.train(); optimizer.zero_grad(set_to_none=True)
    losses = []
    started = time.perf_counter()
    try:
        for row in probes:
            gc.collect(); torch.cuda.empty_cache()
            loss = example_loss(model, row)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite preflight loss")
            losses.append(float(loss.detach()))
            (loss / len(probes)).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        report = {
            "passed": bool(torch.isfinite(grad_norm)), "probe_ids": [r["id"] for r in probes],
            "probe_lengths": [r["full_tokens"] for r in probes], "losses": losses,
            "gradient_norm": float(grad_norm), "seconds": time.perf_counter() - started,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "manifest_sha256": file_hash(DATA / "manifest.json"),
            "train_sha256": file_hash(DATA / "train_R.jsonl"),
        }
    except torch.OutOfMemoryError as exc:
        report = {
            "passed": False, "error": f"OutOfMemoryError: {exc}",
            "probe_ids": [r["id"] for r in probes], "probe_lengths": [r["full_tokens"] for r in probes],
            "seconds": time.perf_counter() - started,
        }
    write_json(path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not report["passed"]:
        raise RuntimeError("E017 longest-sample training preflight failed")


def train(manifest, data_manifest, rows, train_path, output_dir: Path) -> None:
    pre = read_json(OUT / "training_preflight.json")
    if not pre.get("passed") or pre.get("train_sha256") != file_hash(train_path):
        raise RuntimeError("Passing E017 preflight for this exact training file is required")
    if output_dir.exists() or (OUT / "train_log.jsonl").exists():
        raise FileExistsError("E017 training output already exists; refusing to mix runs")
    cfg, seed = manifest["training"], manifest["training"]["seed"]
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats()
    model, tokenizer = load_trainable(manifest)
    model.print_trainable_parameters()

    # Confirm selective target-only CE against the ordinary masked-label objective.
    row = rows[0]
    end = min(len(row["input_ids"]), row["target_start"] + 8)
    start = max(0, row["target_start"] - 24)
    probe = {"input_ids": row["input_ids"][start:end], "target_start": row["target_start"] - start}
    ids = torch.tensor([probe["input_ids"]], dtype=torch.long, device=model.device)
    labels = ids.clone(); labels[:, :probe["target_start"]] = -100
    model.eval()
    with torch.no_grad():
        ordinary = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels, use_cache=False).loss
        selective = example_loss(model, probe)
    alignment = {"ordinary": float(ordinary), "selective": float(selective), "abs_difference": abs(float(ordinary) - float(selective))}
    if alignment["abs_difference"] > 0.005:
        raise ValueError(f"Loss alignment failed: {alignment}")
    write_json(OUT / "loss_alignment.json", alignment)

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=cfg["learning_rate"],
        weight_decay=0.01, fused=True,
    )
    order = []
    for epoch in range(cfg["epochs"]):
        epoch_order = list(range(len(rows)))
        random.Random(seed + epoch).shuffle(epoch_order)
        order.extend(epoch_order)
    accum = cfg["effective_batch"]
    total_steps = math.ceil(len(order) / accum)
    warmup = max(1, round(total_steps * cfg["warmup_ratio"]))
    log_path = OUT / "train_log.jsonl"
    losses, seen_ids = [], []
    started = time.perf_counter()
    model.train(); optimizer.zero_grad(set_to_none=True)
    for step in range(total_steps):
        indices = order[step * accum:(step + 1) * accum]
        progress = max(0.0, (step - warmup) / max(1, total_steps - warmup))
        factor = (step + 1) / warmup if step < warmup else 0.5 * (1 + math.cos(math.pi * progress))
        lr = cfg["learning_rate"] * factor
        for group in optimizer.param_groups:
            group["lr"] = lr
        step_losses = []
        for index in indices:
            gc.collect(); torch.cuda.empty_cache()
            loss = example_loss(model, rows[index])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at optimizer step {step + 1}")
            step_losses.append(float(loss.detach())); seen_ids.append(rows[index]["id"])
            (loss / len(indices)).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"Non-finite gradient at optimizer step {step + 1}")
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        losses.extend(step_losses)
        entry = {
            "step": step + 1, "total_steps": total_steps, "examples": len(indices),
            "mean_example_loss": sum(step_losses) / len(step_losses), "learning_rate": lr,
            "gradient_norm": float(grad_norm), "elapsed_seconds": round(time.perf_counter() - started, 3),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        }
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry) + "\n")
        if (step + 1) % 10 == 0 or step == 0 or step + 1 == total_steps:
            print(json.dumps(entry), flush=True)

    output_dir.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(output_dir); tokenizer.save_pretrained(output_dir)
    report = {
        "experiment": manifest["experiment"], "status": "complete",
        "method": "BF16 LoRA; sequential execution-verified gold-call targets plus per-turn stop targets; complete histories; target-only CE; no truncation",
        "optimizer_steps": total_steps, "example_exposures": len(order), "unique_examples": len(rows),
        "epochs": cfg["epochs"], "mean_example_loss": sum(losses) / len(losses),
        "first_epoch_mean_loss": sum(losses[:len(rows)]) / len(rows),
        "last_epoch_mean_loss": sum(losses[-len(rows):]) / len(rows),
        "training_seconds": round(time.perf_counter() - started, 3),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "train_sha256": file_hash(train_path), "manifest_sha256": file_hash(DATA / "manifest.json"),
        "adapter_sha256": file_hash(output_dir / "adapter_model.safetensors"),
        "script_sha256": file_hash(Path(__file__)),
        "seen_ids_sha256": __import__("hashlib").sha256(json.dumps(seen_ids).encode()).hexdigest(),
    }
    write_json(OUT / "training_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUT / "adapter")
    args = parser.parse_args()
    manifest, data_manifest, rows, train_path = verify_inputs()
    if args.preflight:
        preflight(manifest, rows)
    else:
        train(manifest, data_manifest, rows, train_path, args.output_dir)


if __name__ == "__main__":
    main()
