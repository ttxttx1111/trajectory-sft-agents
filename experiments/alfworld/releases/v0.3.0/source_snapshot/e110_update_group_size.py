#!/usr/bin/env python3
"""Group-size parameterized derivative of the unchanged e088 resumable updater.

The default preserves E066's E026-style trajectory objective. E067's explicit
vendor_dynamic_grpo mode calls pinned verl-agent advantage and policy-loss
functions. Neither mode trains on environment observations as model output.
"""

import argparse
import copy
import json
from pathlib import Path
import random
import signal
import sys
import time

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from e026_codegym_grpo import advantages, file_hash, policy_terms, token_logps
from optimizer_boundary_checkpoint import BoundaryCheckpoints, rng_state


TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def vendor_algorithms():
    """Use the pinned verl-agent implementation for the E067 objective."""
    vendor_root = Path(__file__).resolve().parents[1] / "vendor" / "verl-agent"
    if str(vendor_root) not in sys.path:
        sys.path.insert(0, str(vendor_root))
    from verl.trainer.ppo.core_algos import (
        compute_grpo_outcome_advantage, compute_policy_loss, kl_penalty,
    )
    return compute_grpo_outcome_advantage, compute_policy_loss, kl_penalty


def validate_group_dimensions(source, group_size):
    if group_size < 2 or source.get("group_size", 8) != group_size:
        raise ValueError("Declared rollout group size differs from CLI or is below two")
    if source.get("groups") != len(source.get("results", [])) or source.get("rollouts") != source["groups"] * group_size:
        raise ValueError("Rollout group accounting differs from declared group size")
    for group in source["results"]:
        for key in ("won", "invalid_format", "response_cap_hits", "steps", "records"):
            if len(group.get(key, [])) != group_size:
                raise ValueError("Missing trajectory entries: " + key)
        if "actions" in group and len(group["actions"]) != group_size:
            raise ValueError("Action trajectory count differs from group size")
        if group.get("successes") != sum(group["won"]) or not group.get("mixed") or not 0 < group["successes"] < group_size:
            raise ValueError("Only complete terminal-mixed groups can update")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group-size", type=int, default=8, help="Uniform complete rollouts per prompt group")
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train-base-model", type=Path,
                        help="Fixed model under the persistent LoRA; defaults to --model for legacy runs")
    parser.add_argument("--resume-adapter", type=Path,
                        help="Previous round's trainable LoRA adapter")
    parser.add_argument("--rollout-adapter", type=Path,
                        help="Adapter actually loaded by the fresh rollout actor")
    parser.add_argument("--optimizer-state", type=Path,
                        help="Previous round's AdamW state; required together with --resume-adapter")
    parser.add_argument("--reference-model", type=Path,
                        help="Frozen reference for KL; when omitted, legacy runs use the rollout policy as reference")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--clip-epsilon-high", type=float, default=0.28)
    parser.add_argument("--objective", choices=("legacy_trajectory", "vendor_dynamic_grpo",
                                               "mimo_outcome_grpo"),
                        default="legacy_trajectory")
    parser.add_argument("--kl-beta", type=float, default=0.01)
    parser.add_argument("--loss-reduction", choices=("global_token", "prompt_mean", "trajectory_mean"),
                        default="global_token",
                        help="global_token preserves earlier runs; prompt_mean averages each prompt's tokens first; trajectory_mean gives each rollout equal mass")
    parser.add_argument("--diagnostic-samples", type=int, default=32,
                        help="Read-only post-update probability check on this many stored decisions")
    parser.add_argument("--minibatch-decisions", type=int, default=0,
                        help="0 keeps the legacy single optimizer step per batch; 256 matches the verl-agent actor minibatch size")
    parser.add_argument("--prescreen-rollouts", type=int, default=128)
    parser.add_argument("--additional-initial-probe-rollouts", type=int, default=16)
    parser.add_argument("--prescreen-seed-start", type=int, default=20260924)
    parser.add_argument("--prescreen-seed-end", type=int, default=20260939)
    parser.add_argument("--selection-manifest", type=Path)
    parser.add_argument("--checkpoint-every", type=int, default=8,
                        help="Save unmerged LoRA/AdamW after this many completed optimizer steps")
    parser.add_argument("--resume-update-checkpoint", type=Path,
                        help="Within-batch checkpoint; original rollout adapter/optimizer CLI remains unchanged")
    parser.add_argument("--stop-after-seconds", type=float,
                        help="Pause at a safe record boundary; hard process timeout must leave checkpoint-write time")
    args = parser.parse_args()
    if args.checkpoint_every < 1 or (args.stop_after_seconds is not None and args.stop_after_seconds <= 0):
        raise ValueError("Invalid checkpoint frequency or soft update budget")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    if args.updates < 1:
        raise ValueError("At least one optimizer update is required")
    if args.diagnostic_samples < 1:
        raise ValueError("At least one post-update diagnostic sample is required")
    if args.minibatch_decisions < 0:
        raise ValueError("Minibatch decisions cannot be negative")
    if args.minibatch_decisions and args.loss_reduction != "global_token":
        raise ValueError("Minibatch mode currently supports only global_token loss reduction")
    if bool(args.resume_adapter) != bool(args.optimizer_state):
        raise ValueError("Resume adapter and optimizer state must be supplied together")
    if args.resume_adapter and not args.train_base_model:
        raise ValueError("Resuming LoRA requires a fixed --train-base-model")
    if args.train_base_model and not args.resume_adapter and args.train_base_model != args.model:
        raise ValueError("The first LoRA round must start from its train base model")
    if args.objective == "vendor_dynamic_grpo":
        if not args.reference_model:
            raise ValueError("Vendor objective requires a frozen --reference-model")
        if args.loss_reduction != "global_token":
            raise ValueError("Vendor objective requires global_token loss reduction")
        vendor_advantage, vendor_policy_loss, vendor_kl = vendor_algorithms()
    if args.objective == "mimo_outcome_grpo":
        if args.reference_model or args.kl_beta != 0:
            raise ValueError("MiMo-style outcome GRPO disables the KL loss")
        if args.loss_reduction != "prompt_mean" or args.minibatch_decisions:
            raise ValueError("MiMo-style replay uses one full-batch prompt-mean update")
        if args.updates != 1 or args.clip_epsilon_high != args.clip_epsilon:
            raise ValueError("MiMo-style replay requires one pass and symmetric clipping")

    source = json.loads(args.rollouts.read_text())
    validate_group_dimensions(source, args.group_size)
    if args.objective in ("vendor_dynamic_grpo", "mimo_outcome_grpo"):
        if bool(args.rollout_adapter) != bool(args.resume_adapter):
            raise ValueError("Rollout adapter and resumed training adapter must match")
        if args.rollout_adapter and args.rollout_adapter != args.resume_adapter:
            raise ValueError("Rollout and resumed adapter paths differ")
        if source.get("rollout_adapter") != (str(args.rollout_adapter) if args.rollout_adapter else None):
            raise ValueError("Collector used a different rollout adapter")
        if args.rollout_adapter and source.get("rollout_adapter_sha256") != file_hash(
                args.rollout_adapter / "adapter_model.safetensors"):
            raise ValueError("Rollout adapter changed after collection")
    if source.get("status") not in (None, "complete"):
        raise ValueError("Collector did not reach its mixed-group target")
    if source["groups"] != len(source["results"]) or source["rollouts"] != source["groups"] * args.group_size:
        raise ValueError("Invalid rollout accounting")
    if not all(row["mixed"] and 0 < row["successes"] < args.group_size for row in source["results"]):
        raise ValueError("Only terminal-mixed groups of the declared size can update")
    if any(not row["gamefile"].startswith("/home/codex/.cache/alfworld/json_2.1.1/train/")
           for row in source["results"]):
        raise ValueError("Non-training ALFWorld task in RL batch")
    if len({row["gamefile"] for row in source["results"]}) != source["groups"]:
        raise ValueError("Repeated ALFWorld gamefile within one policy update")
    if any(sum(row["response_cap_hits"]) for row in source["results"]):
        raise ValueError("Response-capped rollout needs separate handling")
    if args.selection_manifest:
        selection = json.loads(args.selection_manifest.read_text())
        if [(item["seed"], item["gamefile"], item["prescreen_successes"])
            for item in selection["selected"]] != [
                (item["seed"], item["gamefile"], item["successes"])
                for item in source["results"]
            ]:
            raise ValueError("Fresh rollouts differ from frozen mixed-group selection")
    records = []
    for group in source["results"]:
        if len(group.get("records", [])) != args.group_size:
            raise ValueError("Missing exact on-policy prompt and completion tokens")
        group_rewards = [10.0 * float(won) - 0.1 * invalid
                         for won, invalid in zip(group["won"], group["invalid_format"])]
        if args.objective == "vendor_dynamic_grpo":
            step_scores = []
            trajectory_ids = []
            for index, trajectory in enumerate(group["records"]):
                for decision in trajectory:
                    if "is_action_valid" not in decision:
                        raise ValueError("Vendor objective requires per-decision is_action_valid")
                    step_scores.append(10.0 * float(group["won"][index])
                                       - 0.1 * float(not decision["is_action_valid"]))
                    trajectory_ids.append(index)
            if sum(not step["is_action_valid"] for trajectory in group["records"]
                   for step in trajectory) != sum(group["invalid_format"]):
                raise ValueError("Per-decision validity differs from trajectory invalid count")
            step_advantages, _ = vendor_advantage(
                torch.tensor(step_scores, dtype=torch.float32).unsqueeze(1),
                torch.ones(len(step_scores), 1),
                np.array([group["gamefile"]] * len(step_scores), dtype=object),
                np.array(trajectory_ids),
            )
            step_advantages = step_advantages.squeeze(1).tolist()
            step_position = 0
        elif args.objective == "mimo_outcome_grpo":
            group_mean = sum(group_rewards) / len(group_rewards)
            group_adv = [reward - group_mean for reward in group_rewards]
        else:
            group_adv = advantages(group_rewards)
        group_tokens = sum(len(step["completion_ids"])
                           for trajectory in group["records"] for step in trajectory)
        if group_tokens <= 0:
            raise ValueError("Empty prompt group completion")
        for index in range(args.group_size):
            if len(group["records"][index]) != group["steps"][index]:
                raise ValueError("Trajectory record count mismatch")
            trajectory_tokens = sum(len(step["completion_ids"]) for step in group["records"][index])
            if trajectory_tokens <= 0:
                raise ValueError("Empty trajectory completion")
            for decision in group["records"][index]:
                if not decision["prompt_ids"] or not decision["completion_ids"]:
                    raise ValueError("Empty on-policy decision")
                if args.objective == "vendor_dynamic_grpo":
                    advantage = step_advantages[step_position]
                    step_position += 1
                else:
                    advantage = group_adv[index]
                records.append({**decision, "advantage": advantage,
                                "seed": group["seed"], "rollout_index": index,
                                "trajectory_tokens": trajectory_tokens,
                                "group_tokens": group_tokens})

    random.seed(20260922)
    torch.manual_seed(20260922)
    started = time.perf_counter()
    fingerprint_paths = [Path(__file__), Path(__file__).with_name("optimizer_boundary_checkpoint.py"),
                         args.rollouts, args.model / "model.safetensors", args.model / "config.json"]
    if args.train_base_model:
        fingerprint_paths += [args.train_base_model / "model.safetensors", args.train_base_model / "config.json"]
    if args.resume_adapter:
        fingerprint_paths += [args.resume_adapter / "adapter_model.safetensors", args.resume_adapter / "adapter_config.json", args.optimizer_state]
    if args.reference_model:
        fingerprint_paths += [args.reference_model / "model.safetensors", args.reference_model / "config.json"]
    if args.objective == "vendor_dynamic_grpo":
        fingerprint_paths += [Path(__file__).resolve().parents[1] / "vendor/verl-agent/verl/trainer/ppo/core_algos.py"]
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                if k not in ("output", "checkpoint_every", "resume_update_checkpoint", "stop_after_seconds")}
    fingerprint = {"settings": settings, "files_sha256": {str(p): file_hash(p) for p in fingerprint_paths}}
    cached = None
    if args.resume_update_checkpoint:
        manifest = BoundaryCheckpoints.inspect(args.resume_update_checkpoint, fingerprint)
        cached = torch.load(manifest["prepared_cache_path"], map_location="cpu", weights_only=True)
        if len(cached["records"]) != len(records):
            raise ValueError("Cached prepared record count changed")
        records = cached["records"]
    if args.reference_model and cached is None:
        reference = AutoModelForCausalLM.from_pretrained(
            args.reference_model, dtype=torch.bfloat16, device_map={"": 0},
            attn_implementation="sdpa", local_files_only=True,
        ).eval()
        with torch.no_grad():
            for index, record in enumerate(records):
                record["ref_logps"] = token_logps(reference, record).cpu().tolist()
                if (index + 1) % 100 == 0:
                    print(json.dumps({"ref_logps": index + 1, "total": len(records)}), flush=True)
        del reference
        torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.train_base_model or args.model, dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="sdpa", local_files_only=True,
    )
    model.config.use_cache = False
    if args.resume_adapter:
        model = PeftModel.from_pretrained(model, args.resume_adapter, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
            bias="none", target_modules=TARGET_MODULES,
        ))
    model.eval()
    rollout_logp_abs_sum = 0.0
    rollout_logp_max_abs = 0.0
    rollout_logp_count = 0
    if cached is not None:
        rollout_logp_abs_sum = cached["rollout_logp_abs_sum"]
        rollout_logp_max_abs = cached["rollout_logp_max_abs"]
        rollout_logp_count = cached["rollout_logp_count"]
    with torch.no_grad():
        for index, record in enumerate(records if cached is None else []):
            recomputed = token_logps(model, record).cpu().float()
            sampled = record.get("rollout_logps")
            if args.objective in ("vendor_dynamic_grpo", "mimo_outcome_grpo") and sampled is None:
                raise ValueError("E067 requires log-probabilities captured during rollout")
            if sampled is not None:
                sampled = torch.tensor(sampled, dtype=torch.float32)
                if sampled.shape != recomputed.shape or not torch.isfinite(sampled).all():
                    raise ValueError("Invalid captured rollout log-probabilities")
                difference = (sampled - recomputed).abs()
                rollout_logp_abs_sum += float(difference.sum())
                rollout_logp_max_abs = max(rollout_logp_max_abs, float(difference.max()))
                rollout_logp_count += difference.numel()
                record["old_logps"] = sampled.tolist()
            else:
                record["old_logps"] = recomputed.tolist()
            if (index + 1) % 100 == 0:
                print(json.dumps({"old_logps": index + 1, "total": len(records)}), flush=True)
    if rollout_logp_count:
        print(json.dumps({"rollout_logp_tokens": rollout_logp_count,
                          "rollout_vs_recomputed_mean_abs": rollout_logp_abs_sum / rollout_logp_count,
                          "rollout_vs_recomputed_max_abs": rollout_logp_max_abs}), flush=True)
        if rollout_logp_abs_sum / rollout_logp_count > 0.05:
            raise ValueError("Rollout and recomputed policy log-probabilities disagree")

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()
    trainable_names = [name for name, item in model.named_parameters() if item.requires_grad]
    parameters = [item for item in model.parameters() if item.requires_grad]
    if args.optimizer_state:
        names_path = args.optimizer_state.parent / "trainable_parameters.json"
        if not names_path.is_file() or json.loads(names_path.read_text()) != trainable_names:
            raise ValueError("Trainable parameter names/order changed across RL rounds")
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.0, fused=True)
    if args.optimizer_state:
        optimizer.load_state_dict(torch.load(args.optimizer_state, map_location="cpu", weights_only=True))
        if any(group["lr"] != args.lr for group in optimizer.param_groups):
            raise ValueError("Optimizer learning rate changed during resume")
    denominator = sum(len(item["completion_ids"]) for item in records)
    torch.cuda.reset_peak_memory_stats()
    args.output.mkdir(parents=True)
    store = BoundaryCheckpoints(args.output / "checkpoints", fingerprint)
    resume_progress = None
    if args.resume_update_checkpoint:
        resume_progress, restored_cache = store.load(args.resume_update_checkpoint, model, optimizer, fingerprint)
        records = restored_cache["records"]
    cache = {"records": records, "rollout_logp_abs_sum": rollout_logp_abs_sum,
             "rollout_logp_max_abs": rollout_logp_max_abs, "rollout_logp_count": rollout_logp_count}
    store.save_cache(cache)
    resumed_steps = resume_progress["completed_optimizer_steps"] if resume_progress else 0
    progress = copy.deepcopy(resume_progress) if resume_progress else {
        "completed_optimizer_steps": 0, "update_index": 0, "next_minibatch": 0,
        "total_loss": 0., "total_kl": 0., "total_ratio": 0., "clipped": 0., "minibatch_metrics": []}
    boundary_rng = rng_state()
    stop_requested = False
    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
    signal.signal(signal.SIGTERM, request_stop)
    def checkpoint():
        path = store.root / f'step_{progress["completed_optimizer_steps"]:06d}'
        if not path.exists():
            path = store.save(model, optimizer, progress, boundary_rng)
        else:
            store.inspect(path, fingerprint)
        return path
    if args.resume_update_checkpoint:
        # A final-step checkpoint may only need diagnostics/export; preserve it
        # in the new run before any later merge removes the LoRA modules.
        checkpoint()
    def pause_if_requested():
        if stop_requested or (args.stop_after_seconds is not None and time.perf_counter() - started >= args.stop_after_seconds):
            # A partial backward has not touched parameters or AdamW. Its RNG
            # and running statistics are discarded; resume from the last step.
            optimizer.zero_grad(set_to_none=True)
            path = checkpoint()
            report = {"status": "paused", "checkpoint": str(path), "completed_optimizer_steps": progress["completed_optimizer_steps"],
                      "optimizer_steps_this_process": progress["completed_optimizer_steps"] - resumed_steps,
                      "next_update_index": progress["update_index"], "next_minibatch": progress["next_minibatch"],
                      "seconds": round(time.perf_counter() - started, 1), "full_batch_complete": False,
                      "resume_with_original_rollout_actor_cli": True}
            (args.output / "paused_update.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report), flush=True)
            raise SystemExit(0)
    log_path = args.output / "update_log.jsonl"
    for update_index in range(args.updates):
        if resume_progress and update_index < resume_progress["update_index"]:
            continue
        order = list(range(len(records)))
        random.Random(20260922 + update_index).shuffle(order)
        total_loss = total_kl = total_ratio = clipped = 0.0
        batch_size = args.minibatch_decisions or len(records)
        chunks = [order[start:start + batch_size]
                  for start in range(0, len(order), batch_size)]
        minibatch_metrics = []
        if resume_progress and update_index == resume_progress["update_index"]:
            total_loss, total_kl, total_ratio, clipped = (resume_progress[k] for k in ("total_loss", "total_kl", "total_ratio", "clipped"))
            minibatch_metrics = copy.deepcopy(resume_progress["minibatch_metrics"])
            if minibatch_metrics:
                norm = minibatch_metrics[-1]["gradient_norm"]
        for minibatch_index, chunk in enumerate(chunks):
            if resume_progress and update_index == resume_progress["update_index"] and minibatch_index < resume_progress["next_minibatch"]:
                continue
            pause_if_requested()
            optimizer.zero_grad(set_to_none=True)
            chunk_denominator = sum(len(records[index]["completion_ids"]) for index in chunk)
            chunk_loss = chunk_kl = chunk_clipped = 0.0
            for local_position, record_index in enumerate(chunk):
                pause_if_requested()
                record = records[record_index]
                current = token_logps(model, record)
                old = torch.tensor(record["old_logps"], device=current.device)
                if args.objective == "vendor_dynamic_grpo":
                    mask = torch.ones_like(current).unsqueeze(0)
                    pg_loss, _, _, _ = vendor_policy_loss(
                        old.unsqueeze(0), current.unsqueeze(0),
                        torch.full_like(current, record["advantage"]).unsqueeze(0),
                        mask, cliprange=args.clip_epsilon,
                        cliprange_low=args.clip_epsilon,
                        cliprange_high=args.clip_epsilon_high,
                        loss_agg_mode="token-mean",
                    )
                    ratio = torch.exp(current - old)
                    terms = pg_loss * current.numel()
                    kl = torch.zeros_like(current)
                else:
                    terms, ratio, kl = policy_terms(
                        current, old, record["advantage"], args.clip_epsilon,
                        0.0 if args.reference_model else args.kl_beta,
                    )
                if args.reference_model:
                    reference_logps = torch.tensor(record["ref_logps"], device=current.device)
                    if args.objective == "vendor_dynamic_grpo":
                        kl = vendor_kl(current, reference_logps, "low_var_kl")
                        terms = terms + args.kl_beta * kl.sum()
                    else:
                        reference_delta = reference_logps - current
                        kl = torch.exp(reference_delta) - reference_delta - 1
                        terms = terms + args.kl_beta * kl
                if args.loss_reduction == "trajectory_mean":
                    loss = terms.sum() / (source["groups"] * args.group_size * record["trajectory_tokens"])
                elif args.loss_reduction == "prompt_mean":
                    loss = terms.sum() / (source["groups"] * record["group_tokens"])
                else:
                    loss = terms.sum() / chunk_denominator
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite online GRPO loss")
                loss.backward()
                chunk_loss += float(loss.detach())
                chunk_kl += float(kl.detach().sum())
                chunk_clipped += float(((ratio.detach() - 1).abs() > args.clip_epsilon).sum())
                total_kl += float(kl.detach().sum()) / denominator
                total_ratio += float(ratio.detach().sum()) / denominator
                clipped += float(((ratio.detach() - 1).abs() > args.clip_epsilon).sum()) / denominator
                position = minibatch_index * batch_size + local_position + 1
                if position % 100 == 0:
                    print(json.dumps({"update": update_index + 1, "decisions": position,
                                      "total": len(records)}), flush=True)
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            if not torch.isfinite(norm):
                raise FloatingPointError("Non-finite online GRPO gradient")
            optimizer.step()
            total_loss += chunk_loss
            minibatch_metrics.append({
                "index": minibatch_index + 1, "decisions": len(chunk),
                "tokens": chunk_denominator, "loss": chunk_loss,
                "mean_kl_to_reference" if args.reference_model else "mean_kl_to_rollout":
                    chunk_kl / chunk_denominator,
                "clipped_fraction": chunk_clipped / chunk_denominator,
                "gradient_norm": float(norm),
            })
            print(json.dumps({"update": update_index + 1,
                              "minibatch": minibatch_metrics[-1]}), flush=True)
            progress = copy.deepcopy({"completed_optimizer_steps": update_index * len(chunks) + minibatch_index + 1,
                                      "update_index": update_index, "next_minibatch": minibatch_index + 1,
                                      "total_loss": total_loss, "total_kl": total_kl, "total_ratio": total_ratio,
                                      "clipped": clipped, "minibatch_metrics": minibatch_metrics})
            boundary_rng = rng_state()
            if progress["completed_optimizer_steps"] % args.checkpoint_every == 0 or minibatch_index + 1 == len(chunks):
                checkpoint()
            pause_if_requested()
        post_delta = post_ratio = post_kl = post_clipped = 0.0
        post_tokens = 0
        probe_indices = random.Random(20260924 + update_index).sample(
            range(len(records)), min(args.diagnostic_samples, len(records)))
        model.eval()
        with torch.no_grad():
            for record_index in probe_indices:
                record = records[record_index]
                current = token_logps(model, record)
                old = torch.tensor(record["old_logps"], device=current.device)
                delta = old - current
                ratio = torch.exp(-delta)
                post_delta += float(delta.abs().sum())
                post_ratio += float(ratio.sum())
                post_kl += float((torch.exp(delta) - delta - 1).sum())
                post_clipped += float(((ratio - 1).abs() > args.clip_epsilon).sum())
                post_tokens += current.numel()
        model.train()
        entry = {
            "update": update_index + 1, "loss": total_loss / len(chunks),
            "mean_kl_to_reference" if args.reference_model else "mean_kl_to_rollout": total_kl,
            "mean_ratio": total_ratio,
            "kl_reference": "frozen_reference" if args.reference_model else "rollout_policy",
            "clipped_fraction": clipped, "gradient_norm": float(norm),
            "optimizer_steps": len(chunks),
            "minibatch_decisions": args.minibatch_decisions or len(records),
            "minibatches": minibatch_metrics,
            "seconds": round(time.perf_counter() - started, 1),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "post_update_probe_decisions": len(probe_indices),
            "post_update_probe_tokens": post_tokens,
            "post_update_mean_abs_logp_shift": post_delta / post_tokens,
            "post_update_mean_ratio": post_ratio / post_tokens,
            "post_update_mean_kl": post_kl / post_tokens,
            "post_update_clipped_fraction": post_clipped / post_tokens,
        }
        with log_path.open("a") as handle:
            handle.write(json.dumps(entry) + "\n")
        print(json.dumps(entry), flush=True)

    model.eval()
    merge_probe_indices = random.Random(20260928).sample(
        range(len(records)), min(8, len(records)))
    with torch.no_grad():
        pre_merge_logps = [token_logps(model, records[index]).cpu().float()
                           for index in merge_probe_indices]
    model.save_pretrained(args.output / "adapter")
    torch.save(optimizer.state_dict(), args.output / "optimizer_state.pt")
    (args.output / "trainable_parameters.json").write_text(json.dumps(trainable_names) + "\n")
    merged = model.merge_and_unload()
    merged.config.use_cache = True
    merge_abs_sum = 0.0
    merge_max_abs = 0.0
    merge_tokens = 0
    with torch.no_grad():
        for index, before in zip(merge_probe_indices, pre_merge_logps):
            after = token_logps(merged, records[index]).cpu().float()
            difference = (after - before).abs()
            merge_abs_sum += float(difference.sum())
            merge_max_abs = max(merge_max_abs, float(difference.max()))
            merge_tokens += difference.numel()
    merge_mean_abs = merge_abs_sum / merge_tokens
    print(json.dumps({"merge_probe_decisions": len(merge_probe_indices),
                      "merge_probe_tokens": merge_tokens,
                      "merge_mean_abs_logp": merge_mean_abs,
                      "merge_max_abs_logp": merge_max_abs}), flush=True)
    if merge_mean_abs > 0.05:
        raise ValueError("LoRA merge changed policy log-probabilities too much")
    merged.save_pretrained(args.output / "merged_hf", safe_serialization=True)
    tokenizer.save_pretrained(args.output / "adapter")
    tokenizer.save_pretrained(args.output / "merged_hf")
    report = {
        "resume_update_checkpoint": str(args.resume_update_checkpoint) if args.resume_update_checkpoint else None,
        "optimizer_steps_this_process": progress["completed_optimizer_steps"] - resumed_steps,
        "checkpoint_every": args.checkpoint_every,
        "prepared_cache_sha256": file_hash(store.cache_path),
        "final_boundary_checkpoint": str(checkpoint()),
        "status": "complete", "model": str(args.model),
        "train_base_model": str(args.train_base_model or args.model),
        "resume_adapter": str(args.resume_adapter) if args.resume_adapter else None,
        "rollout_adapter": str(args.rollout_adapter) if args.rollout_adapter else None,
        "rollout_adapter_sha256": (
            file_hash(args.rollout_adapter / "adapter_model.safetensors")
            if args.rollout_adapter else None
        ),
        "input_optimizer_state": str(args.optimizer_state) if args.optimizer_state else None,
        "reference_model": str(args.reference_model) if args.reference_model else None,
        "model_safetensors_sha256": file_hash(args.model / "model.safetensors"),
        "rollouts_sha256": file_hash(args.rollouts),
        "updater_script_sha256": file_hash(Path(__file__)),
        "vendor_core_algos_sha256": file_hash(
            Path(__file__).resolve().parents[1] / "vendor" / "verl-agent" /
            "verl" / "trainer" / "ppo" / "core_algos.py"
        ) if args.objective == "vendor_dynamic_grpo" else None,
        "rollout_logp_tokens": rollout_logp_count,
        "rollout_vs_recomputed_mean_abs": (
            rollout_logp_abs_sum / rollout_logp_count if rollout_logp_count else None
        ),
        "rollout_vs_recomputed_max_abs": (
            rollout_logp_max_abs if rollout_logp_count else None
        ),
        "merge_probe_decisions": len(merge_probe_indices),
        "merge_probe_tokens": merge_tokens,
        "merge_mean_abs_logp": merge_mean_abs,
        "merge_max_abs_logp": merge_max_abs,
        "selection_manifest_sha256": file_hash(args.selection_manifest) if args.selection_manifest else None,
        "prescreen_seed_range": [args.prescreen_seed_start, args.prescreen_seed_end],
        "prescreen_rollouts": args.prescreen_rollouts,
        "additional_initial_probe_rollouts": args.additional_initial_probe_rollouts,
        "fresh_rollouts": source["rollouts"], "rollout_group_size": args.group_size,
        "candidate_rollouts": source.get("candidate_rollouts"),
        "candidate_groups": source.get("candidate_groups"),
        "mixed_groups": source["groups"], "decision_segments": len(records),
        "target_tokens_per_update": denominator,
        "optimizer_updates": args.updates * ((len(records) + (args.minibatch_decisions or len(records)) - 1)
                                             // (args.minibatch_decisions or len(records))),
        "batch_passes": args.updates, "minibatch_decisions": args.minibatch_decisions,
        "reward_definition": (
            "10*terminal_won - 0.1*invalid_format_at_decision"
            if args.objective == "vendor_dynamic_grpo" else
            "10*terminal_won - 0.1*invalid_format_count_per_trajectory"
        ),
        "objective": args.objective,
        "clip_epsilon_high": (
            args.clip_epsilon_high if args.objective == "vendor_dynamic_grpo" else
            args.clip_epsilon if args.objective == "mimo_outcome_grpo" else None
        ),
        "advantage_rule": (
            "reward-minus-group-mean-no-std" if args.objective == "mimo_outcome_grpo"
            else "vendor-gigpo" if args.objective == "vendor_dynamic_grpo"
            else "reward-minus-group-mean-divide-std"
        ),
        "invalid_actions_in_batch": sum(sum(row["invalid_format"]) for row in source["results"]),
        "lr": args.lr, "clip_epsilon": args.clip_epsilon, "kl_beta": args.kl_beta,
        "kl_reference": ("frozen_reference" if args.reference_model else
                         "none" if args.kl_beta == 0 else "rollout_policy"),
        "loss_reduction": args.loss_reduction,
        "seconds": round(time.perf_counter() - started, 1),
        "adapter_sha256": file_hash(args.output / "adapter" / "adapter_model.safetensors"),
        "optimizer_state_sha256": file_hash(args.output / "optimizer_state.pt"),
    }
    (args.output / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
