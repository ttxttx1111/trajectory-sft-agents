#!/usr/bin/env python3
"""Collect execution-verified ALFWorld hand-coded expert trajectories."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from alfworld.agents.environment import get_environment

TASK_TYPES = (
    "pick_and_place_simple",
    "pick_two_obj_and_place",
    "look_at_obj_in_light",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_clean_then_place_in_recep",
)


def _failed_action_feedback(observation: str) -> bool:
    text = observation.strip().lower()
    return text.startswith(("nothing happens.", "you can't ", "you cannot "))

def _scalar(value: Any) -> Any:
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


def _load_task_metadata(gamefile: str) -> dict[str, Any]:
    task_path = Path(gamefile).with_name("traj_data.json")
    with task_path.open("r", encoding="utf-8") as handle:
        task = json.load(handle)
    return {
        "task_type": task.get("task_type"),
        "task_id": task.get("task_id"),
        "repeat_idx": task.get("repeat_idx"),
        "task_path": str(task_path),
    }


def collect(
    config_path: Path,
    episodes: int,
    seed: int,
    max_steps: int,
    quota_per_type: int | None = None,
    selected_task_type: str | None = None,
) -> list[dict[str, Any]]:
    random.seed(seed)
    np.random.seed(seed)

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    base_env = get_environment(config["env"]["type"])(config, train_eval="train")
    active_types = (selected_task_type,) if selected_task_type else TASK_TYPES
    if selected_task_type:
        base_env.game_files = [
            game for game in base_env.game_files
            if Path(game).parents[1].name.startswith(selected_task_type + "-")
        ]
        base_env.num_games = len(base_env.game_files)
        if not base_env.game_files:
            raise ValueError(f"No eligible games for {selected_task_type}")
    env = base_env.init_env(batch_size=1)
    env.seed(seed)

    records: list[dict[str, Any]] = []
    accepted_counts: Counter[str] = Counter()
    accepted_gamefiles: set[str] = set()
    try:
        for episode_index in range(episodes):
            if quota_per_type is not None and all(
                accepted_counts[task_type] >= quota_per_type for task_type in active_types
            ):
                break
            initial_obs, infos = env.reset()
            initial_text = initial_obs[0]
            gamefile = str(_scalar(infos["extra.gamefile"]))
            metadata = _load_task_metadata(gamefile)
            task_type = metadata["task_type"]
            if task_type not in active_types:
                raise ValueError(f"Unexpected training task type: {task_type}")
            skip_reason = None
            if quota_per_type is not None:
                if accepted_counts[task_type] >= quota_per_type:
                    skip_reason = "type_quota_filled"
                elif gamefile in accepted_gamefiles:
                    skip_reason = "duplicate_accepted_gamefile"
            steps: list[dict[str, Any]] = []
            error: str | None = None
            done = False
            won = False
            score = 0.0

            for step_index in range(0 if skip_reason else max_steps):
                expert_plans = infos.get("extra.expert_plan")
                plan = _scalar(expert_plans) if expert_plans is not None else []
                if not plan:
                    error = f"empty expert plan at step {step_index}"
                    break

                action = str(plan[0])
                admissible = list(_scalar(infos["admissible_commands"]))
                action_is_admissible = action in admissible
                next_obs, scores, dones, next_infos = env.step([action])
                done = bool(dones[0])
                won = bool(_scalar(next_infos["won"]))
                score = float(scores[0])

                steps.append(
                    {
                        "step": step_index,
                        "action": action,
                        "admissible_actions": admissible,
                        "action_is_admissible": action_is_admissible,
                        "observation": next_obs[0],
                        "done": done,
                        "won": won,
                        "score": score,
                    }
                )
                infos = next_infos
                if done:
                    break

            failed_action_feedback_steps = [
                step["step"] for step in steps if _failed_action_feedback(step["observation"])
            ]
            accepted = (
                skip_reason is None and done and won and bool(steps)
                and all(step["action_is_admissible"] for step in steps)
                and not failed_action_feedback_steps
            )
            if accepted and quota_per_type is not None:
                accepted_counts[task_type] += 1
                accepted_gamefiles.add(gamefile)
            records.append(
                {
                    "episode_index": episode_index,
                    "seed": seed,
                    "gamefile": gamefile,
                    **metadata,
                    "initial_observation": initial_text,
                    "steps": steps,
                    "num_steps": len(steps),
                    "done": done,
                    "won": won,
                    "score": score,
                    "all_actions_admissible": all(
                        step["action_is_admissible"] for step in steps
                    ),
                    "failed_action_feedback_steps": failed_action_feedback_steps,
                    "error": error,
                    "accepted": accepted,
                    "skip_reason": skip_reason,
                }
            )
            if quota_per_type is not None and (episode_index + 1) % 100 == 0:
                print(
                    "E066_COLLECTION_PROGRESS "
                    + json.dumps({"attempts": episode_index + 1, "accepted": dict(accepted_counts)}, sort_keys=True),
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        env.close()

    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--quota-per-type", type=int)
    parser.add_argument("--task-type", choices=TASK_TYPES)
    parser.add_argument("--accepted-output", type=Path)
    parser.add_argument("--summary-output", type=Path)
    args = parser.parse_args()

    if args.quota_per_type is not None:
        if args.quota_per_type <= 0 or args.accepted_output is None:
            parser.error("positive --quota-per-type requires --accepted-output")
        if args.accepted_output == args.output:
            parser.error("--accepted-output must differ from --output")
    records = collect(
        args.config, args.episodes, args.seed, args.max_steps, args.quota_per_type,
        args.task_type,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records)
    args.output.write_text(payload, encoding="utf-8")

    accepted = [record for record in records if record["accepted"]]
    accepted_payload = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in accepted
    )
    if args.accepted_output is not None:
        args.accepted_output.parent.mkdir(parents=True, exist_ok=True)
        args.accepted_output.write_text(accepted_payload, encoding="utf-8")

    successful = [record for record in records if record["won"]]
    executed = [record for record in records if record["skip_reason"] is None]
    invalid = [record for record in records if not record["all_actions_admissible"]]
    mean_steps = (
        sum(record["num_steps"] for record in successful) / len(successful)
        if successful
        else 0.0
    )
    summary = {
        "episodes": len(records),
        "expert_executed_episodes": len(executed),
        "successful": len(successful),
        "success_rate_executed": len(successful) / len(executed) if executed else 0.0,
        "executed_by_type": dict(sorted(Counter(record["task_type"] for record in executed).items())),
        "successful_by_type": dict(sorted(Counter(record["task_type"] for record in successful).items())),
        "invalid_action_episodes": len(invalid),
        "failed_action_feedback_episodes": sum(bool(record["failed_action_feedback_steps"]) for record in records),
        "mean_success_steps": mean_steps,
        "min_success_steps": min(
            (record["num_steps"] for record in successful), default=None
        ),
        "max_success_steps": max(
            (record["num_steps"] for record in successful), default=None
        ),
        "errors": sum(record["error"] is not None for record in records),
        "accepted": len(accepted),
        "accepted_by_type": dict(sorted(Counter(record["task_type"] for record in accepted).items())),
        "accepted_unique_gamefiles": len({record["gamefile"] for record in accepted}),
        "accepted_sha256": hashlib.sha256(accepted_payload.encode("utf-8")).hexdigest(),
        "quota_complete": args.quota_per_type is None or all(
            sum(record["task_type"] == task_type for record in accepted) >= args.quota_per_type
            for task_type in ((args.task_type,) if args.task_type else TASK_TYPES)
        ),
        "skip_reasons": dict(sorted(Counter(record["skip_reason"] for record in records if record["skip_reason"]).items())),
        "output": str(args.output),
        "accepted_output": str(args.accepted_output) if args.accepted_output else None,
    }
    summary_text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(summary_text + "\n", encoding="utf-8")
    print(summary_text)
    if not summary["quota_complete"]:
        raise RuntimeError("Balanced expert quota incomplete; do not use this dataset")


if __name__ == "__main__":
    main()
