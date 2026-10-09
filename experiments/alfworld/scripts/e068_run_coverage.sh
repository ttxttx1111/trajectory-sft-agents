#!/usr/bin/env bash
set -euo pipefail

# Bounded single-GPU ALFWorld source-coverage pilot. Not a MiMo reproduction.
# Usage: bash e068_run_coverage.sh RUN_ROOT FROZEN_POOL [ROUNDS=3]
run_root=${1:?run root required}
pool=${2:?frozen training pool required}
rounds=${3:-3}
[[ "$rounds" =~ ^[1-3]$ ]] || { echo 'E068 permits only 1--3 rounds' >&2; exit 1; }
source_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
work_root="$source_root/vendor/verl-agent"
sft_model=${SFT_MODEL:?set SFT_MODEL to the merged SFT directory}
dev_games="$(dirname "$pool")/dev_games.json"
export ALFWORLD_DATA="${ALFWORLD_DATA:-$HOME/.cache/alfworld}"
export PYTHONPATH="$work_root:$source_root/scripts${PYTHONPATH:+:$PYTHONPATH}"
cd "$work_root"
mkdir -p "$run_root"

# Preserve the exact source and refuse silent edits between collection/update.
scripts=(e026_codegym_grpo.py e066_probe_online_group.py e066_collect_fresh_mixed.py
         e066_update_online_grpo.py e066_eval_direct_hf.py e067_eval_fixed_games.py
         e067_probe_fixed_sampled.py e068_collect_coverage.py e068_run_coverage.sh)
if [[ ! -e "$run_root/source_hashes.sha256" ]]; then
  mkdir -p "$run_root/source_snapshot"
  for script in "${scripts[@]}"; do
    cp "$source_root/scripts/$script" "$run_root/source_snapshot/$script"
  done
  (cd "$source_root/scripts" && sha256sum "${scripts[@]}") > "$run_root/source_hashes.sha256"
fi
check_source() { (cd "$source_root/scripts" && sha256sum --quiet -c "$run_root/source_hashes.sha256"); }
check_source
previous_round=""
previous_adapter=""
resume_args=()
actor_args=()
collector_args=()
for ((round=1; round<=rounds; round++)); do
  round_name=$(printf 'round_%02d' "$round")
  round_root="$run_root/$round_name"
  rollouts="$round_root/fresh_mixed_rollouts.json"
  updated="$round_root/update"
  mkdir -p "$round_root"
  if [[ -e "$rollouts" ]]; then
    status=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$rollouts")
    [[ "$status" == collecting || "$status" == complete ]] || {
      echo "Stopped collection at $rollouts: $status; no update" >&2; exit 2;
    }
  else
    status=new
  fi
  if [[ "$status" != complete ]]; then
    check_source
    resume_collection=()
    [[ "$status" == collecting ]] && resume_collection=(--resume)
    python "$source_root/scripts/e068_collect_coverage.py" \
      --model "$sft_model" --pool "$pool" --output "$rollouts" \
      --targets double=8 look=4 maintenance=4 \
      --caps double=24 look=24 maintenance=48 --max-seconds 5400 \
      "${collector_args[@]}" "${resume_collection[@]}" \
      >> "$round_root/collection.log" 2>&1
  fi
  read -r status candidate_rollouts <<< "$(python - "$rollouts" <<'PY'
import json, sys
row = json.load(open(sys.argv[1]))
assert row['source_targets'] == {'double':8, 'look':4, 'maintenance':4}
if row['status'] == 'complete':
    assert row['source_accepted'] == row['source_targets']
    assert row['groups'] == 16 and row['rollouts'] == 128
print(row['status'], row['candidate_rollouts'])
PY
)"
  [[ "$status" == complete ]] || { echo "Source quotas unmet; no update: $rollouts" >&2; exit 2; }
  check_source
  if [[ ! -e "$updated" ]]; then
    python "$source_root/scripts/e066_update_online_grpo.py" \
      --rollouts "$rollouts" --model "$sft_model" --reference-model "$sft_model" \
      --train-base-model "$sft_model" "${resume_args[@]}" "${actor_args[@]}" \
      --output "$updated" --updates 1 --lr 1e-5 \
      --objective vendor_dynamic_grpo \
      --clip-epsilon 0.2 --clip-epsilon-high 0.28 --kl-beta 0.01 \
      --loss-reduction global_token --minibatch-decisions 256 \
      --prescreen-rollouts "$candidate_rollouts" --additional-initial-probe-rollouts 0 \
      > "$round_root/update.log" 2>&1
  fi
  python - "$updated/training_report.json" "$sft_model" "$previous_adapter" <<'PY'
import json, sys
row = json.load(open(sys.argv[1]))
assert row['status'] == 'complete'
assert row['model'] == row['reference_model'] == row['train_base_model'] == sys.argv[2]
assert row.get('rollout_adapter') == (sys.argv[3] or None)
assert row['objective'] == 'vendor_dynamic_grpo'
assert row['batch_passes'] == 1 and row['minibatch_decisions'] == 256
assert row['loss_reduction'] == 'global_token' and row['optimizer_updates'] > 1
PY
  previous_round=$rollouts
  previous_adapter="$updated/adapter"
  resume_args=(--resume-adapter "$previous_adapter" --optimizer-state "$updated/optimizer_state.pt")
  actor_args=(--rollout-adapter "$previous_adapter")
  collector_args=(--adapter "$previous_adapter" --previous-round "$previous_round")
  echo "Completed $round_name: $candidate_rollouts candidate rollouts" >&2

  # Fixed, already-open development games only. The final36 stays unopened.
  if (( round == 1 || round == rounds )); then
    for split in seen unseen; do
      output="$round_root/dev_${split}24.jsonl"
      if [[ ! -e "$output" ]]; then
        check_source
        python "$source_root/scripts/e067_eval_fixed_games.py" \
          --model "$sft_model" --adapter "$previous_adapter" \
          --games "$dev_games" \
          --split "$split" --output "$output" > "$round_root/dev_${split}.log" 2>&1
      fi
      python - "$output" <<'PY'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1])]
assert len(rows) == 24 and len({row['gamefile'] for row in rows}) == 24
PY
    done
  fi
done
check_source
output="$run_root/dev_double8_sampled.json"
if [[ ! -e "$output" ]]; then
  python "$source_root/scripts/e067_probe_fixed_sampled.py" \
    --model "$sft_model" --adapter "$previous_adapter" \
    --games "$dev_games" \
    --task-type pick_two_obj_and_place --output "$output" \
    > "$run_root/dev_double8_sampled.log" 2>&1
fi
echo "$previous_adapter"
