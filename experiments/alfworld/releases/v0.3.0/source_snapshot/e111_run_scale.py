"""Independent nested B=16/32 arms; unchanged parameterized E088 mathematics."""
import argparse
from datetime import datetime, timezone
import fcntl
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from e071_run_mainline import PROJECT, WORK, SFT, read, rows, sha, write
from e071_summarize import NAMES
from e073_eval_regression import paired
from e086_run_continuation import comparison

DATA = PROJECT / 'data/E111_scale'
PREP = PROJECT / 'data/E110_scale_prepare'
INITIAL = Path('/home/codex/agentic-rl/e085_reuse28_20261014/update')
PLAN = PROJECT / 'paper/E111_正式B规模训练登记_20261015.md'
OLDHOLD = Path('/home/codex/agentic-rl/e092_fresh_holdout_20261015')


def compare(left, right):
    result = paired(left, right)
    result['by_type'] = {k: paired([r for r in left if r['task_type'] == k], [r for r in right if r['task_type'] == k]) for k in NAMES}
    result.update(format_errors=sum(r['invalid_format'] for r in right), cap_hits=sum(r['response_cap_hits'] for r in right),
                  before_steps=sum(r['steps'] for r in left), after_steps=sum(r['steps'] for r in right),
                  scope='opened matched regression; not blind test')
    return result


def prepare(arm, run):
    from e110_update_group_size import validate_group_dimensions
    assert read(PREP / 'root_actual_review.json')['actual_postrun_review'] == 'passed'
    proof = read(PREP / 'root_actual_review.json')
    assert proof['cpu_engineering']['g8_bitwise_adapter_adamw_prepared_cache_equal']
    frozen_old = read(PREP / 'frozen_inputs.json')
    assert all(sha(p) == h for p, h in frozen_old['input_sha256'].items())
    assert all(sha(p) == h for p, h in frozen_old['game_sha256'].items())
    source_path = PREP / ('A16_G8.json' if arm == 'A' else 'B32_G8.json')
    source = read(source_path); validate_group_dimensions(source, 8)
    count = 16 if arm == 'A' else 32
    expected = proof['nested_same_policy_batches'][f'{arm}{count}']
    assert source['groups'] == count and sha(source_path) == expected['sha256']
    assert source['rollout_adapter'] == str(INITIAL / 'adapter')
    assert sha(INITIAL / 'adapter/adapter_model.safetensors') == source['rollout_adapter_sha256'] == 'a92b386e2289d473593d1f0ae91c0b40944a1f7f99d3d84071eca32d65a6c70c'
    assert sha(INITIAL / 'optimizer_state.pt') == 'e5caaa26580453d9698aab26b6c77e3ecf2ff53d8dad75f0589c018e3dba235b'
    carry = proof['family_cumulative_seconds']
    if arm == 'B':
        previous = read(DATA / 'A/root_actual_review.json')
        assert previous['actual_postrun_review'] == 'passed' and not previous['direction_stop']
        carry = previous['family_cumulative_seconds']
    dev = rows(INITIAL.parent / 'dev_seen24.jsonl') + rows(INITIAL.parent / 'dev_unseen24.jsonl')
    hold = rows(OLDHOLD / 'e085_block1_24.jsonl') + rows(OLDHOLD / 'e085_block2_24.jsonl')
    assert len(dev) == len(hold) == 48 and sum(r['won'] for r in dev) == 43 and sum(r['won'] for r in hold) == 45
    from e110_test_group_size import verify_math
    verify_math()
    files = [Path(__file__), PLAN, source_path, PREP / 'root_actual_review.json', PREP / 'frozen_inputs.json',
             PREP / 'updater_derivation.json', SFT / 'model.safetensors', SFT / 'config.json', SFT / 'tokenizer.json',
             INITIAL / 'adapter/adapter_model.safetensors', INITIAL / 'adapter/adapter_config.json', INITIAL / 'optimizer_state.pt',
             INITIAL / 'trainable_parameters.json', INITIAL / 'training_report.json',
             INITIAL.parent / 'dev_seen24.jsonl', INITIAL.parent / 'dev_unseen24.jsonl',
             OLDHOLD / 'e085_block1_24.jsonl', OLDHOLD / 'e085_block2_24.jsonl',
             PROJECT / 'vendor/verl-agent/verl/trainer/ppo/core_algos.py',
             PROJECT / 'data/E067_alfworld/e067_dev_games.json', PROJECT / 'data/E092_fresh_holdout/block_01_games24.json',
             PROJECT / 'data/E092_fresh_holdout/block_02_games24.json', PROJECT / 'data/E109_prompt_coverage/future_train_pool1634.json']
    files += [PROJECT / 'scripts' / s for s in ('e110_update_group_size.py', 'e110_test_group_size.py',
              'e111_probe_scale_gpu.py', 'optimizer_boundary_checkpoint.py', 'e026_codegym_grpo.py',
              'e067_eval_fixed_games.py', 'e066_eval_direct_hf.py', 'e071_run_mainline.py', 'e071_summarize.py',
              'e073_eval_regression.py', 'e086_run_continuation.py')]
    frozen = dict(experiment='E111', arm=arm, input_optimizer_step=134, group_size=8, groups=count,
                  batch_passes=1, expected=expected, source=str(source_path), family_previous_seconds=carry,
                  family_cap_seconds=14400, phase_cap_seconds=2400 if arm == 'A' else 3200,
                  input_sha256={str(p): sha(p) for p in files}, game_sha256=frozen_old['game_sha256'])
    if (run / 'frozen_inputs.json').exists(): assert read(run / 'frozen_inputs.json') == frozen
    else: write(run / 'frozen_inputs.json', frozen)
    return frozen, dev, hold


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--arm', choices=('A', 'B'), required=True); ap.add_argument('--preflight', action='store_true'); a = ap.parse_args()
    run = Path(f'/home/codex/agentic-rl/e111_scale_{a.arm}_20261015'); run.mkdir(exist_ok=True)
    DATA.mkdir(exist_ok=True); (DATA / a.arm).mkdir(exist_ok=True)
    os.chdir(WORK); os.environ.update(ALFWORLD_DATA='/home/codex/.cache/alfworld', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    assert os.environ.get('E066_NAVIGATION_MEMORY') != '1'
    frozen, dev, hold = prepare(a.arm, run)
    if a.preflight:
        write(run / 'preflight.json', dict(status='passed', arm=a.arm, input_sha_count=len(frozen['input_sha256']),
              gpu_started=False, expected=frozen['expected'], family_previous_seconds=frozen['family_previous_seconds']))
        print(read(run / 'preflight.json'), flush=True); return
    assert not (run / 'run_state.json').exists() and read(run / 'preflight.json')['status'] == 'passed'
    lock = (WORK / '.agent_experiment.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.monotonic(); pre = read(run / 'preflight_cost.json')['wall_seconds']
    state = dict(experiment='E111', arm=a.arm, status='running', pid=os.getpid(), actual_postrun_review='pending',
                 started_utc=datetime.now(timezone.utc).isoformat(), research_optimizer_updates=0, gpu_jobs_seconds=0.,
                 family_previous_seconds=frozen['family_previous_seconds'], family_cap_seconds=14400,
                 phase_cap_seconds=frozen['phase_cap_seconds'], input_optimizer_step=134)
    def save(**change):
        state.update(change); elapsed = time.monotonic() - started
        state.update(controller_seconds=round(elapsed, 3), inclusive_phase_seconds=round(pre + elapsed, 3),
                     family_cumulative_seconds=round(frozen['family_previous_seconds'] + pre + elapsed, 3),
                     updated_utc=datetime.now(timezone.utc).isoformat()); write(run / 'run_state.json', state)
    def remaining():
        used = pre + time.monotonic() - started
        return min(frozen['phase_cap_seconds'] - used, 14400 - frozen['family_previous_seconds'] - used)
    def child(script, options, label, cap):
        assert all(sha(p) == h for p, h in frozen['input_sha256'].items())
        limit = min(cap, remaining() - 45)
        if limit <= 0: raise TimeoutError('Cumulative scale-study budget exhausted')
        process = None; tick = time.monotonic(); log = run / f'{label}.log'
        try:
            with log.open('x') as handle:
                process = subprocess.Popen([sys.executable, str(PROJECT / 'scripts' / script), *map(str, options)],
                          cwd=WORK, env=os.environ, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                save(stage=label, child_pid=process.pid, log=str(log), stage_hard_seconds=round(limit, 3))
                code = process.wait(timeout=limit)
                if code: raise RuntimeError(f'{label} exit {code}; preserve failed artifacts')
        finally:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try: process.wait(timeout=30)
                except subprocess.TimeoutExpired: os.killpg(process.pid, signal.SIGKILL); process.wait()
            state['gpu_jobs_seconds'] += time.monotonic() - tick; save(child_pid=None)
    def terminate(signum, frame): raise InterruptedError(f'Controller signal {signum}')
    signal.signal(signal.SIGTERM, terminate); save(stage='starting')
    try:
        if a.arm == 'A':
            child('e111_probe_scale_gpu.py', ['--rollouts', frozen['source'], '--output', run / 'gpu_probe.json'], 'gpu_probe', 110)
            assert read(run / 'gpu_probe.json')['status'] == 'passed'
        else:
            assert read(DATA / 'A/root_actual_review.json')['gpu_probe_status'] == 'passed'
        update = run / 'update'; soft = 1250 if a.arm == 'A' else 2100; hard = 1450 if a.arm == 'A' else 2300
        child('e110_update_group_size.py', ['--rollouts', frozen['source'], '--group-size', 8, '--model', SFT,
              '--reference-model', SFT, '--train-base-model', SFT, '--resume-adapter', INITIAL / 'adapter',
              '--optimizer-state', INITIAL / 'optimizer_state.pt', '--rollout-adapter', INITIAL / 'adapter',
              '--output', update, '--updates', 1, '--lr', '1e-5', '--objective', 'vendor_dynamic_grpo',
              '--clip-epsilon', '.2', '--clip-epsilon-high', '.28', '--kl-beta', '.01', '--loss-reduction', 'global_token',
              '--minibatch-decisions', 256, '--checkpoint-every', 4, '--stop-after-seconds', soft,
              '--prescreen-rollouts', 0, '--additional-initial-probe-rollouts', 0], 'update', hard)
        if (update / 'paused_update.json').exists():
            save(status='stopped', reason='incomplete_batch_safe_boundary_preserved', recoverable_checkpoint=read(update / 'paused_update.json')['checkpoint']); return
        t = read(update / 'training_report.json'); expected = frozen['expected']
        assert t['status'] == 'complete' and t['optimizer_updates'] == t['optimizer_steps_this_process'] == expected['expected_updates']
        assert t['decision_segments'] == expected['decisions'] and t['target_tokens_per_update'] == expected['tokens']
        assert t['mixed_groups'] == frozen['groups'] and t['fresh_rollouts'] == frozen['groups'] * 8 and t['rollout_group_size'] == 8
        assert t['rollout_vs_recomputed_mean_abs'] <= .05 and t['rollouts_sha256'] == expected['sha256']
        for key in ('objective', 'lr', 'clip_epsilon', 'clip_epsilon_high', 'kl_beta', 'loss_reduction', 'minibatch_decisions'):
            assert t[key] == read(INITIAL / 'training_report.json')[key]
        import torch
        from safetensors.torch import load_file
        weights = load_file(update / 'adapter/adapter_model.safetensors'); opt = torch.load(update / 'optimizer_state.pt', map_location='cpu', weights_only=True)
        assert len(weights) == len(opt['state']) == 392
        assert all(v.dtype == torch.float32 and bool(torch.isfinite(v).all()) for v in weights.values())
        assert {int(v['step']) for v in opt['state'].values()} == {134 + expected['expected_updates']}
        assert all(bool(torch.isfinite(v[k]).all()) for v in opt['state'].values() for k in ('exp_avg', 'exp_avg_sq'))
        assert sha(update / 'adapter/adapter_model.safetensors') == t['adapter_sha256'] and sha(update / 'optimizer_state.pt') == t['optimizer_state_sha256']
        del weights, opt
        save(research_optimizer_updates=t['optimizer_updates'], output_optimizer_step=134 + t['optimizer_updates'],
             latest_adapter=str(update / 'adapter'), latest_optimizer=str(update / 'optimizer_state.pt'))
        results = {}
        specs = [('dev48', dev, [('seen', PROJECT / 'data/E067_alfworld/e067_dev_games.json'), ('unseen', PROJECT / 'data/E067_alfworld/e067_dev_games.json')], 360),
                 ('holdout48', hold, [('train', PROJECT / 'data/E092_fresh_holdout/block_01_games24.json'), ('train', PROJECT / 'data/E092_fresh_holdout/block_02_games24.json')], 440)]
        for label, baseline, blocks, cap in specs:
            current = []; tick = time.monotonic()
            for number, (split, games) in enumerate(blocks, 1):
                output = run / f'{label}_block{number}.jsonl'
                child('e067_eval_fixed_games.py', ['--model', SFT, '--adapter', update / 'adapter', '--games', games,
                       '--split', split, '--output', output], f'{label}_block{number}', cap - (time.monotonic() - tick))
                batch = rows(output); expected_games = [g['gamefile'] for g in read(games)['games'] if g['split'] == split]
                assert len(batch) == 24 and [r['gamefile'] for r in batch] == expected_games; current += batch
            result = compare(baseline, current)
            if label == 'dev48': result['keep_gate'] = comparison(current, baseline)['keep_gate']
            write(run / f'{label}_paired.json', result); results[label] = result; save(**{f'{label}_wins': result['after']})
        save(status='complete', stage='finished', reason='single_independent_arm_finished', dev_keep_gate=results['dev48']['keep_gate'],
             candidate_small_suites_retained=all(r['after'] >= r['before'] and not r['format_errors'] and not r['cap_hits'] and
                        all(k['before'] - k['after'] < 2 for k in r['by_type'].values()) for r in results.values()))
    except BaseException as error:
        save(status='stopped' if isinstance(error, (InterruptedError, TimeoutError, subprocess.TimeoutExpired)) else 'failed', reason=repr(error)); raise
    finally:
        save(phase_closed=state['status'] != 'running'); write(DATA / a.arm / 'run_state.json', state); lock.close()
        print(state, flush=True)


if __name__ == '__main__': main()
