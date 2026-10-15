"""Post-run CPU verification of a formal scale arm; never launches its successor."""
import argparse
from collections import Counter
import contextlib
import io
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from safetensors.torch import load_file
from e071_run_mainline import PROJECT, WORK, read, rows, sha, write
from e111_run_scale import DATA, INITIAL, OLDHOLD, compare
from e110_update_group_size import validate_group_dimensions, vendor_algorithms
from e110_test_group_size import verify_math
from e086_run_continuation import comparison
from e088_test_boundary_checkpoint import equal
from optimizer_boundary_checkpoint import BoundaryCheckpoints
from e092_review_results import replay
from e067_eval_fixed_games import CONFIG
from agent_system.environments.env_package.alfworld.envs import get_environment, load_config_file


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--arm', choices=('A', 'B'), required=True); a = ap.parse_args()
    started = time.monotonic(); torch.set_num_threads(4); os.chdir(WORK); os.environ['ALFWORLD_DATA'] = '/home/codex/.cache/alfworld'
    assert not torch.cuda.is_initialized()
    run = Path(f'/home/codex/agentic-rl/e111_scale_{a.arm}_20261015')
    state = read(run / 'run_state.json'); assert state['status'] == 'complete' and state['phase_closed']
    frozen = read(run / 'frozen_inputs.json')
    assert all(sha(p) == h for p, h in frozen['input_sha256'].items())
    assert all(sha(p) == h for p, h in frozen['game_sha256'].items()); verify_math()
    source = read(frozen['source']); validate_group_dimensions(source, 8)
    expected = frozen['expected']; update = run / 'update'; t = read(update / 'training_report.json')
    assert t['status'] == 'complete' and t['optimizer_updates'] == t['optimizer_steps_this_process'] == expected['expected_updates']
    assert t['target_tokens_per_update'] == expected['tokens'] and t['decision_segments'] == expected['decisions']
    assert t['rollouts_sha256'] == expected['sha256'] == sha(frozen['source'])
    assert t['updater_script_sha256'] == sha(PROJECT / 'scripts/e110_update_group_size.py')
    assert t['resume_adapter'] == t['rollout_adapter'] == str(INITIAL / 'adapter') and t['input_optimizer_state'] == str(INITIAL / 'optimizer_state.pt')
    assert (t['objective'], t['lr'], t['clip_epsilon'], t['clip_epsilon_high'], t['kl_beta'], t['loss_reduction'], t['minibatch_decisions'], t['batch_passes'], t['rollout_group_size']) == ('vendor_dynamic_grpo', 1e-5, .2, .28, .01, 'global_token', 256, 1, 8)
    assert t['rollout_vs_recomputed_mean_abs'] <= .05 and t['merge_mean_abs_logp'] <= .05
    weights = load_file(update / 'adapter/adapter_model.safetensors'); original = load_file(INITIAL / 'adapter/adapter_model.safetensors')
    opt = torch.load(update / 'optimizer_state.pt', map_location='cpu', weights_only=True)
    assert len(weights) == len(opt['state']) == 392 and any(not torch.equal(v, original[k]) for k, v in weights.items())
    assert all(v.dtype == torch.float32 and bool(torch.isfinite(v).all()) for v in weights.values())
    assert {int(v['step']) for v in opt['state'].values()} == {134 + expected['expected_updates']}
    assert all(bool(torch.isfinite(v[k]).all()) for v in opt['state'].values() for k in ('exp_avg', 'exp_avg_sq'))
    assert sha(update / 'adapter/adapter_model.safetensors') == t['adapter_sha256'] and sha(update / 'optimizer_state.pt') == t['optimizer_state_sha256']
    cp = Path(t['final_boundary_checkpoint']); m = read(cp / 'manifest.json'); BoundaryCheckpoints.inspect(cp, m['fingerprint'])
    assert equal(weights, load_file(cp / 'adapter/adapter_model.safetensors')) and equal(opt, torch.load(cp / 'optimizer_state.pt', map_location='cpu', weights_only=True))
    assert sha(update / 'checkpoints/prepared_records.pt') == t['prepared_cache_sha256']
    cache = torch.load(update / 'checkpoints/prepared_records.pt', map_location='cpu', weights_only=True)['records']
    advantage, _, _ = vendor_algorithms(); cursor = 0
    for g in source['results']:
        rr = [r for tr in g['records'] for r in tr]; indices = [i for i, tr in enumerate(g['records']) for _ in tr]
        scores = torch.tensor([10. * g['won'][i] - .1 * (not r['is_action_valid']) for r, i in zip(rr, indices)]).unsqueeze(1)
        adv, _ = advantage(scores, torch.ones_like(scores), np.array([g['gamefile']] * len(rr), dtype=object), np.array(indices))
        for r, index, value in zip(rr, indices, adv.squeeze(1).tolist()):
            actual = cache[cursor]; assert all(actual[k] == v for k, v in r.items())
            assert actual['advantage'] == value and actual['rollout_index'] == index and actual['seed'] == g['seed']
            assert actual['old_logps'] == r['rollout_logps']
            assert len(actual['ref_logps']) == len(r['completion_ids']) and all(math.isfinite(v) for v in actual['ref_logps'])
            cursor += 1
    assert cursor == len(cache) == expected['decisions']
    order = list(range(len(cache))); random.Random(20260922).shuffle(order)
    logs = rows(update / 'update_log.jsonl'); assert len(logs) == 1
    metrics = logs[0]['minibatches']; assert len(metrics) == expected['expected_updates']
    for number, entry in enumerate(metrics):
        chunk = order[number * 256:(number + 1) * 256]
        assert entry['tokens'] == sum(len(cache[i]['completion_ids']) for i in chunk) and entry['decisions'] == len(chunk)
        assert all(math.isfinite(entry[k]) for k in ('loss', 'gradient_norm', 'clipped_fraction', 'mean_kl_to_reference'))
    assert sum(entry['tokens'] for entry in metrics) == expected['tokens']
    del cache, weights, original, opt
    baselines = {'dev48': rows(INITIAL.parent / 'dev_seen24.jsonl') + rows(INITIAL.parent / 'dev_unseen24.jsonl'),
                 'holdout48': rows(OLDHOLD / 'e085_block1_24.jsonl') + rows(OLDHOLD / 'e085_block2_24.jsonl')}
    suites = {}; changed = []; bases = {}; config = load_config_file(str(CONFIG))
    for label, baseline in baselines.items():
        candidate = rows(run / f'{label}_block1.jsonl') + rows(run / f'{label}_block2.jsonl')
        actual = compare(baseline, candidate); saved = read(run / f'{label}_paired.json')
        assert all(saved[k] == v for k, v in actual.items())
        before = {r['gamefile']: r for r in baseline}
        for r in candidate:
            old = before[r['gamefile']]; assert r['index'] == old['index'] and r['split'] == old['split'] and r['steps'] == len(r['actions']) <= 50
            if r['won'] == old['won']: continue
            split = r['split']
            if split not in bases:
                dataset = {'train': 'train', 'seen': 'eval_in_distribution', 'unseen': 'eval_out_of_distribution'}[split]
                with contextlib.redirect_stdout(io.StringIO()): bases[split] = get_environment(config['env']['type'])(config, train_eval=dataset)
            changed.append(dict(suite=label, direction='gain' if r['won'] else 'loss', task_type=r['task_type'],
                                gamefile=r['gamefile'], before=replay(old, bases[split], config), after=replay(r, bases[split], config)))
        suites[label] = actual
        if label == 'dev48': suites[label]['keep_gate'] = comparison(candidate, baseline)['keep_gate']
    probe = read(Path('/home/codex/agentic-rl/e111_scale_A_20261015/gpu_probe.json')); assert probe['status'] == 'passed'
    cross = None
    if a.arm == 'B':
        arun = Path('/home/codex/agentic-rl/e111_scale_A_20261015')
        cross = {label: compare(rows(arun / f'{label}_block1.jsonl') + rows(arun / f'{label}_block2.jsonl'),
                              rows(run / f'{label}_block1.jsonl') + rows(run / f'{label}_block2.jsonl')) for label in baselines}
    source_cost = {}
    for g in source['results']:
        path = g['source_provenance']['source']
        if path not in source_cost:
            original_source = read(path)
            source_cost[path] = dict(used_groups=0, original_sampling_worker_seconds=original_source.get('seconds'),
                                     original_status=original_source['status'], charge_shared_source_once=True)
        source_cost[path]['used_groups'] += 1
    elapsed = round(time.monotonic() - started, 3)
    # Charge startup, final SHA reread and output I/O conservatively as well.
    inclusive = round(state['inclusive_phase_seconds'] + elapsed + 10, 3)
    result = dict(actual_postrun_review='passed', arm=a.arm, gpu_probe_status='passed',
        direction_stop=not suites['dev48']['keep_gate'] or any(s['format_errors'] or s['cap_hits'] or
                      any(k['before'] - k['after'] >= 2 for k in s['by_type'].values()) for s in suites.values()),
        promoted_over_e085=False, input_sha_count=len(frozen['input_sha256']), game_sha_count=len(frozen['game_sha256']),
        groups=frozen['groups'], group_size=8, rollouts=8 * frozen['groups'], by_type=dict(Counter(g['task_type'] for g in source['results'])),
        decisions=expected['decisions'], assistant_tokens=expected['tokens'], all_cached_advantages_verified=cursor,
        minibatch_denominators_verified=len(metrics), optimizer_steps=[134, 134 + expected['expected_updates']],
        adapter_sha256=t['adapter_sha256'], optimizer_sha256=t['optimizer_state_sha256'],
        suites=suites, paired_with_A=cross, changed_paths=changed, changed_paths_cpu_replayed=2 * len(changed),
        new_sampling_rollouts=0, source_sampling_cost=source_cost, source_cost_allocation='shared historical library; original raw full sampling time per source once, not free algorithm',
        controller_gpu_jobs_seconds=state['gpu_jobs_seconds'], update_worker_seconds=t['seconds'],
        root_worker_seconds=elapsed, root_startup_and_final_checks_charged_seconds=10, inclusive_phase_seconds=inclusive,
        family_cumulative_seconds=round(frozen['family_previous_seconds'] + inclusive, 3), family_cap_seconds=14400,
        phase_cap_seconds=frozen['phase_cap_seconds'], phase_within_cap=inclusive <= frozen['phase_cap_seconds'],
        cuda_initialized=torch.cuda.is_initialized(), review_script_sha256=sha(Path(__file__)))
    assert not torch.cuda.is_initialized() and all(sha(p) == h for p, h in frozen['input_sha256'].items())
    write(run / 'postrun_review.json', result); write(DATA / a.arm / 'root_actual_review.json', result)
    state.update(actual_postrun_review='passed', inclusive_phase_seconds=inclusive, family_cumulative_seconds=result['family_cumulative_seconds'])
    write(run / 'run_state.json', state); write(DATA / a.arm / 'run_state.json', state)
    print({k: v for k, v in result.items() if k not in ('changed_paths', 'source_sampling_cost')}, flush=True)


if __name__ == '__main__': main()
