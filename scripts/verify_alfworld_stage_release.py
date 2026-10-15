"""Check v0.3.0 saved results, training batches, source identity and isolation, on CPU."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

ALF = Path(__file__).resolve().parents[1] / 'experiments/alfworld'
ROOT = ALF / 'releases/v0.3.0'


def load(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]


def combine(directory):
    return sum([rows(directory / f'dev_{split}.jsonl') for split in ('seen', 'unseen')], [])


def counts(records):
    return {t: {'games': sum(r['task_type'] == t for r in records),
                'successes': sum(r['task_type'] == t and bool(r['won']) for r in records)}
            for t in sorted({r['task_type'] for r in records})}


def verify():
    summary = load(ROOT / 'summary.json')
    assert summary['evaluation']['role'] == 'development'
    assert summary['evaluation']['games'] == 48
    manifests = load(ROOT / 'PROVENANCE.json')['files']
    for name, meta in manifests.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == meta['distributed_sha256'], name
    dev_ids = {r['gamefile'] for r in load(ALF / 'data/dev_games.json')['games']}
    expert_ids = {json.loads(l)['gamefile'] for l in gzip.decompress((ALF / 'data/expert_trajectories.jsonl.gz').read_bytes()).splitlines()}
    policies = {}
    for model, score in [('base', 3), ('sft', 33), ('rl_round3', 37), ('previous', 43),
                         ('current', 43), ('scale_B', 43), ('scale_C', 43)]:
        records = combine((ALF if model in ('base', 'sft', 'rl_round3') else ROOT) / 'results' / model)
        assert len(records) == 48 and {r['gamefile'] for r in records} == dev_ids
        assert sum(bool(r['won']) for r in records) == score
        assert counts(records) == summary['scores'][model]['by_task_type']
        assert summary['scores'][model]['successes'] == score
        assert summary['scores'][model]['success_rate_percent'] == 100 * score / 48
        policies[model] = {r['gamefile']: bool(r['won']) for r in records}
    for before, name in [('sft', 'paired_sft_to_current'), ('rl_round3', 'paired_e068_to_current')]:
        actual = {'improved': sum(not policies[before][g] and policies['current'][g] for g in dev_ids),
                  'regressed': sum(policies[before][g] and not policies['current'][g] for g in dev_ids)}
        assert actual == summary[name]
    batches = {}
    for arm, groups, size, decisions, tokens, steps in [('A', 16, 8, 3919, 111241, 16),
                                                       ('B', 32, 8, 7870, 222999, 31),
                                                       ('C', 16, 16, 7558, 214860, 30)]:
        batch = load(ROOT / f'data/rollouts_{arm}.json.gz')
        assert batch['groups'] == groups and batch['rollouts'] == groups * size
        assert batch.get('group_size', 8) == size
        game_ids = {g['gamefile'] for g in batch['results']}
        assert len(game_ids) == groups and not game_ids.intersection(dev_ids | expert_ids)
        assert all(g.startswith('json_2.1.1/train/') for g in game_ids)
        actual_decisions = actual_tokens = 0
        for group in batch['results']:
            assert len(group['records']) == len(group['won']) == size
            assert 0 < sum(group['won']) == group['successes'] < size
            assert sum(group['response_cap_hits']) == 0
            for trajectory, length in zip(group['records'], group['steps']):
                assert len(trajectory) == length <= 50
                for step in trajectory:
                    assert 0 < len(step['prompt_ids']) <= 2048
                    assert 0 < len(step['completion_ids']) == len(step['rollout_logps']) <= 512
                    actual_decisions += 1
                    actual_tokens += len(step['completion_ids'])
        assert (actual_decisions, actual_tokens) == (decisions, tokens)
        assert summary['scale'][arm] == dict(groups=groups, group_size=size, rollouts=groups * size,
                                              decisions=decisions, assistant_tokens=tokens, optimizer_updates=steps)
        report = load(ROOT / f'results/training_{arm}.json')
        assert report['optimizer_updates'] == steps and report['minibatch_decisions'] == 256
        assert report['objective'] == 'vendor_dynamic_grpo' and report['kl_beta'] == .01
        assert report['loss_reduction'] == 'global_token'
        assert report['clip_epsilon'] == .2 and report['clip_epsilon_high'] == .28
        assert report['rollout_vs_recomputed_mean_abs'] <= .05
        assert load(ROOT / f'results/audit_{arm}.json')['actual_postrun_review'] == 'passed'
        batches[arm] = batch
    larger = {g['gamefile']: g for g in batches['B']['results']}
    assert all(group == larger[group['gamefile']] for group in batches['A']['results'])
    for a, c in zip(batches['A']['results'], batches['C']['results']):
        assert a['gamefile'] == c['gamefile'] and a['records'] == c['records'][:8]
    development98 = []
    for file in (ROOT / 'results/current').glob('candidate_*.jsonl'):
        development98 += rows(file)
    assert len(development98) == 98 and sum(r['won'] for r in development98) == 87
    assert len({r['gamefile'] for r in development98}) == 98
    assert not dev_ids.intersection(r['gamefile'] for r in development98)
    assert counts(development98) == summary['development98']['by_task_type']
    for arm, expected in [('A', 46), ('B', 46), ('C', 45)]:
        records = sum([rows(ROOT / f'results/{arm}/train_heldout_block{i}.jsonl') for i in (1, 2)], [])
        ids = {r['gamefile'] for r in records}
        assert len(ids) == len(records) == 48 and sum(r['won'] for r in records) == expected
        assert not ids.intersection(dev_ids | expert_ids | {r['gamefile'] for r in development98})
        assert not ids.intersection(g['gamefile'] for batch in batches.values() for g in batch['results'])
        assert set(Counter(r['task_type'] for r in records).values()) == {8}
    lock = load(ROOT / 'models.lock.json')
    packaging = load(ROOT / 'results/model_packaging_audit.json')
    assert packaging['corrected_merged_tensors_equal'] and not packaging['cuda_initialized']
    assert packaging['model_forward_calls'] == packaging['optimizer_updates'] == 0
    assert packaging['corrected_serialization_sha256'] == lock['merged_sft_safetensors_sha256']
    assert packaging['correction_file_sha256'] == lock['assets']['alfworld-inference-adapters.zip']['files_sha256']['sft/merge_corrections.safetensors']
    entry = load(ROOT / 'results/entrypoint_merge_audit.json')
    assert entry['status'] == 'passed' and entry['cpu_only'] and entry['matches_recorded_sft']
    assert entry['model_forward_calls'] == entry['optimizer_updates'] == 0
    assert entry['safetensors_sha256'] == lock['merged_sft_safetensors_sha256']
    assert lock['assets']['alfworld-inference-adapters.zip']['files_sha256']['rl/adapter_model.safetensors'] == summary['weight_identity']['adapter_sha256']
    assert lock['assets']['alfworld-update-input.zip']['files_sha256']['input_checkpoint/adapter/adapter_model.safetensors'] == batches['A']['rollout_adapter_sha256']
    assert load(ROOT / 'results/audit_development98.json')['actual_postrun_review'] == 'passed'
    return {'passed': True, 'version': '0.3.0', 'dev_scores': {'base': 3, 'sft': 33, 'current_grpo': 43},
            'dev_games': 48, 'development98_successes': 87, 'scale_B_G': ['16x8', '32x8', '16x16'],
            'verification': 'saved evidence only; no model inference, environment rollout or training'}


if __name__ == '__main__':
    print(json.dumps(verify(), ensure_ascii=False, indent=2))
