"""CPU-only consistency checks for the ALFWorld SFT+GRPO snapshot."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1] / 'experiments/alfworld'


def load(relative):
    path = ROOT / relative
    content = gzip.decompress(path.read_bytes()).decode('utf-8') if path.suffix == '.gz' else path.read_text(encoding='utf-8')
    return json.loads(content)


def rows(relative):
    path = ROOT / relative
    text = gzip.decompress(path.read_bytes()).decode('utf-8') if path.suffix == '.gz' else path.read_text(encoding='utf-8')
    return [json.loads(line) for line in text.splitlines() if line]


def verify():
    trajectories = rows('data/expert_trajectories.jsonl.gz')
    decisions = rows('data/expert_decisions.jsonl.gz')
    assert len(trajectories) == 600 and len({r['gamefile'] for r in trajectories}) == 600
    assert all(r['accepted'] and r['won'] and r['all_actions_admissible'] for r in trajectories)
    assert len(Counter(r['task_type'] for r in trajectories)) == 6
    assert set(Counter(r['task_type'] for r in trajectories).values()) == {100}
    assert len(decisions) == 10461
    assert {r['gamefile'] for r in decisions} == {r['gamefile'] for r in trajectories}
    assert all(0 < r['prompt_tokens'] <= 2048 and 0 < r['response_tokens'] <= 512 for r in decisions)
    dev = load('data/dev_games.json')['games']
    final = load('data/final_games.json')['games']
    dev_ids, final_ids = {r['gamefile'] for r in dev}, {r['gamefile'] for r in final}
    assert len(dev_ids) == 48 and len(final_ids) == 36 and not dev_ids.intersection(final_ids)
    assert set(Counter(r['task_type'] for r in dev).values()) == {8}
    evaluated = {}
    for arm, expected in [('base', 3), ('sft', 33), ('rl_round1', 35), ('rl_round3', 37)]:
        arm_rows = rows(f'results/{arm}/dev_seen.jsonl') + rows(f'results/{arm}/dev_unseen.jsonl')
        assert {r['gamefile'] for r in arm_rows} == dev_ids and len(arm_rows) == 48
        assert sum(r['won'] for r in arm_rows) == expected
        evaluated[arm] = {r['gamefile']: r['won'] for r in arm_rows}
    assert sum(not evaluated['sft'][k] and evaluated['rl_round3'][k] for k in dev_ids) == 4
    assert sum(evaluated['sft'][k] and not evaluated['rl_round3'][k] for k in dev_ids) == 0
    accepted = set()
    candidate_count = 0
    decisions_count = 0
    for n in range(1, 4):
        batch = load(f'data/rl_round{n}_rollouts.json.gz')
        assert batch['status'] == 'complete' and batch['groups'] == 16 and batch['rollouts'] == 128
        assert batch['source_accepted'] == {'double': 8, 'look': 4, 'maintenance': 4}
        ids = {r['gamefile'] for r in batch['results']}
        assert len(ids) == 16 and not ids.intersection(accepted | dev_ids | final_ids)
        accepted.update(ids)
        candidate_count += batch['candidate_rollouts']
        for group in batch['results']:
            assert 0 < group['successes'] < 8 and len(group['records']) == 8
            for trajectory in group['records']:
                decisions_count += len(trajectory)
                assert all(len(s['completion_ids']) == len(s['rollout_logps']) > 0 for s in trajectory)
    assert len(accepted) == 48 and candidate_count == 872 and decisions_count == 13358
    for arm, expected in [('base', 1), ('sft', 24), ('rl', 24)]:
        independent = rows(f'results/independent/{arm}_final_seen.jsonl') + rows(f'results/independent/{arm}_final_unseen.jsonl')
        assert len(independent) == 36 and {r['gamefile'] for r in independent} == final_ids
        assert sum(r['won'] for r in independent) == expected
    provenance = load('PROVENANCE.json')['files']
    project_root = ROOT.parents[1]
    if (project_root / '.git').exists():
        tracked = set(subprocess.check_output(['git', 'ls-files', '-z'], cwd=project_root).decode().split('\0'))
        missing = [name for name in provenance if 'experiments/alfworld/' + name not in tracked]
        if missing:
            raise ValueError('Snapshot files present locally but absent from Git: ' + ', '.join(missing))
    for filename, meta in provenance.items():
        assert hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == meta['distributed_sha256'], filename
    return {'passed': True, 'expert_trajectories': 600, 'sft_decisions': 10461,
            'dev_scores': {'base': 3, 'sft': 33, 'rl_round1': 35, 'rl_round3': 37},
            'dev_games': 48, 'accepted_rl_rollouts': 384, 'candidate_rl_rollouts': 872,
            'independent_final36': {'base': 1, 'sft': 24, 'rl': 24},
            'verification': 'saved data and outcome consistency; no new model inference or environment execution'}


if __name__ == '__main__':
    from verify_alfworld_stage_release import verify as verify_latest
    report = verify()
    report['latest_release'] = verify_latest()
    print(json.dumps(report, ensure_ascii=False, indent=2))
