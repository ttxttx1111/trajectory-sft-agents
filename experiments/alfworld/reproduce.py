"""Portable entry points around the published E068 source snapshot."""
import argparse
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / 'vendor/verl-agent'


def materialize(value, dataset):
    if isinstance(value, dict):
        return {k: materialize(v, dataset) for k, v in value.items()}
    if isinstance(value, list):
        return [materialize(v, dataset) for v in value]
    if isinstance(value, str) and value.startswith('json_2.1.1/'):
        return str(dataset / value)
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['prepare', 'train-sft', 'train-rl', 'evaluate'])
    p.add_argument('--run', required=True, type=Path, help='Fresh output directory, separate from published evidence')
    p.add_argument('--model', type=Path, help='Local pinned Base model for SFT, or merged SFT model for RL/evaluation')
    p.add_argument('--adapter', type=Path, help='Optional unmerged RL adapter for evaluation')
    p.add_argument('--split', choices=['seen', 'unseen'], default='seen')
    p.add_argument('--rounds', type=int, choices=[1, 2, 3], default=3)
    p.add_argument('--limit', type=int, default=0, help='Evaluation smoke test only; 0 runs the full frozen split')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    run = args.run.resolve()
    if run == ROOT or ROOT / 'data' in run.parents or ROOT / 'results' in run.parents:
        raise ValueError('Run output must not overwrite released data or evidence')
    dataset = Path(os.environ.get('ALFWORLD_DATA', str(Path.home() / '.cache/alfworld'))).resolve()
    env = dict(os.environ)
    env['ALFWORLD_DATA'] = str(dataset)
    env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env.get('PATH', '')
    env['PYTHONPATH'] = os.pathsep.join([str(RUNTIME), str(ROOT / 'scripts'), env.get('PYTHONPATH', '')])
    if args.stage == 'prepare':
        run.mkdir(parents=True, exist_ok=True)
        for source in ['train_pool.json', 'dev_games.json', 'final_games.json']:
            target = run / source
            if target.exists():
                raise FileExistsError(target)
            value = json.loads((ROOT / 'data' / source).read_text(encoding='utf-8'))
            target.write_text(json.dumps(materialize(value, dataset), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({'prepared': str(run), 'dataset': str(dataset)}))
        return
    if not args.model or not args.model.is_dir():
        raise ValueError('--model must name an existing local model directory')
    model = args.model.resolve()
    if args.stage == 'train-sft':
        command = [sys.executable, str(ROOT / 'scripts/e066_train_alfworld_sft.py'),
                   '--data', str(ROOT / 'data/expert_decisions_h2.parquet'),
                   '--model', str(model), '--output', str(run / 'sft'),
                   '--epochs', '1', '--effective-batch', '8', '--lr', '1e-4']
    elif args.stage == 'train-rl':
        env['SFT_MODEL'] = str(model)
        command = ['bash', str(ROOT / 'scripts/e068_run_coverage.sh'),
                   str(run / 'rl'), str(run / 'train_pool.json'), str(args.rounds)]
    else:
        command = [sys.executable, str(ROOT / 'scripts/e067_eval_fixed_games.py'),
                   '--model', str(model), '--games', str(run / 'dev_games.json'),
                   '--split', args.split, '--output', str(run / f'eval_{args.split}.jsonl')]
        if args.adapter:
            command += ['--adapter', str(args.adapter.resolve())]
        if args.limit:
            if args.limit < 1:
                raise ValueError('--limit must be positive')
            command += ['--limit', str(args.limit)]
    print(json.dumps({'command': command, 'cwd': str(RUNTIME)}, ensure_ascii=False), flush=True)
    if not args.dry_run:
        subprocess.run(command, cwd=RUNTIME, env=env, check=True)


if __name__ == '__main__':
    main()
