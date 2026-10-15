"""Reproduce released checkpoint evaluation or a saved GRPO update, in fresh outputs."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
ALF = ROOT.parents[1]
RUNTIME = ALF / 'vendor/verl-agent'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def check_output(path):
    """Published code, datasets and evidence cannot be an output directory."""
    path = path.resolve()
    if path == ALF or path in ALF.parents or (ALF in path.parents and ALF / 'runs' not in path.parents):
        raise ValueError('Use a fresh runs/<name> directory or a directory outside the repository.')
    repository = ALF.parents[1]
    if path == repository or (repository in path.parents and repository / 'runs' not in path.parents
                              and ALF / 'runs' not in path.parents):
        raise ValueError('Output would overlap published repository files.')
    return path


def materialize(value, dataset):
    if isinstance(value, dict):
        return {k: materialize(v, dataset) for k, v in value.items()}
    if isinstance(value, list):
        return [materialize(v, dataset) for v in value]
    if isinstance(value, str) and value.startswith('json_2.1.1/'):
        return str(dataset / value)
    return value


def verify_assets(assets, name):
    lock = read(ROOT / 'models.lock.json')
    names = {'inference': 'alfworld-inference-adapters.zip', 'update': 'alfworld-update-input.zip'}
    for relative, expected in lock['assets'][names[name]]['files_sha256'].items():
        actual = assets / relative
        if not actual.is_file() or sha(actual) != expected:
            raise ValueError('Missing or changed released model asset: ' + str(actual))


def prepare(run, dataset, arm, initial_adapter=None):
    run.mkdir(parents=True, exist_ok=True)
    target = run / 'dev_games.json'
    if target.exists():
        raise FileExistsError('Do not overwrite a prepared run: ' + str(run))
    value = materialize(read(ALF / 'data/dev_games.json'), dataset)
    target.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    if initial_adapter:
        batch = materialize(json.loads(gzip.decompress((ROOT / f'data/rollouts_{arm}.json.gz').read_bytes())), dataset)
        batch['rollout_adapter'] = str(initial_adapter)
        (run / 'rollouts.json').write_text(json.dumps(batch, ensure_ascii=False) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'merge-sft', 'evaluate', 'update-from-snapshot'])
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--assets', type=Path, help='Directory containing the extracted release assets')
    parser.add_argument('--base', type=Path, help='Pinned original Qwen model, for merge-sft')
    parser.add_argument('--model', type=Path, help='Merged SFT model, for evaluate/update')
    parser.add_argument('--policy', choices=['base', 'sft', 'rl'], default='rl')
    parser.add_argument('--split', choices=['seen', 'unseen'], default='seen')
    parser.add_argument('--arm', choices=['A', 'B', 'C'], default='A')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    run = check_output(args.run)
    dataset = Path(os.environ.get('ALFWORLD_DATA', str(Path.home() / '.cache/alfworld'))).resolve()
    assets = args.assets.resolve() if args.assets else None
    if args.limit < 0:
        raise ValueError('--limit must not be negative')
    if args.stage == 'prepare':
        if not args.dry_run:
            if assets:
                verify_assets(assets, 'update')
            prepare(run, dataset, args.arm, assets / 'input_checkpoint/adapter' if assets else None)
        print(json.dumps({'prepared_run': str(run), 'arm': args.arm, 'dry_run': args.dry_run}))
        return
    if args.stage == 'merge-sft':
        if not args.base or not assets:
            raise ValueError('merge-sft requires --base and --assets')
        base = args.base.resolve()
        output = run / 'merged_sft'
        if output.exists():
            raise FileExistsError(output)
        if args.dry_run:
            print(json.dumps({'base': str(base), 'sft_adapter': str(assets / 'sft'), 'output': str(output)}))
            return
        verify_assets(assets, 'inference')
        lock = read(ROOT / 'models.lock.json')
        if sha(base / 'model.safetensors') != lock['base_safetensors_sha256']:
            raise ValueError('Original Qwen model differs from the pinned base')
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(4)
        model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map='cpu')
        model = PeftModel.from_pretrained(model, assets / 'sft').merge_and_unload()
        # Preserve the originally evaluated GPU-merged values at CPU rounding differences.
        # This is saved checkpoint data, not a new optimization or model modification.
        from safetensors.torch import load_file
        corrections = load_file(assets / 'sft/merge_corrections.safetensors', device='cpu')
        state = model.state_dict()
        with torch.no_grad():
            for key in corrections:
                if key.endswith('.indices'):
                    name = key[:-len('.indices')]
                    state[name].reshape(-1)[corrections[key].long()] = corrections[name + '.values']
        model.config.use_cache = True
        model.save_pretrained(output, safe_serialization=True)
        AutoTokenizer.from_pretrained(base).save_pretrained(output)
        actual = sha(output / 'model.safetensors')
        print(json.dumps({'merged_sft': str(output), 'safetensors_sha256': actual,
                          'matches_recorded_sft': actual == lock['merged_sft_safetensors_sha256']}))
        if actual != lock['merged_sft_safetensors_sha256']:
            raise ValueError('Merged SFT serialization differs; preserve output and check pinned software before training')
        return
    if not args.model:
        raise ValueError('--model is required')
    model = args.model.resolve()
    env = dict(os.environ)
    env['ALFWORLD_DATA'] = str(dataset)
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT / 'scripts'), str(ALF / 'scripts'), str(RUNTIME), env.get('PYTHONPATH', '')])
    if args.stage == 'evaluate':
        command = [sys.executable, str(ALF / 'scripts/e067_eval_fixed_games.py'), '--model', str(model),
                   '--games', str(run / 'dev_games.json'), '--split', args.split,
                   '--output', str(run / f'{args.policy}_{args.split}.jsonl')]
        if args.policy == 'rl':
            if not assets:
                raise ValueError('RL evaluation requires --assets')
            command += ['--adapter', str(assets / 'rl')]
        if args.limit:
            command += ['--limit', str(args.limit)]
        if not args.dry_run:
            if (run / f'{args.policy}_{args.split}.jsonl').exists():
                raise FileExistsError('Refusing to overwrite previous evaluation output')
            lock = read(ROOT / 'models.lock.json')
            expected = lock['base_safetensors_sha256' if args.policy == 'base' else 'merged_sft_safetensors_sha256']
            if sha(model / 'model.safetensors') != expected:
                raise ValueError('Evaluation model differs from the selected policy base')
            if args.policy == 'rl':
                verify_assets(assets, 'inference')
    else:
        if not assets:
            raise ValueError('update-from-snapshot requires --assets')
        initial = assets / 'input_checkpoint'
        group_size = 16 if args.arm == 'C' else 8
        command = [sys.executable, str(ROOT / 'scripts/e110_update_group_size.py'),
                   '--rollouts', str(run / 'rollouts.json'), '--group-size', str(group_size),
                   '--model', str(model), '--train-base-model', str(model), '--reference-model', str(model),
                   '--resume-adapter', str(initial / 'adapter'), '--rollout-adapter', str(initial / 'adapter'),
                   '--optimizer-state', str(initial / 'optimizer_state.pt'), '--output', str(run / 'update'),
                   '--objective', 'vendor_dynamic_grpo', '--updates', '1', '--lr', '1e-5',
                   '--clip-epsilon', '0.2', '--clip-epsilon-high', '0.28', '--kl-beta', '0.01',
                   '--loss-reduction', 'global_token', '--minibatch-decisions', '256',
                   '--checkpoint-every', '4', '--prescreen-rollouts', '0', '--additional-initial-probe-rollouts', '0']
        if not args.dry_run:
            verify_assets(assets, 'update')
            if sha(model / 'model.safetensors') != read(ROOT / 'models.lock.json')['merged_sft_safetensors_sha256']:
                raise ValueError('Merged SFT differs from the original frozen reference')
            batch = read(run / 'rollouts.json')
            frozen = materialize(json.loads(gzip.decompress((ROOT / f'data/rollouts_{args.arm}.json.gz').read_bytes())), dataset)
            frozen['rollout_adapter'] = str(initial / 'adapter')
            if batch != frozen:
                raise ValueError('Prepared batch differs from the released rollout data')
    print(json.dumps({'command': command, 'cwd': str(RUNTIME), 'dry_run': args.dry_run}), flush=True)
    if not args.dry_run:
        subprocess.run(command, cwd=RUNTIME, env=env, check=True)


if __name__ == '__main__':
    main()
