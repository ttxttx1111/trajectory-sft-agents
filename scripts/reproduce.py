"""Public E017 entry point. New outputs are isolated under runs/<name>."""
import argparse
import contextlib
import gzip
import hashlib
import io
import json
import re
import urllib.request
import zipfile
from pathlib import Path

from common import ROOT, file_hash, read_json, write_json


def run_path(name):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', name):
        raise ValueError('Run name must be a simple name, using letters, numbers, _ or -')
    return ROOT/'runs'/name


def unpack_training():
    folder = ROOT/'data/E017_bfcl_indomain'
    blob = gzip.decompress((folder/'train_R.jsonl.gz').read_bytes())
    expected = read_json(folder/'training_data_manifest.json')['train_sha256']
    if hashlib.sha256(blob).hexdigest() != expected:
        raise ValueError('Training dataset checksum mismatch')
    path = folder/'train_R.jsonl'
    if path.exists() and file_hash(path) != expected:
        raise ValueError('An existing decompressed dataset has changed; refusing to overwrite it')
    if not path.exists():
        path.write_bytes(blob)


def download(args):
    if args.only != 'adapter':
        from huggingface_hub import snapshot_download
        source = read_json(ROOT/'configs/sources.lock.json')['model']
        snapshot_download(repo_id=source['repo'], revision=source['revision'], local_dir=args.model)
    if args.only != 'model':
        lock = read_json(ROOT/'configs/adapter.lock.json')
        target = ROOT/'models/e017'
        if target.exists():
            if all(file_hash(target/name) == lock[key] for name,key in
                   [('adapter_model.safetensors','weight_sha256'), ('adapter_config.json','config_sha256')]):
                print('Released adapter already present and verified.'); return
            raise ValueError('Existing adapter differs from release; refusing to overwrite it')
        target.parent.mkdir(parents=True, exist_ok=True)
        archive = target.parent/'e017.download.partial'
        urllib.request.urlretrieve(lock['url'], archive)
        if file_hash(archive) != lock['archive_sha256']:
            raise ValueError('Adapter download checksum mismatch')
        with zipfile.ZipFile(archive) as z:
            expected = {'adapter_model.safetensors','adapter_config.json','LICENSE'}
            if set(z.namelist()) != expected:
                raise ValueError('Unexpected archive contents')
            target.mkdir()
            for name in sorted(expected):
                (target/name).write_bytes(z.read(name))
        archive.unlink()
        print('Released adapter downloaded and verified.')


def runtime(args):
    import evaluate_e017_bfcl as original
    original.MODEL = args.model.resolve()
    original.OUT = run_path(args.run)
    # Frozen source, tokenizer, config, source tasks, and answers are checked here.
    manifest = original.load_manifest()
    return original, manifest


def evaluate(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from bfcl_research_adapter import Limits, TransformersGenerator, evaluate_task
    original, manifest = runtime(args)
    all_ids = manifest['task_ids'][args.split]
    if args.limit is not None and not 1 <= args.limit <= len(all_ids):
        raise ValueError('--limit must be between 1 and 40')
    ids = all_ids[:args.limit] if args.limit is not None else all_ids
    directory = original.OUT/args.arm
    if (directory/f'{args.split}.jsonl').exists() or (directory/f'{args.split}_run.json').exists():
        raise FileExistsError('This arm/split already has output; use a new --run name')
    adapter = args.adapter
    if args.arm == 'tuned' and adapter is None:
        adapter = original.OUT/'adapter' if (original.OUT/'adapter').exists() else ROOT/'models/e017'
    if adapter is not None and args.arm == 'tuned':
        if not (adapter/'adapter_model.safetensors').exists():
            raise FileNotFoundError('Adapter missing. Run download --only adapter, or pass --adapter')
    # Record model shard hashes too: matching config alone does not identify weights.
    weights = sorted(args.model.glob('*.safetensors'))
    if not weights:
        raise FileNotFoundError('Base model safetensors missing; download the pinned model first')
    provenance = {'ids':ids, 'partial':len(ids)!=len(all_ids), 'arm':args.arm,
        'split':args.split, 'manifest_sha256':file_hash(original.DATA/'manifest.json'),
        'model_weight_sha256':{p.name:file_hash(p) for p in weights},
        'adapter_sha256':file_hash(adapter/'adapter_model.safetensors') if args.arm=='tuned' else None,
        'limits':manifest['evaluation']['limits'], 'seed':manifest['training']['seed'],
        'runner_sha256':file_hash(Path(__file__)), 'torch_version':torch.__version__}
    write_json(directory/f'{args.split}_run.json', provenance)
    torch.cuda.set_per_process_memory_fraction(0.80, 0)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
        device_map={'':0}, dtype=torch.bfloat16, attn_implementation='sdpa', low_cpu_mem_usage=True)
    if args.arm == 'tuned':
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    model.eval()
    limits = Limits(**manifest['evaluation']['limits'])
    generator = TransformersGenerator(model, tokenizer, limits)
    raw, gold = original.source_rows()
    done = []
    for task_id in ids:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                row = evaluate_task(raw[task_id], gold[task_id], generator,
                                    limits=limits, seed=manifest['training']['seed'])
        except Exception as exc:
            write_json(directory/'infrastructure_error.json', {'id':task_id,'error':str(exc)})
            raise
        row['arm'], row['split'] = args.arm, args.split
        original.append_jsonl(directory/f'{args.split}.jsonl', row)
        done.append(row)
        print(json.dumps({'completed':len(done),'n':len(ids),'successes':sum(r['success'] for r in done),
                          'last_id':task_id,'last_seconds':round(row['seconds'],2)}), flush=True)
    summary = {'experiment':manifest['experiment'], 'arm':args.arm, 'split':args.split,
        'result':original.summarize(done), 'partial':provenance['partial'],
        'parser_unit_gate':original.parser_unit_gate(), 'manifest_sha256':provenance['manifest_sha256'],
        'script_sha256':file_hash(Path(original.__file__)), 'public_runner':True}
    write_json(directory/f'{args.split}_summary.json', summary)
    print(json.dumps(summary, indent=2))


def train(args):
    original, manifest = runtime(args)
    if not (original.OUT/'base/dev_summary.json').exists():
        raise ValueError('Evaluate Base on all 40 dev tasks using the same --run before training')
    baseline = read_json(original.OUT/'base/dev_run.json')
    weights = sorted(args.model.glob('*.safetensors'))
    if baseline['partial'] or baseline['model_weight_sha256'] != {p.name:file_hash(p) for p in weights}:
        raise ValueError('Training requires a complete Base dev evaluation of these exact model weights')
    import gate_e017_base as gate
    import train_e017_qwen3_17b as training
    gate.OUT = training.OUT = original.OUT
    training.MODEL = args.model.resolve()
    # Cheap executable oracle check + complete baseline gate precede GPU training.
    original.oracle_gate(manifest)
    gate.main()
    unpack_training()
    manifest, data_manifest, rows, path = training.verify_inputs()
    if args.preflight:
        training.preflight(manifest, rows)
    else:
        training.train(manifest, data_manifest, rows, path, original.OUT/'adapter')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('download', help='Fetch pinned base weights and released LoRA')
    p.add_argument('--only', choices=['model','adapter'])
    p.add_argument('--model', type=Path, default=ROOT/'models/Qwen3-1.7B')
    for command in ('evaluate','train'):
        p = sub.add_parser(command)
        p.add_argument('--run', required=True, help='New output directory name under runs/')
        p.add_argument('--model', type=Path, default=ROOT/'models/Qwen3-1.7B')
        if command == 'evaluate':
            p.add_argument('--arm', choices=['base','tuned'], required=True)
            p.add_argument('--split', choices=['dev','test'], default='dev')
            p.add_argument('--limit', type=int, help='First N tasks, for an inexpensive smoke test')
            p.add_argument('--adapter', type=Path)
        else:
            p.add_argument('--preflight', action='store_true', help='Longest two samples: forward/backward/update only')
    args = parser.parse_args()
    {'download':download, 'evaluate':evaluate, 'train':train}[args.command](args)


if __name__ == '__main__':
    main()
