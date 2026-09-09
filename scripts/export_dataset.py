"""Reconstruct readable E017 trajectories and check all 1,129 tokenized targets.

Public dataset derived from Gorilla/BFCL (Apache-2.0). This conversion adds
history/target fields, per-message loss masks, and stable task-family IDs.
The reference actions and executed observations are unchanged.
"""
import argparse
import copy
import gzip
import json
from pathlib import Path

from common import ROOT, file_hash, read_json, read_jsonl, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=ROOT/'models/Qwen3-1.7B')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    from transformers import AutoTokenizer
    import prepare_e017_bfcl_gold as replay
    data = ROOT/'data/E017_bfcl_indomain'
    manifest = read_json(data/'manifest.json')
    with gzip.open(data/'train_R.jsonl.gz', 'rt', encoding='utf-8') as f:
        originals = [json.loads(line) for line in f]
    expected = {row['id']: row for row in originals}
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if file_hash(args.model/'tokenizer.json') != manifest['model']['tokenizer_sha256']:
        raise ValueError('Use the pinned Qwen3-1.7B tokenizer')
    raw, gold = {r['id']: r for r in read_jsonl(replay.TASK_PATH)}, {
        r['id']: r['ground_truth'] for r in read_jsonl(replay.ANSWER_PATH)}
    decisions, trajectories = [], []
    original_encoder = replay.encode_row

    def capture(tokenizer, row):
        encoded = original_encoder(tokenizer, row)
        if encoded != expected[row['id']]:
            raise ValueError(f'Replay/token mismatch: {row["id"]}')
        decisions.append(copy.deepcopy(row))
        return encoded

    replay.encode_row = capture
    for index, task_id in enumerate(manifest['task_ids']['train'], 1):
        _, report = replay.replay_task(raw[task_id], gold[task_id], tokenizer, manifest['training']['seed'])
        if not report['valid']:
            raise ValueError(f'Official replay failed: {task_id}')
        last = decisions[-1]
        messages = last['history'] + [last['candidate_action']]
        trajectories.append({'id':task_id, 'split':'train', 'tools':last['tools'],
                             'messages':messages,
                             'assistant_loss_mask':[m['role']=='assistant' for m in messages],
                             'reference_replay_passed':True})
        if index % 30 == 0:
            print(f'Reconstructed {index}/120; all encoded targets match', flush=True)
    if len(decisions) != 1129 or [r['id'] for r in decisions] != [r['id'] for r in originals]:
        raise ValueError('Decision count/order mismatch')
    if not args.verify_only:
        for name, rows in [('train_decisions.jsonl.gz', decisions), ('train_trajectories.jsonl.gz', trajectories)]:
            blob = ''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows).encode('utf-8')
            (ROOT/'data'/name).write_bytes(gzip.compress(blob, mtime=0))
        example = next(r for r in decisions if r['target_type']=='tool_call' and any(m['role']=='tool' for m in r['history']))
        write_json(ROOT/'data/example.json', example)
        write_json(ROOT/'data/export_verification.json', {
            'trajectories':120, 'decisions':1129, 'exact_encoded_matches':1129,
            'official_replay_passes':120, 'source':'BFCL multi_turn_base reference paths',
            'transform':'Replay reference calls, preserve complete history, add per-user-turn stop target; no error-target filtering in E017',
            'tokenizer_sha256':file_hash(args.model/'tokenizer.json')})
    print('PASS: 120 complete trajectories; 1,129 byte-equivalent encoded target records.')


if __name__ == '__main__':
    main()
