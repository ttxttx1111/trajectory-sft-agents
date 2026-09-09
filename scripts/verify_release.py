"""Verify data, original source hashes, task splits, paired scores, and optional official replays."""
import argparse
import contextlib
import copy
import gzip
import hashlib
import io
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT/path).read_text(encoding='utf-8'))


def rows(path):
    p = ROOT/path
    opener = gzip.open if p.suffix == '.gz' else open
    with opener(p, 'rt', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def verify(replay=False):
    m = read('data/E017_bfcl_indomain/manifest.json')
    dm = read('data/E017_bfcl_indomain/training_data_manifest.json')
    t = rows('data/E017_bfcl_indomain/train_R.jsonl.gz')
    require(hashlib.sha256(gzip.decompress((ROOT/'data/E017_bfcl_indomain/train_R.jsonl.gz').read_bytes())).hexdigest()==dm['train_sha256'], 'Encoded dataset hash')
    require(len(t)==1129 and len({r['id'] for r in t})==1129, 'Unique 1129 targets')
    require(sum(r['target_type']=='tool_call' for r in t)==685, '685 tool calls')
    require(sum(r['target_type']=='terminate' for r in t)==444, '444 stop targets')
    require(all(0<r['target_start']<len(r['input_ids'])==r['full_tokens'] and r['target_tokens']==len(r['input_ids'])-r['target_start'] for r in t), 'Loss boundaries')
    require(max(r['full_tokens'] for r in t)==5745, 'Maximum training length')
    sets = {s:set(ids) for s,ids in m['task_ids'].items()}
    require([len(sets[s]) for s in ('train','dev','test')]==[120,40,40], '120/40/40 split')
    require(len(set.union(*sets.values()))==200, 'Disjoint task families')
    require({r['task_id'] for r in t}==sets['train'], 'Training IDs only')
    for group in m['component_groups']:
        require(sum(bool({f'multi_turn_base_{i}' for i in group}&v) for v in sets.values())==1, 'Component leakage')
    for name, expected in m['script_hashes'].items():
        require(hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest()==expected, f'Original source hash: {name}')
    upstream = read('vendor/gorilla/UPSTREAM.json')
    for name, hashes in upstream['files'].items():
        content=(ROOT/'vendor/gorilla'/name).read_bytes()
        require(hashlib.sha256(content).hexdigest()==hashes['distributed_sha256'], f'Upstream hash: {name}')
        require(hashlib.sha256(content.replace(b'\r\n',b'\n')).hexdigest()==hashes['git_blob_sha256'], f'Upstream content: {name}')
    tr = rows('data/train_trajectories.jsonl.gz')
    de = rows('data/train_decisions.jsonl.gz')
    require(len(tr)==120 and {r['id'] for r in tr}==sets['train'], 'Readable train trajectories')
    require([r['id'] for r in de]==[r['id'] for r in t], 'Readable decisions order')
    require(all(r['assistant_loss_mask']==[x['role']=='assistant' for x in r['messages']] for r in tr), 'Trajectory loss masks')
    from common import digest
    for readable, encoded in zip(de,t):
        require(digest(readable['history'])==encoded['history_digest'], f'History alignment: {readable["id"]}')
        require(digest(readable['tools'])==encoded['tools_digest'], f'Tools alignment: {readable["id"]}')
    for trajectory in tr:
        decisions=[r for r in de if r['task_id']==trajectory['id']]
        last=decisions[-1]
        require(trajectory['messages']==last['history']+[last['candidate_action']], 'Complete trajectory alignment')
        require(trajectory['tools']==last['tools'], 'Trajectory tools alignment')
    clean=read('data/quality_clean.json')
    scores={}
    for split,expected in [('dev',(5,18)),('test',(1,13))]:
        b=rows(f'results/reference/base/{split}.jsonl'); a=rows(f'results/reference/tuned/{split}.jsonl')
        require([r['id'] for r in a]==[r['id'] for r in b]==m['task_ids'][split], f'{split} paired IDs')
        require((sum(r['success'] for r in b),sum(r['success'] for r in a))==expected, f'{split} scores')
        if split=='test':
            require(sum(y['success'] and not x['success'] for x,y in zip(b,a))==12, '12 wins')
            require(sum(x['success'] and not y['success'] for x,y in zip(b,a))==0, '0 regressions')
        ci=set(clean[split]['clean_ids'])
        scores[split]={'base':expected[0],'tuned':expected[1],'n':40,
                       'clean_base':sum(r['success'] for r in b if r['id'] in ci),
                       'clean_tuned':sum(r['success'] for r in a if r['id'] in ci),'clean_n':len(ci)}
    require((scores['test']['clean_base'],scores['test']['clean_tuned'],scores['test']['clean_n'])==(1,10,30),'Clean sensitivity result')
    # Recompute statistics using the original analysis implementation (stdlib only).
    from analyze_e017_results import cluster_bootstrap, mcnemar_exact
    for split in ('dev','test'):
        b={r['id']:r['success'] for r in rows(f'results/reference/base/{split}.jsonl')}
        a={r['id']:r['success'] for r in rows(f'results/reference/tuned/{split}.jsonl')}
        published=read('results/reference/analysis.json')['splits'][split]['paired']
        require(cluster_bootstrap(b,a,m,split)==published['component_cluster_bootstrap_95_ci_pp'], 'Cluster bootstrap')
        require(mcnemar_exact(sum(b[k] and not a[k] for k in b),sum(a[k] and not b[k] for k in b))==published['mcnemar_exact_two_sided_p'], 'McNemar')
    checksum_path=ROOT/'checksums.json'
    if checksum_path.exists():
        for name,h in read('checksums.json')['files'].items():
            require(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h, f'Release checksum: {name}')
    checked=0
    if replay:
        from common import BFCL_ROOT
        from bfcl_research_adapter import prepare_task, clear_run_instances
        from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import multi_turn_checker
        task={r['id']:r for r in rows(BFCL_ROOT/'bfcl_eval/data/BFCL_v4_multi_turn_base.json')}
        gold={r['id']:r['ground_truth'] for r in rows(BFCL_ROOT/'bfcl_eval/data/possible_answer/BFCL_v4_multi_turn_base.json')}
        for split in ('dev','test'):
            for arm in ('base','tuned'):
                for r in rows(f'results/reference/{arm}/{split}.jsonl'):
                    run='release_check_'+uuid.uuid4().hex
                    try:
                        with contextlib.redirect_stdout(io.StringIO()):
                            score=multi_turn_checker(r['decoded'],gold[r['id']],prepare_task(task[r['id']]),'multi_turn_base',run)
                        require((bool(score['valid']) and r['status']=='completed')==bool(r['success']), f'Official replay: {arm}/{r["id"]}')
                        checked+=1
                    finally:
                        clear_run_instances(run)
    return {'passed':True,'training_trajectories':120,'training_targets':1129,'scores':scores,'official_saved_rollouts_replayed':checked}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--replay',action='store_true')
    print(json.dumps(verify(p.parse_args().replay),indent=2))
