#!/usr/bin/env python3
"""Exercise the group-size derivative on synthetic CPU tokens, never Agent data."""
import argparse
import ast
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time

import torch
from safetensors.torch import load_file
import e088_update_resumable as old
import e110_update_group_size as new
from e088_test_trainer_resume import make_model, cpu_logps, Tokenizer, find
from e088_test_boundary_checkpoint import equal
from optimizer_boundary_checkpoint import BoundaryCheckpoints


class SpecializeEight(ast.NodeTransformer):
    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name) and node.value.id == 'args' and node.attr == 'group_size':
            return ast.copy_location(ast.Constant(value=8), node)
        return self.generic_visit(node)


def verify_math():
    a_tree = ast.parse(Path(old.__file__).read_text())
    b_tree = SpecializeEight().visit(ast.parse(Path(new.__file__).read_text()))
    a = next(n for n in a_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    b = next(n for n in b_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    group = lambda n: isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'group'
    record = lambda n: isinstance(n, ast.For) and isinstance(n.target, ast.Tuple) and [getattr(v, 'id', None) for v in n.target.elts] == ['local_position', 'record_index']
    assert ast.dump(find(a, group)) == ast.dump(find(b, group)), 'Advantage/record assembly changed'
    assert ast.dump(find(a, record)) == ast.dump(find(b, record)), 'Policy/KL/backward changed'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path, required=True); args = ap.parse_args()
    started = time.monotonic(); torch.set_num_threads(1); assert not torch.cuda.is_initialized()
    verify_math()
    saved = dict(loader=old.AutoModelForCausalLM.from_pretrained, tokenizer=old.AutoTokenizer.from_pretrained,
                 old_logps=old.token_logps, new_logps=new.token_logps, adam=torch.optim.AdamW,
                 reset=torch.cuda.reset_peak_memory_stats, empty=torch.cuda.empty_cache,
                 peak=torch.cuda.max_memory_allocated, save=BoundaryCheckpoints.save,
                 handler=signal.getsignal(signal.SIGTERM), argv=sys.argv[:])
    old.AutoModelForCausalLM.from_pretrained = make_model
    old.AutoTokenizer.from_pretrained = lambda *a, **k: Tokenizer()
    old.token_logps = new.token_logps = cpu_logps
    torch.cuda.reset_peak_memory_stats = lambda: None
    torch.cuda.empty_cache = lambda: None
    torch.cuda.max_memory_allocated = lambda: 0
    torch.optim.AdamW = lambda p, **k: saved['adam'](p, **{x: y for x, y in k.items() if x != 'fused'})
    try:
        with tempfile.TemporaryDirectory(prefix='e110_group_cpu_') as temp:
            root = Path(temp); base = root / 'base'; model = make_model().eval(); model.save_pretrained(base)
            sources = {}
            for size in (8, 16):
                records = []
                with torch.no_grad():
                    for i in range(size):
                        record = dict(prompt_ids=[i + 1, 2, 3], completion_ids=[i + 4, 5], is_action_valid=True)
                        record['rollout_logps'] = cpu_logps(model, record).tolist(); records.append([record])
                source = dict(status='complete', groups=1, group_size=size, rollouts=size, rollout_adapter=None,
                    results=[dict(gamefile='/home/codex/.cache/alfworld/json_2.1.1/train/ENGINEERING_TEST_ONLY/game.tw-pddl',
                                  seed=1, mixed=True, successes=size // 2, won=[True] * (size // 2) + [False] * (size // 2),
                                  invalid_format=[0] * size, response_cap_hits=[0] * size, steps=[1] * size, records=records)])
                sources[size] = source; (root / f'G{size}.json').write_text(json.dumps(source))
            def run(module, size, name, checkpoint=None):
                out = root / name
                options = ['--rollouts', str(root / f'G{size}.json'), '--model', str(base), '--train-base-model', str(base),
                           '--reference-model', str(base), '--output', str(out), '--updates', '1',
                           '--objective', 'vendor_dynamic_grpo', '--minibatch-decisions', '2',
                           '--checkpoint-every', '1', '--diagnostic-samples', '2']
                if module is new: options += ['--group-size', str(size)]
                if checkpoint: options += ['--resume-update-checkpoint', str(checkpoint)]
                sys.argv = [module.__file__, *options]
                with contextlib.redirect_stdout(io.StringIO()):
                    try: module.main()
                    except SystemExit as error: assert error.code == 0
                return out
            def contents(directory):
                return (load_file(directory / 'adapter/adapter_model.safetensors'),
                        torch.load(directory / 'optimizer_state.pt', weights_only=True),
                        torch.load(directory / 'checkpoints/prepared_records.pt', weights_only=True))
            original = run(old, 8, 'old8'); derivative = run(new, 8, 'new8')
            left, right = contents(original), contents(derivative)
            assert all(equal(x, y) for x, y in zip(left, right)), 'G8 output/AdamW/cache differs'
            whole = run(new, 16, 'whole16'); w = contents(whole)
            cache = w[2]['records']; assert len(cache) == 16
            assert sorted(r['rollout_index'] for r in cache) == list(range(16))
            assert sum(len(r['completion_ids']) for r in cache) == 32
            report = json.loads((whole / 'training_report.json').read_text())
            assert report['optimizer_updates'] == 8 and report['target_tokens_per_update'] == 32 and report['rollout_group_size'] == 16
            def interrupt(self, model, opt, progress, rng):
                path = saved['save'](self, model, opt, progress, rng)
                if progress['completed_optimizer_steps'] == 2: os.kill(os.getpid(), signal.SIGTERM)
                return path
            BoundaryCheckpoints.save = interrupt
            paused = run(new, 16, 'paused16'); BoundaryCheckpoints.save = saved['save']
            pause = json.loads((paused / 'paused_update.json').read_text())
            assert pause['completed_optimizer_steps'] == 2 and not (paused / 'training_report.json').exists()
            resumed = run(new, 16, 'resumed16', pause['checkpoint']); r = contents(resumed)
            assert all(equal(x, y) for x, y in zip(w, r)), 'G16 resume differs'
            rr = json.loads((resumed / 'training_report.json').read_text()); assert rr['optimizer_steps_this_process'] == 6
            exported = run(new, 16, 'export16', rr['final_boundary_checkpoint'])
            er = json.loads((exported / 'training_report.json').read_text())
            assert er['optimizer_steps_this_process'] == 0 and equal(w[0], contents(exported)[0])
            rejected = 0
            for key in ('won', 'records', 'steps'):
                bad = copy.deepcopy(sources[16]); bad['results'][0][key] = bad['results'][0][key][:8]
                try: new.validate_group_dimensions(bad, 16)
                except ValueError: rejected += 1
                else: raise AssertionError('Truncated G16 accepted: ' + key)
            try: new.validate_group_dimensions(sources[16], 8)
            except ValueError: rejected += 1
            else: raise AssertionError('Wrong CLI group size accepted')
    finally:
        sys.argv = saved['argv']; signal.signal(signal.SIGTERM, saved['handler'])
        old.AutoModelForCausalLM.from_pretrained = saved['loader']; old.AutoTokenizer.from_pretrained = saved['tokenizer']
        old.token_logps = saved['old_logps']; new.token_logps = saved['new_logps']; torch.optim.AdamW = saved['adam']
        torch.cuda.reset_peak_memory_stats = saved['reset']; torch.cuda.empty_cache = saved['empty']; torch.cuda.max_memory_allocated = saved['peak']
        BoundaryCheckpoints.save = saved['save']
    result = dict(status='passed', engineering_cpu_seconds=round(time.monotonic() - started, 3),
                  g8_specialized_math_ast_equal=True, g8_bitwise_adapter_adamw_prepared_cache_equal=True,
                  g16_all_16_rollouts_in_loss=True, g16_target_tokens=32, g16_optimizer_updates=8,
                  g16_interrupt_step=2, g16_resume_steps=6, g16_bitwise_resume_equal=True,
                  g16_completed_export_repeated_updates=0, rejected_dimension_cases=rejected,
                  engineering_synthetic_optimizer_steps=24, research_optimizer_steps=0,
                  cuda_initialized=torch.cuda.is_initialized(), real_agent_capability_result=None)
    assert not result['cuda_initialized']; args.output.write_text(json.dumps(result, indent=2) + '\n'); print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
