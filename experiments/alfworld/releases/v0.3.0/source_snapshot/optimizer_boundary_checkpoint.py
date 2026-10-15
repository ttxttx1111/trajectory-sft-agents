#!/usr/bin/env python3
"""Single-process FP32 LoRA checkpoints at completed optimizer boundaries.

The model/optimizer/RNG contract follows verl's checkpoint managers. This helper
also pins the prepared old/reference logps: those must never be recomputed with
an already updated actor during a within-batch resume. No loss or reward code is
implemented here. A caller must provide progress/RNG from its last completed
optimizer step, discard partial gradients, and use a new output directory.
"""
import hashlib
import json
import os
from pathlib import Path
import random
import uuid
import numpy as np
import torch
from peft import set_peft_model_state_dict
from peft.utils.save_and_load import load_peft_weights


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(8*1024*1024),b''):h.update(data)
    return h.hexdigest()


def rng_state():
    n=np.random.get_state()
    return {'python':random.getstate(),'numpy':[n[0],n[1].tolist(),int(n[2]),int(n[3]),float(n[4])],
            'torch_cpu':torch.get_rng_state().clone(),
            'torch_cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def restore_rng(state):
    random.setstate(state['python'])
    n=state['numpy'];np.random.set_state((n[0],np.asarray(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    torch.set_rng_state(state['torch_cpu'])
    if state['torch_cuda']:
        if len(state['torch_cuda'])!=torch.cuda.device_count():raise ValueError('CUDA device count changed')
        torch.cuda.set_rng_state_all(state['torch_cuda'])


def cpu_copy(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_copy(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_copy(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_copy(v) for v in value)
    if value is None or isinstance(value,(str,int,float,bool)):return value
    raise TypeError(f'Unsupported checkpoint value: {type(value).__name__}')


def trainable_names(model):
    names=[]
    for name,param in model.named_parameters():
        if param.requires_grad:
            if 'lora_' not in name or param.dtype!=torch.float32:
                raise ValueError('This checkpoint contract supports unmerged FP32 LoRA only')
            names.append(name)
    if not names:raise ValueError('No trainable LoRA parameters')
    return names


def write_json(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');os.replace(temp,path)


class BoundaryCheckpoints:
    def __init__(self,root,fingerprint):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.fingerprint=json.loads(json.dumps(fingerprint,sort_keys=True))
        self.cache_path=self.root/'prepared_records.pt'

    def save_cache(self,prepared_records):
        if self.cache_path.exists():raise FileExistsError('Do not replace the prepared policy cache')
        temporary=self.root/f'.cache-{uuid.uuid4().hex}.tmp'
        torch.save(cpu_copy(prepared_records),temporary);os.replace(temporary,self.cache_path)
        return sha(self.cache_path)

    def save(self,model,optimizer,progress,boundary_rng):
        if not self.cache_path.is_file():raise ValueError('Prepared original-policy cache missing')
        step=int(progress['completed_optimizer_steps'])
        if step<0:raise ValueError('Invalid completed optimizer step')
        destination=self.root/f'step_{step:06d}'
        if destination.exists():raise FileExistsError('Preserve an existing completed checkpoint')
        temp=self.root/f'.step_{step:06d}-{uuid.uuid4().hex}.tmp';temp.mkdir()
        names=trainable_names(model)
        model.save_pretrained(temp/'adapter',safe_serialization=True,save_embedding_layers=False)
        torch.save(cpu_copy(optimizer.state_dict()),temp/'optimizer_state.pt')
        torch.save({'progress':cpu_copy(progress),'rng':cpu_copy(boundary_rng)},temp/'training_state.pt')
        write_json(temp/'trainable_parameters.json',names)
        files={str(p.relative_to(temp)):sha(p) for p in temp.rglob('*') if p.is_file()}
        manifest={'schema':1,'fingerprint':self.fingerprint,'completed_optimizer_steps':step,
                  'files_sha256':files,'prepared_cache_path':str(self.cache_path.resolve()),
                  'prepared_cache_sha256':sha(self.cache_path),'unmerged_fp32_lora':True,
                  'boundary_rule':'completed optimizer only; caller discards partial gradients'}
        write_json(temp/'manifest.json',manifest)
        # Only a fully populated directory becomes visible as a checkpoint.
        os.replace(temp,destination)
        write_json(self.root/'latest_checkpoint.json',{'path':str(destination.resolve()),'manifest_sha256':sha(destination/'manifest.json')})
        return destination

    @staticmethod
    def inspect(path,fingerprint):
        path=Path(path);manifest=json.loads((path/'manifest.json').read_text())
        if manifest['fingerprint']!=json.loads(json.dumps(fingerprint,sort_keys=True)):
            raise ValueError('Dataset, actor, optimizer or algorithm fingerprint changed')
        for relative,expected in manifest['files_sha256'].items():
            target=(path/relative).resolve()
            if not target.is_relative_to(path.resolve()) or sha(target)!=expected:
                raise ValueError(f'Checkpoint payload changed: {relative}')
        if sha(manifest['prepared_cache_path'])!=manifest['prepared_cache_sha256']:
            raise ValueError('Prepared old/reference policy cache changed')
        return manifest

    @staticmethod
    def load(path,model,optimizer,fingerprint):
        path=Path(path);manifest=BoundaryCheckpoints.inspect(path,fingerprint)
        names=json.loads((path/'trainable_parameters.json').read_text())
        if names!=trainable_names(model):raise ValueError('Trainable parameter names/order changed')
        weights=load_peft_weights(str(path/'adapter'),device='cpu')
        if any(value.dtype!=torch.float32 for value in weights.values()):raise ValueError('LoRA precision changed')
        result=set_peft_model_state_dict(model,weights)
        if result.unexpected_keys or any('lora_' in name for name in result.missing_keys):
            raise ValueError('Incomplete LoRA checkpoint')
        optimizer.load_state_dict(torch.load(path/'optimizer_state.pt',map_location='cpu',weights_only=True))
        state=torch.load(path/'training_state.pt',map_location='cpu',weights_only=True)
        cache=torch.load(manifest['prepared_cache_path'],map_location='cpu',weights_only=True)
        if state['progress']['completed_optimizer_steps']!=manifest['completed_optimizer_steps']:
            raise ValueError('Checkpoint step disagrees with saved progress')
        optimizer.zero_grad(set_to_none=True)
        restore_rng(state['rng'])
        return state['progress'],cache
