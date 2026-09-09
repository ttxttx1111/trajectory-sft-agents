"""Shared local-only runtime and auditable serialization."""
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BFCL_ROOT = ROOT / "vendor/gorilla/berkeley-function-call-leaderboard"
sys.path.insert(0, str(BFCL_ROOT))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def read_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]


def write_jsonl(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def digest(data):
    return hashlib.sha256(json.dumps(data, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def config():
    return read_json(ROOT / "configs/pilot.json")


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True


def tokenizer():
    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained(ROOT / config()["model_path"], local_files_only=True)
    t.pad_token = t.eos_token
    return t


def load_model(adapter=None, training=False):
    import torch
    from transformers import AutoModelForCausalLM
    seed_all(config()["seed"])
    model = AutoModelForCausalLM.from_pretrained(
        ROOT / config()["model_path"], dtype=torch.bfloat16,
        device_map={"": "cuda:0"}, attn_implementation="sdpa", local_files_only=True,
    )
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, is_trainable=training)
    model.config.use_cache = not training
    return model


def normalize_schema(value):
    """BFCL's dict/float schema types -> JSON Schema, retaining all semantic fields."""
    if isinstance(value, list):
        return [normalize_schema(x) for x in value]
    if isinstance(value, dict):
        out = {k: normalize_schema(v) for k, v in value.items() if v is not None}
        if isinstance(out.get("type"), str):
            out["type"] = {"dict": "object", "float": "number", "tuple": "array"}.get(out["type"], out["type"])
        return out
    return value


def make_tools(functions):
    return [{"type": "function", "function": normalize_schema({k: f[k] for k in ("name", "description", "parameters") if k in f})} for f in functions]


def render(t, messages, tools=None, generation=True):
    return t.apply_chat_template(messages, tools=tools or None, tokenize=False,
                                 add_generation_prompt=generation, enable_thinking=False)


def decode_calls(text, tools):
    """Only JSON-native calls; never execute model-produced Python text."""
    text = text.split("</think>")[-1]
    matches = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.S)
    if "<tool_call>" in text and not matches:
        raise ValueError("Malformed/incomplete tool-call tags")
    allowed = {x["function"]["name"] for x in tools}
    calls = []
    for s in matches:
        call = json.loads(s)
        name, args = call.get("name"), call.get("arguments")
        if name not in allowed or not re.fullmatch(r"[A-Za-z][A-Za-z_0-9]*", name):
            raise ValueError(f"Unknown tool: {name}")
        if not isinstance(args, dict) or any(not re.fullmatch(r"[A-Za-z][A-Za-z_0-9]*", k) for k in args):
            raise ValueError("Arguments must be an object with ordinary keyword names")
        calls.append({"type": "function", "function": {"name": name, "arguments": args}})
    return calls


def calls_to_python(calls):
    # All names are allowlisted by decode_calls; JSON values become Python literals via repr.
    return [c["function"]["name"] + "(" + ", ".join(k + "=" + repr(v) for k, v in c["function"]["arguments"].items()) + ")" for c in calls]
