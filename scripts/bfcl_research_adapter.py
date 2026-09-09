"""Research v1 BFCL rollout; separate from the immutable P000 adapter.

Uses upstream function population and FC tool-update semantics. The generation
callback receives only messages and visible tools, never task state or gold.
No benchmark selection, model loading, network calls, or output files on import.
"""
import copy
import random
import re
import time
import uuid
from dataclasses import dataclass

from common import calls_to_python, decode_calls, digest, make_tools, render
from bfcl_eval.constants.default_prompts import DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC, MAXIMUM_STEP_LIMIT
from bfcl_eval.eval_checker.multi_turn_eval import multi_turn_utils as backend
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import multi_turn_checker
from bfcl_eval.utils import populate_test_cases_with_predefined_functions

PROTOCOL_VERSION = "research_bfcl_v1"
SUPPORTED_CATEGORIES = {"multi_turn_base", "multi_turn_miss_param", "multi_turn_miss_func", "multi_turn_long_context"}
SYSTEM_PROMPT = "You are a helpful assistant. Use the available tools to fulfill the user's request. Ask for missing required information. Base answers on actual tool results. At each user turn, continue calling tools until the current request is complete, then respond to the user."


@dataclass(frozen=True)
class Limits:
    input_tokens: int = 32768
    new_tokens: int = 512
    steps_per_turn: int = 21

    def __post_init__(self):
        if min(self.input_tokens, self.new_tokens, self.steps_per_turn) < 1:
            raise ValueError("All generation limits must be positive")


def prepare_task(raw_task):
    task = copy.deepcopy(raw_task)
    category = task["id"].rsplit("_", 1)[0]
    if category not in SUPPORTED_CATEGORIES:
        raise ValueError("This adapter is registered only for the four research categories")
    # Require raw input so the upstream name-to-definition conversion is not run twice.
    if any(not isinstance(name, str) for names in task.get("missed_function", {}).values() for name in names):
        raise ValueError("Expected raw task with missed_function names, not a populated task")
    return populate_test_cases_with_predefined_functions([task])[0]


def turn_inputs(task, turn):
    """Mutate only the private prepared task, matching upstream FC handler order."""
    messages = copy.deepcopy(task["question"][turn])
    heldout = task.get("missed_function", {})
    if str(turn) in heldout:
        if messages:
            raise ValueError("Official heldout-tool turn must have no original user message")
        task["function"].extend(copy.deepcopy(heldout[str(turn)]))
        messages = [{"role": "user", "content": DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC}]
    return messages, make_tools(task["function"])


def clear_run_instances(run_name):
    """Delete only this UUID-scoped run, including checker replay/gold instances."""
    prefix = re.sub(r"[-./:]", "_", run_name) + "_"
    for key in list(vars(backend)):
        if key.startswith(prefix) and key.endswith("_instance"):
            delattr(backend, key)


class TransformersGenerator:
    """Explicit limits, greedy decoding, same Qwen JSON-call template as P000."""
    def __init__(self, model, tokenizer, limits=Limits()):
        self.model, self.tokenizer, self.limits = model, tokenizer, limits

    def __call__(self, messages, tools):
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
        t = self.tokenizer
        prompt = render(t, messages, tools)
        inputs = t(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        length = inputs.input_ids.shape[-1]
        if length > self.limits.input_tokens:
            raise OverflowError(f"Input length {length} exceeds fixed limit {self.limits.input_tokens}")
        start = time.monotonic()
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=self.limits.new_tokens,
                                         do_sample=False, use_cache=True, logits_to_keep=1,
                                         pad_token_id=t.pad_token_id, eos_token_id=t.eos_token_id)
        ids = output[0, length:]
        response = t.decode(ids, skip_special_tokens=False).removesuffix(t.eos_token)
        return response, {"input_tokens": length, "output_tokens": len(ids),
                          "hit_output_limit": len(ids) == self.limits.new_tokens,
                          "seconds": time.monotonic() - start}


def rollout(raw_task, generate, *, limits=Limits(), run_name=None, execute=None):
    """Gold-free rollout. Caller can score its returned decoded calls separately."""
    execute = execute or backend.execute_multi_turn_func_call
    task = prepare_task(raw_task)
    run_name = run_name or "research_" + uuid.uuid4().hex
    category = task["id"].rsplit("_", 1)[0]
    long_context = "long_context" in category or "composite" in category
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    decoded, trace, prompt_schedule = [], [], []
    status = "completed"
    start = time.monotonic()
    try:
        execute([], task["initial_config"], task["involved_classes"], run_name, task["id"],
                long_context=long_context, is_evaL_run=False)
        for turn in range(len(task["question"])):
            new_messages, tools = turn_inputs(task, turn)
            messages.extend(new_messages)
            prompt_schedule.append({"turn": turn, "visible_tools_hash": digest(tools),
                                    "user_messages_hash": digest(new_messages)})
            steps = []
            for step in range(limits.steps_per_turn):
                try:
                    # Copies prevent an inference client from changing adapter history/tools.
                    response, usage = generate(copy.deepcopy(messages), copy.deepcopy(tools))
                except OverflowError as exc:
                    status = "context_limit"
                    trace.append({"turn": turn, "step": step, "error": str(exc)})
                    break
                event = {"turn": turn, "step": step, "response": response, "usage": usage}
                try:
                    calls = decode_calls(response, tools)
                except Exception as exc:
                    # Malformed JSON shapes can raise AttributeError as well as
                    # ValueError. They are model decode failures, not infrastructure.
                    calls = []
                    event["decode_error"] = str(exc)
                pycalls = calls_to_python(calls)
                steps.append(pycalls)
                event["calls"] = pycalls
                if calls:
                    messages.append({"role": "assistant", "content": "", "tool_calls": calls})
                    results, _ = execute(pycalls, task["initial_config"], task["involved_classes"],
                                         run_name, task["id"], long_context=long_context, is_evaL_run=False)
                    event["tool_results"] = results
                    messages.extend({"role": "tool", "content": result} for result in results)
                else:
                    messages.append({"role": "assistant", "content": response})
                trace.append(event)
                if not calls:
                    break
            else:
                # Upstream forces quit after executed-call count >20, i.e. call 21.
                status = "step_limit"
            decoded.append(steps or [[]])
            if status != "completed":
                break
        decoded.extend([[[]] for _ in range(len(task["question"]) - len(decoded))])
        return {"id": task["id"], "protocol": PROTOCOL_VERSION, "status": status,
                "seconds": time.monotonic() - start, "decoded": decoded, "trace": trace,
                "prompt_hash": digest({"system": SYSTEM_PROMPT, "schedule": prompt_schedule}),
                "prompt_schedule": prompt_schedule, "long_context": long_context}
    finally:
        clear_run_instances(run_name)


def evaluate_task(raw_task, gold, generate, *, limits=Limits(), seed=20260908):
    """Gold is first consumed after generation has completed; replay uses fresh state."""
    run_name = "research_" + uuid.uuid4().hex
    random.seed(seed + int(raw_task["id"].rsplit("_", 1)[-1]))
    result = rollout(raw_task, generate, limits=limits, run_name=run_name)
    if len(gold) != len(raw_task["question"]):
        raise ValueError("Gold and question turn counts disagree")
    random.seed(seed + int(raw_task["id"].rsplit("_", 1)[-1]))
    try:
        score = multi_turn_checker(result["decoded"], gold, copy.deepcopy(raw_task),
                                   raw_task["id"].rsplit("_", 1)[0], run_name)
        result.update(score_detail=score, success=bool(score["valid"]) and result["status"] == "completed")
        return result
    finally:
        clear_run_instances(run_name)


assert Limits().steps_per_turn == MAXIMUM_STEP_LIMIT + 1
