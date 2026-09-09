"""CPU-only tests for the runtime boundary and isolation, with real mock environments."""
import copy
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from common import BFCL_ROOT, calls_to_python, decode_calls, read_jsonl
from bfcl_research_adapter import Limits, rollout
from reproduce import run_path


class RuntimeTests(unittest.TestCase):
    tools = [{'type':'function','function':{'name':'lookup','parameters':{'type':'object'}}}]

    def test_untrusted_call_names_and_argument_keys_are_rejected(self):
        for payload in [{'name':'exec','arguments':{}}, {'name':'lookup','arguments':{'x);exec(':'a'}},
                        {'name':'lookup','arguments':[]}, {'name':'lookup','arguments':None}]:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                decode_calls('<tool_call>'+json.dumps(payload)+'</tool_call>',self.tools)
        with self.assertRaises(ValueError):
            decode_calls('<tool_call>{broken}',self.tools)

    def test_argument_text_is_quoted_as_data(self):
        value="'); __import__('os').system('bad'); ('"
        calls=decode_calls('<tool_call>'+json.dumps({'name':'lookup','arguments':{'value':value}})+'</tool_call>',self.tools)
        import ast
        expression=ast.parse(calls_to_python(calls)[0],mode='eval').body
        self.assertIsInstance(expression,ast.Call)
        self.assertEqual(expression.func.id,'lookup')
        self.assertEqual(ast.literal_eval(expression.keywords[0].value),value)

    def test_no_call_ends_each_user_turn_and_preserves_source(self):
        raw=read_jsonl(BFCL_ROOT/'bfcl_eval/data/BFCL_v4_multi_turn_base.json')[0]
        before=copy.deepcopy(raw)
        prompts=[]
        def stop(messages,tools):
            prompts.append(copy.deepcopy(messages))
            # Callback receives only visible messages + tool definitions.
            self.assertTrue(all('role' in m for m in messages))
            return 'The request is complete.', {'input_tokens':1,'output_tokens':5}
        result=rollout(raw,stop)
        self.assertEqual(result['status'],'completed')
        self.assertEqual(len(prompts),len(raw['question']))
        self.assertTrue(all(not event.get('calls') for event in result['trace']))
        self.assertEqual(raw,before)

    def test_run_names_cannot_escape_output_directory(self):
        for name in ('../results','..','a/b','a\\b','C:\\temp',''):
            with self.subTest(name=name), self.assertRaises(ValueError):
                run_path(name)
        self.assertEqual(run_path('my-run_01').parent.name,'runs')

    def test_generation_limits_are_positive(self):
        with self.assertRaises(ValueError):
            Limits(steps_per_turn=0)


if __name__ == '__main__':
    unittest.main()
