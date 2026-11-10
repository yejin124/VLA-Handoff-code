"""Table preflight and unavailable-row contracts, independent of data archives."""
import argparse
import ast
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

CODE = Path(__file__).resolve().parents[1] / 'code'


def functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    body = [x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name in names]
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), 'exec'), namespace)


class RetainedPreflightTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.ns = {'A': root / 'analysis', 'OUT': root / 'output', 'argparse': argparse,
                   'json': json, 'math': math, 'sys': sys,
                   'POLICIES': [('pi05', 'Pi'), ('gr00t', 'GR'), ('molmoact', 'Molmo')],
                   'FAMILIES': [('occluder', 'Occluder'), ('lighting', 'Dim light'),
                                ('moved', 'Moved goal'), ('relocated', 'Relocated goal')],
                   '__doc__': 'test', 'sign_tests': lambda: 'sign results'}
        for name in ('severities', 'within', 'pooled', 'requery', 'transfer', 'probe',
                     'claims', 'lightingdark', 'libero_pro'):
            self.ns[name] = lambda *args: 'rendered table'
        functions(CODE / 'gen_realistic_tables.py', {'required_inputs', 'main'}, self.ns)

    def tearDown(self):
        self.temp.cleanup()

    def test_default_preflight_requires_31_inputs_and_writes_nothing(self):
        paths = self.ns['required_inputs'](['within_realistic', 'object_realistic', 'probe_realistic'])
        self.assertEqual(len(paths), 31)
        with patch.object(sys, 'argv', ['tables']):
            with self.assertRaises(FileNotFoundError):
                self.ns['main']()
        self.assertFalse(self.ns['OUT'].exists())

    def test_all_renders_finish_before_any_output(self):
        for path in self.ns['required_inputs'](['within_realistic', 'probe_realistic']):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{}')
        def failure():
            raise ValueError('Second selected renderer failed')
        self.ns['probe'] = failure
        with patch.object(sys, 'argv', ['tables', '--tables', 'within_realistic', 'probe_realistic']):
            with self.assertRaises(ValueError):
                self.ns['main']()
        self.assertFalse(self.ns['OUT'].exists())

    def test_unavailable_decoding_and_null_holm_keep_pair_counts(self):
        functions(CODE / 'gen_realistic_tables.py', {'probe'}, self.ns)
        self.ns.update(signed=lambda x, digits=2: f'{x:+.{digits}f}',
                       scientific_p=lambda x: str(x), panel=lambda *args: '\n'.join(args[-1]), note=str)
        self.ns['A'].mkdir()
        for suite in ('spatial', 'object'):
            for policy, _ in self.ns['POLICIES']:
                value = {'A': {}, 'B': {}, 'C': {'moved': {'n_pairs': 3, 'higher_in': 2,
                    'median_relative_change': .01, 'wilcoxon_p': .5}},
                    'not_evaluated': {'A/moved': 'Insufficient active tasks'}}
                (self.ns['A'] / f'realistic_probe_{suite}_{policy}.json').write_text(json.dumps(value))
        path = self.ns['A'].parent / 'exp_20261020/e9_probe_null.json'
        path.parent.mkdir()
        path.write_text(json.dumps({'tests': [{'suite': 'spatial', 'policy': 'pi05',
                                               'situation': 'moved', 'holm_p': None}]}))
        rendered = self.ns['probe']()
        self.assertEqual(rendered.count('Moved goal'), 12)  # six data rows plus six status notes
        self.assertIn('-- / --', rendered)
        self.assertIn('2/3', rendered)
        self.assertIn('Insufficient active tasks', rendered)


if __name__ == '__main__':
    unittest.main()
