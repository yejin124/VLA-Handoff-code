"""Prescribed task folds and fixed test families survive unavailable estimands."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import exp_20261020_e9_probe_null as e9
import exp_20261020_e3_analysis as e3
import make_injection_table as table
import gen_exp_20261020_tables as legacy_tables


class DecodabilitySupportTests(unittest.TestCase):
    def test_image_formatter_retains_unavailable_camera_and_pooled_cells(self):
        data = {'decodability': [dict(suite=suite, policy=policy, situations={
            condition: dict(status='not_evaluated', n=0) for condition in ('occluder', 'lighting')})
            for suite in ('spatial', 'object') for policy in ('gr00t', 'pi05', 'molmoact')],
            'within_situation_step80': [], 'pooled_comparisons': {}}
        rendered = legacy_tables.visual(data)
        self.assertIn('Scene decodability', rendered)
        self.assertIn('-- & -- & -- & -- & --', rendered)

    def data(self, tasks=10):
        rng = np.random.default_rng(13)
        return rng.normal(size=(tasks * 2, 4)), np.tile([0, 1], tasks), np.repeat(np.arange(tasks), 2)

    def test_nine_tasks_and_single_class_are_unavailable_not_a_different_cv(self):
        for tasks in (0, 9):
            x, y, groups = self.data(tasks)
            value = e9.null_test(x, y, groups, groups, 5, n_perm=3)
            self.assertEqual(value['status'], 'not_evaluated')
            self.assertIsNone(value['p_phipson_smyth'])
        x, y, groups = self.data()
        value = e9.null_test(x, y * 0, groups, groups, 5, n_perm=3)
        self.assertIn('Both condition labels', value['reason'])
        # Both labels globally, but one fold has only a single training class.
        value = e9.null_test(x, (groups == 0).astype(int), groups, groups, 5, n_perm=3)
        self.assertIn('training fold', value['reason'])

    def test_ten_task_fit_uses_all_prescribed_permutations(self):
        x, y, groups = self.data()
        with threadpool_limits(limits=1):
            value = e9.null_test(x, y, groups, groups, 5, n_perm=3)
        self.assertEqual(value['status'], 'evaluated')
        self.assertEqual(value['n_perm'], 3)
        self.assertTrue(np.isfinite(value['auroc']))
        with patch.object(e9, 'permute_pairs', return_value=np.zeros_like(y)), threadpool_limits(limits=1):
            invalid = e9.null_test(x, y, groups, groups, 5, n_perm=3)
        self.assertEqual(invalid['status'], 'not_evaluated')
        self.assertIsNone(invalid['p_phipson_smyth'])  # do not discard bad permutations

    def test_fixed_twelve_test_holm_family_includes_unavailable_members(self):
        class Pool:
            def __init__(self, *args, **kwargs): pass
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def map(self, function, cells):
                return [('spatial', 'gr00t', {'moved':dict(status='evaluated', p_phipson_smyth=.001,
                        auroc=.7, paper_auroc=None, null_95=.6, null_max=.65)})]
        saved = []
        with tempfile.TemporaryDirectory() as tmp, patch.object(e9.c, 'A', Path(tmp)), \
                patch.object(e9, 'Pool', Pool), \
                patch.object(e9.c, 'write', side_effect=lambda name, value, inputs:saved.append(value) or 'out'), \
                contextlib.redirect_stdout(io.StringIO()):
            e9.main()
        tests = saved[0]['tests']
        self.assertEqual(len(tests), 12)
        available = next(value for value in tests if value['status'] == 'evaluated')
        self.assertAlmostEqual(available['holm_p'], .012)
        missing = next(value for value in tests if value['status'] == 'not_evaluated')
        self.assertEqual(missing['holm_input_p'], 1.)
        self.assertIsNone(missing['holm_p'])

    def test_e3_missing_task_support_precedes_embedding_lookup(self):
        rows = [dict(task=task, episode=0, condition=condition, tag='q40', id=f'data/{task}/{condition}', query=4)
                for task in range(9) for condition in ('clean', 'moved')]
        with patch.object(e3.e9, 'load_rows', return_value=rows), patch.object(e3, 'features', return_value={}):
            _, _, result = e3.decodability(('spatial', 'gr00t'))
        self.assertEqual(result['moved']['status'], 'not_evaluated')
        self.assertNotIn('handoff_auroc', result['moved'])

    def test_injection_table_renders_null_stats_and_creates_output_parent(self):
        unavailable = dict(rho_ci95=None, mean_rho=None, tasks_rho_positive=0, tasks_with_rho=0,
                           sign_p_one_sided=None, holm_p=1., mean_dR_std=None, mean_dS_std=None)
        self.assertEqual(table.fmt(unavailable), '-- & 0/0 & -- & -- & --')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / 'input.json'; output = root / 'nested/tables/injection.tex'
            source.write_text(json.dumps(dict(cells={f'{p}|{s}|{c}':unavailable for p in table.NAMES
                              for s in ('spatial', 'object') for c in ('occluder', 'moved')})))
            with patch.object(sys, 'argv', ['table', str(source), str(output)]):
                table.main()
            self.assertIn('-- & 0/0 & -- & -- & --', output.read_text())
