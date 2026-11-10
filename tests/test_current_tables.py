"""Regression checks for new-result rendering, without GPU/data archives."""
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'code/make_current_tables.py'
SPEC = importlib.util.spec_from_file_location('current_tables', SOURCE)
tables = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tables)


class CurrentTablesTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def save(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    def confirm(self):
        return {'cells': {
            'gr00t_bridge_confirm|20': {'status': 'ineligible', 'valid': 0},
            'gr00t_spatial_confirm|80': {'status': 'evaluated', 'valid': 100,
                'methods': {m: {'estimate': v} for m, v in [('G', .4), ('GH', .6), ('Spread', .5)]},
                'contrasts': {'GH-G': {'estimate': .2, 'ci95': [.1, .3], 'one_sided_p': .02},
                              'GH-Spread': {'estimate': .1, 'ci95': [-.1, .3], 'one_sided_p': .2}}}},
            'cohorts': {c: {'landmarks': {str(step): {'active': 5, 'task_counts': {'0': [5, 1]}}}}
                        for c, step in [('gr00t_bridge_confirm', 20), ('gr00t_spatial_confirm', 80)]}}

    def test_ineligible_primary_keeps_counts_and_correction_family(self):
        source = self.save('confirm.json', self.confirm())
        tables.main_confirm(source, self.root / 'out')
        with (self.root / 'out/main_confirm.csv').open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]['active'], '5')
        self.assertEqual(rows[0]['status'], 'ineligible')
        self.assertEqual(rows[0]['G'], '')
        self.assertAlmostEqual(float(rows[1]['handoff_holm_p']), .04)

    def test_unavailable_probe_keeps_pairs_without_fake_failure_ranking(self):
        for suite in tables.SUITES:
            for policy in tables.POLICIES:
                self.save(f'analysis/realistic_probe_{suite}_{policy}.json', {
                    'A': {}, 'B': {}, 'C': {'moved': {'n_pairs': 3, 'higher_in': 2,
                    'median_relative_change': .01, 'wilcoxon_p': .5}},
                    'not_evaluated': {'A/moved': 'Insufficient active tasks'}})
                self.save(f'analysis/realistic_{suite}_moved_{policy}.json', {
                    'landmarks': [{'landmark': s, 'folds': []} for s in (40, 80)]})
                for episode in (0, 1):
                    self.save(f'episodes/{suite}_main/{policy}/shard00/t0_e{episode}_r0_moved/summary.json', {
                        'task': 0, 'episode': episode, 'replicate': 0, 'complete': True,
                        'success': episode == 0})
        tables.original_tables(self.root / 'analysis', self.root / 'episodes', self.root / 'out', {'main3'})
        with (self.root / 'out/main_goal.csv').open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(r['pairs'] == '3' and r['failure_auroc'] == '' for r in rows))
        self.assertIn('2/3', (self.root / 'out/main_goal.tex').read_text())

    def test_image_table_accepts_new_unavailable_schema(self):
        source = self.save('images.json', {'decodability': [{'suite': 'spatial', 'policy': 'pi05',
            'situations': {'moved': {'status': 'not_evaluated', 'reason': 'Too few tasks', 'n': 3}}}],
            'within_situation_step80': [], 'pooled_comparisons': {}})
        tables.supplementary_images(source, self.root / 'out')
        with (self.root / 'out/supp_image_encoders.csv').open() as handle:
            rows = list(csv.DictReader(handle))
        row = next(r for r in rows if r['part'] == 'A' and r['suite'] == 'spatial'
                   and r['policy'] == 'pi05' and r['condition'] == 'moved')
        self.assertEqual(row['status'], 'not_evaluated')
        self.assertEqual(row['n'], '3')
        self.assertEqual(row['reason'], 'Too few tasks')

    def test_missing_late_input_does_not_publish_partial_outputs(self):
        source = self.save('confirm.json', self.confirm())
        output = self.root / 'out'
        result = subprocess.run([sys.executable, str(SOURCE), '--tables', 'main2', 'supp2',
            '--confirm-results', str(source), '--image-results', str(self.root / 'missing.json'),
            '--output', str(output)], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(output.exists())

    def test_small_positive_p_is_not_printed_as_zero(self):
        self.assertGreater(float(tables.prob(1.7e-6)), 0)


if __name__ == '__main__':
    unittest.main()
