"""New replication outcomes must not manufacture estimates or break aggregation."""
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
import injection_input_level as injection
import realistic_probe as probe
import exp_20261020_e2_calibration as calibration
import exp_20261020_e3_analysis as image
import make_supp_confirm_tables as confirm_tables


class StatisticalSupportTests(unittest.TestCase):
    def test_early_success_before_move_has_no_planned_post_move_query(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / 'queries.jsonl').write_text('\n'.join(
                json.dumps(dict(query=i, step=s)) for i, s in enumerate((0, 10, 20))))
            summary = dict(intervention=dict(start_step=30, duration_steps=10))
            self.assertIsNone(injection.query_index(directory, summary, 'moved'))
            clean = directory / 'clean'; clean.mkdir()
            eps = {(0, 15, 'moved'): (directory, summary),
                   (0, 15, 'clean'): (clean, dict(queries=3))}
            with patch.object(injection, 'record', side_effect=AssertionError('must not read an absent query')):
                with np.errstate(invalid='ignore', divide='ignore'):
                    result = injection.cell(eps, 'moved')
                    self.assertEqual(result['pairs'], 0)
                    self.assertEqual(result['unavailable_pairs'], 1)

    def test_small_matched_cohort_fits_train_only_bounded_pca(self):
        rng = np.random.default_rng(19)
        with tempfile.TemporaryDirectory() as tmp, threadpool_limits(limits=1):
            eps = {}
            for task in range(4):
                for episode in range(6):
                    for condition in ('clean', 'occluder'):
                        directory = Path(tmp) / f'{task}_{episode}_{condition}'; directory.mkdir()
                        np.savez(directory / 'q0000.npz', handoff=rng.normal(size=64),
                                 action_samples=rng.normal(size=(8, 4, 7)))
                        (directory / 'queries.jsonl').write_text(json.dumps(dict(query=0, step=0)))
                        eps[task, episode, condition] = (directory, dict(queries=1, success=bool(episode % 2)))
            result = injection.cell(eps, 'occluder')
            self.assertEqual(result['pairs'], 24)
            self.assertTrue(np.isfinite(result['mean_dR_std']))
            self.assertEqual(injection.readout(12, 64).named_steps['pca'].n_components, 11)

    def test_missing_active_task_preserves_prescribed_ten_folds(self):
        rng = np.random.default_rng(20)
        rows = [dict(task=task, episode=episode, condition=condition, tag='q40',
                     handoff=rng.normal(size=35), image_stats=rng.normal(size=4), spread=float(rng.uniform(.1, 1)))
                for task in range(9) for episode in range(5) for condition in ('clean', 'moved')]
        with patch.object(probe, 'load_rows', return_value=rows), patch.object(probe, 'oof_auroc') as auc, \
                patch.object(probe, 'oof_r2') as r2:
            result = probe.analyse('spatial', 'gr00t')
        auc.assert_not_called(); r2.assert_not_called()
        self.assertEqual(result['A'], {})
        self.assertEqual(result['B'], {})
        self.assertIn('A/moved', result['not_evaluated'])
        self.assertIn('B/moved', result['not_evaluated'])
        self.assertIn('moved', result['C'])  # paired spread remains estimable

    def test_calibration_without_evaluable_cells_never_starts_pool_zero(self):
        saved = []
        with patch.object(calibration.c, 'evaluable_cells', return_value=[]), \
                patch.object(calibration, 'Pool', side_effect=AssertionError('empty pool')), \
                patch.object(calibration.c, 'write', side_effect=lambda name, result, inputs: saved.append(result) or 'out'), \
                patch.object(sys, 'argv', ['calibration']), contextlib.redirect_stdout(io.StringIO()):
            calibration.main()
        self.assertEqual(saved[0]['cells'], [])

    def test_image_analysis_without_ranking_cells_keeps_decoding(self):
        class Pool:
            def __init__(self, *args, **kwargs): pass
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def map(self, function, cells):
                return [('spatial', 'gr00t', {})] if function is image.decodability else []
        saved = []
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'manifest.json').write_text(json.dumps(dict(dinov2={}, resnet18={}, precision='fp16', provenance={}, files={})))
            with patch.object(image.c, 'evaluable_cells', return_value=[]), patch.object(image, 'Pool', Pool), \
                    patch.object(image, 'FEAT', Path(tmp)), \
                    patch.object(image.c, 'write', side_effect=lambda name, result, inputs: saved.append(result) or 'out'), \
                    patch.object(sys, 'argv', ['image']), contextlib.redirect_stdout(io.StringIO()):
                image.main()
        self.assertEqual(saved[0]['within_situation_step80'], [])
        self.assertEqual(saved[0]['pooled_comparisons'], {})
        self.assertEqual(saved[0]['decodability'][0]['policy'], 'gr00t')

    def test_policy_table_keeps_ineligible_members_without_estimates(self):
        def cell():
            return dict(status='evaluated', methods={m:dict(estimate=.6) for m in ('G', 'GH', 'Spread')},
                        contrasts={c:dict(estimate=.1, ci95=[.01, .2], one_sided_p=.03) for c in ('GH-G', 'GH-Spread')})
        cells = {f'{key}|{step}':cell() for key, _, _ in confirm_tables.POLICIES for step in (40, 80)}
        # The other suite of this policy remains in its original two-member family.
        cells['pi05_spatial_confirm|80'] = dict(status='ineligible', methods={}, contrasts={})
        cohorts = {key:dict(landmarks={str(step):dict(active=20, task_counts={'0':[20, 3]}) for step in (40, 80)})
                   for key, _, _ in confirm_tables.POLICIES}
        spatial = {f'gr00t_spatial_confirm|{step}':cell() for step in (40, 80)}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / 'other_policies').mkdir()
            (root / 'other_policies/confirm_results.json').write_text(json.dumps(dict(cells=cells, cohorts=cohorts)))
            (root / 'spread_contrast_results.json').write_text(json.dumps(dict(cells=spatial)))
            (root / 'ral_confirm_results_frozen.json').write_text(json.dumps(dict(cohorts={'gr00t_spatial_confirm':next(iter(cohorts.values()))})))
            confirm_tables.policies(root, root)
            table = (root / 'supp_confirm_policies.tex').read_text()
        unavailable = next(line for line in table.splitlines() if line.startswith('$\\pi_{0.5}$ & Spatial & 80'))
        self.assertIn('20 (3)', unavailable); self.assertIn('--', unavailable)
        available = next(line for line in table.splitlines() if line.startswith('$\\pi_{0.5}$ & Object & 80'))
        self.assertIn('0.060', available)  # unavailable family member is p=1, never removed



if __name__ == '__main__':
    unittest.main()
