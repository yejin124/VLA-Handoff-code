"""Public collection contracts without model weights or stored experiment results."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
import reproduce
import icassp_rollout
import recorded_failure_rollout_20261103 as legacy
import realistic_corruption
import smol_sampling_analysis as smol
import ral_confirm_robustness as robustness


class ReproductionTests(unittest.TestCase):
    def test_cohort_grid_matches_manuscript_and_keeps_conventions_distinct(self):
        records = reproduce.cohorts()
        self.assertEqual(sum(map(reproduce.count, records.values())), 10752)
        self.assertEqual(sum(reproduce.count(r) for n, r in records.items() if '_main' in n), 6000)
        self.assertEqual(reproduce.count(records['bridge_reused']), 288)
        for name, record in records.items():
            expected = 'recorded_failure_rollout_20261103.py' if '_main' in name else (
                'simpler_rollout.py' if name.startswith('bridge_') else 'icassp_rollout.py')
            self.assertEqual(record['collector'], expected)

    def test_official_axis_angle_preserves_nonshortest_branch(self):
        observation = {'robot0_eef_quat': np.array([0., 0., np.sqrt(.5), -np.sqrt(.5)]),
                       'robot0_eef_pos': np.zeros(3), 'robot0_gripper_qpos': np.zeros(2)}
        official = icassp_rollout.robot_state(observation)
        archived = legacy.robot_state(observation)
        self.assertAlmostEqual(float(official[5]), 1.5 * np.pi, places=6)
        self.assertAlmostEqual(float(archived[5]), -.5 * np.pi, places=6)

    def test_shell_characters_are_literal_command_arguments(self):
        record = reproduce.cohorts()['spatial_confirm_gr00t']
        command = reproduce.command(record, '/venv with spaces/python', Path('/tmp/literal $name'), 5662)
        self.assertEqual(command[0], '/venv with spaces/python')
        self.assertIn('/tmp/literal $name/data/icassp2027/realistic/spatial_confirm/gr00t', command)
        self.assertIn('--apply-to-wrist', command)

    def test_invalid_port_shard_and_escape_rejected(self):
        record = reproduce.cohorts()['bridge_new']
        for port, shard, nshards in [(0, 0, 1), (65536, 0, 1), (5612, 1, 1)]:
            with self.assertRaises(ValueError):
                reproduce.command(record, 'python', ROOT, port, shard, nshards)
        with self.assertRaises(ValueError):
            reproduce.command(dict(record, output='../escape'), 'python', ROOT, 5612)

    def test_standard_nested_libero_layout_and_config_conflict(self):
        with tempfile.TemporaryDirectory() as folder:
            checkout = Path(folder) / 'LIBERO'
            package = checkout / 'libero/libero'
            for name in ('bddl_files', 'assets', 'init_files'):
                (package / name).mkdir(parents=True)
            output = Path(folder) / 'local'
            reproduce.configure_libero(checkout, output)
            config = (output / 'config.yaml').read_text()
            self.assertIn(str(package / 'bddl_files'), config)
            reproduce.configure_libero(checkout, output)
            (output / 'config.yaml').write_text('existing: config\n')
            with self.assertRaisesRegex(ValueError, 'Refusing'):
                reproduce.configure_libero(checkout, output)

    def test_fixed_donor_inputs_available_in_clean_package(self):
        bank = realistic_corruption.load_donor_bank()
        self.assertEqual(len(bank), 10)
        self.assertTrue(all(a.ndim == 3 and a.shape[2] == 3 for a in bank))

    def test_baseline_requires_no_published_result_file(self):
        point = {'status': 'evaluated', 'methods': {}, 'contrasts': {}}
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(smol.replay, 'receipt', return_value=({}, 'collection-digest')), \
                patch.object(smol.replay, 'dataset', return_value={}), \
                patch.object(smol.fit, 'evaluate', return_value=point), \
                patch.object(smol.fit, 'compact_fit', return_value={}), \
                patch.object(smol.replay, 'digest', side_effect=AssertionError('No external result should be read')):
            result = smol.baseline(Path(folder))
            self.assertIsNone(result['published_sha256'])
            self.assertEqual(set(result['cells']), set(smol.replay.CELLS))
            self.assertEqual(json.loads((Path(folder) / 'baseline_validation.json').read_text())['collection_sha256'],
                             'collection-digest')

    def test_recomputed_baseline_detects_numeric_and_per_task_changes(self):
        point = {'methods': {k: {'mean_auroc': .6} for k in ('G', 'H', 'GH', 'Spread', 'G_RBF')},
                 'contrasts': {k: {'mean': .1, 'task_effects': {'0': .1}} for k in ('GH-G', 'GH-Spread')}}
        reference = json.loads(json.dumps(point))
        smol.baseline_check(point, 'test', reference)
        reference['methods']['H']['mean_auroc'] = .7
        with self.assertRaises(AssertionError):
            smol.baseline_check(point, 'test', reference)
        reference = json.loads(json.dumps(point))
        reference['contrasts']['GH-G']['task_effects']['0'] = .2
        with self.assertRaisesRegex(ValueError, 'per-task'):
            smol.baseline_check(point, 'test', reference)

    def test_image_cache_rejects_pickle_and_nonfinite_arrays(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder) / 'cache.npz'
            np.savez(cache, emb=np.asarray({'unsafe': 1}, dtype=object))
            with self.assertRaisesRegex(ValueError, 'Legacy'):
                robustness.read_embedding_cache(cache)
            np.savez(cache, ids=np.asarray(['episode|20']), feature_0=np.asarray([np.nan]))
            with self.assertRaisesRegex(ValueError, 'features'):
                robustness.read_embedding_cache(cache)
            np.savez(cache, ids=np.asarray(['episode|20']), feature_0=np.asarray([1., 2.]))
            np.testing.assert_array_equal(robustness.read_embedding_cache(cache)['episode|20'], [1., 2.])


if __name__ == '__main__':
    unittest.main()
