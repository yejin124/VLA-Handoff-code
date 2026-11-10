"""SmolVLA replay provenance, paired analysis and numerical failure gates."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import smol_sampling_replay as replay
import smol_sampling_analysis as analysis
import ral_nested_fit as fit


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.samples = np.random.default_rng(1).normal(size=(64, 10, 7)).astype(np.float32)

    def test_original_server_and_protocol_are_unchanged(self):
        from serve_smol_sampling import ORIGINAL
        for filename, expected in ORIGINAL.items():
            self.assertEqual(replay.digest(replay.CODE_ROOT / 'code' / filename), expected)

    def test_prefix_gate_checks_all_eight_chunks_and_rejects_nonfinite_tail(self):
        replay.validate_prefix(self.samples, self.samples[:8], 'episode')
        for index, value in [((7, 9, 6), 123), ((63, 0, 0), np.nan)]:
            altered = self.samples.copy(); altered[index] = value
            with self.assertRaises(ValueError):
                replay.validate_prefix(altered, self.samples[:8], 'episode')
        with self.assertRaises(ValueError):
            replay.validate_prefix(self.samples[:32], self.samples[:8], 'episode')
        replay.validate_prefix(self.samples[:8], self.samples[:8], 'pilot', draws=8)

    def test_population_spread_features_have_fixed_18_coordinates(self):
        amplitude = np.arange(1, 71, dtype=np.float32).reshape(10, 7)
        samples = np.stack([-amplitude, amplitude])
        scalar, features = replay.spread_features(samples)
        self.assertEqual(scalar, float(amplitude.mean()))
        np.testing.assert_allclose(features[:7], np.log(amplitude + 1e-8).mean(axis=0), rtol=0, atol=0)
        np.testing.assert_allclose(features[7:17], np.log(amplitude + 1e-8).mean(axis=1), rtol=0, atol=0)
        self.assertEqual(features[-1], np.log(scalar + 1e-8))
        self.assertEqual(features.shape, (18,))

    def test_independent_seed_matches_specification_and_uses_relative_identity(self):
        identity = 'data/icassp2027/realistic/spatial_corrected/smolvla/shard00/t0_e15_r0_occluder'
        digest = hashlib.sha256(('smol-J64-20261110|' + identity).encode()).digest()
        self.assertEqual(replay.independent_seed(identity), int.from_bytes(digest[:4], 'big') % 2**31)
        self.assertNotEqual(replay.independent_seed(identity), replay.independent_seed(identity + 'x'))

    def test_full_result_rejects_receipt_drift_and_independent_handoff_drift(self):
        row = dict(id='data/a', cell='smolvla_spatial', artifact_sha256='input', checkpoint='ckpt',
                   device='GPU', seed=1, independent_seed=2)
        values = {k: np.asarray(v) for k, v in dict(id=row['id'], cell=row['cell'], collection_sha256='collection',
                  run_sha256='run', input_sha256='input', checkpoint='ckpt', device='GPU',
                  original_seed=1, independent_seed=2).items()}
        for seed in ('original', 'independent'):
            values[seed] = self.samples.copy(); values[seed + '_handoff'] = np.ones(10240, dtype=np.float32)
        archived = dict(action_samples=self.samples[:8], handoff=np.ones(10240, dtype=np.float16))
        replay.validate_result(values, row, 'collection', 'run', archived)
        with self.assertRaises(ValueError):
            replay.validate_result(values, row, 'wrong', 'run', archived)
        values['independent_handoff'][0] = 1.01
        with self.assertRaises(ValueError):
            replay.validate_result(values, row, 'collection', 'run', archived)

    def test_response_requires_matching_seed_J_code_and_executed_action(self):
        row = {'id': 'data/a'}
        response = dict(seed=1, J=8, code_sha256='code', action_samples=self.samples[:8].tolist(),
                        actions=self.samples[0].tolist(), kv_pooled=np.ones(10240).tolist())
        replay.response_arrays(response, row, 1, 8, 'code')
        for field, value in [('seed', 2), ('J', 64), ('code_sha256', 'wrong'), ('actions', [[0] * 7] * 10)]:
            with self.assertRaises(ValueError):
                replay.response_arrays(dict(response, **{field: value}), row, 1, 8, 'code')

    def test_server_requires_original_runtime_fields_and_fingerprint_integrity(self):
        original = {key: key for key in replay.SERVER_FIELDS}
        runtime = {'packages': [['torch', '2.11.0']]}
        meta = dict(original, sampling_server_sha256='wrapper', sampling_runtime=runtime,
                    sampling_runtime_sha256=hashlib.sha256(json.dumps(runtime, sort_keys=True).encode()).hexdigest())
        sources = {'code/serve_smol_sampling.py': 'wrapper'}
        replay.validate_server(meta, original, sources)
        for field, value in [('device', 'different GPU'), ('protocol_module_sha256', 'different'),
                              ('sampling_runtime_sha256', 'broken')]:
            with self.assertRaises(ValueError):
                replay.validate_server(dict(meta, **{field: value}), original, sources)

    def test_archived_input_hashes_and_state_guard(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(replay, 'ROOT', Path(temporary)):
            parent = Path(temporary) / 'data/a'; parent.mkdir(parents=True)
            image = np.zeros((256, 256, 3), dtype=np.uint8)
            image_hash = hashlib.sha256(str((image.shape, str(image.dtype))).encode() + image.tobytes()).hexdigest()
            np.savez(parent / 'q0008.npz', agent_image=image, wrist_image=image, state=np.zeros(8, dtype=np.float32),
                     action_samples=self.samples[:8], executed_actions=self.samples[0], handoff=np.ones(10240, dtype=np.float16))
            row = dict(id='data/a', artifact='q0008.npz', artifact_sha256=replay.digest(parent / 'q0008.npz'),
                       image_sha256=[image_hash, image_hash], state=[0] * 8)
            replay.read_input(row)
            for field, value in [('artifact_sha256', 'bad'), ('state', [1] * 8), ('artifact', '../q0008.npz')]:
                with self.assertRaises(ValueError):
                    replay.read_input(dict(row, **{field: value}))
            for path in ['/data/a', 'data/../../escape', 'other/a']:
                with self.assertRaises(ValueError):
                    replay.raw_path(path)


class AnalysisTests(unittest.TestCase):
    def synthetic(self):
        rng = np.random.default_rng(51); n = 32
        d = dict(tasks=(0, 1, 2, 3), task=np.repeat(np.arange(4), 8), y=np.tile([0, 1], n // 2),
                 G=rng.normal(size=(n, 2)), H=rng.normal(size=(n, 6)), S=rng.normal(size=(n, 18)),
                 Spread=rng.uniform(size=n))
        return d, dict(d, S=rng.normal(size=(n, 18)), Spread=rng.uniform(size=n))

    def test_cached_transforms_keep_real_refits_and_match_uncached_for_two_resamples(self):
        d, other = self.synthetic(); original = fit.transform_blocks
        with threadpool_limits(limits=1):
            for counts in [np.ones(len(d['y']), dtype=int), np.tile([1, 2, 1, 2], len(d['y']) // 4)]:
                with patch.object(fit.LogisticRegression, 'fit', autospec=True,
                                  side_effect=fit.LogisticRegression.fit) as refit:
                    cached = analysis.evaluate_views({'first': d, 'other': other}, counts)
                    self.assertEqual(refit.call_count, 2 * len(d['tasks']) * len(analysis.BLOCKS))
                self.assertIs(fit.transform_blocks, original)
                for key, data in [('first', d), ('other', other)]:
                    direct = fit.evaluate(data, counts, blocks=analysis.BLOCKS, contrasts=analysis.CONTRASTS)
                    self.assertEqual(fit.compact_fit(cached[key]), fit.compact_fit(direct))

    def test_cache_rejects_different_GH_rows_and_restores_function_after_error(self):
        d, other = self.synthetic(); original = fit.transform_blocks
        changed = dict(other, H=other['H'] + 1)
        with self.assertRaises(ValueError):
            analysis.evaluate_views({'one': d, 'two': changed})
        with patch.object(fit, 'evaluate', side_effect=RuntimeError('fit failed')):
            with self.assertRaises(RuntimeError):
                analysis.evaluate_views({'one': d})
        self.assertIs(fit.transform_blocks, original)

    def test_spearman_handles_undefined_and_reports_every_task(self):
        d, _ = self.synthetic()
        views = {f'{seed}_{j}': d for seed in ('original', 'independent') for j in replay.JS}
        value = analysis.repeatability(views, np.ones(len(d['y']), dtype=int))
        for j in map(str, replay.JS):
            self.assertAlmostEqual(value[j]['pooled'], 1.0)
            self.assertEqual(set(value[j]['task']), {'0', '1', '2', '3'})
        self.assertIsNone(analysis.correlation(np.ones(3), np.arange(3)))

    def test_eight_contrast_holm_family_uses_joint_valid_views_per_cohort(self):
        keys = [f'{seed}_{j}' for seed in ('original', 'independent') for j in replay.JS]
        estimate = dict(status='evaluated', methods={'Spread': {'mean_auroc': .6}, 'S': {'mean_auroc': .7}},
                        contrasts={name: {'mean': .1} for name in analysis.CONTRASTS})
        point = dict(cells={c: {k: deepcopy(estimate) for k in keys} for c in replay.CELLS},
                     repeatability={c: {str(j): dict(pooled=.5, mean_task=.4, task={'0': .4}) for j in replay.JS}
                                    for c in replay.CELLS})
        compact = dict(status='evaluated', methods={'Spread': .6, 'S': .7},
                       contrasts={name: .1 for name in analysis.CONTRASTS})
        draws = [dict(index=i, cells={c: {k: deepcopy(compact) for k in keys} for c in replay.CELLS},
                      repeatability=deepcopy(point['repeatability'])) for i in range(120)]
        draws[0]['cells']['smolvla_spatial']['independent_16'] = {'status': 'invalid_fixed_task_classes'}
        for draw in draws:
            draw['cells']['smolvla_object']['independent_64']['contrasts']['H-S'] = -.1
        summary = analysis.summarize(point, draws)
        self.assertEqual(len(summary['primary_holm']), 8)
        self.assertEqual(summary['cells']['smolvla_spatial']['valid_joint'], 119)
        self.assertEqual(summary['cells']['smolvla_object']['valid_joint'], 120)
        self.assertEqual(summary['cells']['smolvla_spatial']['invalid_indices'], [0])
        self.assertFalse(summary['primary_holm']['smolvla_object/independent_64/H-S']['positive_under_holm'])
        for cell in replay.CELLS:
            self.assertEqual(set(summary['cells'][cell]['views']), set(keys))
            self.assertIn('0', summary['cells'][cell]['repeatability']['64']['task_intervals'])


if __name__ == '__main__':
    unittest.main()
