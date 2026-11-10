"""Numerical/provenance failures that must not enter the Bridge sensitivity analysis."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from bridge_sampling_replay import independent_seed, spread_features, validate_prefix, validate_result


class SamplingValidationTests(unittest.TestCase):
    def setUp(self):
        self.samples = np.random.default_rng(1).normal(size=(64, 4, 7)).astype(np.float32)

    def test_unchanged_prefix_and_reject_changed_or_nonfinite(self):
        original = self.samples[:8].copy()
        validate_prefix(self.samples, original, 'episode')
        for index, value in [((0, 0, 0), 999), ((63, 0, 0), np.nan)]:
            altered = self.samples.copy()
            altered[index] = value
            with self.assertRaises(ValueError):
                validate_prefix(altered, original, 'episode')
        with self.assertRaises(ValueError):
            validate_prefix(self.samples[:32], original, 'episode')

    def test_features_preserve_published_formula(self):
        samples = self.samples[:8]
        scalar, actual = spread_features(samples)
        sd = np.log(samples.std(axis=0, ddof=0) + 1e-8)
        expected = np.concatenate([sd.mean(axis=0), sd.mean(axis=1), [np.log(scalar + 1e-8)]])
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.shape, (12,))

    def test_full_result_rejects_identity_and_handoff_drift(self):
        row = {'id': 'data/episode', 'artifact_sha256': 'input', 'checkpoint': 'ckpt', 'device': 'GPU',
               'seed': 1, 'independent_seed': 2}
        result = {k: np.asarray(v) for k, v in dict(id=row['id'], collection_sha256='collection',
                  run_sha256='run', input_sha256='input', checkpoint='ckpt', device='GPU',
                  original_seed=1, independent_seed=2).items()}
        for kind in ('original', 'independent'):
            result[kind] = self.samples.copy()
            result[kind + '_handoff'] = np.ones(2048, dtype=np.float32)
        src = {'action_samples': self.samples[:8], 'handoff': np.ones(2048, dtype=np.float16)}
        validate_result(result, row, 'collection', 'run', src)
        with self.assertRaises(ValueError):
            validate_result(result, row, 'wrong collection', 'run', src)
        result['independent_handoff'][0] = 2
        with self.assertRaises(ValueError):
            validate_result(result, row, 'collection', 'run', src)

    def test_seed_mapping_is_stable(self):
        self.assertEqual(independent_seed('data/a'), independent_seed('data/a'))
        self.assertNotEqual(independent_seed('data/a'), independent_seed('data/b'))
        self.assertLess(independent_seed('data/a'), 2**31)

    def test_transform_cache_preserves_refits_and_is_cleared(self):
        from bridge_sampling_analysis import evaluate_views, BLOCKS, CONTRASTS
        import ral_nested_fit as fit
        from threadpoolctl import threadpool_limits
        rng = np.random.default_rng(51)
        n = 32
        d = {'tasks': (0, 1, 2, 3), 'task': np.repeat(np.arange(4), 8),
             'y': np.tile([0, 1], n // 2), 'G': rng.normal(size=(n, 2)),
             'H': rng.normal(size=(n, 6)), 'S': rng.normal(size=(n, 12)),
             'Spread': rng.uniform(size=n)}
        other = dict(d, S=rng.normal(size=(n, 12)), Spread=rng.uniform(size=n))
        before = fit.transform_blocks
        with threadpool_limits(limits=1):
            for counts in [np.ones(n, dtype=int), np.tile([1, 2, 1, 2], n // 4)]:
                cached = evaluate_views({'first': d, 'other': other}, counts)
                self.assertIs(fit.transform_blocks, before)
                for key, view in [('first', d), ('other', other)]:
                    direct = fit.evaluate(view, counts, blocks=BLOCKS, contrasts=CONTRASTS)
                    self.assertEqual(fit.compact_fit(cached[key]), fit.compact_fit(direct))


if __name__ == '__main__':
    unittest.main()
