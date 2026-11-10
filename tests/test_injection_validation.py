"""Regression checks for malformed replays and unsafe resume configurations (CPU only)."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from injection_handoff_analysis import load, main
from injection_resume import bind_run, completed_ids, require_new_directions
from injection_validation import CONDITIONS, validate_episode


def episode():
    rows = []
    for name, alpha in sorted(CONDITIONS):
        row = dict(dir=name, alpha=alpha, spread=0.1, mean_action=[0.01] * 28)
        row['parity_max_abs' if name == 'baseline' else 'manip_rel_err'] = 0.0
        rows.append(row)
    return dict(id='data/simpler/confirm/shard0/t0_e0_r0_occluder', task=0, failure=1, results=rows)


class ReplayValidationTests(unittest.TestCase):
    def test_complete_record(self):
        validate_episode(episode())

    def test_zero_baseline_norm_rejected_before_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            row = episode()
            row['results'][0]['mean_action'] = [0.0] * 28
            (out / 'directions.json').write_text(json.dumps({'ids': [row['id']]}))
            (out / 'results_5950.jsonl').write_text(json.dumps(row) + '\n')
            with patch.object(sys, 'argv', ['injection_handoff_analysis.py', tmp]):
                with self.assertRaisesRegex(ValueError, 'Median baseline action norm'):
                    main()
            self.assertFalse((out / 'analysis.json').exists())

    def test_missing_duplicate_nonfinite_and_failed_checks(self):
        base = episode()
        mutations = [
            lambda e: e['results'].pop(),
            lambda e: e['results'].__setitem__(1, copy.deepcopy(e['results'][0])),
            lambda e: e['results'][0].pop('parity_max_abs'),
            lambda e: e['results'][1].pop('manip_rel_err'),
            lambda e: e['results'][0].update(parity_max_abs=float('nan')),
            lambda e: e['results'][1].update(manip_rel_err=float('inf')),
            lambda e: e['results'][0].update(parity_max_abs=0.001),
            lambda e: e['results'][1].update(manip_rel_err=0.1),
            lambda e: e['results'][0].update(spread=0.0),
            lambda e: e['results'][0].update(mean_action=[0.0] * 27),
            lambda e: e['results'][0]['mean_action'].__setitem__(0, float('nan')),
            lambda e: e.update(task=1),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                changed = copy.deepcopy(base)
                mutate(changed)
                with self.assertRaises(ValueError):
                    validate_episode(changed)

    def test_resume_binding_and_new_direction_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            require_new_directions(out)
            manifest = {'ports': [5950], 'directions_sha256': 'original'}
            digest = bind_run(out, manifest)
            self.assertEqual(digest, bind_run(out, manifest))
            with self.assertRaises(ValueError):
                bind_run(out, dict(manifest, ports=[5951]))
            with self.assertRaises(ValueError):
                require_new_directions(out)
            row = episode() | {'run_manifest_sha256': digest}
            path = out / 'results_5950.jsonl'
            path.write_text(json.dumps(row) + '\n')
            self.assertEqual({row['id']}, completed_ids(path, {row['id']}, digest))
            with self.assertRaises(ValueError):
                completed_ids(path, {row['id']}, 'changed')
            with self.assertRaises(ValueError):
                completed_ids(path, set(), digest)
            path.write_text(path.read_text() * 2)
            with self.assertRaises(ValueError):
                completed_ids(path, {row['id']}, digest)

    def test_legacy_analysis_allowed_but_resume_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            row = episode()
            (out / 'directions.json').write_text(json.dumps({'ids': [row['id']]}))
            path = out / 'results_5950.jsonl'
            path.write_text(json.dumps(row) + '\n')
            self.assertEqual(1, len(load(out)))
            with self.assertRaises(ValueError):
                bind_run(out, {'ports': [5950]})
            row['run_manifest_sha256'] = 'missing-manifest'
            path.write_text(json.dumps(row) + '\n')
            with self.assertRaises(ValueError):
                load(out)


if __name__ == '__main__':
    unittest.main()
