"""Synthetic export checks: compact presentation must preserve adverse results."""
import importlib.util
import ast
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

SPEC = importlib.util.spec_from_file_location(
    'compact_supp', Path(__file__).resolve().parents[1] / 'code/make_compact_supp_tables.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_rows(directory, name, rows):
    (directory / name).write_text(r'\midrule' + '\n' + '\n'.join(
        ' & '.join(row) + r'\\' for row in rows) + '\n')


def inputs(directory):
    # Deliberately negative/ineligible estimates, not measurements from the paper.
    bad = '$-0.999$ [$-1.000$, $+0.001$]'
    image = [['A', suite, condition, policy, '--', '0.123', '--', '0.456', bad]
             for suite in ('Spatial', 'Object')
             for condition in ('Moved goal', 'Relocated goal')
             for policy in (MODULE.PI, 'GR00T', 'MolmoAct2')]
    write_rows(directory, 'supp_image_encoders.tex', image)
    cells = [[cohort, step, '--', '--', '--', '--', bad]
             for cohort, steps in [('Bridge (all)', ('10', '20')),
                                   ('Bridge (new only)', ('10', '20')),
                                   ('LIBERO-Spatial', ('40', '80')),
                                   ('LIBERO-Object', ('40', '80'))] for step in steps]
    write_rows(directory, 'supp_confirm_cells.tex', cells)
    write_rows(directory, 'supp_confirm_image.tex', [[c, s, '--', '--', bad]
                for c, s in sorted(MODULE.IMAGE_CONTROL)])
    policies = [[policy, suite, step, '20 (10)', '--', '--', '--', bad, '0.999', '--', '--']
                for step in ('80', '40') for suite in MODULE.SUITES
                for policy in (MODULE.PI, 'GR00T', 'MolmoAct2', MODULE.TAU, 'SmolVLA')
                if (policy, suite, step) != ('GR00T', 'Spatial', '80')]
    write_rows(directory, 'supp_confirm_policies.tex', policies)
    requery = [[policy, condition, step, '20/20 (0)', '--', '--', '--', '--', '--', '--']
               for policy in (MODULE.PI, 'MolmoAct2')
               for condition in ('Occluder', 'Moved goal', 'Relocated goal')
               for step in ('40', '80')]
    write_rows(directory, 'supp_requery.tex', requery)
    lines = [r'\midrule']
    for cohort in ('GR00T / Bridge', 'SmolVLA / Spatial', 'SmolVLA / Object'):
        lines.append(r'\multicolumn{5}{l}{\textbf{' + cohort + r'}}\\')
        for seed in ('Original', 'Independent'):
            for comparison in ('Scene + handoff minus raw spread', 'Handoff minus learned spread'):
                lines.append(' & '.join([seed, comparison, bad, '$0.998$', '$0.999$']) + r'\\')
    (directory / 'supp_sampling_primary.tex').write_text('\n'.join(lines) + '\n')


class CompactSupplementTests(unittest.TestCase):
    def test_negative_and_ineligible_cells_survive_every_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)
            inputs(source)
            output = source / 'out'
            MODULE.render(source, output)
            self.assertEqual(len(list(output.glob('*.tex'))), 5)
            policies = (output / 'supp_policy_summary.tex').read_text()
            self.assertEqual(policies.count('$-0.999$'), 19)
            self.assertEqual(policies.count(' & 0.999 & -- & --'), 19)
            sampling = (output / 'supp_sampling_summary.tex').read_text()
            self.assertEqual(sampling.count(' & $0.998$ & $0.999$'), 12)
            self.assertEqual(sampling.count('Original &'), 6)
            self.assertEqual(sampling.count('Independent &'), 6)
            requery = (output / 'supp_requery_summary.tex').read_text()
            self.assertEqual(requery.count('20/20 (0) & -- & -- & --'), 6)
            goal = (output / 'supp_goal_summary.tex').read_text()
            self.assertEqual(goal.count(' & -- & -- & 0.456 &'), 12)

    def test_missing_late_input_does_not_write_partial_output(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)
            inputs(source)
            (source / 'supp_sampling_primary.tex').unlink()
            with self.assertRaises(FileNotFoundError):
                MODULE.render(source, source / 'out')
            self.assertFalse((source / 'out').exists())

    def test_duplicate_protocol_cell_is_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)
            inputs(source)
            path = source / 'supp_confirm_policies.tex'
            lines = path.read_text().splitlines()
            lines[2] = lines[1]
            path.write_text('\n'.join(lines) + '\n')
            with self.assertRaisesRegex(ValueError, 'distinct protocol cells'):
                MODULE.render(source, source / 'out')
            self.assertFalse((source / 'out').exists())

    def test_changed_input_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)
            inputs(source)
            path = source / 'supp_requery.tex'
            path.write_text(path.read_text().replace('20/20 (0) &', '20/20 (0) & extra &', 1))
            with self.assertRaisesRegex(ValueError, 'expected 10 columns'):
                MODULE.render(source, source / 'out')
            self.assertFalse((source / 'out').exists())

    def test_missing_planned_image_control_is_not_an_untested_dash(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)
            inputs(source)
            path = source / 'supp_confirm_image.tex'
            path.write_text('\n'.join(path.read_text().splitlines()[:-1]) + '\n')
            with self.assertRaisesRegex(ValueError, 'distinct protocol cells'):
                MODULE.render(source, source / 'out')
            self.assertFalse((source / 'out').exists())

    def test_duplicate_sampling_seed_preserves_previous_output_on_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)
            inputs(source)
            output = source / 'out'
            MODULE.render(source, output)
            before = {p.name: p.read_bytes() for p in output.iterdir()}
            path = source / 'supp_sampling_primary.tex'
            path.write_text(path.read_text().replace('Independent &', 'Original &', 1))
            with self.assertRaisesRegex(ValueError, 'distinct protocol cells'):
                MODULE.render(source, output)
            self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})

    def test_fresh_requery_formatter_keeps_all_six_ineligible_cells(self):
        path = Path(__file__).resolve().parents[1] / 'code/gen_realistic_tables.py'
        tree = ast.parse(path.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'requery')
        namespace = dict(POLICIES=[('pi05', MODULE.PI), ('gr00t', 'GR00T'), ('molmoact', 'MolmoAct2')],
            FAMILIES=[('occluder', 'Occluder'), ('lighting', 'Dim light'),
                      ('moved', 'Moved goal'), ('relocated', 'Relocated goal')],
            D='/synthetic', load=lambda name: {}, counts=lambda paths: (20, 20),
            folds=lambda *args: [], aurocs=lambda *args: [], evaluable=lambda *args: False,
            med_iqr=lambda values: '--', glob=SimpleNamespace(glob=lambda path: []))
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)
            inputs(source)
            (source / 'supp_requery.tex').write_text(namespace['requery']())
            MODULE.render(source, source / 'out')
            result = (source / 'out/supp_requery_summary.tex').read_text()
            self.assertEqual(result.count('20/20 (0) & -- & -- & --'), 6)
            self.assertNotIn('Dim light', result)
            self.assertNotIn('GR00T', result)


if __name__ == '__main__':
    unittest.main()
