"""Out-of-fold predictions of the frozen primary pipeline (point estimate, no bootstrap) for the two
primary cells; input to panel (b) of the results figure (Fig. 4). Same code dependency as spread_contrast.py.

Usage: RAL_ROOT=<repo root with the raw records> python code/primary_predictions.py <out.json>
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = os.environ.get('RAL_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ral_confirm_analysis as a  # noqa: E402
from ral_nested_fit import evaluate  # noqa: E402

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--reference-results', type=Path, help='Optional prior predictions to compare')
    args = parser.parse_args()
    reference = json.loads(args.reference_results.read_text()) if args.reference_results else None
    reader, out = a.Reader(), {}
    for cohort, step in a.PRIMARY:
        data, _ = a.load(Path(ROOT), cohort, reader)
        r = evaluate(data[(cohort, step)], save_predictions=True)
        if r['status'] != 'evaluated':
            raise ValueError(f'Cohort not evaluable: {cohort}, step {step}')
        key = f'{cohort}|{step}'
        gh, sp = r['methods']['GH']['mean_auroc'], r['methods']['Spread']['mean_auroc']
        if reference is not None:
            if abs(gh - reference[key]['GH']) > 1e-12 or abs(sp - reference[key]['Spread']) > 1e-12:
                raise ValueError(f'Predictions differ from supplied reference: {key}')
        out[key] = {'GH': gh, 'Spread': sp, 'predictions': r['predictions']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, allow_nan=False))


if __name__ == '__main__':
    main()
