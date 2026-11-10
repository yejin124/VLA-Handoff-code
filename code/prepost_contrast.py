"""Pre/post-interface readouts on the frozen confirmatory bootstrap (docs/PROTOCOLS.md).

Adds the mean-pooled PRE (backbone output) and MID (after vlln LayerNorm) features captured by prepost_driver.py as
blocks P and M, fitted with exactly the frozen pipeline (standardize, PCA <= 30 fitted on the training rows, standardize,
L2 logistic C=1, task-held-out folds with a reserved task, 2,000 full-refit draws, seed 20261106). Episodes that failed
a no-op check are excluded (the plan allows at most 1% per cell).

Contrasts (one-sided bootstrap p as in the article): P-H (primary: pre minus post), M-H, P-M, GP-GH, P-G, GP-G, P-Spread.
GH-G is refitted from the same raw inputs and must agree across equivalent pipelines.

Usage: RAL_ROOT=<repo> [PREPOST_PCA=30] [PREPOST_C=1] python code/prepost_contrast.py <workers> <features_dir> <out_dir> [cells]
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = os.environ.get('RAL_ROOT', str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault('RAL_RAW_ROOT', ROOT)
import ral_nested_fit as f  # noqa: E402
import ral_confirm_analysis as a  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

PCA_COMPONENTS = int(os.environ.get('PREPOST_PCA', '30'))
C_VALUE = float(os.environ.get('PREPOST_C', '1'))
FEATURES = {}   # relative episode id -> (pre, mid)
RAW_ROOT = Path(os.environ['RAL_RAW_ROOT']).resolve()


def load_features(features_dir, cells):
    for cell in cells:
        with np.load(Path(features_dir) / f'{cell}.npz', allow_pickle=False) as z:
            for i, ok, pre, mid in zip(z['ids'], z['ok'], z['pre'], z['mid']):
                if ok:
                    FEATURES[str(i)] = (pre.astype(np.float64), mid.astype(np.float64))


def transform_blocks(dataset, train_indices, names=('G', 'E', 'H'), max_components=30):
    """ral_nested_fit.transform_blocks with PCA also applied to the wide blocks P and M."""
    transformed = {}
    for block in names:
        full = np.asarray(dataset[block], dtype=np.float64)
        scaler = StandardScaler().fit(full[train_indices])
        z = scaler.transform(full)
        if block in ('H', 'I', 'P', 'M'):
            components = min(max_components, len(train_indices) - 1, full.shape[1])
            if components < 1:
                raise ValueError('Insufficient PCA training rows')
            pca = PCA(n_components=components, svd_solver='randomized', random_state=0).fit(z[train_indices])
            z = pca.transform(z)
            scaler = StandardScaler().fit(z[train_indices])
            z = scaler.transform(z)
        transformed[block] = z
    return transformed


_load = a.load


def load(root, cell, reader):
    data, audit = _load(root, cell, reader)
    primary = dict(a.PRIMARY)[cell]
    data = {k: d for k, d in data.items() if k[1] == primary}   # the plan covers the primary landmark only
    for key, d in data.items():
        rel = [str(Path(i).resolve().relative_to(RAW_ROOT)) for i in d['ids']]
        keep = np.array([r in FEATURES for r in rel])
        dropped = int((~keep).sum())
        if dropped > 0.01 * len(rel):
            raise RuntimeError(f'{key}: {dropped} of {len(rel)} episodes lack a verified pre/post replay (plan limit 1%)')
        for name in ('task', 'scene', 'replica', 'query_step', 'y', 'G', 'E', 'H', 'Spread'):
            d[name] = d[name][keep]
        d['ids'] = [i for i, k in zip(d['ids'], keep) if k]
        rel = [r for r, k in zip(rel, keep) if k]
        d['P'] = np.asarray([FEATURES[r][0] for r in rel])
        d['M'] = np.asarray([FEATURES[r][1] for r in rel])
        audit.setdefault('prepost_excluded', {})[str(key[1])] = dropped
    return data, audit


_evaluate = f.evaluate


def evaluate(dataset, counts=None, **kwargs):
    kwargs.setdefault('C', C_VALUE)
    kwargs.setdefault('max_components', PCA_COMPONENTS)
    return _evaluate(dataset, counts, **kwargs)


def holm(ps):
    order, out, run = np.argsort(ps), [0.0] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, (len(ps) - r) * ps[i])
        out[i] = min(1.0, run)
    return out


def main():
    workers, features_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    cells = sys.argv[4] if len(sys.argv) > 4 else 'gr00t_bridge_confirm,gr00t_spatial_confirm'
    load_features(features_dir, cells.split(','))
    f.transform_blocks = transform_blocks
    a.load, a.evaluate = load, evaluate
    a.ATTEMPTS = int(os.environ.get('SMOKE_ATTEMPTS', a.ATTEMPTS))   # smoke tests only; the reported run uses 2,000
    f.BLOCKS.update({'P': ('P',), 'M': ('M',), 'GP': ('G', 'P'), 'GM': ('G', 'M')})
    f.CONTRASTS.update({'P-H': ('P', 'H'), 'M-H': ('M', 'H'), 'P-M': ('P', 'M'), 'GP-GH': ('GP', 'GH'),
                        'P-G': ('P', 'G'), 'GP-G': ('GP', 'G'), 'P-Spread': ('P', 'Spread'), 'H-Spread': ('H', 'Spread')})
    a.CONTRASTS, a.METHODS = f.CONTRASTS, tuple(f.BLOCKS) + ('Spread', 'G_RBF')
    sys.argv = ['prepost_contrast', '--workers', workers, '--output', out_dir, '--cells', cells]
    a.main()
    path = Path(out_dir) / 'confirm_results.json'
    res = json.loads(path.read_text())
    keys = [k for k in (f'{c}|{s}' for c, s in a.PRIMARY) if k in res['cells'] and res['cells'][k]['status'] == 'evaluated']
    ps = [res['cells'][k]['contrasts']['P-H'].get('one_sided_p', 1.0) for k in keys]
    res['prepost'] = {'plan': 'docs/PROTOCOLS.md', 'pca_components': PCA_COMPONENTS, 'C': C_VALUE,
                      'P-H': {k: {'estimate': res['cells'][k]['contrasts']['P-H']['estimate'],
                                  'ci95': res['cells'][k]['contrasts']['P-H'].get('ci95'),
                                  'one_sided_p': p, 'holm_p': h, 'loss_detected': h < .05}
                              for k, p, h in zip(keys, ps, holm(ps))}}
    path.write_text(json.dumps(res, indent=2) + '\n')
    print(json.dumps(res['prepost'], indent=1))


if __name__ == '__main__':
    main()
