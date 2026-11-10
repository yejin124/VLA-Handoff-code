"""Learned action-spread readout versus the handoff readout: a supervision-matched control specified before running.

Question: is the handoff advantage over spread only an advantage of a supervised readout over a raw statistic?
Spread block S (per query, from the J=8 recorded action chunks, shape J x H_a x 7):
    log std over draws, averaged over the horizon, per coordinate        (7)
    log std over draws, averaged over the coordinates, per horizon step   (H_a)
    log of the scalar spread used in the paper                            (1)
S is fitted with exactly the frozen pipeline (standardize, L2 logistic C=1, task-held-out folds with a reserved
task, same 2,000 full-refit bootstrap draws, seed 20261106). No PCA on S (it has at most 24 features).
Contrasts, one-sided bootstrap p as in the paper, reported for both primary cells, no further selection:
    H-S    handoff readout minus learned spread readout          (main: is C a supervision artifact?)
    GH-GS  scene+handoff minus scene+learned spread
    S-Spread  learned minus raw spread
Decision rule fixed now: if H-S is not significantly positive (p >= 0.05) on Bridge, the paper may not claim that
System 1 fails to express what the handoff carries on Bridge; it must say the gap is between a supervised readout
and the raw spread statistic.
Sanity: GH-G must agree with the raw-feature baseline fitted on the same inputs.

Usage: RAL_ROOT=<repo root with the raw records> python code/learned_spread_contrast.py <workers> <out_dir> [cells]
"""
import os
import sys

import numpy as np

ROOT = os.environ.get('RAL_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('RAL_RAW_ROOT', ROOT)
import ral_nested_fit as f  # noqa: E402
import ral_confirm_analysis as a  # noqa: E402

STASH = {}
_artifact = a.artifact


def spread_features(directory, query):
    with np.load(directory / query['artifact'], allow_pickle=False) as src:
        sd = np.log(src['action_samples'].std(axis=0, ddof=0) + 1e-8)  # H_a x 7
    return np.concatenate([sd.mean(axis=0), sd.mean(axis=1), [np.log(query['spread'] + 1e-8)]])


def artifact(directory, query, hashes):
    out = _artifact(directory, query, hashes)  # keeps every digest and consistency check
    STASH[(str(directory), query['step'])] = spread_features(directory, query)
    return out


_load = a.load


def load(root, cell, reader):
    data, audit = _load(root, cell, reader)
    for d in data.values():
        d['S'] = np.asarray([STASH[(i, int(q))] for i, q in zip(d['ids'], d['query_step'])])
    return data, audit


a.artifact, a.load = artifact, load
a.ATTEMPTS = int(os.environ.get('SMOKE_ATTEMPTS', a.ATTEMPTS))  # smoke test only; the reported run uses 2,000
f.BLOCKS.update({'S': ('S',), 'GS': ('G', 'S')})
f.CONTRASTS.update({'H-S': ('H', 'S'), 'GH-GS': ('GH', 'GS'), 'S-Spread': ('S', 'Spread'),
                    'H-Spread': ('H', 'Spread'), 'GH-Spread': ('GH', 'Spread')})
a.CONTRASTS, a.METHODS = f.CONTRASTS, tuple(f.BLOCKS) + ('Spread', 'G_RBF')
cells = sys.argv[3] if len(sys.argv) > 3 else 'gr00t_bridge_confirm,gr00t_spatial_confirm'
sys.argv = ['learned_spread_contrast', '--workers', sys.argv[1], '--output', sys.argv[2], '--cells', cells]
a.main()
