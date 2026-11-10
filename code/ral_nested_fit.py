"""Shared train-only transformations and full-refit scene bootstrap."""
from __future__ import annotations

from fractions import Fraction
import warnings

import numpy as np
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

METHODS = ('G', 'E', 'H', 'GH', 'GE', 'GEH', 'EH', 'Spread', 'G_RBF')
BLOCKS = {'G': ('G',), 'E': ('E',), 'H': ('H',), 'GH': ('G', 'H'),
          'GE': ('G', 'E'), 'GEH': ('G', 'E', 'H'), 'EH': ('E', 'H')}
CONTRASTS = {'GH-G': ('GH', 'G'), 'GEH-GE': ('GEH', 'GE'),
             'EH-E': ('EH', 'E'), 'H-G': ('H', 'G'), 'GH-G_RBF': ('GH', 'G_RBF')}


def weighted_auc(y, score, counts):
    y, score, counts = np.asarray(y), np.asarray(score), np.asarray(counts)
    if y.ndim != 1 or score.shape != y.shape or counts.shape != y.shape:
        raise ValueError('Mismatched AUROC inputs')
    if not np.isfinite(score).all() or np.any(counts < 0) or np.any(counts != counts.astype(int)):
        raise ValueError('Invalid scores or noninteger multiplicities')
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError('AUROC labels must be binary')
    pos, neg = y == 1, y == 0
    denominator = int(counts[pos].sum()) * int(counts[neg].sum())
    if not denominator:
        raise ValueError('AUROC requires both outcome classes')
    pair_weights = counts[pos, None] * counts[None, neg]
    difference = score[pos, None] - score[None, neg]
    twice_wins = int(np.sum(pair_weights * (2 * (difference > 0) + (difference == 0))))
    return Fraction(twice_wins, 2 * denominator)


def invalid_tasks(dataset, counts):
    result = []
    for task in dataset['tasks']:
        mask = dataset['task'] == task
        y, w = dataset['y'][mask], counts[mask]
        if not w[y == 0].sum() or not w[y == 1].sum():
            result.append(task)
    return result


def transform_blocks(dataset, train_indices, names=('G', 'E', 'H'), max_components=30):
    """Fit each block once; every nested readout shares these exact features. High-dimensional blocks
    (H, and the image embedding I when present) get training-only PCA."""
    transformed = {}
    for block in names:
        full = np.asarray(dataset[block], dtype=np.float64)
        scaler = StandardScaler().fit(full[train_indices])
        z = scaler.transform(full)
        if block in ('H', 'I'):
            components = min(max_components, len(train_indices) - 1, full.shape[1])
            if components < 1:
                raise ValueError('Insufficient PCA training rows')
            pca = PCA(n_components=components, svd_solver='randomized', random_state=0).fit(z[train_indices])
            z = pca.transform(z)
            scaler = StandardScaler().fit(z[train_indices])
            z = scaler.transform(z)
        transformed[block] = z
    return transformed


def evaluate(dataset, counts=None, save_predictions=False, blocks=None, contrasts=None, C=1, max_components=30):
    """Defaults reproduce the frozen pipeline; blocks/contrasts/C/max_components serve the prespecified
    robustness analyses only."""
    blocks = BLOCKS if blocks is None else blocks
    contrasts = CONTRASTS if contrasts is None else contrasts
    methods = tuple(blocks) + ('Spread', 'G_RBF')
    names = tuple(sorted({b for bs in blocks.values() for b in bs} | {'G'}))
    n = len(dataset['y'])
    counts = np.ones(n, dtype=int) if counts is None else np.asarray(counts, dtype=int)
    bad = invalid_tasks(dataset, counts)
    if bad:
        return {'status': 'invalid_fixed_task_classes', 'invalid_tasks': bad}
    scores = {m: np.full(n, np.nan) for m in methods}
    scores['Spread'] = dataset['Spread'].copy()
    folds = []
    for index, task in enumerate(dataset['tasks']):
        reserved = dataset['tasks'][(index + 1) % len(dataset['tasks'])]
        train_mask = (dataset['task'] != task) & (dataset['task'] != reserved)
        training = np.repeat(np.flatnonzero(train_mask), counts[train_mask])
        test = np.flatnonzero(dataset['task'] == task)
        if len(np.unique(dataset['y'][training])) != 2:
            return {'status': 'invalid_training_classes', 'invalid_tasks': [task]}
        z = transform_blocks(dataset, training, names, max_components)
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            for method, parts in blocks.items():
                x = np.concatenate([z[b] for b in parts], axis=1)
                fit = LogisticRegression(C=C, solver='liblinear', max_iter=2000, random_state=0)
                fit.fit(x[training], dataset['y'][training])
                scores[method][test] = fit.decision_function(x[test])
            model = SVC(C=1, kernel='rbf', gamma='scale')
            model.fit(z['G'][training], dataset['y'][training])
            scores['G_RBF'][test] = model.decision_function(z['G'][test])
        folds.append({'test_task': task, 'reserved_task': reserved,
                      'train_tasks': [t for t in dataset['tasks'] if t not in (task, reserved)],
                      'train_rows': len(training), 'test_rows_with_multiplicity': int(counts[test].sum())})
    task_auc = {}
    for task in dataset['tasks']:
        keep = dataset['task'] == task
        task_auc[task] = {m: weighted_auc(dataset['y'][keep], scores[m][keep], counts[keep]) for m in methods}
    out = {'status': 'evaluated', 'folds': folds, 'methods': {}, 'contrasts': {}}
    for method in methods:
        values = [task_auc[t][method] for t in dataset['tasks']]
        out['methods'][method] = {'mean_auroc': float(sum(values, Fraction()) / len(values)),
                                 'task_auroc': {str(t): float(task_auc[t][method]) for t in dataset['tasks']}}
    for name, (left, right) in contrasts.items():
        effects = {t: task_auc[t][left] - task_auc[t][right] for t in dataset['tasks']}
        out['contrasts'][name] = {'mean': float(sum(effects.values(), Fraction()) / len(effects)),
            'task_effects': {str(t): float(x) for t, x in effects.items()},
            'positive': sum(x > 0 for x in effects.values()), 'negative': sum(x < 0 for x in effects.values()),
            'tied': sum(x == 0 for x in effects.values())}
    if save_predictions:
        out['predictions'] = [{'id': dataset['ids'][i], 'task': int(dataset['task'][i]),
            'scene': int(dataset['scene'][i]), 'replica': int(dataset['replica'][i]),
            'query_step': int(dataset['query_step'][i]), 'failure': int(dataset['y'][i]),
            'scores': {m: float(scores[m][i]) for m in methods}} for i in range(n)]
    return out


def compact_fit(result):
    if result['status'] != 'evaluated':
        return result
    return {'status': 'evaluated',
            'methods': {m: v['mean_auroc'] for m, v in result['methods'].items()},
            'contrasts': {m: v['mean'] for m, v in result['contrasts'].items()}}
