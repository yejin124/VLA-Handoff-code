"""Full-refit, paired scene bootstrap for the frozen Bridge J=64 follow-up.

python code/bridge_sampling_analysis.py COLLECTION --workers 24
Requires the complete validated collection, including raw handoff features.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

import bridge_sampling_replay as replay
import ral_confirm_analysis as confirm
import ral_nested_fit as fit

BLOCKS = {'G': ('G',), 'H': ('H',), 'GH': ('G', 'H'), 'S': ('S',), 'GS': ('G', 'S')}
CONTRASTS = {'GH-G': ('GH', 'G'), 'GH-Spread': ('GH', 'Spread'), 'H-S': ('H', 'S'),
             'GH-GS': ('GH', 'GS'), 'S-Spread': ('S', 'Spread')}
PRIMARY = [(f'{seed}_64', contrast) for seed in ('original', 'independent')
           for contrast in ('GH-Spread', 'H-S')]
N_BOOTSTRAP = 2000


def plain(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def load_views(out):
    replay.verify(out)
    collection, _ = replay.receipt(out)
    with np.load(out / 'dataset.npz', allow_pickle=False) as z:
        dataset = {k: z[k].item() if z[k].ndim == 0 else z[k] for k in z.files}
    for k in ('tasks', 'units'):
        dataset[k] = tuple(int(x) for x in dataset[k])
    dataset['ids'] = [r['id'] for r in collection['rows']]
    parts = {f'{seed}_{j}': {'Spread': [], 'S': []}
             for seed in ('original', 'independent') for j in replay.JS}
    for row in collection['rows']:
        with np.load(replay.result_path(out, row['id']), allow_pickle=False) as z:
            for seed in ('original', 'independent'):
                for j in replay.JS:
                    scalar, features = replay.spread_features(z[seed][:j])
                    parts[f'{seed}_{j}']['Spread'].append(scalar)
                    parts[f'{seed}_{j}']['S'].append(features)
    views = {k: dict(dataset, **{n: np.asarray(v) for n, v in p.items()}) for k, p in parts.items()}
    np.testing.assert_array_equal(views['original_8']['Spread'], dataset['Spread'])
    np.testing.assert_array_equal(views['original_8']['S'], dataset['S8'])
    keep = np.asarray([i.startswith('data/simpler/confirm/') for i in dataset['ids']])
    row_keys = ('ids', 'task', 'scene', 'replica', 'query_step', 'y', 'G', 'E', 'H', 'Spread', 'S', 'S8')
    new = {k: dict(d, **{n: np.asarray(d[n])[keep] for n in row_keys})
           for k, d in views.items() if k.endswith('_64')}
    return {'full': views, 'new_only': new}


def evaluate_views(views, counts=None, predictions=False):
    # The same training rows and G/H appear at every J and seed within one draw.
    # Cache only their deterministic transformations; discard this cache after every draw/scope.
    original_transform = fit.transform_blocks
    cache = {}

    def transform(dataset, training, names, max_components):
        if tuple(names) != ('G', 'H', 'S'):
            raise ValueError('Unexpected feature blocks')
        key = (training.tobytes(), max_components)
        if key not in cache:
            cache[key] = original_transform(dataset, training, ('G', 'H'), max_components)
        return dict(cache[key], **original_transform(dataset, training, ('S',), max_components))

    fit.transform_blocks = transform
    try:
        return {k: fit.evaluate(d, counts, save_predictions=predictions and k in ('original_8', 'original_64', 'independent_64'),
                                blocks=BLOCKS, contrasts=CONTRASTS) for k, d in views.items()}
    finally:
        fit.transform_blocks = original_transform


def correlation(left, right):
    if len(left) < 2 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return None
    value = float(spearmanr(left, right).statistic)
    return value if np.isfinite(value) else None


def repeatability(views, counts):
    ref = views['original_8']
    indices = np.repeat(np.arange(len(counts)), counts)
    results = {}
    for j in replay.JS:
        left, right = views[f'original_{j}']['Spread'], views[f'independent_{j}']['Spread']
        task = {}
        for t in ref['tasks']:
            ix = indices[ref['task'][indices] == t]
            task[str(t)] = correlation(left[ix], right[ix])
        results[str(j)] = {'pooled': correlation(left[indices], right[indices]), 'task': task,
                           'mean_task': float(np.mean(list(task.values()))) if all(v is not None for v in task.values()) else None}
    return results


def initialize(datasets):
    global DATASETS
    DATASETS = datasets
    threadpool_limits(limits=1)


def attempt(index):
    result = {'index': index}
    for scope, views in DATASETS.items():
        d = next(iter(views.values()))
        counts = confirm.counts_for(index, {(replay.CELL, replay.STEP): d})[(replay.CELL, replay.STEP)]
        result[scope] = {k: fit.compact_fit(v) for k, v in evaluate_views(views, counts).items()}
        if scope == 'full':
            result['repeatability'] = repeatability(views, counts)
    return result


def summarize(point, draws):
    valid = [r for r in draws if all(v['status'] == 'evaluated' for scope in ('full', 'new_only') for v in r[scope].values())]
    valid_indices = {r['index'] for r in valid}
    output = {'attempts': len(draws), 'valid_joint': len(valid), 'invalid_indices': [r['index'] for r in draws if r['index'] not in valid_indices],
              'bootstrap_seed': confirm.SEED, 'scopes': {}, 'primary_holm': {}, 'changes': {}, 'repeatability': {}}
    if len(valid) < 100:
        raise ValueError(f'Only {len(valid)} valid joint bootstrap draws; insufficient')
    for scope in ('full', 'new_only'):
        output['scopes'][scope] = {}
        for key, estimate in point[scope].items():
            item = {k: v for k, v in estimate.items() if k != 'predictions'}
            for group, field in [('methods', 'mean_auroc'), ('contrasts', 'mean')]:
                for method, value in item[group].items():
                    vector = np.asarray([r[scope][key][group][method] for r in valid])
                    value['ci95'] = np.quantile(vector, [.025, .975]).tolist()
                    if group == 'contrasts':
                        value['one_sided_p'] = float((1 + (vector <= 0).sum()) / (1 + len(vector)))
            output['scopes'][scope][key] = item
    ps = [output['scopes']['full'][k]['contrasts'][c]['one_sided_p'] for k, c in PRIMARY]
    for (key, contrast), p_holm in zip(PRIMARY, confirm.holm(ps)):
        value = output['scopes']['full'][key]['contrasts'][contrast]
        output['primary_holm'][key + '/' + contrast] = dict(value, holm_p=p_holm, positive_under_holm=p_holm < .05)
    for seed in ('original', 'independent'):
        for earlier in (8, 32):
            for group, field, methods in [('methods', 'mean_auroc', ('Spread', 'S')),
                                           ('contrasts', 'mean', ('GH-Spread', 'H-S'))]:
                for method in methods:
                    old, new = f'{seed}_{earlier}', f'{seed}_64'
                    name = f'{seed}/{method}/64-minus-{earlier}'
                    vector = np.asarray([r['full'][new][group][method] - r['full'][old][group][method] for r in valid])
                    output['changes'][name] = {'estimate': point['full'][new][group][method][field] - point['full'][old][group][method][field],
                                              'ci95': np.quantile(vector, [.025, .975]).tolist()}
    for j, values in point['repeatability'].items():
        output['repeatability'][j] = values
        for field in ('pooled', 'mean_task'):
            vector = [r['repeatability'][j][field] for r in valid if r['repeatability'][j][field] is not None]
            values[field + '_ci95'] = np.quantile(vector, [.025, .975]).tolist() if len(vector) >= 100 else None
            values[field + '_valid'] = len(vector)
        values['task_intervals'] = {}
        for task, estimate in values['task'].items():
            vector = [r['repeatability'][j]['task'][task] for r in valid if r['repeatability'][j]['task'][task] is not None]
            values['task_intervals'][task] = {'estimate': estimate, 'valid': len(vector),
                                             'ci95': np.quantile(vector, [.025, .975]).tolist() if len(vector) >= 100 else None}
    return output


def baseline_check(point, reference):
    published = json.loads(Path(reference).read_text())
    expected = published['cells'][replay.CELL + '|20']
    for group, names, field in [('methods', ('H', 'GH', 'S', 'Spread'), 'mean_auroc'),
                                ('contrasts', ('GH-G', 'GH-Spread', 'H-S'), 'mean')]:
        for name in names:
            np.testing.assert_allclose(point[group][name][field], expected[group][name]['estimate'], rtol=0, atol=1e-12)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('collection', type=Path)
    parser.add_argument('--workers', type=int, default=24)
    parser.add_argument('--reference-results', type=Path,
                        help='Optional archived J8 analysis for numerical comparison; raw data suffice without it')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    out = args.collection / 'analysis'
    out.mkdir(exist_ok=False)
    receipt = {'created_utc': datetime.now(timezone.utc).isoformat(), 'code_sha256': replay.digest(__file__),
               'collection_sha256': replay.digest(args.collection / 'collection.json'),
               'run_manifest_sha256': replay.digest(args.collection / 'run_manifest.json'), 'bootstrap_draws': N_BOOTSTRAP,
               'numpy': np.__version__, 'workers': args.workers,
               'reference_results_sha256': replay.digest(args.reference_results) if args.reference_results else None}
    (out / 'analysis_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    threadpool_limits(limits=1)
    datasets = load_views(args.collection)
    point = {scope: evaluate_views(views, predictions=True) for scope, views in datasets.items()}
    if any(v['status'] != 'evaluated' for s in point.values() for v in s.values()):
        raise ValueError('Point estimates not evaluable')
    if args.reference_results:
        baseline_check(point['full']['original_8'], args.reference_results)
    # Independent cache sanity: the original function must reproduce the cached original-J8 fit.
    uncached = fit.evaluate(datasets['full']['original_8'], blocks=BLOCKS, contrasts=CONTRASTS)
    if fit.compact_fit(uncached) != fit.compact_fit(point['full']['original_8']):
        raise ValueError('Transformation cache changes the fit')
    point['repeatability'] = repeatability(datasets['full'], np.ones(len(datasets['full']['original_8']['y']), dtype=int))
    (out / 'point_results.json').write_text(json.dumps(point, indent=2, default=plain, allow_nan=False) + '\n')
    draws = []
    with (out / 'bootstrap.jsonl').open('x') as log:
        with get_context('fork').Pool(args.workers, initializer=initialize, initargs=(datasets,)) as pool:
            for r in pool.imap(attempt, range(N_BOOTSTRAP), chunksize=5):
                draws.append(r)
                log.write(json.dumps(r, default=plain, allow_nan=False) + '\n')
                if len(draws) % 50 == 0:
                    log.flush()
                    print(f'Bootstrap {len(draws)}/{N_BOOTSTRAP}', flush=True)
    result = summarize(point, draws)
    result['analysis_receipt_sha256'] = replay.digest(out / 'analysis_receipt.json')
    (out / 'results.json').write_text(json.dumps(result, indent=2, default=plain, allow_nan=False) + '\n')
    print(json.dumps(result['primary_holm'], indent=2), flush=True)


if __name__ == '__main__':
    main()
