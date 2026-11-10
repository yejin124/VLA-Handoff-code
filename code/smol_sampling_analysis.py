"""Paired full-refit scene bootstrap for both SmolVLA J64 sensitivity cohorts.

python code/smol_sampling_analysis.py baseline COLLECTION
python code/smol_sampling_analysis.py analyze COLLECTION --workers 24
The baseline command uses archived J8 only; analyze requires all 384 validated replays.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import importlib.metadata
import json
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

import smol_sampling_replay as replay
import ral_confirm_analysis as confirm
import ral_nested_fit as fit

BLOCKS = {'G': ('G',), 'H': ('H',), 'GH': ('G', 'H'), 'S': ('S',), 'GS': ('G', 'S')}
CONTRASTS = {'GH-G': ('GH', 'G'), 'GH-Spread': ('GH', 'Spread'), 'H-S': ('H', 'S'),
             'GH-GS': ('GH', 'GS'), 'S-Spread': ('S', 'Spread'), 'H-Spread': ('H', 'Spread')}
PRIMARY = [(cell, f'{seed}_64', contrast) for cell in replay.CELLS for seed in ('original', 'independent')
           for contrast in ('GH-Spread', 'H-S')]
N_BOOTSTRAP, MIN_VALID = 2000, 100


def plain(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def baseline_check(point: dict, cell: str, published: dict, all_methods: bool = False) -> None:
    methods = tuple(published['methods']) if all_methods else ('G', 'H', 'GH', 'Spread', 'G_RBF')
    for group, names, field in [('methods', methods, 'mean_auroc'),
                                ('contrasts', ('GH-G', 'GH-Spread'), 'mean')]:
        for name in names:
            expected = published[group][name]
            np.testing.assert_allclose(point[group][name][field], expected.get('estimate', expected.get(field)),
                                       rtol=0, atol=1e-12, err_msg=f'{cell}/{group}/{name}')
    for contrast in ('GH-G', 'GH-Spread'):
        if point['contrasts'][contrast]['task_effects'] != published['contrasts'][contrast]['task_effects']:
            raise ValueError(f'Original per-task effects differ: {cell}/{contrast}')


def baseline(out: Path, reference: Path | None = None) -> dict:
    _, collection_hash = replay.receipt(out)
    result = dict(collection_sha256=collection_hash,
                  published_sha256=replay.digest(reference) if reference else None, cells={})
    published = json.loads(reference.read_text())['cells'] if reference else None
    threadpool_limits(limits=1)
    for cell in replay.CELLS:
        d = replay.dataset(out, cell)
        contrasts = dict(fit.CONTRASTS, **{'GH-Spread': ('GH', 'Spread')})
        point = fit.evaluate(d, contrasts=contrasts, save_predictions=True)
        if point['status'] != 'evaluated':
            raise ValueError(f'Original baseline is not evaluable: {cell}')
        if published is not None:
            baseline_check(point, cell, published[cell + '|80'], all_methods=True)
        result['cells'][cell] = point
    path = out / 'baseline_validation.json'
    with path.open('x') as handle:
        handle.write(json.dumps(result, indent=2, default=plain, allow_nan=False) + '\n')
    print(json.dumps({c: fit.compact_fit(v) for c, v in result['cells'].items()}, indent=2), flush=True)
    return result


def load_views(out: Path) -> dict:
    replay.verify(out)
    collection, _ = replay.receipt(out)
    cells = {}
    for cell in replay.CELLS:
        d = replay.dataset(out, cell)
        rows = [r for r in collection['rows'] if r['cell'] == cell]
        if d['ids'] != [r['id'] for r in rows]:
            raise ValueError('Dataset/result row order mismatch')
        parts = {f'{seed}_{j}': {'Spread': [], 'S': []}
                 for seed in ('original', 'independent') for j in replay.JS}
        for row in rows:
            with np.load(replay.result_path(out, row['id']), allow_pickle=False) as z:
                for seed in ('original', 'independent'):
                    for j in replay.JS:
                        scalar, features = replay.spread_features(z[seed][:j])
                        parts[f'{seed}_{j}']['Spread'].append(scalar)
                        parts[f'{seed}_{j}']['S'].append(features)
        cells[cell] = {k: dict(d, **{n: np.asarray(v) for n, v in p.items()}) for k, p in parts.items()}
        np.testing.assert_array_equal(cells[cell]['original_8']['Spread'], d['Spread'])
        np.testing.assert_array_equal(cells[cell]['original_8']['S'], d['S8'])
    return cells


def evaluate_views(views: dict, counts=None, predictions: bool = False) -> dict:
    # Only G/H deterministic transforms are cached within this cohort and this draw.
    # All readouts still pass through fit.evaluate and are actually fitted for each view.
    original, cache = fit.transform_blocks, {}
    reference = next(iter(views.values()))
    for d in views.values():
        for name in ('task', 'y', 'G', 'H'):
            if not np.array_equal(d[name], reference[name]):
                raise ValueError(f'Cannot share transforms across different {name} rows')

    def transform(d, training, names, max_components):
        if tuple(names) != ('G', 'H', 'S'):
            raise ValueError('Unexpected feature blocks')
        key = (training.tobytes(), max_components)
        if key not in cache:
            cache[key] = original(d, training, ('G', 'H'), max_components)
        return dict(cache[key], **original(d, training, ('S',), max_components))

    fit.transform_blocks = transform
    try:
        return {k: fit.evaluate(d, counts, save_predictions=predictions,
                                blocks=BLOCKS, contrasts=CONTRASTS) for k, d in views.items()}
    finally:
        fit.transform_blocks = original


def correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return None
    value = float(spearmanr(left, right).statistic)
    return value if np.isfinite(value) else None


def repeatability(views: dict, counts: np.ndarray) -> dict:
    ref = views['original_8']; indices = np.repeat(np.arange(len(counts)), counts)
    results = {}
    for j in replay.JS:
        left, right = views[f'original_{j}']['Spread'], views[f'independent_{j}']['Spread']
        task = {}
        for t in ref['tasks']:
            ix = indices[ref['task'][indices] == t]
            task[str(t)] = correlation(left[ix], right[ix])
        results[str(j)] = dict(pooled=correlation(left[indices], right[indices]), task=task,
                               mean_task=float(np.mean(list(task.values())))
                               if all(v is not None for v in task.values()) else None)
    return results


def initialize(datasets: dict) -> None:
    global DATASETS
    DATASETS = datasets
    threadpool_limits(limits=1)


def attempt(index: int) -> dict:
    data = {(cell, replay.STEP): views['original_8'] for cell, views in DATASETS.items()}
    counts = confirm.counts_for(index, data)
    result = dict(index=index, cells={}, repeatability={})
    for cell, views in DATASETS.items():
        weights = counts[(cell, replay.STEP)]
        result['cells'][cell] = {k: fit.compact_fit(v) for k, v in evaluate_views(views, weights).items()}
        result['repeatability'][cell] = repeatability(views, weights)
    return result


def interval(values: list) -> dict:
    finite = [v for v in values if v is not None]
    return dict(ci95=np.quantile(finite, [.025, .975]).tolist() if len(finite) >= MIN_VALID else None,
                valid=len(finite))


def summarize(point: dict, draws: list[dict]) -> dict:
    output = dict(attempts=len(draws), bootstrap_seed=confirm.SEED, cells={}, primary_holm={})
    for cell in replay.CELLS:
        valid = [r for r in draws if all(v['status'] == 'evaluated' for v in r['cells'][cell].values())]
        valid_ids = {r['index'] for r in valid}
        if len(valid) < MIN_VALID:
            raise ValueError(f'Insufficient valid draws: {cell}/{len(valid)}')
        cellout = dict(valid_joint=len(valid), invalid_indices=[r['index'] for r in draws if r['index'] not in valid_ids],
                       views={}, changes={}, repeatability={})
        for key, estimate in point['cells'][cell].items():
            item = deepcopy({k: v for k, v in estimate.items() if k != 'predictions'})
            for group in ('methods', 'contrasts'):
                for method, value in item[group].items():
                    vector = np.asarray([r['cells'][cell][key][group][method] for r in valid])
                    value['ci95'] = np.quantile(vector, [.025, .975]).tolist()
                    if group == 'contrasts':
                        value['one_sided_p'] = float((1 + (vector <= 0).sum()) / (1 + len(vector)))
            cellout['views'][key] = item
        for seed in ('original', 'independent'):
            for earlier in (8, 32):
                for group, field, methods in [('methods', 'mean_auroc', ('Spread', 'S')),
                                               ('contrasts', 'mean', ('GH-Spread', 'H-S'))]:
                    for method in methods:
                        old, new = f'{seed}_{earlier}', f'{seed}_64'
                        vector = [r['cells'][cell][new][group][method] - r['cells'][cell][old][group][method] for r in valid]
                        cellout['changes'][f'{seed}/{method}/64-minus-{earlier}'] = dict(
                            estimate=point['cells'][cell][new][group][method][field] - point['cells'][cell][old][group][method][field],
                            **interval(vector))
        for j, values in point['repeatability'][cell].items():
            entry = deepcopy(values)
            for field in ('pooled', 'mean_task'):
                entry[field + '_interval'] = interval([r['repeatability'][cell][j][field] for r in valid])
            entry['task_intervals'] = {t: interval([r['repeatability'][cell][j]['task'][t] for r in valid])
                                       for t in values['task']}
            cellout['repeatability'][j] = entry
        output['cells'][cell] = cellout
    ps = [output['cells'][c]['views'][k]['contrasts'][name]['one_sided_p'] for c, k, name in PRIMARY]
    for (cell, key, name), p_holm in zip(PRIMARY, confirm.holm(ps)):
        output['primary_holm'][f'{cell}/{key}/{name}'] = dict(
            output['cells'][cell]['views'][key]['contrasts'][name], holm_p=p_holm, positive_under_holm=p_holm < .05)
    return output


def analyze(collection: Path, workers: int) -> None:
    if workers < 1:
        raise ValueError('workers must be positive')
    threadpool_limits(limits=1)
    datasets = load_views(collection)
    archived_baseline = json.loads((collection / 'baseline_validation.json').read_text())
    if archived_baseline['collection_sha256'] != replay.digest(collection / 'collection.json'):
        raise ValueError('Baseline validation belongs to another collection')
    point = dict(cells={}, repeatability={})
    for cell, views in datasets.items():
        point['cells'][cell] = evaluate_views(views, predictions=True)
        if any(v['status'] != 'evaluated' for v in point['cells'][cell].values()):
            raise ValueError('Point estimates not evaluable')
        baseline_check(point['cells'][cell]['original_8'], cell, archived_baseline['cells'][cell])
        original = replay.dataset(collection, cell)
        fresh_baseline = fit.evaluate(original, contrasts=dict(fit.CONTRASTS, **{'GH-Spread': ('GH', 'Spread')}))
        baseline_check(fresh_baseline, cell, archived_baseline['cells'][cell], all_methods=True)
        uncached = fit.evaluate(views['original_8'], blocks=BLOCKS, contrasts=CONTRASTS)
        if fit.compact_fit(uncached) != fit.compact_fit(point['cells'][cell]['original_8']):
            raise ValueError('Transformation cache changes the fit')
        point['repeatability'][cell] = repeatability(views, np.ones(len(views['original_8']['y']), dtype=int))
    out = collection / 'analysis'; out.mkdir(exist_ok=False)
    receipt = dict(created_utc=datetime.now(timezone.utc).isoformat(), code_sha256=replay.digest(__file__),
                   collection_sha256=replay.digest(collection / 'collection.json'),
                   run_manifest_sha256=replay.digest(collection / 'run_manifest.json'), bootstrap_draws=N_BOOTSTRAP,
                   baseline_sha256=replay.digest(collection / 'baseline_validation.json'),
                   packages={name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'scikit-learn', 'threadpoolctl')},
                   workers=workers, primary_family_size=len(PRIMARY))
    (out / 'analysis_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    (out / 'point_results.json').write_text(json.dumps(point, indent=2, default=plain, allow_nan=False) + '\n')
    draws = []
    with (out / 'bootstrap.jsonl').open('x') as log:
        with get_context('fork').Pool(workers, initializer=initialize, initargs=(datasets,)) as pool:
            for result in pool.imap(attempt, range(N_BOOTSTRAP), chunksize=5):
                draws.append(result); log.write(json.dumps(result, default=plain, allow_nan=False) + '\n')
                if len(draws) % 50 == 0:
                    log.flush(); print(f'Bootstrap {len(draws)}/{N_BOOTSTRAP}', flush=True)
    result = summarize(point, draws)
    result['analysis_receipt_sha256'] = replay.digest(out / 'analysis_receipt.json')
    (out / 'results.json').write_text(json.dumps(result, indent=2, default=plain, allow_nan=False) + '\n')
    print(json.dumps(result['primary_holm'], indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('baseline', 'analyze')); parser.add_argument('collection', type=Path)
    parser.add_argument('--workers', type=int, default=24)
    parser.add_argument('--reference-results', type=Path, help='Optional archived analysis to check during baseline')
    args = parser.parse_args()
    if args.reference_results and args.command != 'baseline':
        parser.error('--reference-results applies only to baseline')
    if args.command == 'baseline':
        baseline(args.collection.resolve(), args.reference_results)
    else:
        analyze(args.collection.resolve(), args.workers)
