"""Confirmatory cohorts (plan committed 2026-11-05 before the new collection; Section IV-B and Supplementary Section S3 of the article).

The frozen nested pipeline (ral_nested_fit.evaluate: G, E, H blocks, readouts, folds) on episodes that no
earlier analysis has seen. Only the loader (new seeds, replicate counts, unused initial states) and the
resampling unit table differ from the earlier nested-controls analysis (not included). CPU only.
    RAL_RAW_ROOT=<repo root with the raw records> python code/ral_confirm_analysis.py --workers 12
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import zlib
from multiprocessing import get_context
import os
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from analyze_icassp_trials import artifact
from icassp_conditions import image_seeds
from ral_nested_features import external40, geometry, policy_seed
from ral_nested_fit import CONTRASTS, METHODS, compact_fit, evaluate
from recheck_restoration_allocation import Reader

ATTEMPTS, SEED, MIN_VALID = 2000, 20261106, 100
UNUSED = tuple(range(15)) + tuple(range(35, 50))
# cell -> (sources [(folder, seed, replicates)], suite, landmarks, severity, max_steps)
COHORTS = {
    'gr00t_bridge_confirm': ([('data/simpler/restore/continue', 20261105, 3), ('data/simpler/confirm', 20261106, 9)],
                             'simpler_bridge', (10, 20), .75, 120),
    'gr00t_spatial_confirm': ([('data/icassp2027/realistic/spatial_confirm/gr00t', 20261109, 2)],
                              'libero_spatial', (40, 80), 1.0, 280),
    'gr00t_object_confirm': ([('data/icassp2027/realistic/object_confirm/gr00t', 20261110, 2)],
                             'libero_object', (40, 80), 1.0, 280),
    # secondary cells added by the 2026-11-06 amendment
    'gr00t_bridge_confirm_new': ([('data/simpler/confirm', 20261106, 9)], 'simpler_bridge', (10, 20), .75, 120),
    'pi05_spatial_confirm': ([('data/icassp2027/realistic/spatial_confirm/pi05', 20261111, 2)],
                             'libero_spatial', (40, 80), 1.0, 280),
    'pi05_object_confirm': ([('data/icassp2027/realistic/object_confirm/pi05', 20261112, 2)],
                            'libero_object', (40, 80), 1.0, 280),
    # tau0 cohorts (amendment fixed before collection; Holm family over the two suites)
    'tau0_spatial': ([('data/icassp2027/realistic/spatial_tau0/tau0', 20261009, 1)], 'libero_spatial', (40, 80), 1.0, 280),
    'tau0_object': ([('data/icassp2027/realistic/object_tau0/tau0', 20261010, 1)], 'libero_object', (40, 80), 1.0, 280),
    # SmolVLA cohorts (amendment fixed before collection; Holm family over the two suites)
    'smolvla_spatial': ([('data/icassp2027/realistic/spatial_corrected/smolvla', 20261009, 1)], 'libero_spatial', (40, 80), 1.0, 280),
    'smolvla_object': ([('data/icassp2027/realistic/object_corrected/smolvla', 20261010, 1)], 'libero_object', (40, 80), 1.0, 280),
    # exploratory cells added by the plan amendment (reference rotation conversion, states 15-34)
    'molmoact_spatial': ([('data/icassp2027/realistic/spatial_corrected/molmoact', 20261009, 1)],
                         'libero_spatial', (40, 80), 1.0, 280),
    'molmoact_object': ([('data/icassp2027/realistic/object_corrected/molmoact', 20261010, 1)],
                        'libero_object', (40, 80), 1.0, 280),
}
MAIN_STATES = tuple(range(15, 35))
PRIMARY = (('gr00t_bridge_confirm', 20), ('gr00t_spatial_confirm', 80))
# Holm families of one-sided GH-G tests, each fixed before its cohorts were collected or analysed
FAMILIES = {'primary': PRIMARY,                                               # the two primary tests
            'tau0': (('tau0_spatial', 80), ('tau0_object', 80)),              # tau0: Spatial, Object
            'smolvla': (('smolvla_spatial', 80), ('smolvla_object', 80))}     # SmolVLA: Spatial, Object


def load(root, cell, reader):
    sources, suite, landmarks, severity, max_steps = COHORTS[cell]
    bridge = suite == 'simpler_bridge'
    tasks, scenes = (tuple(range(4)), tuple(range(24))) if bridge else (tuple(range(10)),
                     MAIN_STATES if cell.startswith(('molmoact', 'tau0', 'smolvla')) else UNUSED)
    rows, episodes, failures, offset, checkpoints = {s: [] for s in landmarks}, 0, 0, 0, set()
    for folder, seed, reps in sources:
        base = root / folder
        configs = {}
        for path in sorted(base.glob('shard*/config.json')):
            cfg = reader.json(path)
            if (cfg['seed'], cfg['suite'], cfg['replicates'], cfg['draws'], cfg['max_steps']) != (seed, suite, reps, 8, max_steps):
                raise ValueError(f'Unexpected configuration {path}')
            if (cfg['severity'] if bridge else cfg['severities']['occluder']) != severity:
                raise ValueError(f'Unexpected severity {path}')
            if not bridge and cfg.get('apply_to_wrist') is not True:
                raise ValueError('Expected both LIBERO cameras corrupted')
            checkpoints.add(reader.json(path.parent / 'server.json')['checkpoint_id'])
            configs[path.parent] = cfg
        paths = sorted(base.glob('shard*/t*_e*_r*_occluder/summary.json'))
        records = [reader.json(p) for p in paths]
        ids = {(r['task'], r['episode'], r['replicate']) for r in records}
        if len(ids) != len(records) or ids != {(t, e, r) for t in tasks for e in scenes for r in range(reps)}:
            raise ValueError(f'Incomplete or unexpected grid in {folder}: {len(records)} records')
        for path, s in zip(paths, records):
            cfg = configs[path.parent.parent]
            if s.get('complete') is not True or s['condition'] != 'occluder' or s['config_sha256'] != cfg['config_sha256']:
                raise ValueError(f'Bad summary {path}')
            task, scene, rep = s['task'], s['episode'], s['replicate']
            trace = [json.loads(x) for x in reader.read(path.parent / 'queries.jsonl').splitlines() if x.strip()]
            pair = (seed, suite, task, scene, rep)
            for i, q in enumerate(trace):
                if q['query'] != i or q['policy_seed'] != policy_seed(*pair, 'policy', i):
                    raise ValueError(f'Query order or seed mismatch {path}')
                if q['corruption_seeds'] != image_seeds(pair, 'occluder', q['step'])[:1 if bridge else 2]:
                    raise ValueError(f'Corruption seed mismatch {path}')
            g = geometry(seed, suite, task, scene, rep, 1 if bridge else 2, severity)
            episodes += 1
            failures += not s['success']
            for step in landmarks:
                if s['steps'] <= step:
                    continue
                q = max((q for q in trace if q['step'] <= step), key=lambda q: q['step'])
                spread, old_e, h = artifact(path.parent, q, reader.hashes)
                rows[step].append({'id': str(path.parent), 'task': task, 'scene': scene, 'replica': offset + rep,
                                   'query_step': q['step'], 'y': int(not s['success']), 'G': g,
                                   'E': external40(old_e, 'euler_sxyz' if bridge else 'axisangle'),
                                   'H': h.astype(np.float64), 'Spread': spread})
        offset += reps
    if len(checkpoints) != 1:
        raise ValueError('A cohort mixes checkpoints')
    audit = {'episodes': episodes, 'failures': failures, 'prevalence': failures / episodes, 'landmarks': {}}
    data = {}
    for step, rs in rows.items():
        counts = {t: Counter(r['y'] for r in rs if r['task'] == t) for t in tasks}
        eligible = .1 <= failures / episodes <= .8 and all(counts[t][0] and counts[t][1] for t in tasks)
        audit['landmarks'][str(step)] = {'active': len(rs), 'eligible': eligible,
                                         'task_counts': {str(t): [sum(counts[t].values()), counts[t][1]] for t in tasks}}
        data[(cell, step)] = {'cell': cell, 'landmark': step, 'bridge': bridge, 'tasks': tasks, 'units': scenes,
                              'eligible': eligible, 'ids': [r['id'] for r in rs],
                              **{k: np.asarray([r[k] for r in rs]) for k in
                                 ('task', 'scene', 'replica', 'query_step', 'y', 'G', 'E', 'H', 'Spread')}}
    return data, audit


def counts_for(index, datasets):
    """Whole-unit bootstrap: scene (Bridge) or initial state (LIBERO) IDs per task, all replicates together;
    one draw of unit multiplicities per (cohort, task), shared by both landmarks and every fold."""
    # one independent stream per (cohort, task): a cell's draws do not depend on which other cells are analysed
    table, out = {}, {}
    for (cell, _), d in sorted(datasets.items()):
        for t in d['tasks']:
            if (cell, t) not in table:
                rng = np.random.default_rng(np.random.SeedSequence([SEED, index, zlib.crc32(cell.encode()), int(t)]))
                units = np.asarray(d['units'])
                drawn = Counter(rng.choice(units, len(units)).tolist())
                table[(cell, t)] = drawn
    for key, d in datasets.items():
        out[key] = np.asarray([table[(d['cell'], int(t))][int(s)] for t, s in zip(d['task'], d['scene'])], dtype=int)
    return out


def init(datasets):
    global DATASETS
    DATASETS = datasets
    threadpool_limits(limits=1)


def attempt(index):
    counts = counts_for(index, DATASETS)
    return {f'{c}|{s}': compact_fit(evaluate(d, counts[(c, s)])) if d['eligible'] else {'status': 'ineligible'}
            for (c, s), d in DATASETS.items()}


def holm(ps):
    order, out, run = np.argsort(ps), [0.0] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, (len(ps) - r) * ps[i])
        out[i] = min(1.0, run)
    return out


def family_tests(cells):
    """Holm-adjusted one-sided GH-G tests for every family whose cells are all present; a partly present family is
    an error. An ineligible or unevaluated cell counts as p = 1 (not confirmed), as the plans state."""
    out = {}
    for family, members in FAMILIES.items():
        keys = [f'{c}|{s}' for c, s in members]
        present = [k in cells for k in keys]
        if not any(present):
            continue
        if not all(present):
            raise ValueError(f'Holm family {family} needs all of {keys} in one run')
        ps = [cells[k]['contrasts']['GH-G'].get('one_sided_p', 1.0) if cells[k]['status'] == 'evaluated' else 1.0
              for k in keys]
        out[family] = {k: {'GH-G': cells[k]['contrasts'].get('GH-G'), 'holm_p': h, 'confirmed': h < .05}
                       for k, h in zip(keys, holm(ps))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--output', type=Path, default=Path('data/confirmatory_rerun'))
    ap.add_argument('--cells', default=','.join(COHORTS), help='comma-separated cohorts to analyse (default: all)')
    a = ap.parse_args()
    cells = a.cells.split(',')
    if not set(cells) <= set(COHORTS):
        raise ValueError(f'Unknown cohorts: {set(cells) - set(COHORTS)}')
    root = Path(os.environ.get('RAL_RAW_ROOT', Path(__file__).resolve().parents[1]))
    reader, datasets, audit = Reader(), {}, {}
    for cell in cells:
        d, au = load(root, cell, reader)
        datasets.update(d)
        audit[cell] = au
    point = {f'{c}|{s}': evaluate(d) if d['eligible'] else {'status': 'ineligible'} for (c, s), d in datasets.items()}
    with get_context('fork').Pool(a.workers, initializer=init, initargs=(datasets,)) as pool:  # workers inherit module state
        draws = pool.map(attempt, range(ATTEMPTS), chunksize=10)
    res = {'plan': 'confirmatory plan of 2026-11-05 (article Section IV-B, Supplementary Section S3)', 'attempts': ATTEMPTS, 'seed': SEED, 'cohorts': audit, 'cells': {}}
    for key, est in point.items():
        valid = [d[key] for d in draws if d[key]['status'] == 'evaluated']
        cell = {'status': est['status'], 'valid': len(valid), 'methods': {}, 'contrasts': {}}
        if est['status'] == 'evaluated':
            for kind, names, field in (('methods', METHODS, 'mean_auroc'), ('contrasts', CONTRASTS, 'mean')):
                for n in names:
                    v = {'estimate': est[kind][n][field]}
                    if kind == 'contrasts':
                        v['task_effects'] = est[kind][n]['task_effects']
                    if len(valid) >= MIN_VALID:
                        arr = np.asarray([x[kind][n] for x in valid])
                        v['ci95'] = np.quantile(arr, [.025, .975]).tolist()
                        if kind == 'contrasts':
                            v['one_sided_p'] = float((1 + np.sum(arr <= 0)) / (1 + len(arr)))
                    cell[kind][n] = v
        res['cells'][key] = cell
    res['families'] = family_tests(res['cells'])
    res['primary'] = res['families'].get('primary')
    res['input_sha256'] = reader.hashes and {'files': len(reader.hashes)}
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / 'confirm_results.json').write_text(json.dumps(res, indent=2) + '\n')
    print(json.dumps({'cohorts': audit, 'families': res['families']}, indent=1))


if __name__ == '__main__':
    main()
