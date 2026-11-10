"""Experiment 1 of docs/PROTOCOLS.md: input-level injection.

For each (policy, suite, condition in {occluder, moved}) pair the clean and perturbed episodes from the same
(task, initial state); compute dR (System 2 risk score change) and dS (log spread change) at the planned query and
test whether they are coupled within tasks.

Usage: python code/injection_input_level.py <raw_root> <out.json>
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

POLICIES = ('gr00t', 'molmoact', 'pi05')
SUITES = ('spatial', 'object')
CONDITIONS = ('occluder', 'moved')
RNG = np.random.default_rng(20261106)


def episodes(root, suite, policy):
    out = {}
    for d in sorted((root / f'{suite}_main' / policy).glob('shard*/t*_e*_r0_*')):
        summ = json.loads((d / 'summary.json').read_text())
        if not summ.get('complete'):
            continue
        out[(summ['task'], summ['episode'], summ['condition'])] = (d, summ)
    return out


def query_index(d, summ, condition):
    rows = [json.loads(line) for line in (d / 'queries.jsonl').read_text().splitlines()]
    if condition == 'moved':
        iv = summ['intervention']
        after = iv['start_step'] + iv['duration_steps']
        return next((r['query'] for r in rows if r['step'] >= after), None)
    return 0


def record(d, q):
    z = np.load(d / f'q{q:04d}.npz')
    samples = z['action_samples'].astype(np.float64)
    return z['handoff'].astype(np.float32), float(samples.std(axis=0, ddof=0).mean())


def readout(n_train, n_features):
    components = min(30, n_train - 1, n_features)
    if components < 1:
        raise ValueError('Insufficient training support for the planned readout')
    return make_pipeline(StandardScaler(), PCA(n_components=components, svd_solver='full'), StandardScaler(),
                         LogisticRegression(C=1.0, solver='liblinear', max_iter=2000, random_state=0))


def cell(eps, condition):
    pairs, unavailable_pairs = [], 0
    for (task, ep, cond), (d, summ) in eps.items():
        if cond != condition or (task, ep, 'clean') not in eps:
            continue
        dc, sc = eps[(task, ep, 'clean')]
        q = query_index(d, summ, condition)
        if q is None or q >= sc['queries']:
            unavailable_pairs += 1
            continue  # either episode ended before the matched post-move query
        hp, sp = record(d, q)
        hc, spc = record(dc, q)
        pairs.append(dict(task=task, episode=ep, fail=int(not summ['success']), hp=hp, hc=hc, sp=sp, sc=spc))
    tasks = sorted({p['task'] for p in pairs})
    for i, t in enumerate(tasks):
        reserved = tasks[(i + 1) % len(tasks)]
        train = [p for p in pairs if p['task'] not in (t, reserved)]
        y = np.array([p['fail'] for p in train])
        test = [p for p in pairs if p['task'] == t]
        if len(np.unique(y)) < 2:
            for p in test:
                p['dR'] = np.nan
            continue
        training = np.stack([p['hp'] for p in train])
        model = readout(*training.shape).fit(training, y)
        rp = model.decision_function(np.stack([p['hp'] for p in test]))
        rc = model.decision_function(np.stack([p['hc'] for p in test]))
        for p, a, b in zip(test, rp, rc):
            p['dR'], p['rc'] = float(a - b), float(b)
    for p in pairs:
        p['dS'] = float(np.log(p['sp']) - np.log(p['sc']))
    rhos = {}
    for t in tasks:
        tp = [p for p in pairs if p['task'] == t and np.isfinite(p['dR'])]
        if len(tp) >= 5:
            r = spearmanr([p['dR'] for p in tp], [p['dS'] for p in tp]).statistic
            if np.isfinite(r):
                rhos[t] = float(r)
    vals = np.array(list(rhos.values()))
    boot = [RNG.choice(vals, len(vals)).mean() for _ in range(2000)] if len(vals) else [np.nan]
    pos = int((vals > 0).sum())
    ok = [p for p in pairs if np.isfinite(p['dR'])]
    sd_r = float(np.std([p['rc'] for p in ok])) if ok else 0.0
    sd_s = float(np.std(np.log([p['sc'] for p in ok]))) if ok else 0.0
    return {'pairs': len(pairs), 'unavailable_pairs': unavailable_pairs, 'tasks_with_rho': len(vals), 'rho_by_task': rhos,
            'mean_rho': float(vals.mean()) if len(vals) else None,
            'rho_ci95': np.quantile(boot, [.025, .975]).tolist() if len(vals) else None,
            'tasks_rho_positive': pos,
            'sign_p_one_sided': float(binomtest(pos, len(vals), 0.5, alternative='greater').pvalue) if len(vals) else None,
            'mean_dR_std': float(np.mean([p['dR'] for p in ok]) / sd_r) if sd_r > 0 else None,
            'mean_dS_std': float(np.mean([p['dS'] for p in ok]) / sd_s) if sd_s > 0 else None,
            'share_dR_positive': float(np.mean([p['dR'] > 0 for p in ok])) if ok else None,
            'share_dS_positive': float(np.mean([p['dS'] > 0 for p in ok])) if ok else None}


def holm(ps):
    order, m, out, run = np.argsort(ps), len(ps), [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[i]))
        out[i] = run
    return out


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    res = {}
    for suite in SUITES:
        for policy in POLICIES:
            eps = episodes(root, suite, policy)
            for condition in CONDITIONS:
                res[f'{policy}|{suite}|{condition}'] = cell(eps, condition)
                r = res[f'{policy}|{suite}|{condition}']
                print(policy, suite, condition, r['pairs'], r['mean_rho'], r['tasks_rho_positive'], r['tasks_with_rho'],
                      r['sign_p_one_sided'], round(r['mean_dR_std'], 2) if r['mean_dR_std'] is not None else None,
                      round(r['mean_dS_std'], 2) if r['mean_dS_std'] is not None else None, flush=True)
    keys = list(res)
    adj = holm([res[k]['sign_p_one_sided'] if res[k]['sign_p_one_sided'] is not None else 1.0 for k in keys])
    for k, a in zip(keys, adj):
        res[k]['holm_p'] = a
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'plan': 'docs/PROTOCOLS.md', 'cells': res}, indent=1) + '\n')


if __name__ == '__main__':
    main()
