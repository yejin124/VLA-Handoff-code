"""Experiment 2 analysis (docs/PROTOCOLS.md).

Usage: python code/injection_handoff_analysis.py data/injection/exp2_bridge
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
from injection_validation import validate_episode
from injection_resume import sha256

LOG110 = float(np.log(1.10))
B, SEED = 2000, 20261106


def load(out):
    eps = []
    for f in sorted(glob.glob(str(out / 'results_*.jsonl'))):
        eps += [json.loads(x) for x in Path(f).read_text().splitlines()]
    expected_ids = json.loads((out / 'directions.json').read_text())['ids']
    expected = set(expected_ids)
    if not expected or len(expected) != len(expected_ids):
        raise ValueError('Direction metadata must contain a nonempty, unique cohort')
    ids = [e['id'] for e in eps]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate episode ids in the replay result files')
    if set(ids) != expected:
        raise ValueError(f'Replay cohort mismatch: {len(expected - set(ids))} missing, {len(set(ids) - expected)} unexpected episodes')
    for episode in eps:
        validate_episode(episode)
    manifest = out / 'run_manifest.json'
    if manifest.exists():
        record = json.loads(manifest.read_text())
        for key, name in [('directions_sha256', 'directions.npz'), ('direction_metadata_sha256', 'directions.json')]:
            if record[key] != sha256(out / name):
                raise ValueError('Direction files do not match the replay manifest')
        manifest_hash = sha256(manifest)
        if any(e.get('run_manifest_sha256') != manifest_hash for e in eps):
            raise ValueError('Replay records do not match the run manifest')
    elif any('run_manifest_sha256' in e for e in eps):
        raise ValueError('Replay records reference a missing run manifest')
    return eps


def scene_key(eid):
    # .../t{task}_e{episode}_r{rep}_occluder -> scene = (task, episode): replicates from every collection source
    # share the physical scene (4 tasks x 24 scenes = 96 clusters)
    t, e = eid.rsplit('/', 1)[1].split('_')[:2]
    return f'{t}|{e}'


def boot(values, tasks, scenes, stat=np.mean):
    """Resample scenes within tasks; values are per-episode."""
    rng = np.random.default_rng(SEED)
    values, tasks, scenes = np.asarray(values), np.asarray(tasks), np.asarray(scenes)
    groups = {t: {s: np.flatnonzero((tasks == t) & (scenes == s)) for s in np.unique(scenes[tasks == t])}
              for t in np.unique(tasks)}
    draws = []
    for _ in range(B):
        idx = []
        for t, g in groups.items():
            keys = list(g)
            for s in rng.choice(keys, len(keys)):
                idx.extend(g[s])
        draws.append(stat(values[idx]))
    return float(stat(values)), np.quantile(draws, [.025, .975]).tolist()


def main():
    out = Path(sys.argv[1])
    eps = load(out)
    rows = []
    for e in eps:
        res = {(x['dir'], x['alpha']): x for x in e['results']}
        base = res[('baseline', 0.0)]
        s0, m0 = np.log(base['spread']), np.asarray(base['mean_action'])
        rows.append(dict(id=e['id'], task=e['task'], scene=scene_key(e['id']), parity=base['parity_max_abs'],
                         manip=max(x['manip_rel_err'] for x in e['results'] if x['dir'] != 'baseline'),
                         dS={k: np.log(v['spread']) - s0 for k, v in res.items() if k[0] != 'baseline'},
                         dA_raw={k: float(np.linalg.norm(np.asarray(v['mean_action']) - m0)) for k, v in res.items() if k[0] != 'baseline'},
                         base_norm=float(np.linalg.norm(m0))))
    norm = np.median([r['base_norm'] for r in rows])
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError('Median baseline action norm must be finite and positive')
    tasks, scenes = [r['task'] for r in rows], [r['scene'] for r in rows]
    checks = {'episodes': len(rows), 'parity_failures': int(sum(r['parity'] > 1e-5 for r in rows)),
              'manip_failures': int(sum(r['manip'] > 0.05 for r in rows)), 'max_manip_rel_err': max(r['manip'] for r in rows)}
    if checks['parity_failures'] or checks['manip_failures']:
        raise RuntimeError(f'Invalid replay; the plan aborts the cell: {checks}')
    rand = [f'rand{k}' for k in range(1, 5)]
    res = {'checks': checks, 'alpha': {}}
    for a in (1.0, 2.0, 4.0):
        block = {}
        for name in ['risk', 'spread'] + rand:
            block[f'dS_{name}'] = boot([r['dS'][(name, a)] for r in rows], tasks, scenes)
            block[f'dA_{name}'] = boot([r['dA_raw'][(name, a)] / norm for r in rows], tasks, scenes)
        block['dS_rand_mean'] = boot([np.mean([r['dS'][(k, a)] for k in rand]) for r in rows], tasks, scenes)
        block['dA_rand_mean'] = boot([np.mean([r['dA_raw'][(k, a)] for k in rand]) / norm for r in rows], tasks, scenes)
        block['D_risk_minus_rand'] = boot([r['dS'][('risk', a)] - np.mean([r['dS'][(k, a)] for k in rand]) for r in rows], tasks, scenes)
        block['DA_risk_minus_rand'] = boot([(r['dA_raw'][('risk', a)] - np.mean([r['dA_raw'][(k, a)] for k in rand])) / norm
                                            for r in rows], tasks, scenes)
        res['alpha'][str(a)] = block
    pc = res['alpha']['2.0']['dS_spread']
    prim = res['alpha']['2.0']['D_risk_minus_rand']
    positive_control = pc[1][0] > LOG110
    if not positive_control:
        verdict = 'inconclusive: positive control failed (d_spread does not raise spread by 10%)'
    elif prim[1][1] < LOG110:
        verdict = 'C supported: d_risk raises spread no more than 10% beyond random directions'
    elif prim[1][0] > LOG110:
        verdict = 'transfer: spread follows the risk direction'
    else:
        verdict = 'inconclusive'
    res['primary'] = {'alpha': 2.0, 'D': prim, 'margin_log': LOG110, 'positive_control_dS_spread': pc,
                      'positive_control_passed': bool(positive_control), 'verdict': verdict}
    (out / 'analysis.json').write_text(json.dumps(res, indent=1) + '\n')
    print(json.dumps({'checks': checks, 'primary': res['primary']}, indent=1))
    for a in ('1.0', '2.0', '4.0'):
        b = res['alpha'][a]
        print(a, {k: (round(v[0], 3), [round(x, 3) for x in v[1]]) for k, v in b.items()
                  if k in ('dS_risk', 'dS_spread', 'dS_rand_mean', 'D_risk_minus_rand', 'dA_risk', 'dA_rand_mean')})


if __name__ == '__main__':
    main()
