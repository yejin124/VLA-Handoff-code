"""Experiment 2 driver (docs/PROTOCOLS.md): handoff-level injection on GR00T Bridge.

  python code/injection_handoff_driver.py directions <out_dir>
  python code/injection_handoff_driver.py run <out_dir> <port> [<port> ...]

`directions` fits d_risk, d_spread and four random directions on the recorded step-20 handoffs of the Bridge
confirmatory cohort and writes directions.npz. `run` sends 19 queries per episode (baseline, d_risk x 3, d_spread x 3,
d_rand x 4 x 3) to injection servers and appends one JSON line per episode to results_<port>.jsonl. Restart with the
same inputs, directions, ordered ports and server configuration. A run manifest binds these settings;
legacy results without a manifest can be analyzed but cannot be resumed by this version.
"""
import json
import os
import socket
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RAL_ROOT = Path(os.environ.get('RAL_ROOT', Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepost_driver import encode, query  # noqa: E402
from injection_resume import bind_run, completed_ids, require_new_directions, sha256  # noqa: E402
from injection_validation import validate_episode  # noqa: E402

SOURCES = ('data/simpler/restore/continue', 'data/simpler/confirm')
STEP, ALPHAS, N_RAND = 20, (1.0, 2.0, 4.0), 4


def cohort():
    rows = []
    for src in SOURCES:
        for d in sorted((RAL_ROOT / src).glob('shard*/t*_occluder')):
            summ = json.loads((d / 'summary.json').read_text())
            if not summ.get('complete'):
                continue
            q = [json.loads(x) for x in (d / 'queries.jsonl').read_text().splitlines()]
            hit = [r for r in q if r['step'] == STEP]
            if not hit:
                continue  # ended before the landmark
            r = hit[0]
            z = np.load(d / r['artifact'])
            if 'agent_image' not in z.files:
                raise RuntimeError(f'{d.name}: no image at step {STEP}')
            device = json.loads((d.parent / 'server.json').read_text())['device']
            rows.append(dict(id=f'{src}/{d.parent.name}/{d.name}', dir=str(d), artifact=r['artifact'], task=summ['task'], device=device,
                             language=summ['language'], seed=r['policy_seed'], spread=r['spread'],
                             failure=int(not summ['success'])))
    return rows


def gradient(model, width):
    return model.decision_function(np.eye(width, dtype=np.float32)) - model.decision_function(np.zeros((1, width), np.float32))


def directions(out):
    require_new_directions(out)
    rows = cohort()
    H = np.stack([np.load(Path(r['dir']) / r['artifact'])['handoff'].astype(np.float32) for r in rows])
    y = np.array([r['failure'] for r in rows])
    s = np.log(np.array([r['spread'] for r in rows]))
    width = H.shape[1]
    pre = lambda: [StandardScaler(), PCA(n_components=30, svd_solver='full'), StandardScaler()]
    risk = make_pipeline(*pre(), LogisticRegression(C=1.0, solver='liblinear', max_iter=2000, random_state=0)).fit(H, y)
    sprd = make_pipeline(*pre(), Ridge(alpha=1.0)).fit(H, s)
    d_risk = gradient(risk, width)
    d_spread = sprd.predict(np.eye(width, dtype=np.float32)) - sprd.predict(np.zeros((1, width), np.float32))
    unit = lambda v: v / np.linalg.norm(v)
    u_risk = unit(d_risk)
    comps = PCA(n_components=30, svd_solver='full').fit(H).components_
    rand = []
    for k in range(1, N_RAND + 1):
        v = np.random.default_rng(k).standard_normal(30) @ comps
        v = v - (v @ u_risk) * u_risk
        rand.append(unit(v))
    dirs = {'risk': u_risk, 'spread': unit(d_spread), **{f'rand{k}': r for k, r in enumerate(rand, 1)}}
    sigma = {k: float(np.std(H @ v)) for k, v in dirs.items()}
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / 'directions.npz', **{k: v.astype(np.float32) for k, v in dirs.items()})
    meta = {'episodes': len(rows), 'failures': int(y.sum()), 'sigma': sigma,
            'cos_risk_spread': float(u_risk @ dirs['spread']),
            'ids': [r['id'] for r in rows]}
    (out / 'directions.json').write_text(json.dumps(meta, indent=1) + '\n')
    print(json.dumps({k: meta[k] for k in ('episodes', 'failures', 'sigma', 'cos_risk_spread')}, indent=1))


def conditions(out):
    z = np.load(out / 'directions.npz')
    sigma = json.loads((out / 'directions.json').read_text())['sigma']
    conds = [('baseline', 0.0, None)]
    for name in ['risk', 'spread'] + [f'rand{k}' for k in range(1, N_RAND + 1)]:
        for a in ALPHAS:
            conds.append((name, a, (a * sigma[name] * z[name]).astype(np.float32)))
    return conds


def worker(port, rows, conds, out, lock, manifest_sha256):
    path = out / f'results_{port}.jsonl'
    done = completed_ids(path, {r['id'] for r in rows}, manifest_sha256)
    with socket.create_connection(('127.0.0.1', port)) as conn:
        for r in rows:
            if r['id'] in done:
                continue
            rec = np.load(Path(r['dir']) / r['artifact'])
            base = {'images': [encode(rec['agent_image'])], 'state': rec['state'].astype(np.float32).tolist(),
                    'task': r['language'], 'seed': int(r['seed']), 'J': 8}
            res, kv0 = [], None
            for name, a, delta in conds:
                resp = query(conn, dict(base, **({} if delta is None else {'delta': delta.tolist()})))
                samples = np.asarray(resp['action_samples'], dtype=np.float32)
                kv = np.asarray(resp['kv_pooled'], dtype=np.float64)
                res.append({'dir': name, 'alpha': a, 'spread': float(samples.std(axis=0, ddof=0).mean()),
                            'mean_action': samples.mean(axis=0).ravel().tolist(), 'calls': resp.get('injection_calls', 0)})
                if name == 'baseline':
                    kv0 = kv
                    res[-1]['device'] = r['device']
                    res[-1]['parity_max_abs'] = float(np.abs(samples - rec['action_samples']).max())
                    res[-1]['handoff_vs_recorded_max_abs'] = float(np.abs(kv - rec['handoff'].astype(np.float64)).max())
                else:   # manipulation check: the pooled handoff moved by delta
                    res[-1]['manip_rel_err'] = float(np.linalg.norm((kv - kv0) - delta) / np.linalg.norm(delta))
            episode = {'id': r['id'], 'task': r['task'], 'failure': r['failure'], 'results': res,
                       'run_manifest_sha256': manifest_sha256}
            validate_episode(episode)
            line = json.dumps(episode)
            with lock, path.open('a') as f:
                f.write(line + '\n')


def run(out, ports):
    if not ports or len(set(ports)) != len(ports) or any(not 1 <= p <= 65535 for p in ports):
        raise ValueError('Provide unique TCP ports in [1, 65535]')
    rows, conds = cohort(), conditions(out)
    ids = [r['id'] for r in rows]
    if not ids or len(set(ids)) != len(ids) or ids != json.loads((out / 'directions.json').read_text())['ids']:
        raise ValueError('Current cohort differs from the cohort used to fit directions')
    if {p.name for p in out.glob('results_*.jsonl')} - {f'results_{p}.jsonl' for p in ports}:
        raise ValueError('Results from other ports exist; use the original ordered port list')
    lock = threading.Lock()
    devices, servers = {}, {}
    for p in ports:   # parity needs the recorded GPU model (plan amendment 1)
        with socket.create_connection(('127.0.0.1', p)) as conn:
            servers[str(p)] = query(conn, {'cmd': 'ping'})
            devices[p] = servers[str(p)]['device']
    shards = {p: [] for p in ports}
    for dev in sorted({r['device'] for r in rows}):
        match = [p for p in ports if devices[p] == dev]
        if not match:
            raise RuntimeError(f'no server on {dev}')
        sub = [r for r in rows if r['device'] == dev]
        for i, p in enumerate(match):
            shards[p] += sub[i::len(match)]
    manifest = {'version': 1, 'ports': ports, 'servers': servers, 'step': STEP, 'draws': 8,
                'conditions': [[name, alpha] for name, alpha, _ in conds],
                'directions_sha256': sha256(out / 'directions.npz'),
                'direction_metadata_sha256': sha256(out / 'directions.json'),
                'driver_sha256': sha256(Path(__file__)),
                'cohort': [{k: r[k] for k in ('id', 'task', 'failure', 'device', 'language', 'seed')} |
                           {'artifact_sha256': sha256(Path(r['dir']) / r['artifact'])} for r in rows],
                'shards': {str(p): [r['id'] for r in shards[p]] for p in ports}}
    manifest_sha256 = bind_run(out, manifest)
    for p in ports:
        completed_ids(out / f'results_{p}.jsonl', set(manifest['shards'][str(p)]), manifest_sha256)
    print({p: (devices[p], len(shards[p])) for p in ports}, flush=True)
    shards = [shards[p] for p in ports]
    with ThreadPoolExecutor(max_workers=len(ports)) as executor:
        futures = [executor.submit(worker, p, s, conds, out, lock, manifest_sha256) for p, s in zip(ports, shards)]
        for future in futures:
            future.result()  # re-raise a worker failure instead of exiting with a partial cohort


if __name__ == '__main__':
    mode, out = sys.argv[1], Path(sys.argv[2])
    if mode == 'directions':
        directions(out)
    elif mode == 'run':
        run(out, [int(p) for p in sys.argv[3:]])
    else:
        raise SystemExit(__doc__)
