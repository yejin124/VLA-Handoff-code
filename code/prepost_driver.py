"""Pre/post-interface replay driver (docs/PROTOCOLS.md).

  python code/prepost_driver.py run <out_dir> <cell> <port> [<port> ...]
  python code/prepost_driver.py collect <out_dir> <cell>

`run` replays the primary-analysis query of every episode of a confirmatory cell (gr00t_bridge_confirm at step 20 or
gr00t_spatial_confirm at step 80) through serve_prepost_gr00t.py servers, routing each episode to a server on the GPU
model that recorded it, and writes one npz per episode with the pooled PRE/MID/POST features and the parity checks.
Finished episodes are skipped on restart. `collect` aggregates the episode files into <out_dir>/<cell>.npz and
<out_dir>/<cell>_summary.json.
"""
import base64
import io
import json
import os
import socket
import struct
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ral_confirm_analysis import COHORTS, PRIMARY  # noqa: E402

ROOT = Path(os.environ.get('RAL_RAW_ROOT', HERE.parent)).resolve()
LANDMARK = dict(PRIMARY)


def encode(value):
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(value), allow_pickle=False)
    return base64.b64encode(buffer.getvalue()).decode()


def exact_read(conn, n):
    chunks = bytearray()
    while len(chunks) < n:
        part = conn.recv(n - len(chunks))
        if not part:
            raise ConnectionError("Policy server disconnected")
        chunks.extend(part)
    return bytes(chunks)


def query(conn, request):
    data = json.dumps(request, allow_nan=False).encode()
    conn.sendall(struct.pack('>I', len(data)) + data)
    n = struct.unpack('>I', exact_read(conn, 4))[0]
    if n > 64 * 1024 * 1024:
        raise ValueError('Unexpectedly large response')
    response = json.loads(exact_read(conn, n))
    if 'error' in response:
        raise RuntimeError(response['error'])
    return response


def cohort(cell):
    """Same episode and query selection as ral_confirm_analysis.load for the primary landmark."""
    sources, suite, _landmarks, _severity, _max_steps = COHORTS[cell]
    step, is_bridge, rows = LANDMARK[cell], suite == 'simpler_bridge', []
    for folder, _seed, _reps in sources:
        for summ_path in sorted((ROOT / folder).glob('shard*/t*_e*_r*_occluder/summary.json')):
            s = json.loads(summ_path.read_text())
            if s.get('complete') is not True or s['condition'] != 'occluder' or s['steps'] <= step:
                continue
            d = summ_path.parent
            trace = [json.loads(x) for x in (d / 'queries.jsonl').read_text().splitlines() if x.strip()]
            q = max((q for q in trace if q['step'] <= step), key=lambda q: q['step'])
            device = json.loads((d.parent / 'server.json').read_text())['device']
            rows.append(dict(id=str(d.relative_to(ROOT)), dir=str(d), artifact=q['artifact'], step=int(q['step']),
                             task=int(s['task']), language=s['language'], seed=int(q['policy_seed']), device=device,
                             bridge=is_bridge, failure=int(not s['success'])))
    return rows


def episode_file(out, cell, row):
    return out / cell / (row['id'].replace('/', '__') + '.npz')


def worker(port, rows, out, cell):
    with socket.create_connection(('127.0.0.1', port)) as conn:
        for r in rows:
            target = episode_file(out, cell, r)
            if target.exists():
                continue
            with np.load(Path(r['dir']) / r['artifact'], allow_pickle=False) as rec:
                images = [encode(rec['agent_image'])] + ([] if r['bridge'] else [encode(rec['wrist_image'])])
                req = {'images': images, 'state': rec['state'].astype(np.float32).tolist(), 'task': r['language'],
                       'seed': r['seed'], 'J': 8}
                resp = query(conn, req)
                samples = np.asarray(resp['action_samples'], dtype=np.float32)
                post = np.asarray(resp['kv_pooled'], dtype=np.float32)
                recorded_handoff = rec['handoff'].astype(np.float32)
                recorded_samples = rec['action_samples'].astype(np.float32)
            np.savez(target, pre=np.asarray(resp['pre_pooled'], dtype=np.float32), mid=np.asarray(resp['mid_pooled'], dtype=np.float32),
                     post=post, recorded_handoff=recorded_handoff,
                     parity_actions_max_abs=np.float32(np.abs(samples - recorded_samples).max()),
                     post_vs_recorded_max_abs=np.float32(np.abs(post - recorded_handoff).max()),
                     recorded_handoff_max_abs=np.float32(np.abs(recorded_handoff).max()),
                     device=np.array(r['device']), server_device=np.array(resp.get('device', '')), step=np.int64(r['step']),
                     task=np.int64(r['task']), failure=np.int64(r['failure']), id=np.array(r['id']))


def run(out, cell, ports):
    rows = cohort(cell)
    devices = {}
    for p in ports:
        with socket.create_connection(('127.0.0.1', p)) as conn:
            devices[p] = query(conn, {'cmd': 'ping'})['device']
    shards = {p: [] for p in ports}
    for dev in sorted({r['device'] for r in rows}):
        match = [p for p in ports if devices[p] == dev]
        if not match:
            raise RuntimeError(f'no server on {dev}; recorded devices: {sorted({r["device"] for r in rows})}')
        sub = [r for r in rows if r['device'] == dev]
        for i, p in enumerate(match):
            shards[p] += sub[i::len(match)]
    (out / cell).mkdir(parents=True, exist_ok=True)
    print(json.dumps({'cell': cell, 'episodes': len(rows), 'servers': {str(p): (devices[p], len(shards[p])) for p in ports}}), flush=True)
    with ThreadPoolExecutor(max_workers=len(ports)) as executor:
        futures = [executor.submit(worker, p, shards[p], out, cell) for p in ports]
        for future in futures:
            future.result()  # re-raise a worker failure instead of exiting with a partial cohort
    print('done', sum(1 for r in rows if episode_file(out, cell, r).exists()), 'of', len(rows), flush=True)


def collect(out, cell):
    rows = cohort(cell)
    keep = {k: [] for k in ('pre', 'mid', 'post')}
    ids, parity, post_diff, rel_post_diff, devices, missing = [], [], [], [], [], []
    for r in rows:
        f = episode_file(out, cell, r)
        if not f.exists():
            missing.append(r['id'])
            continue
        with np.load(f, allow_pickle=False) as z:
            for k in keep:
                keep[k].append(z[k])
            ids.append(str(z['id']))
            parity.append(float(z['parity_actions_max_abs']))
            post_diff.append(float(z['post_vs_recorded_max_abs']))
            rel_post_diff.append(float(z['post_vs_recorded_max_abs'] / max(float(z['recorded_handoff_max_abs']), 1e-12)))
            devices.append(str(z['device']))
    parity, rel = np.asarray(parity), np.asarray(rel_post_diff)
    ok = (parity == 0) & (rel <= 2 ** -9)
    summary = {'cell': cell, 'step': LANDMARK[cell], 'episodes': len(rows), 'replayed': len(ids), 'missing': missing,
               'exact_action_parity': int((parity == 0).sum()), 'max_action_abs_diff': float(parity.max()) if len(parity) else None,
               'post_rel_diff_max': float(rel.max()) if len(rel) else None, 'post_rel_diff_median': float(np.median(rel)) if len(rel) else None,
               'passed_both_checks': int(ok.sum()), 'failed_ids': [i for i, good in zip(ids, ok) if not good],
               'devices': {d: int(sum(1 for x in devices if x == d)) for d in sorted(set(devices))}}
    np.savez(out / f'{cell}.npz', ids=np.array(ids), ok=ok, **{k: np.stack(v).astype(np.float32) for k, v in keep.items()})
    (out / f'{cell}_summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('missing', 'failed_ids')}, indent=1))
    print('missing', len(missing), 'failed checks', len(summary['failed_ids']))


if __name__ == '__main__':
    cmd, out, cell = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
    if cmd == 'run':
        run(out, cell, [int(p) for p in sys.argv[4:]])
    elif cmd == 'collect':
        collect(out, cell)
    else:
        raise SystemExit(__doc__)
