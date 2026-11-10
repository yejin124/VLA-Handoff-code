"""Collect paired J=64 draws at the frozen Bridge primary queries.

python code/bridge_sampling_replay.py prepare OUTPUT
python code/bridge_sampling_replay.py run OUTPUT PORT [PORT ...]
python code/bridge_sampling_replay.py verify OUTPUT
Raw records and the original GR00T environment are required; results are never overwritten.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import tempfile
import threading
import time

import numpy as np

import prepost_driver as replay
from ral_confirm_analysis import load
from recheck_restoration_allocation import Reader

CODE_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get('RAL_RAW_ROOT', CODE_ROOT)).resolve()
CELL, STEP, JS = 'gr00t_bridge_confirm', 20, (8, 16, 32, 64)
PLAN = 'docs/PROTOCOLS.md'
SOURCES = ('code/bridge_sampling_replay.py', 'code/serve_bridge_sampling.py', 'code/prepost_driver.py', 'code/serve_simpler_gr00t.py',
           'code/serve_icassp_gr00t.py', 'code/ral_confirm_analysis.py', 'code/ral_nested_fit.py',
           'code/ral_nested_features.py', 'code/icassp_conditions.py', PLAN)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def independent_seed(identity):
    value = hashlib.sha256(('bridge-J64-20261110|' + identity).encode('utf-8')).digest()
    return int.from_bytes(value[:4], 'big') % 2**31


def spread_features(samples):
    sd = np.log(samples.std(axis=0, ddof=0) + 1e-8)
    spread = float(samples.std(axis=0, ddof=0).mean())
    return spread, np.concatenate([sd.mean(axis=0), sd.mean(axis=1), [np.log(spread + 1e-8)]])


def atomic_npz(path, **values):
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npz', delete=False) as f:
        temporary = Path(f.name)
        np.savez_compressed(f, **values)
    os.replace(temporary, path)


def prepare(out):
    if (out / 'collection.json').exists() or (out / 'dataset.npz').exists():
        raise ValueError('Prepared collection already exists; use run/verify rather than overwrite')
    reader = Reader()
    data, audit = load(ROOT, CELL, reader)
    d = data[(CELL, STEP)]
    rows = replay.cohort(CELL)
    by_id = {r['id']: r for r in rows}
    if not rows or len(by_id) != len(rows) or len(rows) != len(d['ids']) or not d['eligible']:
        raise ValueError('Unexpected or duplicate Bridge primary cohort')
    expected_ids = {Path(i).relative_to(ROOT).as_posix() for i in d['ids']}
    if len(expected_ids) != len(d['ids']) or set(by_id) != expected_ids:
        raise ValueError('Bridge replay identities differ from the validated loader')
    selected, original_s, checkpoints = [], [], set()
    for index, absolute in enumerate(d['ids']):
        identity = Path(absolute).relative_to(ROOT).as_posix()
        r = dict(by_id.pop(identity))
        if (r['step'], r['task'], r['failure']) != (int(d['query_step'][index]), int(d['task'][index]), int(d['y'][index])):
            raise ValueError(f'Primary query mismatch: {identity}')
        path = Path(r['dir']) / r['artifact']
        r['artifact_sha256'] = digest(path)
        r['independent_seed'] = independent_seed(identity)
        r['scene'], r['replica'] = int(d['scene'][index]), int(d['replica'][index])
        server = json.loads((Path(r['dir']).parent / 'server.json').read_text())
        r['checkpoint'] = server['checkpoint_id']
        checkpoints.add(r['checkpoint'])
        with np.load(path, allow_pickle=False) as src:
            samples = src['action_samples'].astype(np.float32)
            value, features = spread_features(samples)
            if samples.shape != (8, 4, 7) or value != d['Spread'][index]:
                raise ValueError(f'Archived spread mismatch: {identity}')
            original_s.append(features)
        selected.append(r)
    if by_id or len(checkpoints) != 1:
        raise ValueError('Cohort or checkpoint mismatch')
    seeds = [r['independent_seed'] for r in selected]
    if len(set(seeds)) != len(seeds) or any(r['seed'] == r['independent_seed'] for r in selected):
        raise ValueError('Policy seed collision')
    out.mkdir(parents=True, exist_ok=True)
    d = dict(d, S8=np.asarray(original_s))
    atomic_npz(out / 'dataset.npz', **d)
    receipt = {'version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(), 'cell': CELL,
               'landmark': STEP, 'queries': len(selected), 'J': 64, 'rows': selected,
               'checkpoint': checkpoints.pop(), 'source_sha256': {p: digest(CODE_ROOT / p) for p in SOURCES},
               'dataset_sha256': digest(out / 'dataset.npz'), 'audit': audit,
               'input_sha256': reader.hashes}
    (out / 'collection.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'queries': len(selected), 'collection_sha256': digest(out / 'collection.json')}), flush=True)


def receipt(out):
    value = json.loads((out / 'collection.json').read_text())
    for path, expected in value['source_sha256'].items():
        if digest(CODE_ROOT / path) != expected:
            raise ValueError(f'Collection source changed: {path}')
    if digest(out / 'dataset.npz') != value['dataset_sha256']:
        raise ValueError('Prepared dataset changed')
    return value, digest(out / 'collection.json')


def result_path(out, identity):
    return out / 'episodes' / (hashlib.sha256(identity.encode()).hexdigest() + '.npz')


def validate_result(values, row, collection_hash, run_hash, archived):
    for name, expected in [('id', row['id']), ('collection_sha256', collection_hash),
                           ('run_sha256', run_hash), ('input_sha256', row['artifact_sha256']),
                           ('checkpoint', row['checkpoint']), ('device', row['device']),
                           ('original_seed', row['seed']), ('independent_seed', row['independent_seed'])]:
        if values[name].item() != expected:
            raise ValueError(f'Replay metadata mismatch: {name}, {row["id"]}')
    for kind in ('original', 'independent'):
        samples = values[kind]
        if samples.dtype != np.float32 or samples.shape != (64, 4, 7) or not np.isfinite(samples).all():
            raise ValueError(f'Invalid action samples: {row["id"]}/{kind}')
        handoff = values[kind + '_handoff']
        ref = archived['handoff'].astype(np.float32)
        if handoff.shape != ref.shape or not np.isfinite(handoff).all():
            raise ValueError('Invalid handoff output')
        relative = float(np.abs(handoff - ref).max()) / max(float(np.abs(ref).max()), 1e-12)
        if relative > 2**-9:
            raise ValueError(f'Handoff parity failed: {row["id"]}/{kind}: {relative}')
    validate_prefix(values['original'], archived['action_samples'], row['id'])


def validate_prefix(samples, archived, identity):
    if samples.dtype != np.float32 or samples.shape != (64, 4, 7) or not np.isfinite(samples).all():
        raise ValueError(f'Invalid action samples: {identity}')
    if samples[:8].tobytes() != archived.astype(np.float32).tobytes():
        raise ValueError(f'Action prefix is not bitwise identical: {identity}')


def read_input(row):
    path = Path(row['dir']) / row['artifact']
    if digest(path) != row['artifact_sha256']:
        raise ValueError(f'Input changed: {row["id"]}')
    with np.load(path, allow_pickle=False) as src:
        return {k: src[k] for k in ('agent_image', 'state', 'action_samples', 'handoff')}


def run(out, ports):
    collection, collection_hash = receipt(out)
    if len(set(ports)) != len(ports):
        raise ValueError('Duplicate ports')
    servers = {}
    for port in ports:
        with socket.create_connection(('127.0.0.1', port), timeout=600) as conn:
            meta = replay.query(conn, {'cmd': 'ping'})
        if meta['checkpoint_id'] != collection['checkpoint'] or meta['code_sha256'] != collection['source_sha256']['code/serve_simpler_gr00t.py']:
            raise ValueError(f'Checkpoint or code mismatch on port {port}')
        if meta.get('sampling_server_sha256') != collection['source_sha256']['code/serve_bridge_sampling.py'] or not meta.get('sampling_runtime_sha256'):
            raise ValueError(f'Missing sampling runtime provenance on port {port}')
        servers[str(port)] = meta
    if len({m['sampling_runtime_sha256'] for m in servers.values()}) != 1:
        raise ValueError('Servers disagree on the GR00T source/runtime fingerprint')
    manifest = {'collection_sha256': collection_hash, 'servers': servers}
    manifest_path = out / 'run_manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Resume server configuration changed')
    else:
        if (out / 'episodes').exists() and any((out / 'episodes').glob('*.npz')):
            raise ValueError('Existing replay files without a run manifest')
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    run_hash = digest(manifest_path)
    shards = {p: [] for p in ports}
    for device in sorted({r['device'] for r in collection['rows']}):
        matched = [p for p in ports if servers[str(p)]['device'] == device]
        if not matched:
            raise ValueError(f'No server on original GPU model: {device}')
        for i, row in enumerate(r for r in collection['rows'] if r['device'] == device):
            shards[matched[i % len(matched)]].append(row)
    (out / 'episodes').mkdir(exist_ok=True)
    stop, lock = threading.Event(), threading.Lock()
    progress = {'completed': 0, 'total': len(collection['rows'])}

    def worker(port):
        with socket.create_connection(('127.0.0.1', port), timeout=600) as conn:
            for row in shards[port]:
                if stop.is_set():
                    return
                started = time.monotonic()
                target = result_path(out, row['id'])
                src = read_input(row)
                if target.exists():
                    with np.load(target, allow_pickle=False) as old:
                        validate_result(old, row, collection_hash, run_hash, src)
                else:
                    result = {k: np.asarray(v) for k, v in dict(id=row['id'], collection_sha256=collection_hash,
                              run_sha256=run_hash, input_sha256=row['artifact_sha256'], checkpoint=row['checkpoint'],
                              device=row['device'], original_seed=row['seed'], independent_seed=row['independent_seed']).items()}
                    for kind, seed in [('original', row['seed']), ('independent', row['independent_seed'])]:
                        request = {'images': [replay.encode(src['agent_image'])], 'state': src['state'].astype(np.float32).tolist(),
                                   'task': row['language'], 'seed': seed, 'J': 64}
                        response = replay.query(conn, request)
                        if response.get('checkpoint_sha256') != row['checkpoint'] or response.get('seed') != seed or response.get('J') != 64:
                            raise ValueError(f'Response provenance mismatch: {row["id"]}')
                        result[kind] = np.asarray(response['action_samples'], dtype=np.float32)
                        result[kind + '_handoff'] = np.asarray(response['kv_pooled'], dtype=np.float32)
                        if kind == 'original':
                            try:
                                validate_prefix(result[kind], src['action_samples'], row['id'])
                            except ValueError as error:
                                stop.set()
                                failed = out / 'failed'
                                failed.mkdir(exist_ok=True)
                                atomic_npz(failed / target.name, **result, error=np.asarray(str(error)))
                                raise
                    try:
                        validate_result(result, row, collection_hash, run_hash, src)
                    except (ValueError, KeyError) as error:
                        stop.set()
                        failed = out / 'failed'
                        failed.mkdir(exist_ok=True)
                        atomic_npz(failed / target.name, **result, error=np.asarray(str(error)))
                        raise
                    atomic_npz(target, **result)
                with lock:
                    progress['completed'] += 1
                    print(json.dumps(dict(progress, port=port, seconds=round(time.monotonic() - started, 2))), flush=True)
    with ThreadPoolExecutor(max_workers=len(ports)) as pool:
        futures = [pool.submit(worker, p) for p in ports]
        try:
            for f in as_completed(futures):
                f.result()
        except BaseException:
            stop.set()
            for f in futures:
                f.cancel()
            raise
    verify(out)


def verify(out):
    collection, collection_hash = receipt(out)
    run_hash = digest(out / 'run_manifest.json')
    missing = []
    for row in collection['rows']:
        path = result_path(out, row['id'])
        if not path.exists():
            missing.append(row['id'])
            continue
        with np.load(path, allow_pickle=False) as result:
            validate_result(result, row, collection_hash, run_hash, read_input(row))
    expected = {result_path(out, row['id']).name for row in collection['rows']}
    unexpected = sorted(p.name for p in (out / 'episodes').glob('*.npz') if p.name not in expected)
    summary = {'complete': not missing and not unexpected, 'queries': len(collection['rows']),
               'verified': len(collection['rows']) - len(missing), 'missing': missing, 'unexpected': unexpected,
               'collection_sha256': collection_hash, 'run_sha256': run_hash}
    (out / 'collection_validation.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('missing', 'unexpected')}), flush=True)
    if not summary['complete']:
        raise ValueError('Collection incomplete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run', 'verify'))
    parser.add_argument('output', type=Path)
    parser.add_argument('ports', nargs='*', type=int)
    args = parser.parse_args()
    if args.command == 'run':
        run(args.output, args.ports)
    else:
        globals()[args.command](args.output)
