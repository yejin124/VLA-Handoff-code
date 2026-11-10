"""Frozen-input SmolVLA J64 follow-up: prepare, pilot (J8 only), run, verify.

Set RAL_RAW_ROOT to the original raw checkout. Code and raw roots are separate.
The pilot never collects J64; full collection requires an explicit run command.
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import socket
import struct
import tempfile
import threading
import time

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get('RAL_RAW_ROOT', CODE_ROOT)).resolve()
CELLS, STEP, JS = ('smolvla_spatial', 'smolvla_object'), 80, (8, 16, 32, 64)
PLAN = 'docs/PROTOCOLS.md'
SOURCES = ('code/smol_sampling_replay.py', 'code/smol_sampling_analysis.py', 'code/serve_smol_sampling.py',
           'code/serve_icassp_smolvla.py', 'code/serve_icassp_pi05.py', 'code/ral_confirm_analysis.py',
           'code/ral_nested_fit.py', 'code/ral_nested_features.py', 'code/icassp_conditions.py',
           'code/analyze_icassp_trials.py', 'code/recheck_restoration_allocation.py', PLAN)
SERVER_FIELDS = ('checkpoint_id', 'checkpoint_files_sha256', 'code_sha256', 'protocol_module_sha256',
                 'lerobot_source_sha256', 'torch_version', 'transformers_version', 'numpy_version',
                 'lerobot_version', 'device', 'execute_steps', 'chunk_size', 'num_flow_steps',
                 'num_vlm_layers', 'image_features', 'resize_imgs_with_padding', 'execution_sample', 'spread_ddof')


def digest(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def independent_seed(identity: str) -> int:
    return int.from_bytes(hashlib.sha256(('smol-J64-20261110|' + identity).encode()).digest()[:4], 'big') % 2**31


def spread_features(samples: np.ndarray) -> tuple[float, np.ndarray]:
    if samples.ndim != 3 or samples.shape[1:] != (10, 7) or len(samples) < 2 or not np.isfinite(samples).all():
        raise ValueError('Expected finite SmolVLA samples [J, 10, 7]')
    sd = np.log(samples.std(axis=0, ddof=0) + 1e-8)
    spread = float(samples.std(axis=0, ddof=0).mean())
    return spread, np.concatenate([sd.mean(axis=0), sd.mean(axis=1), [np.log(spread + 1e-8)]])


def atomic_npz(path: Path, **values) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npz', delete=False) as f:
        temporary = Path(f.name)
        np.savez_compressed(f, **values)
    os.replace(temporary, path)


def raw_path(relative: str) -> Path:
    rel = PurePosixPath(relative)
    if rel.is_absolute() or '..' in rel.parts or not rel.parts or rel.parts[0] != 'data':
        raise ValueError('Unsafe raw-relative path')
    path = (ROOT / str(rel)).resolve(strict=True)
    if not path.is_relative_to(ROOT):
        raise ValueError('Raw input escapes root')
    return path


def cohort(cell: str) -> list[dict]:
    from ral_confirm_analysis import COHORTS
    rows = []
    for folder, _, _ in COHORTS[cell][0]:
        for path in sorted((ROOT / folder).glob('shard*/t*_e*_r*_occluder/summary.json')):
            s = json.loads(path.read_text())
            if not s['complete'] or s['condition'] != 'occluder' or s['steps'] <= STEP:
                continue
            trace = [json.loads(x) for x in (path.parent / 'queries.jsonl').read_text().splitlines() if x.strip()]
            q = max((q for q in trace if q['step'] <= STEP), key=lambda q: q['step'])
            server = path.parent.parent / 'server.json'
            meta = json.loads(server.read_text())
            identity = path.parent.relative_to(ROOT).as_posix()
            rows.append(dict(id=identity, cell=cell, artifact=q['artifact'], artifact_sha256=q['artifact_sha256'],
                             step=q['step'], task=s['task'], scene=s['episode'], replica=s['replicate'],
                             failure=int(not s['success']), language=s['language'], seed=q['policy_seed'],
                             independent_seed=independent_seed(identity), image_sha256=q['image_sha256'],
                             state=q['state'], checkpoint=meta['checkpoint_id'], device=meta['device'],
                             server=server.relative_to(ROOT).as_posix()))
    return rows


def read_input(row: dict) -> dict:
    if Path(row['artifact']).name != row['artifact'] or not row['artifact'].endswith('.npz'):
        raise ValueError('Unsafe artifact name')
    path = raw_path(row['id'] + '/' + row['artifact'])
    if digest(path) != row['artifact_sha256']:
        raise ValueError(f'Input digest mismatch: {row["id"]}')
    with np.load(path, allow_pickle=False) as src:
        result = {k: src[k] for k in ('agent_image', 'wrist_image', 'state', 'action_samples', 'handoff', 'executed_actions')}
    for i, key in enumerate(('agent_image', 'wrist_image')):
        image = np.ascontiguousarray(result[key])
        actual = hashlib.sha256(str((image.shape, str(image.dtype))).encode() + image.tobytes()).hexdigest()
        if image.dtype != np.uint8 or image.shape != (256, 256, 3) or actual != row['image_sha256'][i]:
            raise ValueError('Transmitted image mismatch')
    for key, shape, dtype in [('state', (8,), np.float32), ('action_samples', (8, 10, 7), np.float32),
                               ('handoff', (10240,), np.float16), ('executed_actions', (10, 7), np.float32)]:
        if result[key].shape != shape or result[key].dtype != dtype or not np.isfinite(result[key]).all():
            raise ValueError(f'Invalid archived {key}')
    if not np.array_equal(result['state'], np.asarray(row['state'], dtype=np.float32)):
        raise ValueError('State differs from query record')
    if result['executed_actions'].tobytes() != result['action_samples'][0].tobytes():
        raise ValueError('Executed action differs from sample zero')
    return result


def prepare(out: Path) -> None:
    from ral_confirm_analysis import load
    from recheck_restoration_allocation import Reader
    if (out / 'collection.json').exists() or any((out / f'{c}.npz').exists() for c in CELLS):
        raise ValueError('Prepared files already exist; never overwrite a collection')
    reader, rows, datasets, audits, servers = Reader(), [], {}, {}, {}
    for cell in CELLS:
        data, audits[cell] = load(ROOT, cell, reader)
        d, selected = data[(cell, STEP)], cohort(cell)
        by_id = {r['id']: r for r in selected}
        if not selected or len(by_id) != len(selected) or len(selected) != len(d['ids']) or not d['eligible']:
            raise ValueError(f'Unexpected cohort: {cell}')
        expected_ids = {Path(i).relative_to(ROOT).as_posix() for i in d['ids']}
        if len(expected_ids) != len(d['ids']) or set(by_id) != expected_ids:
            raise ValueError(f'Replay identities differ from the validated loader: {cell}')
        s8 = []
        for i, absolute in enumerate(d['ids']):
            row = by_id.pop(Path(absolute).relative_to(ROOT).as_posix())
            for name, field in [('task', 'task'), ('scene', 'scene'), ('replica', 'replica'),
                                ('failure', 'y'), ('step', 'query_step')]:
                if row[name] != int(d[field][i]):
                    raise ValueError(f'Validated dataset mismatch: {name}/{row["id"]}')
            if row['step'] != STEP or row['artifact'] != 'q0008.npz':
                raise ValueError('Unexpected primary query')
            src = read_input(row)
            scalar, features = spread_features(src['action_samples'])
            if scalar != d['Spread'][i] or not np.array_equal(src['handoff'].astype(np.float64), d['H'][i]):
                raise ValueError('Archived features disagree with the frozen loader')
            rows.append(row); s8.append(features)
            servers[row['server']] = json.loads(raw_path(row['server']).read_text())
        if by_id:
            raise ValueError('Replay has episodes absent from frozen analysis')
        datasets[cell] = dict(d, S8=np.asarray(s8))
    seeds = [r['independent_seed'] for r in rows]
    if len(set(seeds)) != len(seeds) or any(r['seed'] == r['independent_seed'] for r in rows):
        raise ValueError('Independent policy seed collision')
    exemplar = next(iter(servers.values()))
    if any(any(m[k] != exemplar[k] for k in SERVER_FIELDS) for m in servers.values()):
        raise ValueError('Original server scientific/runtime configurations differ')
    out.mkdir(parents=True, exist_ok=True)
    for cell, d in datasets.items():
        atomic_npz(out / f'{cell}.npz', **d)
    value = dict(version=1, created_utc=datetime.now(timezone.utc).isoformat(), cells=list(CELLS),
                 landmark=STEP, queries=len(rows), J=64, rows=rows, original_servers=servers, audit=audits,
                 source_sha256={p: digest(CODE_ROOT / p) for p in SOURCES},
                 dataset_sha256={c: digest(out / f'{c}.npz') for c in CELLS}, input_sha256=reader.hashes)
    (out / 'collection.json').write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps({'queries': len(rows), 'collection_sha256': digest(out / 'collection.json')}), flush=True)


def receipt(out: Path) -> tuple[dict, str]:
    value = json.loads((out / 'collection.json').read_text())
    for path, expected in value['source_sha256'].items():
        if digest(CODE_ROOT / path) != expected:
            raise ValueError(f'Collection source changed: {path}')
    for cell, expected in value['dataset_sha256'].items():
        if digest(out / f'{cell}.npz') != expected:
            raise ValueError(f'Prepared dataset changed: {cell}')
    return value, digest(out / 'collection.json')


def dataset(out: Path, cell: str) -> dict:
    with np.load(out / f'{cell}.npz', allow_pickle=False) as z:
        d = {k: z[k].item() if z[k].ndim == 0 else z[k] for k in z.files}
    for k in ('tasks', 'units'):
        d[k] = tuple(int(x) for x in d[k])
    d['ids'] = [Path(i).relative_to(ROOT).as_posix() for i in d['ids']]
    return d


def encode(value: np.ndarray) -> str:
    buffer = io.BytesIO(); np.save(buffer, np.ascontiguousarray(value), allow_pickle=False)
    return base64.b64encode(buffer.getvalue()).decode()


def query(conn: socket.socket, request: dict) -> dict:
    def read(n):
        data = bytearray()
        while len(data) < n:
            chunk = conn.recv(n - len(data))
            if not chunk:
                raise ConnectionError('Policy server disconnected')
            data.extend(chunk)
        return bytes(data)
    data = json.dumps(request, allow_nan=False).encode(); conn.sendall(struct.pack('>I', len(data)) + data)
    n = struct.unpack('>I', read(4))[0]
    if not 0 < n <= 64 * 1024 * 1024:
        raise ValueError('Response size out of bounds')
    response = json.loads(read(n))
    if response.get('ok') is not True or 'error' in response:
        raise RuntimeError(f'Inference failed: {response.get("error")} {response.get("message", "")}')
    return response


def validate_server(meta: dict, original: dict, source: dict) -> None:
    for key in SERVER_FIELDS:
        if meta.get(key) != original[key]:
            raise ValueError(f'Original server metadata mismatch: {key}')
    if meta.get('sampling_server_sha256') != source['code/serve_smol_sampling.py']:
        raise ValueError('Sampling wrapper mismatch')
    runtime = meta.get('sampling_runtime')
    if not runtime or meta.get('sampling_runtime_sha256') != hashlib.sha256(json.dumps(runtime, sort_keys=True).encode()).hexdigest():
        raise ValueError('Invalid runtime fingerprint')


def request_for(row: dict, src: dict, seed: int, draws: int) -> dict:
    return dict(images=[encode(src[k]) for k in ('agent_image', 'wrist_image')],
                state=src['state'].tolist(), task=row['language'], seed=seed, J=draws)


def response_arrays(response: dict, row: dict, seed: int, draws: int, code_hash: str) -> tuple[np.ndarray, np.ndarray]:
    if response.get('seed') != seed or response.get('J') != draws or response.get('code_sha256') != code_hash:
        raise ValueError(f'Response provenance mismatch: {row["id"]}')
    samples = np.asarray(response['action_samples'], dtype=np.float32)
    if samples.shape != (draws, 10, 7) or not np.isfinite(samples).all():
        raise ValueError('Invalid replay action samples')
    if samples[0].tobytes() != np.asarray(response['actions'], dtype=np.float32).tobytes():
        raise ValueError('Response actions differ from sample zero')
    return samples, np.asarray(response['kv_pooled'], dtype=np.float32)


def validate_prefix(samples: np.ndarray, archived: np.ndarray, identity: str, draws: int = 64) -> None:
    if samples.dtype != np.float32 or samples.shape != (draws, 10, 7) or not np.isfinite(samples).all():
        raise ValueError(f'Invalid action samples: {identity}')
    if samples[:8].tobytes() != archived.astype(np.float32).tobytes():
        raise ValueError(f'Action prefix is not bitwise identical: {identity}')


def validate_handoff(value: np.ndarray, archived: np.ndarray) -> float:
    if value.dtype != np.float32 or value.shape != (10240,) or not np.isfinite(value).all():
        raise ValueError('Invalid handoff')
    ref = archived.astype(np.float32)
    relative = float(np.abs(value - ref).max()) / max(float(np.abs(ref).max()), 1e-12)
    if relative > 2**-9:
        raise ValueError(f'Handoff parity failed: {relative}')
    return relative


def result_path(out: Path, identity: str) -> Path:
    return out / 'episodes' / (hashlib.sha256(identity.encode()).hexdigest() + '.npz')


def validate_result(values: dict, row: dict, collection_hash: str, run_hash: str, archived: dict) -> None:
    for name, expected in [('id', row['id']), ('cell', row['cell']), ('collection_sha256', collection_hash),
                           ('run_sha256', run_hash), ('input_sha256', row['artifact_sha256']),
                           ('checkpoint', row['checkpoint']), ('device', row['device']),
                           ('original_seed', row['seed']), ('independent_seed', row['independent_seed'])]:
        if values[name].item() != expected:
            raise ValueError(f'Replay metadata mismatch: {name}/{row["id"]}')
    for kind in ('original', 'independent'):
        samples = values[kind]
        if samples.dtype != np.float32 or samples.shape != (64, 10, 7) or not np.isfinite(samples).all():
            raise ValueError('Invalid action samples')
        validate_handoff(values[kind + '_handoff'], archived['handoff'])
    validate_prefix(values['original'], archived['action_samples'], row['id'])


def pilot(out: Path, port: int) -> None:
    collection, collection_hash = receipt(out)
    destination = out / 'pilot'; destination.mkdir(exist_ok=False)
    results = []
    with socket.create_connection(('127.0.0.1', port), timeout=600) as conn:
        meta = query(conn, {'cmd': 'ping'})
        for cell in CELLS:
            row = next(r for r in collection['rows'] if r['cell'] == cell)
            validate_server(meta, collection['original_servers'][row['server']], collection['source_sha256'])
            src = read_input(row)
            response = query(conn, request_for(row, src, row['seed'], 8))
            samples, handoff = response_arrays(response, row, row['seed'], 8, meta['code_sha256'])
            atomic_npz(destination / f'{cell}.npz', samples=samples, handoff=handoff, id=np.asarray(row['id']))
            validate_prefix(samples, src['action_samples'], row['id'], draws=8)
            relative = validate_handoff(handoff, src['handoff'])
            results.append(dict(cell=cell, id=row['id'], J=8, seed=row['seed'], exact_action_prefix=True,
                                handoff_relative_max_error=relative))
    value = dict(collection_sha256=collection_hash, server=meta, results=results, J64_collected=False)
    (destination / 'pilot.json').write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(results, indent=2), flush=True)


def run(out: Path, ports: list[int]) -> None:
    collection, collection_hash = receipt(out)
    if not ports or len(set(ports)) != len(ports) or any(not 1 <= p <= 65535 for p in ports):
        raise ValueError('Need distinct valid server ports')
    servers = {}
    for port in ports:
        with socket.create_connection(('127.0.0.1', port), timeout=600) as conn:
            meta = query(conn, {'cmd': 'ping'})
        for original in collection['original_servers'].values():
            validate_server(meta, original, collection['source_sha256'])
        servers[str(port)] = meta
    if len({m['sampling_runtime_sha256'] for m in servers.values()}) != 1:
        raise ValueError('Server runtimes differ')
    manifest = dict(collection_sha256=collection_hash, servers=servers)
    path = out / 'run_manifest.json'
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError('Resume server configuration changed')
    else:
        if any((out / 'episodes').glob('*.npz')):
            raise ValueError('Existing episodes without run manifest')
        path.write_text(json.dumps(manifest, indent=2) + '\n')
    run_hash, stop, lock = digest(path), threading.Event(), threading.Lock()
    (out / 'episodes').mkdir(exist_ok=True); progress = {'completed': 0, 'total': len(collection['rows'])}

    def worker(port, rows):
        with socket.create_connection(('127.0.0.1', port), timeout=600) as conn:
            for row in rows:
                if stop.is_set():
                    return
                started = time.monotonic(); target = result_path(out, row['id']); src = read_input(row)
                if target.exists():
                    with np.load(target, allow_pickle=False) as previous:
                        validate_result(previous, row, collection_hash, run_hash, src)
                else:
                    values = {k: np.asarray(v) for k, v in dict(id=row['id'], cell=row['cell'],
                              collection_sha256=collection_hash, run_sha256=run_hash,
                              input_sha256=row['artifact_sha256'], checkpoint=row['checkpoint'], device=row['device'],
                              original_seed=row['seed'], independent_seed=row['independent_seed']).items()}
                    try:
                        for kind, seed in [('original', row['seed']), ('independent', row['independent_seed'])]:
                            response = query(conn, request_for(row, src, seed, 64))
                            values[kind], values[kind + '_handoff'] = response_arrays(
                                response, row, seed, 64, servers[str(port)]['code_sha256'])
                            if kind == 'original':
                                validate_prefix(values[kind], src['action_samples'], row['id'])
                            validate_handoff(values[kind + '_handoff'], src['handoff'])
                        validate_result(values, row, collection_hash, run_hash, src)
                    except Exception as error:
                        stop.set(); failed = out / 'failed'; failed.mkdir(exist_ok=True)
                        atomic_npz(failed / target.name, **values, error=np.asarray(str(error)))
                        raise
                    atomic_npz(target, **values)
                with lock:
                    progress['completed'] += 1
                    print(json.dumps(dict(progress, cell=row['cell'], port=port,
                                          seconds=round(time.monotonic() - started, 2))), flush=True)
    with ThreadPoolExecutor(max_workers=len(ports)) as pool:
        futures = [pool.submit(worker, p, collection['rows'][i::len(ports)]) for i, p in enumerate(ports)]
        try:
            for future in as_completed(futures):
                future.result()
        except BaseException:
            stop.set()
            for future in futures:
                future.cancel()
            raise
    verify(out)


def verify(out: Path) -> dict:
    collection, collection_hash = receipt(out); run_path = out / 'run_manifest.json'
    manifest = json.loads(run_path.read_text()); run_hash = digest(run_path)
    if manifest['collection_sha256'] != collection_hash or not manifest['servers']:
        raise ValueError('Run manifest belongs to another collection')
    for meta in manifest['servers'].values():
        for original in collection['original_servers'].values():
            validate_server(meta, original, collection['source_sha256'])
    missing = []
    for row in collection['rows']:
        path = result_path(out, row['id'])
        if not path.exists():
            missing.append(row['id']); continue
        with np.load(path, allow_pickle=False) as values:
            validate_result(values, row, collection_hash, run_hash, read_input(row))
    expected = {result_path(out, r['id']).name for r in collection['rows']}
    unexpected = sorted(p.name for p in (out / 'episodes').glob('*.npz') if p.name not in expected)
    summary = dict(complete=not missing and not unexpected, queries=len(collection['rows']),
                   verified=len(collection['rows']) - len(missing), missing=missing, unexpected=unexpected,
                   collection_sha256=collection_hash, run_sha256=run_hash)
    (out / 'collection_validation.json').write_text(json.dumps(summary, indent=2) + '\n')
    if not summary['complete']:
        raise ValueError('Collection incomplete')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'pilot', 'run', 'verify'))
    parser.add_argument('output', type=Path); parser.add_argument('ports', nargs='*', type=int)
    args = parser.parse_args(); output = args.output.resolve()
    if output.is_relative_to(ROOT):
        parser.error('Collection outputs must be outside the original raw checkout')
    if args.command == 'run':
        run(output, args.ports)
    elif args.command == 'pilot':
        if len(args.ports) != 1:
            parser.error('Pilot needs exactly one port')
        pilot(output, args.ports[0])
    else:
        globals()[args.command](output)
