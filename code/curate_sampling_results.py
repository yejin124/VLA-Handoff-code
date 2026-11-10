"""Export completed sampling summaries and compact, source-bound provenance.

Raw observations, features, sampled actions and large runtime manifests remain
external. The exported results are byte-identical to the completed analyses.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def export_one(source, output, name):
    collection = read(source / 'collection.json')
    validation = read(source / 'collection_validation.json')
    run = read(source / 'run_manifest.json')
    analysis = read(source / 'analysis/analysis_receipt.json')
    results = read(source / 'analysis/results.json')
    for field, path in [('collection_sha256', 'collection.json'), ('run_sha256', 'run_manifest.json')]:
        if validation[field] != digest(source / path):
            raise ValueError(f'Collection validation mismatch: {name}/{field}')
    for field, path in [('collection_sha256', 'collection.json'), ('run_manifest_sha256', 'run_manifest.json')]:
        if analysis[field] != digest(source / path):
            raise ValueError(f'Analysis receipt mismatch: {name}/{field}')
    if (not validation['complete'] or validation['verified'] != collection['queries'] or
            validation['missing'] or validation['unexpected'] or results['attempts'] != 2000):
        raise ValueError(f'Incomplete experiment: {name}')
    if results['analysis_receipt_sha256'] != digest(source / 'analysis/analysis_receipt.json'):
        raise ValueError(f'Results not bound to analysis: {name}')
    code_root = Path(__file__).resolve().parents[1]
    for path, expected in collection['source_sha256'].items():
        if digest(code_root / path) != expected:
            raise ValueError(f'Frozen source changed: {name}/{path}')
    analysis_name = f'code/{name}_sampling_analysis.py'
    if digest(code_root / analysis_name) != analysis['code_sha256']:
        raise ValueError(f'Analysis source changed: {name}')
    servers = []
    seen = set()
    for meta in run['servers'].values():
        identity = (meta['device'], meta['sampling_runtime_sha256'])
        if identity in seen:
            continue
        seen.add(identity)
        keys = ('device', 'checkpoint_id', 'checkpoint_sha256', 'checkpoint_files_sha256',
                'code_sha256', 'protocol_module_sha256', 'sampling_server_sha256', 'sampling_runtime_sha256',
                'python', 'torch', 'transformers', 'torch_version', 'transformers_version',
                'numpy_version', 'lerobot_version', 'execute_steps', 'action_horizon', 'num_flow_steps')
        servers.append({k: meta[k] for k in keys if k in meta})
    provenance = dict(collection_created_utc=collection['created_utc'], queries=collection['queries'],
                      landmark=collection['landmark'], J=collection['J'],
                      dataset_sha256=collection['dataset_sha256'], source_sha256=collection['source_sha256'],
                      device_counts=dict(Counter(row['device'] for row in collection['rows'])),
                      collection_validation=validation, analysis_receipt=analysis, servers=servers,
                      archived_files_sha256={p: digest(source / p) for p in
                          ('collection.json', 'run_manifest.json', 'collection_validation.json',
                           'analysis/analysis_receipt.json', 'analysis/point_results.json',
                           'analysis/bootstrap.jsonl', 'analysis/results.json')},
                      scope='Fixed-input resampling; original episode outcomes; no new closed-loop episodes.')
    shutil.copyfile(source / 'analysis/results.json', output / f'{name}_results.json')
    (output / f'{name}_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bridge', type=Path)
    parser.add_argument('smol', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    export_one(args.bridge, args.output, 'bridge')
    export_one(args.smol, args.output, 'smol')
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                    files_sha256={p.name: digest(p) for p in sorted(args.output.glob('*.json'))},
                    primary_families={'bridge': 4, 'smol': 8},
                    note='Separate prespecified follow-up families; not a joint 12-test confirmation.')
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
