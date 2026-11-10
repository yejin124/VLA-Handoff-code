"""Bind resumable GPU replays to their inputs, directions, servers and port allocation."""
import hashlib
import json

from injection_validation import validate_episode


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require_new_directions(out):
    if any((out / name).exists() for name in ('directions.npz', 'directions.json', 'run_manifest.json')) or list(out.glob('results_*.jsonl')):
        raise ValueError('Direction or replay files already exist; choose a new output directory')


def bind_run(out, manifest):
    """Refuse legacy or changed resumes; create metadata before starting any worker."""
    path = out / 'run_manifest.json'
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError('Replay configuration changed; use the original configuration or a new output directory')
    else:
        if list(out.glob('results_*.jsonl')):
            raise ValueError('Existing replay has no run manifest; keep it for analysis and start a new output directory')
        with path.open('x') as stream:
            stream.write(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    return sha256(path)


def completed_ids(path, expected_ids, manifest_sha256):
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            validate_episode(row)
            if row['id'] not in expected_ids or row['id'] in done:
                raise ValueError(f'Unexpected or duplicate replay episode in {path.name}')
            if row.get('run_manifest_sha256') != manifest_sha256:
                raise ValueError(f'Replay record does not match the run manifest: {path.name}')
            done.add(row['id'])
    return done
