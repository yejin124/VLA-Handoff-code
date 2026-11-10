"""Portable entry point for the paper's 10,752-episode collection protocol.

List cohorts, inspect a command without starting a model, or collect against an
already running localhost policy server. Use a simulator-specific Python.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'config/reproduction/cohorts.json'
COLLECTORS = {'icassp_rollout.py', 'recorded_failure_rollout_20261103.py', 'simpler_rollout.py'}


def cohorts() -> dict:
    """Load and validate the public collection configuration."""
    records = json.loads(CONFIG.read_text())
    for name, record in records.items():
        target = Path(record['output'])
        if target.is_absolute() or '..' in target.parts or target.parts[0] != 'data':
            raise ValueError(f'Unsafe output path in cohort {name}')
        if record['collector'] not in COLLECTORS:
            raise ValueError(f'Unknown collector in cohort {name}')
        if not isinstance(record['options'], dict):
            raise ValueError(f'Invalid options in cohort {name}')
    return records


def command(record: dict, python: str, raw_root: Path, port: int,
            shard: int = 0, nshards: int = 1) -> list[str]:
    """Build a shell-free collector command for one shard."""
    if not 1 <= port <= 65535 or not 0 <= shard < nshards:
        raise ValueError('Invalid port or shard allocation')
    target = (raw_root.resolve() / record['output']).resolve()
    if not target.is_relative_to(raw_root.resolve()):
        raise ValueError('Collection output escapes raw root')
    result = [python, str(ROOT / 'code' / record['collector']), '--out', str(target),
              '--port', str(port), '--shard', str(shard), '--nshards', str(nshards)]
    for key, value in record['options'].items():
        flag = '--' + key.replace('_', '-')
        if isinstance(value, bool):
            if value:
                result.append(flag)
        elif isinstance(value, dict):
            result.extend([flag, ','.join(f'{k}={v}' for k, v in value.items())])
        else:
            result.extend([flag, str(value)])
    return result


def count(record: dict) -> int:
    """Count planned episodes, without reading any experimental result."""
    options = record['options']
    episodes = options['episodes']
    if ':' in episodes:
        first, last = map(int, episodes.split(':'))
        n = last - first + 1
    else:
        n = len(episodes.split(','))
    return len(options['tasks'].split(',')) * n * options['replicates'] * len(options['conditions'].split(','))


def configure_libero(checkout: Path, destination: Path) -> None:
    """Create LIBERO's local path configuration without a private machine path."""
    checkout = checkout.expanduser().resolve()
    candidates = (checkout / 'libero/libero', checkout / 'libero', checkout)
    package = next((p for p in candidates if (p / 'bddl_files').is_dir()), candidates[0])
    for name in ('bddl_files', 'init_files', 'assets'):
        if not (package / name).is_dir():
            raise ValueError(f'LIBERO checkout is missing {package / name}')
    destination.mkdir(parents=True, exist_ok=True)
    paths = {'benchmark_root': package, 'bddl_files': package / 'bddl_files',
             'init_states': package / 'init_files', 'datasets': package.parent / 'datasets',
             'assets': package / 'assets'}
    # JSON strings are valid YAML scalars, including paths with spaces.
    content = ''.join(f'{k}: {json.dumps(str(v))}\n' for k, v in paths.items())
    target = destination / 'config.yaml'
    if target.exists() and target.read_text() != content:
        raise ValueError(f'Refusing to replace a different configuration: {target}')
    target.write_text(content)
    print(f'export LIBERO_CONFIG_PATH={shlex.quote(str(destination.resolve()))}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('list', help='List all cohorts and their planned episode counts')
    setup = sub.add_parser('configure-libero')
    setup.add_argument('checkout', type=Path)
    setup.add_argument('--output', type=Path, default=ROOT / 'config/local/libero')
    collect = sub.add_parser('collect')
    collect.add_argument('cohort')
    collect.add_argument('--python', default=sys.executable, help='Python with the required simulator installed')
    collect.add_argument('--raw-root', type=Path, default=ROOT)
    collect.add_argument('--port', type=int, required=True)
    collect.add_argument('--shard', type=int, default=0)
    collect.add_argument('--nshards', type=int, default=1)
    collect.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.action == 'configure-libero':
        configure_libero(args.checkout, args.output.expanduser().resolve())
        return
    records = cohorts()
    if args.action == 'list':
        for name, record in records.items():
            print(f'{name:35s} {count(record):5d}  {record["collector"]}')
        print(f'Total: {sum(map(count, records.values())):,} episodes')
        return
    if args.cohort not in records:
        parser.error('Unknown cohort; use reproduce.py list')
    cmd = command(records[args.cohort], args.python, args.raw_root.expanduser(),
                  args.port, args.shard, args.nshards)
    print(shlex.join(cmd), flush=True)
    if not args.dry_run:
        env = dict(os.environ)
        env.setdefault('VLA_DONOR_BANK', str(ROOT / 'assets/occluder'))
        subprocess.run(cmd, check=True, cwd=ROOT, env=env)


if __name__ == '__main__':
    main()
