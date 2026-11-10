"""Original Bridge inference with an additional runtime provenance receipt.

Uses the same environment variables and wire protocol as serve_simpler_gr00t.py.
No sampling, precision, action decoding or feature computation is changed.
"""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess

import serve_simpler_gr00t as base


def fingerprint():
    root = Path(os.environ['GR00T_ROOT']).resolve(strict=True)
    sources = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted((root / 'gr00t').rglob('*.py'))}
    packages = {d.metadata['Name']: d.version for d in importlib.metadata.distributions() if d.metadata['Name']}
    return {'gr00t_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
            'gr00t_python_sha256': sources, 'packages': packages,
            'driver': sorted(set(subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version',
                                                        '--format=csv,noheader'], text=True).splitlines())),
            'environment': {k: os.environ.get(k) for k in ('CUDA_DEVICE_ORDER', 'OMP_NUM_THREADS',
                            'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'TOKENIZERS_PARALLELISM')}}


class SamplingBridge(base.Bridge):
    def __init__(self, *args, **kwargs):
        before = fingerprint()
        super().__init__(*args, **kwargs)
        if fingerprint() != before:
            raise RuntimeError('GR00T source/runtime changed during model initialization')
        before['torch_cuda'] = self.torch.version.cuda
        before['cudnn'] = self.torch.backends.cudnn.version()
        self.metadata['sampling_runtime'] = before
        self.metadata['sampling_runtime_sha256'] = hashlib.sha256(
            json.dumps(before, sort_keys=True).encode()).hexdigest()
        self.metadata['sampling_server_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


if __name__ == '__main__':
    base.Bridge = SamplingBridge
    base.main()
