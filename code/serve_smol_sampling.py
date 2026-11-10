"""Add provenance to the unchanged dd97790e SmolVLA server (same wire protocol).

Only ping metadata is extended. Prediction, RNG, precision, processors and hooks
remain in the restored original server. HF_HUB_OFFLINE=1 is required.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess

import serve_icassp_smolvla as base

ORIGINAL = {
    'serve_icassp_smolvla.py': '7792b5bce74d5140957e4f64fb03bdcfb484756df5c212915f9e4561aef86955',
    'serve_icassp_pi05.py': '35c59e8ed7a6b71f57b909bb0d9b1945604f7e9d9c86e3d6c9c33edcd3f831d7',
}
LEROBOT_SOURCES = (
    'policies/smolvla/modeling_smolvla.py', 'policies/smolvla/smolvlm_with_expert.py',
    'policies/smolvla/processor_smolvla.py', 'policies/smolvla/configuration_smolvla.py',
    'policies/common/flow_matching.py', 'policies/factory.py', 'policies/pretrained.py',
    'processor/env_processor.py', 'processor/pipeline.py', 'processor/normalize_processor.py',
    'processor/batch_processor.py', 'processor/newline_task_processor.py', 'processor/tokenizer_processor.py',
    'processor/device_processor.py', 'processor/rename_processor.py', 'processor/factory.py',
    'envs/libero.py', 'envs/utils.py', 'scripts/lerobot_eval.py', 'utils/import_utils.py',
)
TRANSFORMERS_SOURCES = ('models/smolvlm/modeling_smolvlm.py', 'models/smolvlm/processing_smolvlm.py',
                        'models/llama/modeling_llama.py', 'models/idefics3/modeling_idefics3.py',
                        'cache_utils.py', 'modeling_utils.py', 'processing_utils.py')


def fingerprint() -> dict:
    import lerobot
    import torch
    import transformers
    root = Path(lerobot.__file__).resolve().parent
    transformers_root = Path(transformers.__file__).resolve().parent
    packages = sorted((d.metadata['Name'], d.version) for d in importlib.metadata.distributions()
                      if d.metadata['Name'])
    return {
        'lerobot_python_sha256': {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in LEROBOT_SOURCES},
        'transformers_python_sha256': {p: hashlib.sha256((transformers_root / p).read_bytes()).hexdigest()
                                      for p in TRANSFORMERS_SOURCES},
        'packages': packages, 'python': platform.python_version(),
        'torch_cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
        'driver': sorted(set(subprocess.check_output(
            ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True).splitlines())),
        'environment': {k: os.environ.get(k) for k in ('CUDA_DEVICE_ORDER', 'OMP_NUM_THREADS',
                         'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'TOKENIZERS_PARALLELISM', 'HF_HUB_OFFLINE')},
    }


def main() -> None:
    if os.environ.get('HF_HUB_OFFLINE') != '1':
        raise ValueError('Offline checkpoint/processor loading is required')
    for name, expected in ORIGINAL.items():
        if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Original server source changed: {name}')
    before, original_send, runtime = fingerprint(), base.send_msg, None

    def send(conn, value):
        nonlocal runtime
        if 'checkpoint_id' in value and 'checkpoint_files_sha256' in value:
            if runtime is None:
                if fingerprint() != before:
                    raise RuntimeError('SmolVLA source/runtime changed during initialization')
                runtime = dict(before)
            value = dict(value, sampling_runtime=runtime,
                         sampling_runtime_sha256=hashlib.sha256(json.dumps(runtime, sort_keys=True).encode()).hexdigest(),
                         sampling_server_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        original_send(conn, value)

    base.send_msg = send
    base.main()


if __name__ == '__main__':
    main()
