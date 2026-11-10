#!/usr/bin/env python3
"""GR00T-N1.7-SimplerEnv-Bridge server with a handoff intervention (docs/PROTOCOLS.md).

Identical to the Bridge collection server (code/serve_simpler_gr00t.py) except for one
extra DiT pre-hook, registered in front of the capture hook: when a request carries "delta" (a vector of the
handoff's width), it is added to every token of encoder_hidden_states at every DiT call. The captured pooled handoff
is therefore the intervened one. Requests without "delta" are served exactly as before.

Run in the GR00T environment, e.g.
  GR00T_ROOT=<Isaac-GR00T checkout> GR00T_MODEL_PATH=<snapshot> POLICY_SERVER_PORT=5950 python code/serve_injection_gr00t.py
"""
import json
import os
import socket
import sys
import traceback
from pathlib import Path

import numpy as np

RAL_ROOT = Path(os.environ.get('RAL_ROOT', Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import serve_simpler_gr00t as base  # noqa: E402
from serve_icassp_gr00t import check_fork_switches, check_port, recv_msg, send_msg  # noqa: E402


class InjectionBridge(base.Bridge):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delta = None
        self.calls = 0
        torch = self.torch

        def inject(_module, args, kwargs):
            if self.delta is None:
                return None
            ehs = kwargs['encoder_hidden_states']
            if ehs.shape[-1] != self.delta.shape[0]:
                raise RuntimeError(f'delta width {self.delta.shape[0]} != handoff width {ehs.shape[-1]}')
            kwargs = dict(kwargs, encoder_hidden_states=ehs + torch.as_tensor(self.delta, dtype=ehs.dtype, device=ehs.device))
            self.calls += 1
            return args, kwargs
        self.dit.register_forward_pre_hook(inject, with_kwargs=True, prepend=True)
        self.metadata = dict(self.metadata, injection_code_sha256=base.sha256_file(__file__))

    def predict(self, req):
        delta = req.get('delta')
        req = {k: v for k, v in req.items() if k != 'delta'}
        self.delta = None if delta is None else np.asarray(delta, dtype=np.float32)
        if self.delta is not None and (self.delta.ndim != 1 or not np.isfinite(self.delta).all()):
            raise ValueError('delta must be a finite vector')
        self.calls = 0
        try:
            out = super().predict(req)
        finally:
            used, self.delta = self.calls, None
        if delta is not None and used == 0:
            raise RuntimeError('injection hook was not invoked')
        out['injection_calls'] = used
        return out


def main():
    check_fork_switches()
    gr00t_root = Path(os.environ['GR00T_ROOT']).resolve(strict=True)  # Isaac-GR00T checkout
    sys.path.insert(0, str(gr00t_root))
    port = check_port(int(os.environ.get('POLICY_SERVER_PORT', '5950')))
    bridge = InjectionBridge(Path(os.environ['GR00T_MODEL_PATH']).resolve(strict=True),
                             int(os.environ.get('POLICY_EXECUTE_STEPS', '4')), int(os.environ.get('POLICY_SERVER_J', '8')))
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('127.0.0.1', port))
        server.listen(8)
        print(f'[gr00t-injection] READY 127.0.0.1:{port} {json.dumps(bridge.metadata)}', flush=True)
        while True:
            conn, _ = server.accept()
            with conn:
                while True:
                    try:
                        req = recv_msg(conn)
                        if req is None:
                            break
                        send_msg(conn, bridge.metadata if req.get('cmd') == 'ping' else bridge.predict(req))
                    except (EOFError, ConnectionError):
                        break
                    except Exception as error:
                        traceback.print_exc()
                        try:
                            send_msg(conn, {'ok': False, 'error': str(error)})
                        except OSError:
                            break


if __name__ == '__main__':
    main()
