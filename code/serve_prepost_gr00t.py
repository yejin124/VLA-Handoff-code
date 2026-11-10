#!/usr/bin/env python3
"""GR00T-N1.7 servers with pre-interface capture (docs/PROTOCOLS.md).

Identical to the recorded servers (serve_simpler_gr00t.py for SimplerEnv-Bridge, serve_icassp_gr00t.py for LIBERO)
plus two capture points in front of the recorded handoff, all mean-pooled over tokens with the same pool_feature and
taken on the first DiT call of draw 0:
    pre_pooled   input of action_head.vlln   (backbone output handed to the action head: System 2's output)
    mid_pooled   output of action_head.vlln  (after LayerNorm, before the four VL self-attention layers)
    kv_pooled    DiT encoder_hidden_states   (the recorded handoff, captured by the base server)
Requests are served exactly as by the base servers, so action samples must reproduce the recordings bitwise on the
recorded GPU model.

Run in the GR00T env with CUDA_DEVICE_ORDER=PCI_BUS_ID:
  GR00T_ROOT=<Isaac-GR00T checkout> PREPOST_MODE=bridge GR00T_MODEL_PATH=<bridge snapshot> POLICY_SERVER_PORT=5960 python code/serve_prepost_gr00t.py
  GR00T_ROOT=<Isaac-GR00T checkout> PREPOST_MODE=libero GR00T_MODEL_PATH=<.../GR00T-N1.7-LIBERO/libero_spatial> POLICY_SERVER_PORT=5961 python code/serve_prepost_gr00t.py
"""
import json
import os
import socket
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import serve_icassp_gr00t as libero  # noqa: E402
import serve_simpler_gr00t as bridge  # noqa: E402
from serve_icassp_gr00t import check_fork_switches, check_port, pool_feature, recv_msg, send_msg, sha256_file  # noqa: E402

SITES = {'pre': 'forward pre-hook on action_head.vlln: its input, the backbone output tokens handed to the action head',
         'mid': 'forward hook on action_head.vlln: its output, after LayerNorm and before the VL self-attention layers',
         'post': 'kv_pooled of the base server: DiT encoder_hidden_states after the VL self-attention (recorded handoff)'}


def attach_prepost(server):
    head = server.policy.model.action_head
    if not hasattr(head, 'vlln'):
        raise RuntimeError('action head has no vlln module; the interface layout differs from the recorded servers')

    def first_tensor(args, kwargs):
        return args[0] if args else next(iter(kwargs.values()))

    def pre_in(_module, args, kwargs):
        if server.capturing and 'pre' not in server.capture:
            server.capture['pre'] = pool_feature(first_tensor(args, kwargs))
        return None

    def mid_out(_module, _args, _kwargs, output):
        if server.capturing and 'mid' not in server.capture:
            server.capture['mid'] = pool_feature(output)
        return None

    head.vlln.register_forward_pre_hook(pre_in, with_kwargs=True)
    head.vlln.register_forward_hook(mid_out, with_kwargs=True)
    server.metadata = dict(server.metadata, prepost_sites=SITES, prepost_code_sha256=sha256_file(__file__),
                           vlln=type(head.vlln).__name__)


def extend(server, out):
    for key in ('pre', 'mid'):
        if key not in server.capture:
            raise RuntimeError(f'{key} hook was not invoked')
        out[f'{key}_pooled'] = server.capture[key].tolist()
    return out


class BridgePrePost(bridge.Bridge):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        attach_prepost(self)

    def predict(self, req):
        return extend(self, super().predict(req))


class LiberoPrePost(libero.Bridge):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        attach_prepost(self)

    def predict(self, req, audit_dir):
        return extend(self, super().predict(req, audit_dir))


def main():
    check_fork_switches()
    mode = os.environ['PREPOST_MODE']
    gr00t_root = Path(os.environ['GR00T_ROOT']).resolve(strict=True)  # Isaac-GR00T checkout
    sys.path.insert(0, str(gr00t_root))
    port = check_port(int(os.environ['POLICY_SERVER_PORT']))
    model_path = Path(os.environ['GR00T_MODEL_PATH']).resolve(strict=True)
    draws = int(os.environ.get('POLICY_SERVER_J', '8'))
    if mode == 'bridge':
        server = BridgePrePost(model_path, int(os.environ.get('POLICY_EXECUTE_STEPS', '4')), draws)
        handle = server.predict
    elif mode == 'libero':
        audit_dir = Path(os.environ.get('AUDIT_DIR', 'logs/gr00t_audits')).resolve()
        audit_dir.mkdir(parents=True, exist_ok=True)
        server = LiberoPrePost(model_path, os.environ.get('GR00T_EMBODIMENT', 'LIBERO_PANDA'),
                               int(os.environ.get('POLICY_EXECUTE_STEPS', '8')), draws, gr00t_root)
        handle = lambda req: server.predict(req, audit_dir)  # noqa: E731
    else:
        raise ValueError('PREPOST_MODE must be bridge or libero')
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('127.0.0.1', port))
        sock.listen(8)
        print(f'[gr00t-prepost-{mode}] READY 127.0.0.1:{port} {json.dumps(server.metadata)}', flush=True)
        while True:
            conn, _ = sock.accept()
            with conn:
                while True:
                    try:
                        req = recv_msg(conn)
                        if req is None:
                            break
                        send_msg(conn, server.metadata if req.get('cmd') == 'ping' else handle(req))
                    except (EOFError, ConnectionError):
                        break
                    except Exception as error:  # noqa: BLE001
                        traceback.print_exc()
                        try:
                            send_msg(conn, {'ok': False, 'error': str(error), 'error_type': type(error).__name__})
                        except OSError:
                            break


if __name__ == '__main__':
    main()
