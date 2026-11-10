#!/usr/bin/env python3
"""Deterministic pi0.5 LIBERO measurement bridge using official policy transforms.

Wire protocol: uint32 big-endian byte length followed by JSON. Inference requires
images=[base64-npy agent RGB, base64-npy wrist RGB], state[8], task, seed.
Images must already have the official LIBERO camera rotation. J may be 1..64.
Actions and every raw sample are post-output-transform, valid 7D LIBERO actions.
The handoff summary is explicitly PRE-RoPE K/V projections, not actual KV cache.
"""
import base64
import dataclasses
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import time
import traceback

import numpy as np

MAX_MESSAGE = 16 * 1024 * 1024
HANDOFF = "first prefix pre-RoPE K/V projection outputs; mean over all tokens; layer-sorted K then V; not post-RoPE cache"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def recv_exact(conn, size):
    output = bytearray()
    while len(output) < size:
        block = conn.recv(size - len(output))
        if not block:
            if output:
                raise EOFError("partial request")
            return None
        output.extend(block)
    return bytes(output)


def recv_msg(conn):
    header = recv_exact(conn, 4)
    if header is None:
        return None
    size = struct.unpack(">I", header)[0]
    if not 0 < size <= MAX_MESSAGE:
        raise ValueError("request size out of bounds")
    body = recv_exact(conn, size)
    if body is None:
        raise EOFError("missing request body")
    request = json.loads(body)
    if not isinstance(request, dict):
        raise ValueError("request must be a JSON object")
    return request


def send_msg(conn, obj):
    data = json.dumps(obj, allow_nan=False).encode()
    conn.sendall(struct.pack(">I", len(data)) + data)


def decode_request(req, default_j):
    seed = req.get("seed")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed is mandatory and must be an integer in [0, 2**63)")
    draws = req.get("J", default_j)
    if type(draws) is not int or not 1 <= draws <= 64:
        raise ValueError("J must be an integer in [1, 64]")
    if type(req.get("audit", False)) is not bool:
        raise ValueError("audit must be boolean")
    task = req.get("task")
    if not isinstance(task, str) or not task.strip() or len(task) > 4096:
        raise ValueError("task must be a nonempty instruction of at most 4096 characters")
    state = np.asarray(req.get("state"), dtype=np.float32)
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError("state must contain 8 finite LIBERO state values")
    encoded = req.get("images")
    if not isinstance(encoded, list) or len(encoded) != 2:
        raise ValueError("images must contain agent and wrist base64-npy arrays")
    images = []
    for value in encoded:
        image = np.load(io.BytesIO(base64.b64decode(value, validate=True)), allow_pickle=False)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("images must be uint8 HWC RGB arrays")
        if not 1 <= image.shape[0] <= 2048 or not 1 <= image.shape[1] <= 2048:
            raise ValueError("image dimensions out of bounds")
        images.append(image)
    return seed, draws, task, state, images


def main():
    openpi = Path(os.environ.get("OPENPI_ROOT", "~/scratch/openpi")).expanduser().resolve()
    checkpoint = Path(os.environ.get("PI05_CHECKPOINT", "~/.cache/openpi/pi05_libero_pytorch")).expanduser().resolve()
    port = int(os.environ.get("POLICY_SERVER_PORT", os.environ.get("POLICY_PORT", "5655")))
    draws_default = int(os.environ.get("POLICY_SERVER_J", "8"))
    if not 1 <= draws_default <= 64:
        raise ValueError("POLICY_SERVER_J must be 1..64")
    audit_dir = Path(os.environ.get("AUDIT_DIR", "logs/pi05_audits")).resolve()
    audit_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(openpi / "src"))
    import torch
    from openpi.training import config
    from openpi.policies import policy_config
    from openpi_client import image_tools

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    source_files = ["models_pytorch/pi0_pytorch.py", "models_pytorch/gemma_pytorch.py",
                    "policies/policy.py", "policies/policy_config.py", "policies/libero_policy.py",
                    "transforms.py", "training/config.py"]
    metadata = {
        "ok": True, "schema_version": 1, "model": "pi0.5-LIBERO", "J": draws_default,
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint / "model.safetensors"),
        "code_sha256": sha256(__file__), "handoff_description": HANDOFF,
        "action_dimensions": 7, "state_dimensions": 8,
        "execution_sample": 0, "spread_ddof": 0, "seed_required": True,
        "torch_version": torch.__version__, "numpy_version": np.__version__,
        "openpi_commit": subprocess.check_output(["git", "-C", str(openpi), "rev-parse", "HEAD"], text=True).strip(),
        "openpi_source_sha256": {p: sha256(openpi / "src/openpi" / p) for p in source_files},
        "checkpoint_assets_sha256": {str(p.relative_to(checkpoint)): sha256(p)
                                     for p in sorted((checkpoint / "assets").rglob("*")) if p.is_file()},
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    metadata["checkpoint_id"] = "sha256:" + metadata["checkpoint_sha256"]
    metadata["pytorch_compile_mode"] = None
    print("[pi05] loading official pi05_libero policy", flush=True)
    cfg = config.get_config("pi05_libero")
    # Eager inference preserves transparent hooks and avoids compiler cache/graph
    # effects during the hooked/unhooked audit; weights/transforms are unchanged.
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, pytorch_compile_mode=None))
    policy = policy_config.create_trained_policy(cfg, str(checkpoint), pytorch_device="cuda")
    model = policy._model
    lm = model.paligemma_with_expert.paligemma.language_model
    layers = lm.model.layers if hasattr(lm, "model") else lm.layers
    metadata["action_horizon"] = cfg.model.action_horizon
    capture, handles = {}, []
    capturing = False

    def make_hook(name):
        def hook(_module, _inputs, output):
            if capturing and name not in capture:
                capture[name] = output.detach().float().reshape(-1, output.shape[-1]).mean(0).cpu().numpy()
        return hook

    def attach_hooks():
        for index, layer in enumerate(layers):
            for kind in ("k", "v"):
                projection = getattr(layer.self_attn, f"{kind}_proj")
                handles.append(projection.register_forward_hook(make_hook(f"{index:02d}_{kind}")))

    attach_hooks()

    def seed_rng(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def infer(element):
        with torch.inference_mode():
            action = np.asarray(policy.infer(element)["actions"], dtype=np.float32)
        if action.shape != (cfg.model.action_horizon, 7) or not np.isfinite(action).all():
            raise RuntimeError(f"invalid official action output: {action.shape}")
        return action

    def predict(req):
        nonlocal capturing
        seed, draws, task, state, images = decode_request(req, draws_default)
        element = {
            "observation/image": image_tools.convert_to_uint8(image_tools.resize_with_pad(images[0], 224, 224)),
            "observation/wrist_image": image_tools.convert_to_uint8(image_tools.resize_with_pad(images[1], 224, 224)),
            "observation/state": state, "prompt": task,
        }
        capture.clear()
        started = time.monotonic()
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            seed_rng(seed)
            samples = []
            try:
                for index in range(draws):
                    capturing = index == 0
                    samples.append(infer(element))
            finally:
                capturing = False
            if len(capture) != 2 * len(layers):
                raise RuntimeError("incomplete handoff capture")
            handoff = np.concatenate([capture[key] for key in sorted(capture)])
            audit = None
            if req.get("audit", False):
                # Physically remove hooks: parity compares ordinary official inference.
                for handle in handles:
                    handle.remove()
                handles.clear()
                try:
                    seed_rng(seed)
                    unhooked = infer(element)
                finally:
                    attach_hooks()
                maximum = float(np.abs(samples[0] - unhooked).max())
                audit = {"hooked_unhooked_equal": bool(np.array_equal(samples[0], unhooked)),
                         "max_absolute_action_difference": maximum, "seed": seed,
                         "valid_action_shape": list(samples[0].shape),
                         "handoff_dimensions": int(handoff.size), "handoff_description": HANDOFF,
                         "code_sha256": metadata["code_sha256"], "checkpoint_sha256": metadata["checkpoint_sha256"]}
                identity = hashlib.sha256(json.dumps(req, sort_keys=True).encode()).hexdigest()
                audit["request_sha256"] = identity
                (audit_dir / f"{identity}.json").write_text(json.dumps(audit, indent=2))
                if not audit["hooked_unhooked_equal"]:
                    raise RuntimeError(f"hook parity failed: {audit}")
        stacked = np.stack(samples)
        return {"ok": True, "actions": samples[0].tolist(), "action_samples": stacked.tolist(),
                "kv_pooled": handoff.tolist(), "action_std": float(stacked.std(axis=0, ddof=0).mean()),
                "seed": seed, "J": draws, "audit": audit, "token_file": None,
                "elapsed_seconds": time.monotonic() - started,
                "code_sha256": metadata["code_sha256"]}

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", port))
        server.listen(8)
        print(f"[pi05] READY 127.0.0.1:{port} {json.dumps(metadata)}", flush=True)
        while True:
            conn, _ = server.accept()
            with conn:
                while True:
                    try:
                        req = recv_msg(conn)
                        if req is None:
                            break
                        if req.get("cmd") == "ping":
                            response = metadata
                        else:
                            response = predict(req)
                        send_msg(conn, response)
                    except (EOFError, ConnectionError):
                        break
                    except Exception as error:
                        traceback.print_exc()
                        try:
                            send_msg(conn, {"ok": False, "error": type(error).__name__, "message": str(error)})
                        except OSError:
                            break


if __name__ == "__main__":
    main()
