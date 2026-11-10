#!/usr/bin/env python3
"""E5 variant of serve_icassp_molmo.py: adds a read-only backbone hidden-state capture.

Identical inference, seed contract, J handling and audit; the only additions are the
`backbone_pooled` feature (final RMSNorm output of the vision-language backbone,
same first forward pass as `kv_pooled`) and its ping description. Default port 5660.
No recovery/teacher/monitor behavior. Protocol: 4-byte big-endian JSON length,
then UTF-8 JSON. Images are base64-encoded uint8 HWC .npy arrays. Every inference
requires an explicit seed. This server is sequential per model, threaded for I/O.
"""
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import random
import socket
import struct
import threading
import time
from datetime import datetime, timezone

import numpy as np

MAX_MESSAGE = 64 * 1024 * 1024
HANDOFF = {
    "site": "ActionExpert._prepare_kv_context incoming encoder_kv_states",
    "capture": "first call of first action sample; before original method",
    "pooling": "each layer K then V: flatten all axes except last, mean; concatenate",
    "meaning": "incoming context cache, not evidence of an exclusively visual representation",
    "dtype": "float32",
}
BACKBONE = {
    "site": "MolmoAct2TextModel.ln_f forward output (module path model.model.transformer.ln_f)",
    "layer": "final RMSNorm applied after the last of 36 decoder blocks; this is the "
             "backbone last_hidden_state whose K/V cache feeds the action expert",
    "capture": "first ln_f call of the first action sample; same forward pass as kv_pooled",
    "pooling": "mean over all sequence positions, then the last sequence position; concatenated",
    "positions": "batch 1, processor attention mask all ones (dropped as trivial); every token pooled",
    "positional_encoding": "RoPE is applied to Q/K inside attention only; no positional "
                           "encoding is added to this residual-stream hidden state",
    "dtype": "float32",
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def checkpoint_manifest(root):
    """Content identity includes weights, local model code, tokenizer and configs."""
    root = Path(root).resolve(strict=True)
    entries = []
    for p in sorted(root.iterdir()):
        if p.is_file() and p.suffix in {".safetensors", ".json", ".py", ".jinja"}:
            entries.append({"path": p.name, "size": p.stat().st_size,
                            "sha256": sha256_file(p)})
    if not any(e["path"].endswith(".safetensors") for e in entries):
        raise ValueError("checkpoint has no safetensors weights")
    digest = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
    return {"path": str(root), "sha256": digest, "files": entries}


def recv_exact(conn, n):
    chunks = bytearray()
    while len(chunks) < n:
        part = conn.recv(n - len(chunks))
        if not part:
            if not chunks:
                return None
            raise ConnectionError("incomplete message")
        chunks.extend(part)
    return bytes(chunks)


def recv_msg(conn):
    hdr = recv_exact(conn, 4)
    if hdr is None:
        return None
    size = struct.unpack(">I", hdr)[0]
    if size < 2 or size > MAX_MESSAGE:
        raise ValueError("invalid message length")
    payload = recv_exact(conn, size)
    if payload is None:
        raise ConnectionError("missing message body")
    obj = json.loads(payload)
    if not isinstance(obj, dict):
        raise ValueError("request must be a JSON object")
    return obj


def send_msg(conn, obj):
    data = json.dumps(obj, allow_nan=False).encode()
    conn.sendall(struct.pack(">I", len(data)) + data)


def validate_request(req, default_j):
    seed = req.get("seed")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    if "J" in req and "j" in req and req["J"] != req["j"]:
        raise ValueError("J and j disagree")
    j = req.get("J", req.get("j", default_j))
    if type(j) is not int or not 1 <= j <= 64:
        raise ValueError("J must be an integer in [1, 64]")
    if not isinstance(req.get("task"), str) or not req["task"].strip():
        raise ValueError("task must be a nonempty string")
    state = np.asarray(req.get("state"), dtype=np.float32)
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError("state must contain 8 finite floats")
    encoded = req.get("images")
    if not isinstance(encoded, list) or len(encoded) != 2:
        raise ValueError("images must contain agentview and wrist arrays")
    images = []
    for value in encoded:
        arr = np.load(io.BytesIO(base64.b64decode(value, validate=True)), allow_pickle=False)
        if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError("images must be uint8 HWC RGB arrays")
        if not (1 <= arr.shape[0] <= 2048 and 1 <= arr.shape[1] <= 2048):
            raise ValueError("invalid image dimensions")
        images.append(arr)
    if type(req.get("audit", False)) is not bool:
        raise ValueError("audit must be boolean")
    return seed, j, state, images


class PlainPolicy:
    def __init__(self, root, default_j):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        import transformers

        self.torch = torch
        self.default_j = default_j
        self.lock = threading.Lock()
        self.capture = {}
        self.capture_enabled = False
        torch.set_num_threads(int(os.environ.get("POLICY_CPU_THREADS", "4")))
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        print(f"[{utc_now()}] hashing checkpoint {root}", flush=True)
        manifest = checkpoint_manifest(root)
        print(f"[{utc_now()}] checkpoint sha256={manifest['sha256']}; loading", flush=True)
        self.processor = AutoProcessor.from_pretrained(root, trust_remote_code=True,
                                                       local_files_only=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            root, trust_remote_code=True, local_files_only=True,
            dtype=torch.bfloat16).to("cuda").eval()
        experts = [m for m in self.model.modules() if type(m).__name__ == "ActionExpert"]
        if len(experts) != 1:
            raise RuntimeError(f"expected one ActionExpert, found {len(experts)}")
        self.expert = experts[0]
        self.original = self.expert._prepare_kv_context
        self.expert._prepare_kv_context = self.wrapped_context
        norms = [m for n, m in self.model.named_modules() if n.endswith("transformer.ln_f")]
        if len(norms) != 1:
            raise RuntimeError(f"expected one backbone ln_f, found {len(norms)}")
        self.ln_f = norms[0]
        self.ln_f_handle = self.ln_f.register_forward_hook(self.backbone_hook)
        text = self.model.config.text_config if hasattr(self.model.config, "text_config") else self.model.config
        kv_dim = text.num_hidden_layers * 2 * text.num_key_value_heads * text.head_dim
        self.metadata = {
            "ok": True, "protocol": "vla-icassp-plain-v1", "model": "MolmoAct2-LIBERO",
            "checkpoint": manifest, "checkpoint_id": manifest["sha256"],
            "J": default_j, "max_J": 64,
            "code_sha256": sha256_file(__file__), "handoff": HANDOFF, "backbone": BACKBONE,
            "feature_dims": {"kv_pooled": kv_dim, "backbone_pooled": 2 * text.hidden_size},
            "base_server": "code/serve_icassp_molmo.py", "e5": True,
            "started_at": utc_now(), "python": platform.python_version(),
            "torch": torch.__version__, "transformers": transformers.__version__,
            "numpy": np.__version__, "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "action_mode": "continuous", "norm_tag": "libero", "num_steps": 10,
            "depth_reasoning": False, "cuda_graph": False,
            "seed_contract": "request torch/python/numpy seeds; sequential J draws; first draw executed",
            "determinism_scope": "same software/device; cross-GPU bitwise parity not asserted",
        }

    def backbone_hook(self, _module, _inputs, output):
        if self.capture_enabled and "backbone" not in self.capture:
            x = output.detach().float()
            if x.ndim != 3 or x.shape[0] != 1:
                raise RuntimeError(f"unexpected backbone hidden state shape {list(x.shape)}")
            pooled = self.torch.cat([x[0].mean(0), x[0, -1]])
            self.capture["backbone"] = pooled.cpu().numpy().astype(np.float32)
            self.capture["backbone_shape"] = list(x.shape)

    def wrapped_context(self, encoder_kv_states):
        if self.capture_enabled and "kv" not in self.capture:
            vecs, shapes = [], []
            for layer in encoder_kv_states:
                for tensor in layer:
                    x = tensor.detach().float()
                    shapes.append(list(x.shape))
                    vecs.append(x.reshape(-1, x.shape[-1]).mean(0))
            if not vecs:
                raise RuntimeError("empty incoming context cache")
            self.capture["kv"] = self.torch.cat(vecs).cpu().numpy().astype(np.float32)
            self.capture["shapes"] = shapes
        return self.original(encoder_kv_states)

    def predict(self, images, task, state):
        torch = self.torch
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = self.model.predict_action(
                processor=self.processor, images=images, task=task, state=state,
                norm_tag="libero", inference_action_mode="continuous",
                enable_depth_reasoning=False, num_steps=10, enable_cuda_graph=False)
        actions = output.actions
        if torch.is_tensor(actions):
            actions = actions.detach().float().cpu().numpy()
        actions = np.asarray(actions, dtype=np.float32)
        actions = actions.reshape(-1, actions.shape[-1])
        if actions.ndim != 2 or actions.shape[-1] != 7 or not np.isfinite(actions).all():
            raise RuntimeError(f"invalid LIBERO action output: {actions.shape}")
        return actions

    def seed(self, value):
        self.torch.manual_seed(value)
        self.torch.cuda.manual_seed_all(value)
        random.seed(value)
        np.random.seed(value % 2**32)

    def infer(self, req):
        from PIL import Image

        seed, j, state, arrays = validate_request(req, self.default_j)
        images = [Image.fromarray(a) for a in arrays]
        torch = self.torch
        with self.lock:
            np_state, py_state = np.random.get_state(), random.getstate()
            try:
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    self.seed(seed)
                    self.capture.clear()
                    self.capture_enabled = True
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    first = self.predict(images, req["task"], state)
                    torch.cuda.synchronize()
                    first_seconds = time.perf_counter() - start
                    self.capture_enabled = False
                    if "kv" not in self.capture:
                        raise RuntimeError("incoming context hook was not invoked")
                    if "backbone" not in self.capture:
                        raise RuntimeError("backbone ln_f hook was not invoked")
                    actions = [first]
                    for _ in range(j - 1):
                        actions.append(self.predict(images, req["task"], state))
                    torch.cuda.synchronize()
                    inference_seconds = time.perf_counter() - start
                    audit = None
                    if req.get("audit", False):
                        self.seed(seed)
                        self.expert._prepare_kv_context = self.original
                        self.ln_f_handle.remove()
                        audit_start = time.perf_counter()
                        try:
                            unhooked = self.predict(images, req["task"], state)
                            torch.cuda.synchronize()
                        finally:
                            self.expert._prepare_kv_context = self.wrapped_context
                            self.ln_f_handle = self.ln_f.register_forward_hook(self.backbone_hook)
                        audit = {"exact_equal": bool(np.array_equal(first, unhooked)),
                                 "max_abs_diff": float(np.max(np.abs(first - unhooked))),
                                 "seconds": time.perf_counter() - audit_start,
                                 "scope": "same input/seed, first action chunk, kv and ln_f hooks on versus both removed"}
                    stack = np.stack(actions)
                    return {
                        "ok": True, "actions": first.tolist(), "action_samples": stack.tolist(),
                        "action_std": float(stack.std(axis=0, ddof=0).mean()),
                        "kv_pooled": self.capture["kv"].tolist(),
                        "kv_shapes": self.capture["shapes"], "handoff": HANDOFF,
                        "backbone_pooled": self.capture["backbone"].tolist(),
                        "backbone_shape": self.capture["backbone_shape"], "backbone": BACKBONE,
                        "seed": seed, "J": j, "timestamp": utc_now(),
                        "first_sample_seconds": first_seconds,
                        "inference_seconds": inference_seconds,
                        "checkpoint_sha256": self.metadata["checkpoint"]["sha256"],
                        "code_sha256": self.metadata["code_sha256"], "audit": audit,
                    }
            finally:
                self.capture_enabled = False
                np.random.set_state(np_state)
                random.setstate(py_state)


def main():
    root = os.environ.get("MOLMO_MODEL_PATH")
    if not root:
        raise ValueError("MOLMO_MODEL_PATH must identify the local checkpoint directory")
    port = int(os.environ.get("POLICY_SERVER_PORT", "5660"))
    default_j = int(os.environ.get("POLICY_SERVER_J", "8"))
    if not 1 <= default_j <= 64:
        raise ValueError("POLICY_SERVER_J must be between 1 and 64")
    policy = PlainPolicy(str(Path(root).resolve(strict=True)), default_j)
    metadata_path = os.environ.get("POLICY_METADATA_PATH")
    if metadata_path:
        path = Path(metadata_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(policy.metadata, indent=2) + "\n")

    def serve(conn):
        with conn:
            while True:
                try:
                    request = recv_msg(conn)
                    if request is None:
                        return
                    if request.get("cmd") == "ping":
                        send_msg(conn, policy.metadata)
                    elif "cmd" in request:
                        raise ValueError("only ping is supported as a command")
                    else:
                        send_msg(conn, policy.infer(request))
                except Exception as exc:
                    print(f"[{utc_now()}] request error: {type(exc).__name__}: {exc}", flush=True)
                    try:
                        send_msg(conn, {"ok": False, "error": str(exc),
                                        "error_type": type(exc).__name__})
                    except OSError:
                        pass
                    return

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", port))
        server.listen(16)
        print(f"[{utc_now()}] READY 127.0.0.1:{port} J={default_j} "
              f"device={policy.metadata['device']} code={policy.metadata['code_sha256']}", flush=True)
        while True:
            conn, _ = server.accept()
            threading.Thread(target=serve, args=(conn,), daemon=True).start()


if __name__ == "__main__":
    main()
