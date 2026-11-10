#!/usr/bin/env python3
"""Deterministic tau-0-VLA-LIBERO measurement bridge over the official Tau0VLAPolicy.

Wire protocol: uint32 big-endian byte length followed by JSON, identical to the
pi0.5 / GR00T / MolmoAct2 bridges. Inference requires images=[base64-npy agent RGB,
base64-npy wrist RGB], state[8], task, seed. Images must already carry the
180-degree LIBERO rotation (obs[::-1, ::-1]) that code/icassp_rollout.py applies.
The server undoes that flip and calls the official client's own preprocessing
(deploy/libero/main.py::_prepare_image: flip + PIL bilinear resize to 224), so the
model sees exactly what the official evaluator would send for the same render.
The payload then goes through the official LIBERO payload adapter and
Tau0VLAPolicy.infer (encode_payload -> Qwen3.5 tokenization -> flow matching ->
restore_action). Actions are the first POLICY_EXECUTE_STEPS (default 8, the official
--replan-steps) of the 10-step chunk, unchanged (the official client steps them as-is).
J may be 1..64; sample 0 is executed.

Handoff: the expert reads the backbone only through the prefix KV cache of the six
full-attention Qwen3.5 layers (linear-attention layers of the expert see suffix
tokens only). The feature is that cache as the expert reads it (handle_kv_cache with
fill_kv_cache=False) at denoising step 0 of sample 0. Module hooks cannot be used:
predict_velocity calls qwenvl_with_expert.forward(...) directly.
"""
import base64
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import random
import socket
import struct
import subprocess
import sys
import time
import traceback

import numpy as np

MAX_MESSAGE = 64 * 1024 * 1024
RESERVED_PORTS = (range(5650, 5657), range(5740, 5836))   # running project servers
RESIZE = 224   # deploy/libero/main.py:75 resize_size
SOURCE_FILES = ["deploy/libero/main.py", "deploy/libero_server.py", "deploy/policy.py", "deploy/warmup.py",
                "deploy/_bootstrap.py", "src/tau0_vla/adapters/libero/deploy_io.py",
                "src/tau0_vla/adapters/libero/layout.py", "src/tau0_vla/data/data_spec.py",
                "src/tau0_vla/data/pipeline.py",
                "src/tau0_vla/models/vision_language_action_models/tau0_vla/modeling_tau_vla.py",
                "src/tau0_vla/models/vision_language_action_models/tau0_vla/qwen35vl_in_vla.py"]
HANDOFF = {
    "site": "read-only wrapper on flow_matching.qwenvl_with_expert.handle_kv_cache (instance attribute), "
            "calls with fill_kv_cache=False: the expert reading the cached prefix K/V, denoising step 0",
    "meaning": "prefix (VLM) KV cache of the full_attention layers, the only backbone information the action "
               "expert reads (modeling_tau_vla.py: handle_kv_cache, joint attention on full_attention layers; "
               "linear_attention expert layers attend to suffix tokens only)",
    "positional_encoding": "K is post-MRoPE (RoPE applied before cache fill); V has no positional encoding",
    "capture": "first expert read per layer of the first action sample only; wrapper returns the original result",
    "pooling": "per full_attention layer (ascending index): mean over batch and prefix tokens of K "
               "[kv_heads*head_dim], then of V; concatenated -> float32",
    "dtype": "float32 (cache is bfloat16)",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_port(port):
    if any(port in r for r in RESERVED_PORTS) or not 1024 <= port <= 65535:
        raise ValueError(f"port {port} is reserved or out of range")
    return port


def recv_exact(conn, size):
    output = bytearray()
    while len(output) < size:
        block = conn.recv(size - len(output))
        if not block:
            if output:
                raise ConnectionError("incomplete message")
            return None
        output.extend(block)
    return bytes(output)


def recv_msg(conn):
    header = recv_exact(conn, 4)
    if header is None:
        return None
    size = struct.unpack(">I", header)[0]
    if not 2 <= size <= MAX_MESSAGE:
        raise ValueError("invalid message length")
    body = recv_exact(conn, size)
    if body is None:
        raise ConnectionError("missing message body")
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
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or image.shape[0] != image.shape[1]:
            raise ValueError("images must be square uint8 HWC RGB arrays")
        if not 1 <= image.shape[0] <= 2048:
            raise ValueError("image dimensions out of bounds")
        images.append(np.ascontiguousarray(image))
    return seed, draws, task, state, images


@contextlib.contextmanager
def seed_scope(seed, torch):
    """Fork Torch (CPU+CUDA)/Python/NumPy RNG, seed like the official /reset_episode, restore on exit."""
    np_state, py_state = np.random.get_state(), random.getstate()
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(seed)          # libero_server.py reset_episode: torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed % 2**32)
            yield
    finally:
        np.random.set_state(np_state)
        random.setstate(py_state)


class Bridge:
    def __init__(self, tau0_root, checkpoint, execute_steps, default_j):
        os.chdir(tau0_root)   # run_spec.json names scripts/deepspeed/zero1.json relative to the repo root
        sys.path.insert(0, str(tau0_root))
        import torch
        import transformers
        import tau0_vla
        from deploy._bootstrap import (discover_checkpoint_config_modules, ensure_configs_registered,
                                       ensure_policy_manifest, resolve_deploy_io)
        from deploy.libero.main import _prepare_image
        from deploy.libero_server import _validate_action_chunk, _validate_libero_checkpoint
        from deploy.policy import Tau0VLAPolicy
        from deploy.warmup import _configure_inference_mode

        if not Path(tau0_vla.__file__).resolve().is_relative_to(Path(tau0_root).resolve()):
            raise RuntimeError(f"tau0_vla imported from {tau0_vla.__file__}, not TAU0_ROOT={tau0_root}")
        self.torch, self.default_j = torch, default_j
        self.prepare_image, self.validate_chunk = _prepare_image, _validate_action_chunk
        self.capture, self.capturing, self.handles = {}, False, []
        # Backend flags stay at the PyTorch defaults the official deploy.libero_server runs with; forcing
        # cudnn.allow_tf32=False / deterministic=True changed actions by bf16 steps versus the official server.
        if torch.backends.cudnn.benchmark:
            raise RuntimeError("cudnn.benchmark must stay off (default) for repeatable algorithm choice")
        # libero_server.main: register configs, then load (deploy/libero_server.py:157-167)
        ensure_configs_registered()
        ensure_policy_manifest(str(checkpoint), verbose=False)
        discover_checkpoint_config_modules(str(checkpoint))
        self.policy = Tau0VLAPolicy.from_checkpoint(str(checkpoint), device="cuda")
        _validate_libero_checkpoint(self.policy)
        _configure_inference_mode(self.policy, "eager", 0)   # README: --infer-mode eager
        spec = self.policy.data_spec
        deploy_io = resolve_deploy_io(spec)
        state_fd = deploy_io.load_state_field_descriptions(spec.artifacts_dir)
        self.adapt = deploy_io.build_payload_adapter(cam_keys=spec.cam_keys, state_fd=state_fd,
                                                     state_dim=deploy_io.state_dim_from_field_descriptions(state_fd))
        self.horizon = int(spec.action_chunk_size)
        if not 1 <= execute_steps <= self.horizon:
            raise ValueError(f"POLICY_EXECUTE_STEPS must be in [1, {self.horizon}]")
        self.execute_steps = execute_steps
        fm = self.policy.model.flow_matching
        if int(fm.config.max_prefix_len) != 0:
            raise RuntimeError("eager (dynamic KV cache) inference required")
        self.joint = fm.qwenvl_with_expert
        if type(self.joint).__name__ != "Qwen35vlWithExpertModel":
            raise RuntimeError(f"unexpected joint model {type(self.joint).__name__}")
        text_cfg = self.joint.qwenvl.config.text_config
        self.full_layers = [i for i, t in enumerate(text_cfg.layer_types) if t == "full_attention"]
        kv_width = int(text_cfg.num_key_value_heads * text_cfg.head_dim)
        self.attach_hooks()
        manifest = {p: sha256_file(Path(checkpoint) / p) for p in
                    ["model.safetensors", "config.json", "run_spec.json", "policy_manifest.json",
                     "processor_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]
                    + [str(p.relative_to(checkpoint)) for p in sorted(Path(checkpoint, "finch_data_spec").rglob("*.json"))]}
        git = lambda *a: subprocess.check_output(["git", "-C", str(tau0_root), *a], text=True).strip()
        self.metadata = {
            "ok": True, "protocol": "vla-icassp-plain-v1", "model": "tau0-VLA-LIBERO/" + Path(checkpoint).name,
            "checkpoint": str(checkpoint), "checkpoint_files_sha256": manifest,
            "checkpoint_sha256": manifest["model.safetensors"], "checkpoint_id": "sha256:" + manifest["model.safetensors"],
            "code_sha256": sha256_file(__file__),
            "tau0_source": {"root": str(tau0_root), "commit": git("rev-parse", "HEAD"),
                            "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
                            "files_sha256": {p: sha256_file(Path(tau0_root) / p) for p in SOURCE_FILES}},
            "J": default_j, "max_J": 64, "execution_sample": 0, "spread_ddof": 0, "seed_required": True,
            "action_horizon": execute_steps, "decoded_action_horizon": self.horizon,
            "num_denoising_steps": int(fm.config.num_steps), "infer_mode": "eager",
            "action_dimensions": 7, "state_dimensions": 8,
            "action_units": "restore_action output (mean_std un-normalized delta xyz, delta axis-angle, gripper), "
                            "stepped unchanged as deploy/libero/main.py:267-268 does",
            "image_contract": "client sends obs[::-1, ::-1]; server undoes it and applies deploy/libero/main.py "
                              "_prepare_image (flip + PIL bilinear resize to 224); policy resize_with_pad to 224 is identity",
            "state_contract": "eef_pos(3) + quat2axisangle(robot0_eef_quat) without wrapping (3) + gripper_qpos(2)",
            "prompt_template": spec.format_instruction("{instruction}"),
            "handoff": dict(HANDOFF, layers=self.full_layers, dimensions=2 * kv_width * len(self.full_layers)),
            "seed_contract": "fork_rng scope; torch.manual_seed(seed) as /reset_episode does, python/numpy seeded; "
                             "sequential J draws; sample 0 executed",
            "determinism_scope": "same process is bitwise repeatable; fla Triton kernels are autotuned from timings, so "
                                 "processes can differ by bf16 steps unless they share TRITON_CACHE_DIR (cache_results)",
            "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR"),
            "fla_cache_results": os.environ.get("FLA_CACHE_RESULTS", "1"),
            "python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
            "numpy": np.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }

    def attach_hooks(self):
        """Read-only wrapper on the instance's handle_kv_cache (modeling_tau_vla.py:560-582).

        predict_velocity calls qwenvl_with_expert.forward(...) directly (modeling_tau_vla.py:1553),
        which bypasses nn.Module hooks, so the tap sits where the expert reads the cache: calls with
        fill_kv_cache=False, one per full_attention layer, the first being denoising step 0.
        """
        original = type(self.joint).handle_kv_cache.__get__(self.joint)

        def tap(key_states, value_states, layer_idx, past_key_values=None, use_cache=None, fill_kv_cache=None):
            if self.capturing and use_cache and not fill_kv_cache and layer_idx not in self.capture.setdefault("layers", {}):
                cached_k, cached_v = past_key_values.get(layer_idx)   # [B, L_prefix, kv_heads, head_dim]
                self.capture["layers"][layer_idx] = [t.detach().float().mean(dim=(0, 1)).reshape(-1)
                                                     for t in (cached_k, cached_v)]
                self.capture["shape"] = list(cached_k.shape)
                if sorted(self.capture["layers"]) == self.full_layers and "kv" not in self.capture:
                    parts = [t for layer in self.full_layers for t in self.capture["layers"][layer]]
                    self.capture["kv"] = self.torch.cat(parts).cpu().numpy().astype(np.float32)
                    self.capture["shapes"] = {"key": self.capture["shape"], "layers": self.full_layers}
            return original(key_states, value_states, layer_idx, past_key_values=past_key_values,
                            use_cache=use_cache, fill_kv_cache=fill_kv_cache)
        self.joint.handle_kv_cache = tap
        self.handles.append(True)

    def remove_hooks(self):
        self.joint.__dict__.pop("handle_kv_cache", None)   # back to the unmodified class method
        self.handles.clear()

    def element(self, images, state, task):
        """The official evaluator's request dict (deploy/libero/main.py:251-258) for the same render."""
        agent, wrist = (self.prepare_image(image[::-1, ::-1], RESIZE) for image in images)
        return {"observation/image": agent, "observation/wrist_image": wrist,
                "observation/state": np.asarray(state, dtype=np.float32), "prompt": str(task)}

    def infer_once(self, element):
        actions = self.policy.infer(self.adapt(element))["actions"]   # libero_server.act_libero (no_grad inside)
        chunk = self.validate_chunk(actions, horizon=self.horizon)[: self.execute_steps]
        return np.ascontiguousarray(chunk, dtype=np.float32)

    def predict(self, req, audit_dir):
        torch = self.torch
        seed, draws, task, state, images = decode_request(req, self.default_j)
        element = self.element(images, state, task)
        self.capture.clear()
        torch.cuda.synchronize()
        started = time.perf_counter()
        samples = []
        with seed_scope(seed, torch):
            try:
                for index in range(draws):
                    self.capturing = index == 0
                    samples.append(self.infer_once(element))
                    if index == 0:
                        torch.cuda.synchronize()
                        first_seconds = time.perf_counter() - started
            finally:
                self.capturing = False
        torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - started
        if "kv" not in self.capture:
            raise RuntimeError("expert KV hook was not invoked")
        first, audit = samples[0], None
        if req.get("audit", False):
            self.remove_hooks()
            try:
                with seed_scope(seed, torch):
                    unhooked = self.infer_once(element)
            finally:
                self.attach_hooks()
            audit = {"exact_equal": bool(np.array_equal(first, unhooked)),
                     "hooked_unhooked_equal": bool(np.array_equal(first, unhooked)),
                     "max_abs_diff": float(np.max(np.abs(first - unhooked))), "seed": seed,
                     "scope": "same input/seed, executed chunk, cache tap attached versus removed (class method restored)",
                     "handoff_dimensions": int(self.capture["kv"].size), "kv_shapes": self.capture["shapes"],
                     "code_sha256": self.metadata["code_sha256"], "checkpoint_sha256": self.metadata["checkpoint_sha256"],
                     "request_sha256": hashlib.sha256(json.dumps(req, sort_keys=True).encode()).hexdigest()}
            (audit_dir / f"{audit['request_sha256']}.json").write_text(json.dumps(audit, indent=2))
            if not audit["exact_equal"]:
                raise RuntimeError(f"hook parity failed: {audit}")
        stacked = np.stack(samples)
        return {"ok": True, "actions": first.tolist(), "action_samples": stacked.tolist(),
                "action_std": float(stacked.std(axis=0, ddof=0).mean()),
                "kv_pooled": self.capture["kv"].tolist(), "kv_shapes": self.capture["shapes"],
                "seed": seed, "J": draws, "audit": audit,
                "timing": {"first_sample_seconds": first_seconds, "inference_seconds": inference_seconds},
                "elapsed_seconds": inference_seconds,
                "checkpoint_sha256": self.metadata["checkpoint_sha256"], "code_sha256": self.metadata["code_sha256"]}


def main():
    here = Path(__file__).resolve().parents[1]
    tau0_root = Path(os.environ.get("TAU0_ROOT", here / "ext/tau-0-vla")).resolve(strict=True)
    checkpoint = Path(os.environ.get("TAU0_CHECKPOINT", here / "ext/ckpt/tau-0-vla-libero")).resolve(strict=True)
    port = check_port(int(os.environ.get("POLICY_SERVER_PORT", "5920")))
    default_j = int(os.environ.get("POLICY_SERVER_J", "8"))
    if not 1 <= default_j <= 64:
        raise ValueError("POLICY_SERVER_J must be 1..64")
    execute_steps = int(os.environ.get("POLICY_EXECUTE_STEPS", "8"))   # deploy/libero/main.py:76 replan_steps
    audit_dir = Path(os.environ.get("AUDIT_DIR", here / "logs/tau0_audits")).resolve()
    audit_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = os.environ.get("POLICY_METADATA_PATH")
    metadata_path = Path(metadata_path).resolve() if metadata_path else None
    bridge = Bridge(tau0_root, checkpoint, execute_steps, default_j)
    if metadata_path:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(bridge.metadata, indent=2) + "\n")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", port))
        server.listen(8)
        print(f"[tau0] READY 127.0.0.1:{port} {json.dumps(bridge.metadata)}", flush=True)
        while True:
            conn, _ = server.accept()
            with conn:
                while True:
                    try:
                        req = recv_msg(conn)
                        if req is None:
                            break
                        if req.get("cmd") == "ping":
                            response = bridge.metadata
                        elif "cmd" in req:
                            raise ValueError("only ping is supported as a command")
                        else:
                            response = bridge.predict(req, audit_dir)
                        send_msg(conn, response)
                    except (EOFError, ConnectionError):
                        break
                    except Exception as error:
                        traceback.print_exc()
                        try:
                            send_msg(conn, {"ok": False, "error": str(error), "error_type": type(error).__name__,
                                            "message": str(error)})
                        except OSError:
                            break


if __name__ == "__main__":
    main()
