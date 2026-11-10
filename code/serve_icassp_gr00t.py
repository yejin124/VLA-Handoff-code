#!/usr/bin/env python3
"""Deterministic GR00T-N1.7-LIBERO measurement bridge over the official Gr00tPolicy.

Wire protocol: uint32 big-endian byte length followed by JSON, identical to the
MolmoAct2 and pi0.5 bridges. Inference requires images=[base64-npy agent RGB,
base64-npy wrist RGB], state[8], task, seed. Images must already carry the
180-degree LIBERO rotation (obs[::-1, ::-1]) that Isaac-GR00T's own libero_env
applies; the server applies no further orientation change. J may be 1..64.
Actions are the first POLICY_EXECUTE_STEPS (default 8, the official LIBERO
n_action_steps) of the decoded 16-step chunk, with the official gripper
normalize/binarize/invert applied, so they are directly steppable in LIBERO.
The handoff feature is the DiT's incoming encoder_hidden_states (post-vlln,
post-VL self-attention backbone tokens), mean-pooled to the 2048-d feature axis.
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
FORBIDDEN_PORTS = range(5650, 5657)
STATE_KEYS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
VIDEO_KEYS = ["image", "wrist_image"]
# Fork-only switches in this Isaac-GR00T checkout; any of them changes the forward pass.
FORK_ENV_SWITCHES = ["GR00T_BOUNDARY_MONITOR", "GR00T_DEGRADE_FILM", "GR00T_DIT_SDPA_MODE",
                     "GR00T_MONITOR_GATE_ALPHA", "GR00T_MONITOR_LOGSCALE", "GR00T_MONITOR_MODE",
                     "GR00T_OCCLUDER_PATH", "GR00T_TOKEN_ACTION_RELEVANCE",
                     "GR00T_TOKEN_BOUNDARY_MONITOR", "GR00T_TOKEN_CONDITIONED_FILM",
                     "GR00T_TOKEN_LOWER_ALPHA", "GR00T_TOKEN_MONITOR_MODE", "GR00T_TOKEN_SMOOTH_PASSES"]
SOURCE_FILES = ["gr00t/model/gr00t_n1d7/gr00t_n1d7.py", "gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py",
                "gr00t/model/gr00t_n1d7/image_augmentations.py", "gr00t/model/modules/dit.py",
                "gr00t/model/modules/qwen3_backbone.py", "gr00t/policy/gr00t_policy.py",
                "gr00t/policy/policy.py", "gr00t/data/embodiment_tags.py",
                "gr00t/eval/sim/LIBERO/libero_env.py"]
HANDOFF = {
    "site": "forward pre-hook on action_head.model (AlternateVLDiT); kwarg encoder_hidden_states",
    "meaning": "backbone_features after action_head.vlln LayerNorm and the 4-layer VL self-attention; "
               "the exact cross-attention context the DiT consumes",
    "positional_encoding": "RoPE applied inside the Qwen3-VL backbone (layer select_layer); no "
                           "positional embedding is added to these tokens in the action head "
                           "(add_pos_embed applies to action tokens only)",
    "capture": "first DiT call (denoising step 0) of the first action sample only; hook returns None",
    "pooling": "mean over batch and token axes -> [backbone_embedding_dim] float32; padding tokens "
               "are included (batch is 1, so none are expected; kv_shapes reports mask sum)",
    "dtype": "float32",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_manifest(root):
    root = Path(root).resolve(strict=True)
    entries = [{"path": str(p.relative_to(root)), "size": p.stat().st_size, "sha256": sha256_file(p)}
               for p in sorted(root.rglob("*")) if p.is_file() and p.suffix in {".safetensors", ".json", ".yaml"}]
    if not any(e["path"].endswith(".safetensors") for e in entries):
        raise ValueError("checkpoint has no safetensors weights")
    digest = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
    return {"path": str(root), "sha256": digest, "files": entries}


def git_state(root):
    def run(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL).strip()
    state = {"root": str(root), "commit": run("rev-parse", "HEAD"),
             "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}
    try:
        state["commits_ahead_of_origin_main"] = int(run("rev-list", "--count", "origin/main..HEAD"))
    except subprocess.CalledProcessError:
        state["commits_ahead_of_origin_main"] = None
    state["files_sha256"] = {p: sha256_file(Path(root) / p) for p in SOURCE_FILES}
    return state


def check_port(port):
    if port in FORBIDDEN_PORTS:
        raise ValueError(f"port {port} is in the range the other policy servers used during collection")
    if not 1024 <= port <= 65535:
        raise ValueError("port out of range")
    return port


def check_fork_switches(environ=os.environ):
    active = [name for name in FORK_ENV_SWITCHES if environ.get(name)]
    if active:
        raise RuntimeError(f"fork-only GR00T switches must be unset for plain inference: {active}")


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
    if "J" in req and "j" in req and req["J"] != req["j"]:
        raise ValueError("J and j disagree")
    draws = req.get("J", req.get("j", default_j))
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
        images.append(np.ascontiguousarray(image))
    return seed, draws, task, state, images


def build_observation(images, state, task, language_key):
    """Nested Gr00tPolicy observation, batch 1, one frame; mirrors libero_env._process_observation."""
    state = np.asarray(state, dtype=np.float32)
    slices = {"x": state[0:1], "y": state[1:2], "z": state[2:3], "roll": state[3:4],
              "pitch": state[4:5], "yaw": state[5:6], "gripper": state[6:8]}
    return {"video": {key: image[None, None] for key, image in zip(VIDEO_KEYS, images)},
            "state": {key: slices[key][None, None] for key in STATE_KEYS},
            "language": {language_key: [[task]]}}


def chunk_from_action(action):
    """Gr00tPolicy action dict -> [T, 7] in modality-key order (LIBERO action order)."""
    parts = [np.asarray(action[key], dtype=np.float32) for key in STATE_KEYS]
    if any(p.ndim != 3 or p.shape[0] != 1 for p in parts):
        raise RuntimeError(f"unexpected action shapes {[p.shape for p in parts]}")
    return np.concatenate([p[0] for p in parts], axis=-1)


def to_libero_action(chunk):
    """Official libero_env.step gripper post-processing: [0,1] -> [-1,1], binarize, invert."""
    out = np.array(chunk, dtype=np.float32, copy=True)
    out[..., -1] = -np.sign(2.0 * out[..., -1] - 1.0)
    return out


def pool_feature(tensor):
    """Mean over every axis except the last; float32 numpy."""
    if tensor.ndim < 2:
        raise RuntimeError("feature tensor needs a token axis and a feature axis")
    return tensor.detach().float().reshape(-1, tensor.shape[-1]).mean(0).cpu().numpy().astype(np.float32)


@contextlib.contextmanager
def seed_scope(seed, torch, device=None):
    """Fork Torch/Python/NumPy RNG state, seed everything once, restore on exit."""
    devices = [device] if device is not None else []
    np_state, py_state = np.random.get_state(), random.getstate()
    try:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            if devices:
                torch.cuda.manual_seed_all(seed)
            random.seed(seed)
            np.random.seed(seed % 2**32)
            yield
    finally:
        np.random.set_state(np_state)
        random.setstate(py_state)


def draw_samples(infer, draws, capture_first):
    """Sequential seeded draws inside an already-seeded scope; capture toggled for draw 0 only."""
    samples = []
    for index in range(draws):
        capture_first(index == 0)
        samples.append(infer())
    capture_first(False)
    return samples


class Bridge:
    def __init__(self, model_path, embodiment, execute_steps, default_j, gr00t_root):
        import torch
        import transformers
        import gr00t
        from gr00t.policy.gr00t_policy import Gr00tPolicy

        self.torch = torch
        self.default_j = default_j
        self.capture, self.capturing, self.handles = {}, False, []
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if not Path(gr00t.__file__).resolve().is_relative_to(Path(gr00t_root).resolve()):
            raise RuntimeError(f"gr00t imported from {gr00t.__file__}, not from GR00T_ROOT={gr00t_root}")
        manifest = checkpoint_manifest(model_path)
        self.policy = Gr00tPolicy(embodiment_tag=embodiment, model_path=str(model_path), device="cuda", strict=True)
        cfg = self.policy.modality_configs
        if list(cfg["video"].modality_keys) != VIDEO_KEYS or list(cfg["video"].delta_indices) != [0]:
            raise RuntimeError(f"unexpected video config {cfg['video']}")
        if list(cfg["state"].modality_keys) != STATE_KEYS or list(cfg["action"].modality_keys) != STATE_KEYS:
            raise RuntimeError("unexpected state/action keys")
        self.language_key = self.policy.language_key
        decoded_horizon = len(cfg["action"].delta_indices)
        if not 1 <= execute_steps <= decoded_horizon:
            raise ValueError(f"POLICY_EXECUTE_STEPS must be in [1, {decoded_horizon}]")
        self.execute_steps = execute_steps
        head = self.policy.model.action_head
        for attr in ("degrade_film", "boundary_monitor", "token_boundary_monitor"):
            if getattr(head, attr, None) is not None:
                raise RuntimeError(f"fork module {attr} is active; plain inference required")
        if getattr(head, "token_action_relevance", False):
            raise RuntimeError("fork token_action_relevance is active")
        self.dit = head.model
        if type(self.dit).__name__ != "AlternateVLDiT":
            raise RuntimeError(f"unexpected action-head transformer {type(self.dit).__name__}")
        self.attach_hooks()
        model_cfg = self.policy.model.config
        self.metadata = {
            "ok": True, "protocol": "vla-icassp-plain-v1",
            "model": "GR00T-N1.7-LIBERO/" + Path(model_path).name,
            "checkpoint": manifest, "checkpoint_id": manifest["sha256"], "checkpoint_sha256": manifest["sha256"],
            "code_sha256": sha256_file(__file__), "gr00t_source": git_state(gr00t_root),
            "embodiment_tag": str(self.policy.embodiment_tag.name), "embodiment_value": self.policy.embodiment_tag.value,
            "J": default_j, "max_J": 64, "execution_sample": 0, "spread_ddof": 0, "seed_required": True,
            "action_horizon": execute_steps, "decoded_action_horizon": decoded_horizon,
            "model_action_horizon": int(model_cfg.action_horizon),
            "num_inference_timesteps": int(model_cfg.num_inference_timesteps),
            "action_dimensions": 7, "state_dimensions": 8,
            "action_units": "LIBERO OSC_POSE normalized deltas in [-1,1] for dims 0-5 (dataset q01/q99 "
                            "unnormalization); gripper binarized to -1 (open) / +1 (close) as in libero_env.step",
            "image_contract": "client sends obs[::-1, ::-1] (official LIBERO rotation); server applies no rotation; "
                              "processor eval transform: SmallestMaxSize 256, FractionalCenterCrop 0.95, SmallestMaxSize 256",
            "handoff": dict(HANDOFF, dimensions=int(model_cfg.backbone_embedding_dim)),
            "seed_contract": "fork_rng scope; torch/cuda/python/numpy seeded once; sequential J draws; sample 0 executed",
            "determinism_scope": "same software/device; cross-GPU bitwise parity not asserted",
            "python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
            "numpy": np.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }

    def attach_hooks(self):
        def pre_hook(_module, _args, kwargs):
            if self.capturing and "kv" not in self.capture:
                context = kwargs["encoder_hidden_states"]
                mask = kwargs.get("backbone_attention_mask")
                self.capture["kv"] = pool_feature(context)
                self.capture["shapes"] = {"encoder_hidden_states": list(context.shape),
                                          "mask_sum": None if mask is None else int(mask.sum().item())}
            return None
        self.handles.append(self.dit.register_forward_pre_hook(pre_hook, with_kwargs=True))

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def set_capture(self, enabled):
        self.capturing = enabled

    def infer_once(self, observation):
        with self.torch.inference_mode():
            action, _ = self.policy.get_action(observation)
        chunk = to_libero_action(chunk_from_action(action))[: self.execute_steps]
        if chunk.shape != (self.execute_steps, 7) or not np.isfinite(chunk).all():
            raise RuntimeError(f"invalid action output {chunk.shape}")
        return chunk

    def predict(self, req, audit_dir):
        torch = self.torch
        seed, draws, task, state, images = decode_request(req, self.default_j)
        observation = build_observation(images, state, task, self.language_key)
        finished = []

        def infer():
            chunk = self.infer_once(observation)
            torch.cuda.synchronize()
            finished.append(time.perf_counter())
            return chunk

        self.capture.clear()
        device = torch.cuda.current_device()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with seed_scope(seed, torch, device):
            samples = draw_samples(infer, draws, self.set_capture)
        if "kv" not in self.capture:
            raise RuntimeError("DiT context hook was not invoked")
        first, first_seconds, inference_seconds = samples[0], finished[0] - started, finished[-1] - started
        audit = None
        if req.get("audit", False):
            self.remove_hooks()
            try:
                audit_start = time.perf_counter()
                with seed_scope(seed, torch, device):
                    unhooked = infer()
                    torch.cuda.synchronize()
            finally:
                self.attach_hooks()
            audit = {"exact_equal": bool(np.array_equal(first, unhooked)),
                     "hooked_unhooked_equal": bool(np.array_equal(first, unhooked)),
                     "max_abs_diff": float(np.max(np.abs(first - unhooked))), "seed": seed,
                     "seconds": time.perf_counter() - audit_start,
                     "scope": "same input/seed, executed chunk, DiT pre-hook attached versus physically removed",
                     "handoff_dimensions": int(self.capture["kv"].size),
                     "code_sha256": self.metadata["code_sha256"], "checkpoint_sha256": self.metadata["checkpoint_sha256"],
                     "request_sha256": hashlib.sha256(json.dumps(req, sort_keys=True).encode()).hexdigest()}
            (audit_dir / f"{audit['request_sha256']}.json").write_text(json.dumps(audit, indent=2))
            if not audit["exact_equal"]:
                raise RuntimeError(f"hook parity failed: {audit}")
        stacked = np.stack(samples)
        return {"ok": True, "actions": first.tolist(), "action_samples": stacked.tolist(),
                "action_std": float(stacked.std(axis=0, ddof=0).mean()),
                "kv_pooled": self.capture["kv"].tolist(), "kv_shapes": self.capture["shapes"],
                "handoff": self.metadata["handoff"], "seed": seed, "J": draws, "audit": audit,
                "timing": {"first_sample_seconds": first_seconds, "inference_seconds": inference_seconds},
                "elapsed_seconds": inference_seconds,
                "checkpoint_sha256": self.metadata["checkpoint_sha256"], "code_sha256": self.metadata["code_sha256"]}


def main():
    check_fork_switches()
    model_path = os.environ.get("GR00T_MODEL_PATH")
    if not model_path:
        raise ValueError("GR00T_MODEL_PATH must name one suite checkpoint directory (e.g. .../GR00T-N1.7-LIBERO/libero_object)")
    gr00t_root = Path(os.environ["GR00T_ROOT"]).resolve(strict=True)  # Isaac-GR00T checkout
    sys.path.insert(0, str(gr00t_root))
    port = check_port(int(os.environ.get("POLICY_SERVER_PORT", "5662")))
    default_j = int(os.environ.get("POLICY_SERVER_J", "8"))
    if not 1 <= default_j <= 64:
        raise ValueError("POLICY_SERVER_J must be 1..64")
    execute_steps = int(os.environ.get("POLICY_EXECUTE_STEPS", "8"))
    audit_dir = Path(os.environ.get("AUDIT_DIR", "logs/gr00t_audits")).resolve()
    audit_dir.mkdir(parents=True, exist_ok=True)
    bridge = Bridge(Path(model_path).resolve(strict=True), os.environ.get("GR00T_EMBODIMENT", "LIBERO_PANDA"),
                    execute_steps, default_j, gr00t_root)
    metadata_path = os.environ.get("POLICY_METADATA_PATH")
    if metadata_path:
        Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
        Path(metadata_path).write_text(json.dumps(bridge.metadata, indent=2) + "\n")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", port))
        server.listen(8)
        print(f"[gr00t] READY 127.0.0.1:{port} {json.dumps(bridge.metadata)}", flush=True)
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
