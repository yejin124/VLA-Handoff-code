#!/usr/bin/env python3
"""GR00T-N1.7-SimplerEnv-Bridge measurement bridge, same wire protocol as serve_icassp_gr00t.

Request: images=[base64-npy RGB image_0], state[8] in WidowXBridgeEnv order
(x, y, z, roll, pitch, yaw, pad, gripper), task, seed, J. Response actions are the
first POLICY_EXECUTE_STEPS (default 4, the official Bridge n_action_steps) of the
decoded chunk in the policy's raw units; WidowXBridgeEnv.step applies the official
gripper binarization. The handoff is captured exactly as in the LIBERO bridge.
"""
import json
import os
from pathlib import Path
import platform
import socket
import sys
import time
import traceback

import numpy as np

from serve_icassp_gr00t import (HANDOFF, check_fork_switches, check_port, checkpoint_manifest, decode_request as _decode,
                                draw_samples, pool_feature, recv_msg, seed_scope, send_msg, sha256_file)

STATE_KEYS = ["x", "y", "z", "roll", "pitch", "yaw", "pad", "gripper"]
ACTION_KEYS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


def decode_request(req, default_j):
    # The LIBERO decoder demands two cameras; Bridge has one, so duplicate it for the checks and keep the first.
    images = req.get("images")
    if not isinstance(images, list) or len(images) != 1:
        raise ValueError("images must contain one base64-npy array (image_0)")
    seed, draws, task, state, images = _decode(dict(req, images=images * 2), default_j)
    return seed, draws, task, state, images[0]


class Bridge:
    def __init__(self, model_path, execute_steps, default_j):
        import torch
        import transformers
        from gr00t.policy.gr00t_policy import Gr00tPolicy

        self.torch, self.default_j = torch, default_j
        self.capture, self.capturing = {}, False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        manifest = checkpoint_manifest(model_path)
        self.policy = Gr00tPolicy(embodiment_tag="simpler_env_widowx", model_path=str(model_path), device="cuda", strict=True)
        cfg = self.policy.modality_configs
        if (list(cfg["video"].modality_keys) != ["image_0"] or list(cfg["state"].modality_keys) != STATE_KEYS
                or list(cfg["action"].modality_keys) != ACTION_KEYS):
            raise RuntimeError(f"unexpected modality config {cfg}")
        self.language_key = self.policy.language_key
        decoded = len(cfg["action"].delta_indices)
        if not 1 <= execute_steps <= decoded:
            raise ValueError(f"POLICY_EXECUTE_STEPS must be in [1, {decoded}]")
        self.execute_steps = execute_steps
        self.dit = self.policy.model.action_head.model
        if type(self.dit).__name__ != "AlternateVLDiT":
            raise RuntimeError(f"unexpected action-head transformer {type(self.dit).__name__}")

        def pre_hook(_module, _args, kwargs):
            if self.capturing and "kv" not in self.capture:
                self.capture["kv"] = pool_feature(kwargs["encoder_hidden_states"])
            return None
        self.dit.register_forward_pre_hook(pre_hook, with_kwargs=True)
        model_cfg = self.policy.model.config
        self.metadata = {
            "ok": True, "protocol": "vla-icassp-plain-v1", "model": "GR00T-N1.7-SimplerEnv-Bridge",
            "checkpoint_id": manifest["sha256"], "checkpoint_sha256": manifest["sha256"], "checkpoint_path": str(model_path),
            "code_sha256": sha256_file(__file__), "embodiment_tag": "simpler_env_widowx",
            "J": default_j, "execution_sample": 0, "spread_ddof": 0, "seed_required": True,
            "action_horizon": execute_steps, "decoded_action_horizon": decoded,
            "action_units": "raw policy output; gripper in [0,1], binarized by WidowXBridgeEnv.step",
            "handoff": dict(HANDOFF, dimensions=int(model_cfg.backbone_embedding_dim)),
            "python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
            "device": torch.cuda.get_device_name(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}

    def infer_once(self, observation):
        with self.torch.inference_mode():
            action, _ = self.policy.get_action(observation)
        chunk = np.concatenate([np.asarray(action[k], dtype=np.float32)[0] for k in ACTION_KEYS], axis=-1)
        chunk = chunk[: self.execute_steps]
        if chunk.shape != (self.execute_steps, 7) or not np.isfinite(chunk).all():
            raise RuntimeError(f"invalid action output {chunk.shape}")
        return chunk

    def predict(self, req):
        torch = self.torch
        seed, draws, task, state, image = decode_request(req, self.default_j)
        observation = {"video": {"image_0": image[None, None]},
                       "state": {k: state[i:i + 1][None, None] for i, k in enumerate(STATE_KEYS)},
                       "language": {self.language_key: [[task]]}}
        self.capture.clear()
        device = torch.cuda.current_device()
        started = time.perf_counter()
        with seed_scope(seed, torch, device):
            samples = draw_samples(lambda: self.infer_once(observation), draws, lambda on: setattr(self, "capturing", on))
        torch.cuda.synchronize()
        if "kv" not in self.capture:
            raise RuntimeError("DiT context hook was not invoked")
        stacked = np.stack(samples)
        return {"ok": True, "actions": samples[0].tolist(), "action_samples": stacked.tolist(),
                "action_std": float(stacked.std(axis=0, ddof=0).mean()), "kv_pooled": self.capture["kv"].tolist(),
                "seed": seed, "J": draws, "timing": {"inference_seconds": time.perf_counter() - started},
                "checkpoint_sha256": self.metadata["checkpoint_sha256"], "code_sha256": self.metadata["code_sha256"]}


def main():
    check_fork_switches()
    gr00t_root = Path(os.environ["GR00T_ROOT"]).resolve(strict=True)  # Isaac-GR00T checkout
    sys.path.insert(0, str(gr00t_root))
    port = check_port(int(os.environ.get("POLICY_SERVER_PORT", "5612")))
    bridge = Bridge(Path(os.environ["GR00T_MODEL_PATH"]).resolve(strict=True),
                    int(os.environ.get("POLICY_EXECUTE_STEPS", "4")), int(os.environ.get("POLICY_SERVER_J", "8")))
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", port))
        server.listen(8)
        print(f"[gr00t-simpler] READY 127.0.0.1:{port} {json.dumps(bridge.metadata)}", flush=True)
        while True:
            conn, _ = server.accept()
            with conn:
                while True:
                    try:
                        req = recv_msg(conn)
                        if req is None:
                            break
                        send_msg(conn, bridge.metadata if req.get("cmd") == "ping" else bridge.predict(req))
                    except (EOFError, ConnectionError):
                        break
                    except Exception as error:
                        traceback.print_exc()
                        try:
                            send_msg(conn, {"ok": False, "error": str(error)})
                        except OSError:
                            break


if __name__ == "__main__":
    main()
