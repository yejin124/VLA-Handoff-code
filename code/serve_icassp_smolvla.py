#!/usr/bin/env python3
"""Deterministic SmolVLA LIBERO measurement bridge using the official LeRobot policy and processors.

Wire protocol (shared with serve_icassp_pi05.py, whose framing and request validation are imported):
uint32 big-endian byte length followed by JSON. Inference requires images=[base64-npy agent RGB,
base64-npy wrist RGB] already rotated 180 degrees (LeRobot LiberoProcessorStep convention, which the
collector applies), state[8] = eef pos, axis-angle (2*acos(w), no wrap), gripper qpos, task, seed.
J may be 1..64; sample 0 is executed. Each sample is the official 50-step chunk after the checkpoint's
unnormalizer; the first POLICY_EXECUTE_STEPS actions are returned.

Handoff: the prefix KV cache that the action expert reads at its first flow step, i.e. the post-RoPE
VLM keys and values of every layer over the prefix (2 images, instruction, state token), captured
on the first denoise_step call of sample 0. Each layer's K and V are averaged over valid
(non-padding) prefix tokens and flattened (kv-head major); layers in order, K then V.
"""
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from serve_icassp_pi05 import decode_request, recv_msg, send_msg, sha256  # noqa: E402

HANDOFF = ("SmolVLA prefix KV cache read by the action expert at the first flow step (post-RoPE VLM K/V, "
           "all VLM layers); per layer mean over valid prefix tokens (images, instruction, state); K then V")


def main():
    import torch
    import lerobot
    import transformers
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    checkpoint = Path(os.environ["SMOLVLA_CHECKPOINT"]).expanduser().resolve(strict=True)
    port = int(os.environ.get("POLICY_SERVER_PORT", "5910"))
    draws_default = int(os.environ.get("POLICY_SERVER_J", "8"))
    execute = int(os.environ.get("POLICY_EXECUTE_STEPS", "10"))
    if not 1 <= draws_default <= 64:
        raise ValueError("POLICY_SERVER_J must be 1..64")
    audit_dir = Path(os.environ.get("AUDIT_DIR", "logs/smolvla_audits")).resolve()
    audit_dir.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    print("[smolvla] loading official SmolVLA policy and processors", flush=True)
    policy = SmolVLAPolicy.from_pretrained(str(checkpoint))
    policy.to("cuda").eval()
    cfg = policy.config
    if cfg.compile_model or (cfg.rtc_config is not None and cfg.rtc_config.enabled):
        raise RuntimeError("compiled or RTC inference is not the measured configuration")
    if not 1 <= execute <= cfg.chunk_size:
        raise ValueError("POLICY_EXECUTE_STEPS must be 1..chunk_size")
    # The checkpoint's own processors (rename image->camera1, newline, tokenizer, MEAN_STD), as lerobot-eval
    # loads them; only the device is overridden there too.
    preprocess, postprocess = make_pre_post_processors(
        cfg, pretrained_path=str(checkpoint), preprocessor_overrides={"device_processor": {"device": "cuda"}})
    model = policy.model
    lerobot_root = Path(lerobot.__file__).parent
    sources = ["policies/smolvla/modeling_smolvla.py", "policies/smolvla/smolvlm_with_expert.py",
               "policies/smolvla/processor_smolvla.py", "policies/common/flow_matching.py",
               "processor/env_processor.py", "envs/libero.py", "envs/utils.py", "scripts/lerobot_eval.py"]
    files = sorted(p for p in checkpoint.iterdir() if p.is_file())
    manifest = {p.name: sha256(p) for p in files}
    metadata = {
        "ok": True, "schema_version": 1, "model": "SmolVLA-LIBERO", "J": draws_default,
        "checkpoint": str(checkpoint), "checkpoint_files_sha256": manifest,
        "checkpoint_id": "sha256:" + hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
        "code_sha256": sha256(__file__), "protocol_module_sha256": sha256(Path(__file__).with_name("serve_icassp_pi05.py")),
        "handoff_description": HANDOFF, "action_dimensions": 7, "state_dimensions": 8,
        "chunk_size": cfg.chunk_size, "execute_steps": execute, "num_flow_steps": cfg.num_steps,
        "vlm_model_name": cfg.vlm_model_name, "num_vlm_layers": model.vlm_with_expert.num_vlm_layers,
        "image_features": list(cfg.image_features), "resize_imgs_with_padding": list(cfg.resize_imgs_with_padding),
        "execution_sample": 0, "spread_ddof": 0, "seed_required": True,
        "lerobot_version": lerobot.__version__, "lerobot_source_sha256": {s: sha256(lerobot_root / s) for s in sources},
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "numpy_version": np.__version__, "device": torch.cuda.get_device_name(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "determinism_scope": "same software/device; cross-GPU bitwise parity not asserted",
    }
    capture = {}
    capturing = False
    original = model.denoise_step

    def hooked(prefix_pad_masks, past_key_values, x_t, timestep):
        if capturing and "kv" not in capture:
            valid = prefix_pad_masks[0].bool()
            parts = []
            for layer in past_key_values.layers:   # [B, kv_heads, L, head_dim]
                for tensor in (layer.keys, layer.values):
                    x = tensor[0].detach().float().permute(1, 0, 2)[valid]
                    parts.append(x.reshape(x.shape[0], -1).mean(0))
            capture["kv"] = torch.cat(parts).cpu().numpy()
            capture["tokens"] = [int(valid.sum()), int(valid.numel())]
        return original(prefix_pad_masks=prefix_pad_masks, past_key_values=past_key_values,
                        x_t=x_t, timestep=timestep)

    model.denoise_step = hooked

    def seed_rng(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def batch_for(task, state, images):
        obs = {"observation.state": torch.from_numpy(state)[None], "task": [task]}
        for key, image in zip(("observation.images.image", "observation.images.image2"), images):
            # envs/utils.py preprocess_observation: uint8 HWC -> float32 CHW in [0, 1]
            obs[key] = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)[None].float() / 255
        return preprocess(obs)

    def infer(batch):
        with torch.inference_mode():
            chunk = postprocess(policy.predict_action_chunk(batch))
        action = chunk[0].float().cpu().numpy()
        if action.shape != (cfg.chunk_size, 7) or not np.isfinite(action).all():
            raise RuntimeError(f"invalid official action output: {action.shape}")
        return action[:execute]

    def predict(req):
        nonlocal capturing
        seed, draws, task, state, images = decode_request(req, draws_default)
        batch = batch_for(task, state, images)
        capture.clear()
        started = time.monotonic()
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            seed_rng(seed)
            samples = []
            try:
                for index in range(draws):
                    capturing = index == 0
                    samples.append(infer(batch))
            finally:
                capturing = False
            if "kv" not in capture:
                raise RuntimeError("handoff hook was not invoked")
            handoff = capture["kv"]
            audit = None
            if req.get("audit", False):
                model.denoise_step = original      # parity compares ordinary official inference
                try:
                    seed_rng(seed)
                    unhooked = infer(batch)
                finally:
                    model.denoise_step = hooked
                audit = {"hooked_unhooked_equal": bool(np.array_equal(samples[0], unhooked)),
                         "max_absolute_action_difference": float(np.abs(samples[0] - unhooked).max()),
                         "seed": seed, "valid_action_shape": list(samples[0].shape),
                         "handoff_dimensions": int(handoff.size), "prefix_tokens_valid_total": capture["tokens"],
                         "handoff_description": HANDOFF, "code_sha256": metadata["code_sha256"],
                         "checkpoint_id": metadata["checkpoint_id"]}
                identity = hashlib.sha256(json.dumps(req, sort_keys=True).encode()).hexdigest()
                audit["request_sha256"] = identity
                (audit_dir / f"{identity}.json").write_text(json.dumps(audit, indent=2))
                if not audit["hooked_unhooked_equal"]:
                    raise RuntimeError(f"hook parity failed: {audit}")
        stacked = np.stack(samples)
        return {"ok": True, "actions": samples[0].tolist(), "action_samples": stacked.tolist(),
                "kv_pooled": handoff.tolist(), "prefix_tokens_valid_total": capture["tokens"],
                "action_std": float(stacked.std(axis=0, ddof=0).mean()),
                "seed": seed, "J": draws, "audit": audit, "elapsed_seconds": time.monotonic() - started,
                "code_sha256": metadata["code_sha256"]}

    metadata_path = os.environ.get("POLICY_METADATA_PATH")
    if metadata_path:
        Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
        Path(metadata_path).write_text(json.dumps(metadata, indent=2) + "\n")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", port))
        server.listen(8)
        print(f"[smolvla] READY 127.0.0.1:{port} {json.dumps(metadata)}", flush=True)
        while True:
            conn, _ = server.accept()
            with conn:
                while True:
                    try:
                        req = recv_msg(conn)
                        if req is None:
                            break
                        send_msg(conn, metadata if req.get("cmd") == "ping" else predict(req))
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
