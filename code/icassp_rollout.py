"""Seeded LIBERO rollout collection with query-level evidence, no intervention.

All conditions of a source initial state stay together in a deterministic shard.
Incomplete episodes are quarantined on restart; completed episodes are immutable.
"""
import argparse
import base64
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import socket
import struct
import sys
import time

import numpy as np

DONORS = None
ROBOT_STATE_CONVENTION = 'libero_official_axis_angle_no_wrap'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # tests load this file by path
from icassp_conditions import ALL_CONDITIONS, MOVE_FAMILIES, RELOCATE_FAMILY, VIEW_FAMILY, DestinationMover, degrade, image_seeds, load_donor_bank, shift_viewpoint
from icassp_conditions import legacy_corrupt as corrupt  # noqa: F401  (legacy families; exact as before)


def stable_seed(*parts):
    raw = json.dumps(parts, separators=(",", ":"), ensure_ascii=True).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "big") % (2**31 - 1)


def sha(value):
    return hashlib.sha256(value).hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value)
    return sha(str((value.shape, str(value.dtype))).encode() + value.tobytes())


def encode(value):
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(value), allow_pickle=False)
    return base64.b64encode(buffer.getvalue()).decode()


def exact_read(conn, n):
    chunks = bytearray()
    while len(chunks) < n:
        block = conn.recv(n - len(chunks))
        if not block:
            raise ConnectionError("Policy server disconnected")
        chunks.extend(block)
    return bytes(chunks)


def query(conn, request):
    data = json.dumps(request, allow_nan=False).encode()
    conn.sendall(struct.pack(">I", len(data)) + data)
    n = struct.unpack(">I", exact_read(conn, 4))[0]
    if n > 64 * 1024 * 1024:
        raise ValueError("Unexpectedly large response")
    response = json.loads(exact_read(conn, n))
    if "error" in response:
        raise RuntimeError(response["error"])
    return response




def robot_state(obs):
    """Use the released LIBERO interfaces' axis-angle branch, without wrapping.

    q and -q denote the same orientation, but their numerical axis-angle inputs
    are different for a learned state encoder. Do not replace the official
    2*acos(w) convention with the shortest-angle representation.
    """
    q = np.asarray(obs["robot0_eef_quat"], dtype=np.float64).copy()
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) == 0:
        raise ValueError('Expected a finite nonzero xyzw quaternion')
    w = np.clip(q[3], -1, 1)
    scale = np.sqrt(1 - w * w)
    rotation = np.zeros(3) if math.isclose(scale, 0.) else q[:3] * 2 * math.acos(w) / scale
    return np.concatenate((obs["robot0_eef_pos"], rotation,
                           obs["robot0_gripper_qpos"])).astype(np.float32)


def validate_response(response, draws):
    actions = np.asarray(response["actions"], dtype=np.float32)
    samples = np.asarray(response["action_samples"], dtype=np.float32)
    features = np.asarray(response["kv_pooled"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 7 or not len(actions):
        raise ValueError(f"Invalid executed action shape {actions.shape}")
    if samples.shape != (draws, *actions.shape):
        raise ValueError(f"Invalid samples {samples.shape}, requested {draws}")
    if features.ndim != 1 or len(features) < 2:
        raise ValueError("Missing handoff feature")
    if not all(np.isfinite(v).all() for v in (actions, samples, features)):
        raise ValueError("Nonfinite model output")
    if not np.array_equal(actions, samples[0]):
        raise ValueError("Executed action is not draw zero")
    spread = float(samples.std(axis=0).mean())
    if not np.isclose(spread, response["action_std"], rtol=1e-4, atol=1e-7):
        raise ValueError("Server/client spread mismatch")
    return actions, samples, features, spread


def json_write(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def trial_grid(tasks, episodes, replicates, conditions, shard, nshards):
    for k, (task, episode) in enumerate((t, e) for t in tasks for e in episodes):
        if k % nshards != shard:
            continue
        for rep in range(replicates):
            for condition in conditions:
                yield task, episode, rep, condition


def severity_for(cfg, condition):
    """Per-condition severity from --severities (e.g. occluder=1.0,moved=0.15), else --severity."""
    return float(cfg.get("severities", {}).get(condition, cfg["severity"]))


def collect(env, conn, language, initial, trial, cfg, directory, audit):
    task, episode, rep, condition = trial
    pair = (cfg["seed"], cfg["suite"], task, episode, rep)
    env_seed = stable_seed(*pair, "environment")
    env.seed(env_seed)
    env.reset()
    env.set_init_state(initial)
    view = (shift_viewpoint(env, cfg["suite"], stable_seed(*pair, "viewpoint"), severity_for(cfg, condition))
            if condition == VIEW_FAMILY else None)
    for _ in range(10):
        obs, _, _, _ = env.step([0.] * 6 + [-1.])
    steps, qi, done = 0, 0, False
    mover = (DestinationMover(env, cfg["suite"], stable_seed(*pair, "move"), severity_for(cfg, condition), cfg.get("move_step", 30), cfg.get("move_duration", 10),
                               relocate=(condition == RELOCATE_FAMILY))
             if condition in MOVE_FAMILIES else None)
    started = time.monotonic()
    trace = (directory / "queries.jsonl").open("w")
    events = (directory / "steps.jsonl").open("w")
    try:
        while not done and steps < cfg["max_steps"]:
            if shutil.disk_usage(directory).free < 4 * 1024**3:
                raise RuntimeError("Disk reserve reached: stop before compromising artifacts")
            active = condition if steps >= cfg["onset"] else "clean"
            seeds = image_seeds(pair, active, steps)
            images = degrade([obs[key][::-1, ::-1] for key in ("agentview_image", "robot0_eye_in_hand_image")],
                             active, severity_for(cfg, active), seeds, donors=DONORS, apply_to_wrist=cfg.get("apply_to_wrist", False))
            state = robot_state(obs)
            seed = stable_seed(*pair, "policy", qi)
            req = {"images": [encode(v) for v in images], "state": state.tolist(),
                   "task": language, "seed": seed, "J": cfg["draws"]}
            if audit and qi == 0:
                check = query(conn, dict(req, J=2, audit=True))
                check_actions, check_samples, _, _ = validate_response(check, 2)
                parity = check.get("audit", {})
                if not (parity.get("exact_equal") or parity.get("hooked_unhooked_equal")):
                    raise ValueError(f"Hook parity was not verified: {parity}")
                again = query(conn, dict(req, J=1))
                one, _, _, _ = validate_response(again, 1)
                if not np.array_equal(check_actions, one):
                    raise ValueError("Same seed draw zero changes with J / repeated request")
                json_write(directory / "adapter_audit.json", {
                    "same_seed_j1_j2_first_action_exact": True,
                    "server_audit": check.get("audit"), "request_seed": seed,
                    "input_image_sha256": [array_hash(v) for v in images],
                    "state": state.tolist(), "task": language,
                })
            t0 = time.monotonic()
            response = query(conn, req)
            latency = time.monotonic() - t0
            actions, samples, feature, spread = validate_response(response, cfg["draws"])
            artifact = f"q{qi:04d}.npz"
            # Record exactly transmitted images around fixed landmarks, hash every image.
            keep_images = qi == 0 or any(steps <= mark < steps + len(actions)
                                         for mark in (40, 80, 120))
            extra = {"agent_image": images[0], "wrist_image": images[1]} if keep_images else {}
            np.savez_compressed(directory / artifact, handoff=feature.astype(np.float16),
                                action_samples=samples, executed_actions=actions, state=state, **extra)
            line = {"query": qi, "step": steps, "policy_seed": seed,
                    "corruption_seeds": seeds, "active_condition": active,
                    "image_sha256": [array_hash(v) for v in images], "artifact": artifact,
                    "artifact_sha256": sha((directory / artifact).read_bytes()),
                    "spread": spread, "latency_seconds": latency, "draws": cfg["draws"],
                    "image_stats": [float(np.mean(v)) for v in images] +
                                   [float(np.std(v)) for v in images],
                    "state": state.tolist(), "action_l2": float(np.linalg.norm(actions)),
                    "server_timing": response.get("timing")}
            trace.write(json.dumps(line, allow_nan=False) + "\n")
            trace.flush()
            for action in actions:
                moving = mover.before_step(steps) if mover else False
                obs, reward, done, _ = env.step(action.tolist())
                steps += 1
                events.write(json.dumps({"step": steps, "query": qi, "success": bool(env.check_success()), "moving": moving,
                                         "state": robot_state(obs).tolist()}) + "\n")
                if done or steps >= cfg["max_steps"]:
                    break
            events.flush()
            qi += 1
        summary = {"task": task, "episode": episode, "replicate": rep, "condition": condition,
                   "language": language, "initial_state_sha256": array_hash(np.asarray(initial)),
                   "environment_seed": env_seed, "success": bool(env.check_success()),
                   "steps": steps, "queries": qi, "wall_seconds": time.monotonic() - started,
                   "terminal_reason": "success" if env.check_success() else ("terminated" if done else "timeout"),
                   "config_sha256": cfg["config_sha256"], "complete": True,
                   "intervention": mover.final() if mover else view}
    finally:
        trace.close()
        events.close()
    json_write(directory / "summary.json", summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--tasks", default="0,1")
    p.add_argument("--episodes", default="0,1,2")
    p.add_argument("--replicates", type=int, default=2)
    p.add_argument("--conditions", default="clean,noise,occlusion,combined")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--draws", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=280)
    p.add_argument("--onset", type=int, default=40)
    p.add_argument("--severity", type=float, default=.5, help="level in [0,1]; for `moved` the displacement in metres")
    p.add_argument("--severities", default="", help="per-condition overrides, e.g. occluder=1.0,lighting=1.5,moved=0.15")
    p.add_argument("--move-step", type=int, default=30, help="simulator step at which the destination starts to move")
    p.add_argument("--move-duration", type=int, default=10, help="steps over which the destination slides")
    p.add_argument("--apply-to-wrist", action="store_true", help="also degrade the wrist camera for occluder (pilot fallback only)")
    p.add_argument("--seed", type=int, default=20260916)
    p.add_argument("--suite", default="libero_spatial")
    a = p.parse_args()
    tasks = [int(v) for v in a.tasks.split(",")]
    episodes = [int(v) for v in a.episodes.split(",")]
    conditions = a.conditions.split(",")
    if len(set(tasks)) != len(tasks) or len(set(episodes)) != len(episodes) or min(tasks + episodes) < 0:
        p.error("Task/episode IDs must be unique and nonnegative")
    if not 0 <= a.shard < a.nshards or not 1 <= a.draws <= 64 or a.replicates < 1:
        p.error("Invalid shard/draw/replicate configuration")
    if len(set(conditions)) != len(conditions) or not set(conditions) <= set(ALL_CONDITIONS):
        p.error("Unknown or duplicate conditions")
    if a.max_steps < 1 or not 0 <= a.onset < a.max_steps or not 0 <= a.severity <= 1.5:
        p.error("Invalid horizon, onset or severity")
    try:
        a.severities = {k: float(v) for k, v in (item.split("=") for item in a.severities.split(",") if item)}
    except ValueError:
        p.error("--severities must look like family=value,family=value")
    if not set(a.severities) <= set(conditions) or not all(0 <= v <= 1.5 for v in a.severities.values()):
        p.error("--severities names an unknown condition or an out-of-range value")
    global DONORS
    DONORS = load_donor_bank() if {"occluder"} & set(conditions) else None
    root = a.out.expanduser().resolve() / f"shard{a.shard:02d}"
    root.mkdir(parents=True, exist_ok=True)
    conn = socket.create_connection(("127.0.0.1", a.port), timeout=300)
    ping = query(conn, {"cmd": "ping"})
    if not ping.get("ok") or a.model.lower() not in ping.get("model", "").lower():
        raise ValueError(f"Wrong server: {ping}")
    cfg = vars(a).copy()
    cfg.pop("out")
    cfg["collector_sha256"] = sha(Path(__file__).read_bytes())
    cfg["server_identity"] = {k: ping.get(k) for k in ("model", "code_sha256", "checkpoint_id")}
    cfg["config_sha256"] = sha(json.dumps(cfg, sort_keys=True).encode())
    cfg_path = root / "config.json"
    if cfg_path.exists() and json.loads(cfg_path.read_text()) != cfg:
        raise ValueError("Resume configuration mismatch; use a new output directory")
    json_write(cfg_path, cfg)
    json_write(root / "server.json", ping)
    import torch
    original_load = torch.load
    torch.load = lambda *args, **kwargs: original_load(*args, **dict(kwargs, weights_only=False))
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    bench = benchmark.get_benchmark_dict()[a.suite]()
    env, current = None, None
    audited = (root / "audit_passed.json").exists()
    try:
        for trial in trial_grid(tasks, episodes, a.replicates, conditions, a.shard, a.nshards):
            task, episode, rep, condition = trial
            directory = root / f"t{task}_e{episode}_r{rep}_{condition}"
            if (directory / "summary.json").exists():
                previous = json.loads((directory / "summary.json").read_text())
                if previous.get("complete") and previous["config_sha256"] == cfg["config_sha256"]:
                    continue
                raise ValueError("Conflicting completed episode")
            if directory.exists():
                directory.rename(directory.with_name(directory.name + f".incomplete-{time.time_ns()}"))
            directory.mkdir()
            if task != current:
                if env is not None:
                    env.close()
                spec = bench.get_task(task)
                bddl = Path(get_libero_path("bddl_files")) / spec.problem_folder / spec.bddl_file
                env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
                initial_states = bench.get_task_init_states(task)
                current = task
                json_write(root / f"task{task}.json", {"bddl_path": str(bddl), "sha256": sha(bddl.read_bytes())})
            if episode >= len(initial_states):
                raise ValueError("Requested initial state out of range")
            result = collect(env, conn, spec.language, initial_states[episode], trial, cfg, directory, not audited)
            if not audited:
                json_write(root / "audit_passed.json", {"source": str(directory / "adapter_audit.json")})
                audited = True
            print(json.dumps(result), flush=True)
    finally:
        if env is not None:
            env.close()
        conn.close()


if __name__ == "__main__":
    main()
