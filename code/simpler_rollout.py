"""Seeded SimplerEnv-Bridge (WidowX) rollouts in the record format of icassp_rollout.py.

Run with the SimplerEnv venv of the Isaac-GR00T checkout. One camera (image_0); the
occluder covers it from step 0. Episodes end at the first success, as in LIBERO.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from icassp_conditions import degrade, image_seeds, load_donor_bank
from icassp_rollout import array_hash, encode, json_write, query, sha, stable_seed, validate_response

TASKS = ["widowx_spoon_on_towel", "widowx_carrot_on_plate", "widowx_stack_cube", "widowx_put_eggplant_in_basket",
         "widowx_put_eggplant_in_sink", "widowx_open_drawer", "widowx_close_drawer"]
SCENE_TASKS = (0, 1, 2, 3)
STATE_KEYS = ["x", "y", "z", "roll", "pitch", "yaw", "pad", "gripper"]
ACTION_KEYS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
SUITE = "simpler_bridge"


def collect(env, conn, trial, cfg, directory, donors):
    task, episode, rep, condition = trial
    pair = (cfg["seed"], SUITE, task, episode, rep)
    env_seed = stable_seed(*pair[:-1], "environment")   # same scene for every replicate and condition
    wrapper = env.unwrapped
    # The four put/stack tasks have 24 canonical scenes chosen by episode_id; the drawer tasks have one scene.
    options = {"obj_init_options": {"episode_id": episode}} if task in SCENE_TASKS else None
    obs, _ = env.reset(seed=env_seed)                    # the outer wrapper must see a reset before step
    if options:                                          # its reset drops options, so pick the scene directly
        obs = wrapper._process_observation(wrapper.env.reset(seed=env_seed, options=options)[0])
    language = obs["annotation.human.action.task_description"]
    scene = getattr(wrapper.env.unwrapped, "obj_init_options", None) or {}
    initial = sha(json.dumps([TASKS[task], episode,
                              {k: np.asarray(v).tolist() for k, v in scene.items()}], sort_keys=True).encode())
    steps, qi, success, done = 0, 0, False, False
    # Fork mode: replay the source episode's executed actions open-loop for every query that starts at or
    # before replay_through (the rendered images are not bitwise reproducible, the physics is), then hand over.
    source, src_q = None, []
    if cfg["replay_source"]:
        found = sorted(Path(cfg["replay_source"]).glob(f"shard*/{directory.name}/summary.json"))
        if len(found) != 1 or not json.loads(found[0].read_text()).get("complete"):
            raise ValueError(f"Fork mode needs exactly one complete source episode for {directory.name}")
        source = found[0].parent
        src_q = [json.loads(x) for x in (source / "queries.jsonl").read_text().splitlines()]
    replay_error = 0.0
    started = time.monotonic()
    with (directory / "queries.jsonl").open("w") as trace:
        while not (success or done) and steps < cfg["max_steps"]:
            seeds = image_seeds(pair, condition, steps)[:1]
            shown = "clean" if 0 <= cfg["restore_after"] < steps else condition   # restore branch: view cleared
            image = degrade([obs["video.image_0"]], shown, cfg["severity"], seeds, donors=donors)[0]
            state = np.array([obs[f"state.{k}"][0] for k in STATE_KEYS], dtype=np.float32)
            seed = stable_seed(*pair, "policy", qi)
            t0 = time.monotonic()
            replayed = qi < len(src_q) and src_q[qi]["step"] <= cfg["replay_through"]
            if replayed and src_q[qi]["step"] != steps:
                raise ValueError(f"Replay out of step in {directory.name}: query {qi} at {steps}, source {src_q[qi]['step']}")
            if source and not replayed and steps <= cfg["replay_through"]:   # source ended or skipped a query
                raise ValueError(f"Source {source.name} ends its replay prefix early at query {qi}")
            if replayed:
                rec = np.load(source / src_q[qi]["artifact"])
                replay_error = max(replay_error, float(np.abs(state - rec["state"]).max()))
                actions, samples, feature = rec["executed_actions"], rec["action_samples"], rec["handoff"]
                spread, response = src_q[qi]["spread"], {}
            else:
                response = query(conn, {"images": [encode(image)], "state": state.tolist(), "task": language,
                                        "seed": seed, "J": cfg["draws"]})
                actions, samples, feature, spread = validate_response(response, cfg["draws"])
            latency = time.monotonic() - t0
            artifact = f"q{qi:04d}.npz"
            keep = qi == 0 or any(steps <= m < steps + len(actions) for m in cfg["landmarks"])
            np.savez_compressed(directory / artifact, handoff=feature.astype(np.float16), action_samples=samples,
                                executed_actions=actions, state=state, **({"agent_image": image} if keep else {}))
            # The analyzer expects two means and two stds; the halves of the single frame stand in for two cameras.
            halves = (image[:, :128], image[:, 128:])
            trace.write(json.dumps({
                "query": qi, "step": steps, "policy_seed": seed, "corruption_seeds": seeds, "active_condition": condition,
                "image_sha256": [array_hash(image)], "artifact": artifact, "shown_condition": shown, "replayed": replayed,
                "artifact_sha256": sha((directory / artifact).read_bytes()), "spread": spread, "latency_seconds": latency,
                "draws": cfg["draws"], "image_stats": [float(np.mean(h)) for h in halves] + [float(np.std(h)) for h in halves],
                "state": state.tolist(), "action_l2": float(np.linalg.norm(actions)),
                "server_timing": response.get("timing")}, allow_nan=False) + "\n")
            for action in actions:
                obs, _, done, _, info = env.step({f"action.{k}": action[i:i + 1] for i, k in enumerate(ACTION_KEYS)})
                steps += 1
                success = bool(info["success"])
                if success or done or steps >= cfg["max_steps"]:
                    break
            qi += 1
    summary = {"task": task, "env_name": TASKS[task], "episode": episode, "replicate": rep, "condition": condition,
               "language": language, "initial_state_sha256": initial, "environment_seed": env_seed,
               "success": success, "steps": steps, "queries": qi, "wall_seconds": time.monotonic() - started,
               "terminal_reason": "success" if success else ("terminated" if done else "timeout"),
               "restore_after": cfg["restore_after"], "replay_max_state_error": replay_error if source else None, "config_sha256": cfg["config_sha256"], "complete": True}
    json_write(directory / "summary.json", summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--tasks", default="0,1,2,3,5,6")
    p.add_argument("--replicates", type=int, default=1)
    p.add_argument("--episodes", required=True, help="first:last (inclusive) episode indices")
    p.add_argument("--conditions", default="clean,occluder")
    p.add_argument("--severity", type=float, required=True)
    p.add_argument("--draws", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=120)
    p.add_argument("--landmarks", default="10,20")
    p.add_argument("--seed", type=int, default=20261103)
    p.add_argument("--restore-after", type=int, default=-1, help="clear the occluder for queries starting after this step")
    p.add_argument("--replay-source", default="", help="source cohort root whose episodes are replayed (fork mode)")
    p.add_argument("--replay-through", type=int, default=-1, help="replay source queries starting at or before this step")
    p.add_argument("--fresh-env", action="store_true", help="create a new environment for every episode")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--gr00t-root", default=os.environ.get("GR00T_ROOT"))
    a = p.parse_args()
    if not a.gr00t_root:
        p.error("Set GR00T_ROOT or pass --gr00t-root for the Isaac-GR00T checkout")
    first, last = map(int, a.episodes.split(":"))
    tasks = [int(t) for t in a.tasks.split(",")]
    conditions = a.conditions.split(",")
    if not set(conditions) <= {"clean", "occluder"} or not 0 <= a.shard < a.nshards or not 0 <= first <= last:
        p.error("invalid conditions, shard or episode range")
    root = a.out.expanduser().resolve() / f"shard{a.shard:02d}"
    root.mkdir(parents=True, exist_ok=True)
    conn = socket.create_connection(("127.0.0.1", a.port), timeout=300)
    ping = query(conn, {"cmd": "ping"})
    if "SimplerEnv-Bridge" not in ping.get("model", ""):
        raise ValueError(f"Wrong server: {ping}")
    cfg = {k: v for k, v in vars(a).items() if k not in ("out", "port", "gr00t_root")}
    cfg.update(suite=SUITE, task_names=TASKS, landmarks=[int(v) for v in a.landmarks.split(",")],
               collector_sha256=sha(Path(__file__).read_bytes()),
               server_identity={k: ping.get(k) for k in ("model", "code_sha256", "checkpoint_id")})
    cfg["config_sha256"] = sha(json.dumps(cfg, sort_keys=True).encode())
    if (root / "config.json").exists() and json.loads((root / "config.json").read_text()) != cfg:
        raise ValueError("Resume configuration mismatch; use a new output directory")
    json_write(root / "config.json", cfg)
    json_write(root / "server.json", ping)
    sys.path.insert(0, a.gr00t_root)
    import gymnasium as gym
    from gr00t.eval.sim.SimplerEnv.simpler_env import register_simpler_envs
    register_simpler_envs()
    donors = load_donor_bank() if "occluder" in conditions else None
    pairs = [(t, e) for t in tasks for e in range(first, last + 1)]
    for task in tasks:
        env = None
        for k, (t, episode) in enumerate(pairs):
            if t != task or k % a.nshards != a.shard:
                continue
            for rep, condition in ((r, c) for r in range(a.replicates) for c in conditions):
                directory = root / f"t{task}_e{episode}_r{rep}_{condition}"
                if (directory / "summary.json").exists():
                    continue
                if directory.exists():
                    directory.rename(directory.with_name(directory.name + f".incomplete-{time.time_ns()}"))
                directory.mkdir()
                env = env or gym.make(f"simpler_env_widowx/{TASKS[task]}")
                print(json.dumps(collect(env, conn, (task, episode, rep, condition), cfg, directory, donors)), flush=True)
                if a.fresh_env:          # physics carries over across resets; a fresh env makes replays reproducible
                    env.close()
                    env = None
        if env is not None:
            env.close()
    conn.close()


if __name__ == "__main__":
    main()
