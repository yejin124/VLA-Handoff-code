"""Paired continue / view-restore branches from stored occluder prefixes (experiment 2026-10-29).

For each main-campaign occluder episode still active at step 80, the stored executed actions are
replayed in a fresh simulator up to the end of the step-80 query's chunk; every replayed robot state
is checked against the stored trace. Two branches then run to the 280-step horizon:

  continue  occluder kept on both cameras (same policy and image seeds as the source episode)
  restore   occluder removed from the next query on: a person clears the obstruction

The monitor decision uses only the source episode's step-80 held-out score, so no branch outcome
enters fitting or thresholds. J=1 continuations reuse the seed, but batch-size changes need not preserve the J=8 draw zero.
Replay checks cover robot-state coordinates, not the full simulator state.
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
import icassp_rollout as R
from icassp_conditions import degrade, image_seeds, load_donor_bank
from failure_state_convention import select_state_convention, require_same_state_convention

DECISION_STEP = 80
STATE_ATOL = 1e-5


class ReplayMismatch(Exception):
    """The fresh-simulator replay left the stored trajectory; the episode is excluded and recorded."""


def replay(env, initial, cfg, task, episode, src, state_fn=None):
    """Fresh-env replay of the source prefix through the step-80 chunk; returns obs, step, next query."""
    state_fn = R.robot_state if state_fn is None else state_fn
    pair = (cfg["seed"], cfg["suite"], task, episode, 0)
    env.seed(R.stable_seed(*pair, "environment"))
    env.reset()
    env.set_init_state(initial)
    for _ in range(10):
        obs, _, _, _ = env.step([0.] * 6 + [-1.])
    queries = [json.loads(l) for l in (src / "queries.jsonl").read_text().splitlines()]
    trace = [json.loads(l) for l in (src / "steps.jsonl").read_text().splitlines()]
    decision = max(q["query"] for q in queries if q["step"] <= DECISION_STEP)
    steps, worst = 0, 0.0
    for q in queries[:decision + 1]:
        gap = float(np.abs(state_fn(obs) - np.asarray(q["state"], np.float32)).max())
        if gap > STATE_ATOL:
            raise ReplayMismatch(q["query"], gap)
        for action in np.load(src / q["artifact"])["executed_actions"]:
            obs, _, done, _ = env.step(action.tolist())
            steps += 1
            worst = max(worst, float(np.abs(state_fn(obs) - trace[steps - 1]["state"]).max()))
            if done or steps >= cfg["max_steps"]:
                break
        if done:
            break
    if worst > STATE_ATOL:
        raise ReplayMismatch(decision, worst)
    nxt = queries[decision + 1] if decision + 1 < len(queries) else None
    return obs, steps, decision + 1, done, worst, nxt


def branch(env, conn, language, obs, steps, qi, done, cfg, task, episode, active, donors, state_fn=None):
    state_fn = R.robot_state if state_fn is None else state_fn
    pair = (cfg["seed"], cfg["suite"], task, episode, 0)
    first_hash, first_action, n = None, None, 0
    while not done and steps < cfg["max_steps"] and not env.check_success():
        seeds = image_seeds(pair, active, steps)
        images = degrade([obs[k][::-1, ::-1] for k in ("agentview_image", "robot0_eye_in_hand_image")],
                         active, 1.0, seeds, donors=donors, apply_to_wrist=True)
        req = {"images": [R.encode(v) for v in images], "state": state_fn(obs).tolist(),
               "task": language, "seed": R.stable_seed(*pair, "policy", qi), "J": 1}
        actions, _, _, _ = R.validate_response(R.query(conn, req), 1)
        if first_hash is None:
            first_hash, first_action = [R.array_hash(v) for v in images], actions
        for action in actions:
            obs, _, done, _ = env.step(action.tolist())
            steps += 1
            if done or steps >= cfg["max_steps"]:
                break
        qi, n = qi + 1, n + 1
    return {"success": bool(env.check_success()), "steps": steps, "queries_after_branch": n,
            "first_image_sha256": first_hash, "first_actions": None if first_action is None else first_action.tolist()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True, help="realistic/<suite>_main/<policy> directory")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--reverse", action="store_true", help="walk the shard backwards (extra client sharing a shard)")
    p.add_argument("--replay-only", action="store_true", help="check prefix replay fidelity without a server")
    p.add_argument("--state-convention", choices=("official", "recorded"), default="official",
                   help="recorded reproduces the historical wrapped orientation input")
    a = p.parse_args()
    state_fn, state_receipt = select_state_convention(a.state_convention)
    srcs = sorted(d for d in a.source.glob("shard*/t*_occluder")
                  if json.loads((d / "summary.json").read_text())["steps"] > DECISION_STEP)
    srcs = srcs[a.shard::a.nshards][::-1 if a.reverse else 1]
    a.out.mkdir(parents=True, exist_ok=True)
    conn = None
    if not a.replay_only:
        conn = socket.create_connection(("127.0.0.1", a.port), timeout=600)
        ping = R.query(conn, {"cmd": "ping"})
        if a.model.lower() not in ping.get("model", "").lower():
            raise ValueError(f"Wrong server: {ping}")
    donors = load_donor_bank()
    import torch
    original_load = torch.load
    torch.load = lambda *x, **k: original_load(*x, **dict(k, weights_only=False))
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    cfg0 = json.loads(next(a.source.glob("shard*/config.json")).read_text())
    bench = benchmark.get_benchmark_dict()[cfg0["suite"]]()
    for src in srcs:
        rec = a.out / f"{src.parent.name}_{src.name}.json"
        if rec.exists():
            require_same_state_convention(json.loads(rec.read_text()), state_receipt)
            continue
        cfg = json.loads((src.parent / "config.json").read_text())
        s = json.loads((src / "summary.json").read_text())
        task, episode = s["task"], s["episode"]
        spec = bench.get_task(task)
        bddl = Path(get_libero_path("bddl_files")) / spec.problem_folder / spec.bddl_file
        initial = bench.get_task_init_states(task)[episode]
        result = {"source": str(src), "task": task, "episode": episode, "source_success": s["success"],
                  "source_steps": s["steps"], "decision_step": DECISION_STEP, "branches": {},
                  "state_convention": state_receipt}
        started = time.monotonic()
        for name, active in (("continue", "occluder"), ("restore", "clean")):
            env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
            try:
                try:
                    obs, steps, qi, done, worst, nxt = replay(env, initial, cfg, task, episode, src, state_fn=state_fn)
                except ReplayMismatch as e:
                    result["replay_mismatch"] = {"query": e.args[0], "max_state_error": e.args[1]}
                    break
                result["replay_max_state_error"] = worst
                result["branch_step"], result["branch_query"] = steps, qi
                if a.replay_only:
                    result["branches"][name] = {"replayed_to": steps}
                    continue
                out = branch(env, conn, s["language"], obs, steps, qi, done, cfg, task, episode, active, donors,
                             state_fn=state_fn)
                if name == "continue" and nxt is not None:
                    out["first_image_matches_source"] = out["first_image_sha256"] == nxt["image_sha256"]
                    out["first_actions_match_source"] = bool(np.array_equal(
                        np.asarray(out["first_actions"], np.float32), np.load(src / nxt["artifact"])["executed_actions"]))
                result["branches"][name] = out
            finally:
                env.close()
        result["wall_seconds"] = time.monotonic() - started
        R.json_write(rec, result)
        print(json.dumps({k: result[k] for k in ("source", "source_success")} |
                         {b: v.get("success") for b, v in result["branches"].items()}), flush=True)
    if conn:
        conn.close()


if __name__ == "__main__":
    main()
