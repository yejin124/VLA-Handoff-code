"""Stage 1 of the tau-0 adapter parity check (LIBERO interpreter).

Dumps raw LIBERO observations (unflipped renders, eef pose/quat, gripper qpos) plus
what code/icassp_rollout.py would transmit for them (flipped images, robot_state,
and an occluder-degraded variant) so code/tau0_parity_check.py can compare the
collector + bridge path with the official tau-0 evaluator + server path.
"""
import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import icassp_rollout as rollout
from icassp_conditions import degrade, load_donor_bank


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--suites", default="libero_spatial,libero_object")
    p.add_argument("--tasks", default="0,5")
    p.add_argument("--episode", type=int, default=15)
    p.add_argument("--snapshots", type=int, default=3)
    a = p.parse_args()
    import torch
    original_load = torch.load
    torch.load = lambda *args, **kw: original_load(*args, **dict(kw, weights_only=False))
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    donors = load_donor_bank()
    rng = np.random.default_rng(20261106)
    rows = []
    for suite in a.suites.split(","):
        bench = benchmark.get_benchmark_dict()[suite]()
        for task in (int(t) for t in a.tasks.split(",")):
            spec = bench.get_task(task)
            bddl = Path(get_libero_path("bddl_files")) / spec.problem_folder / spec.bddl_file
            env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
            env.seed(7)
            env.reset()
            obs = env.set_init_state(bench.get_task_init_states(task)[a.episode])
            for _ in range(10):
                obs, _, _, _ = env.step([0.] * 6 + [-1.])
            for k in range(a.snapshots):
                sent = [obs[key][::-1, ::-1] for key in ("agentview_image", "robot0_eye_in_hand_image")]
                pair = (20261106, suite, task, a.episode, k)
                occluded = degrade(sent, "occluder", 1.0, rollout.image_seeds(pair, "occluder", 0), donors=donors,
                                   apply_to_wrist=True)
                rows.append(dict(suite=suite, task=task, language=spec.language, snapshot=k,
                                 raw_agent=obs["agentview_image"].copy(), raw_wrist=obs["robot0_eye_in_hand_image"].copy(),
                                 eef_pos=obs["robot0_eef_pos"].copy(), eef_quat=obs["robot0_eef_quat"].copy(),
                                 gripper_qpos=obs["robot0_gripper_qpos"].copy(), sent_agent=sent[0].copy(),
                                 sent_wrist=sent[1].copy(), occ_agent=occluded[0], occ_wrist=occluded[1],
                                 state=rollout.robot_state(obs)))
                for _ in range(25):   # move the arm to a new pose (rotation changes too)
                    obs, _, _, _ = env.step(np.r_[rng.uniform(-.6, .6, 6), rng.choice([-1., 1.])].tolist())
            env.close()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **{f"{i}_{k}": v for i, row in enumerate(rows) for k, v in row.items()})
    print(f"wrote {len(rows)} snapshots to {a.out}")


if __name__ == "__main__":
    main()
