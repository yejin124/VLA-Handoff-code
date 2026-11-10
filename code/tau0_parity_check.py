"""Stage 2 of the tau-0 adapter parity check (tau-0 venv).

For every snapshot from code/tau0_parity_dump.py, compares
  official: deploy/libero/main.py element construction (_prepare_image on the raw render,
            _quat2axisangle state) -> official deploy.libero_server (/reset_episode seed, /act_libero)
  ours:     code/icassp_rollout.py transmitted images/state -> code/serve_icassp_tau0.py (seed, J)
Executed actions must be bitwise equal to the first POLICY_EXECUTE_STEPS official actions.
"""
import argparse
import base64
import io
import json
from pathlib import Path
import pickle
import socket
import struct
import sys

import numpy as np
import requests


def encode(value):
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(value), allow_pickle=False)
    return base64.b64encode(buffer.getvalue()).decode()


def bridge_query(conn, request):
    data = json.dumps(request).encode()
    conn.sendall(struct.pack(">I", len(data)) + data)
    size = struct.unpack(">I", conn.recv(4, socket.MSG_WAITALL))[0]
    body = bytearray()
    while len(body) < size:
        body.extend(conn.recv(size - len(body)))
    response = json.loads(body)
    if not response.get("ok"):
        raise RuntimeError(response)
    return response


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dump", type=Path, required=True)
    p.add_argument("--tau0-root", type=Path, required=True)
    p.add_argument("--official-port", type=int, required=True)
    p.add_argument("--bridge-port", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.tau0_root))
    from deploy.libero.main import _prepare_image, _quat2axisangle   # the official evaluator's own functions
    data = np.load(a.dump)
    count = len({k.split("_", 1)[0] for k in data.files})
    official_url = f"http://127.0.0.1:{a.official_port}"
    conn = socket.create_connection(("127.0.0.1", a.bridge_port), timeout=600)
    steps = bridge_query(conn, {"cmd": "ping"})["action_horizon"]
    rows = []
    for i in range(count):
        row = {k.split("_", 1)[1]: data[k] for k in data.files if k.split("_", 1)[0] == str(i)}
        language = str(row["language"])
        official_state = np.concatenate((row["eef_pos"], _quat2axisangle(row["eef_quat"]), row["gripper_qpos"]))
        state_equal = bool(np.array_equal(row["state"], official_state.astype(np.float32)))
        for variant, raw, sent in (("clean", (row["raw_agent"], row["raw_wrist"]), (row["sent_agent"], row["sent_wrist"])),
                                   ("occluder", (row["occ_agent"][::-1, ::-1], row["occ_wrist"][::-1, ::-1]),
                                    (row["occ_agent"], row["occ_wrist"]))):
            seed = 1000 + 17 * i + (variant == "occluder")
            element = {"observation/image": _prepare_image(raw[0], 224),
                       "observation/wrist_image": _prepare_image(raw[1], 224),
                       "observation/state": official_state, "prompt": language}
            requests.post(f"{official_url}/reset_episode", params={"seed": seed}, timeout=60).raise_for_status()
            response = requests.post(f"{official_url}/act_libero", data=pickle.dumps(element), timeout=600)
            response.raise_for_status()
            official = np.asarray(response.json(), dtype=np.float32)
            request = {"images": [encode(v) for v in sent], "state": row["state"].tolist(), "task": language, "seed": seed}
            one = bridge_query(conn, dict(request, J=1))
            eight = bridge_query(conn, dict(request, J=8, audit=True))
            ours = np.asarray(one["actions"], dtype=np.float32)
            rows.append({"suite": str(row["suite"]), "task": int(row["task"]), "snapshot": int(row["snapshot"]),
                         "variant": variant, "seed": seed, "state_bitwise_equal": state_equal,
                         "official_chunk_shape": list(official.shape),
                         "actions_bitwise_equal_official": bool(np.array_equal(ours, official[:steps])),
                         "max_abs_diff_official": float(np.abs(ours - official[:steps]).max()),
                         "j1_j8_sample0_equal": bool(np.array_equal(ours, np.asarray(eight["actions"], dtype=np.float32))),
                         "hook_parity": eight["audit"]["exact_equal"], "handoff_dims": len(eight["kv_pooled"]),
                         "kv_shapes": eight["kv_shapes"], "spread": eight["action_std"]})
            print(json.dumps(rows[-1]), flush=True)
    summary = {"snapshots": count, "comparisons": len(rows),
               "all_actions_bitwise_equal": all(r["actions_bitwise_equal_official"] for r in rows),
               "all_states_bitwise_equal": all(r["state_bitwise_equal"] for r in rows),
               "all_j_invariant": all(r["j1_j8_sample0_equal"] for r in rows),
               "all_hook_parity": all(r["hook_parity"] for r in rows), "rows": rows}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}))


if __name__ == "__main__":
    main()
