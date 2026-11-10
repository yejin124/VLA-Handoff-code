#!/usr/bin/env python3
"""Episode conditions shared by the closed-loop rollout clients (full-chunk and re-query).

Supported situations:

  occluder  an out-of-focus object near the agent-view lens (realistic_corruption.occluder)
  lighting  a dim, warm room on both cameras (realistic_corruption.lighting)
  moved     a person slides the task's destination (plate or basket) sideways during the
            episode; the task stays achievable and success still requires the goal predicate
  relocated a person lifts the destination, carries it over the table and puts it down
            somewhere else: a seeded free spot at least `severity` metres away, clear of every
            other object, reached along a lifted straight path that prefers a clear corridor
  viewpoint the agent-view camera is mounted elsewhere: the renderer's camera pose is turned
            about the vertical axis through the point where its optical axis meets the table,
            by VIEW_DEG * severity degrees in a seeded direction, for the whole episode
            (amendment 2026-10-21; no image postprocessing, the wrist camera is unchanged)

The two camera families are fixed for the whole episode: their seed depends on the episode
and the camera, not on the step. `moved` is a simulator intervention applied over
`move_duration` consecutive simulator steps starting at `move_step`; its severity is the
displacement in metres. Direction is chosen by the episode seed among the eight compass
directions in the table plane that keep the destination inside the object layout and clear
of every other movable object; if none qualifies, the direction with the largest clearance is
used and the shortfall is recorded. Everything is deterministic in the episode seed.

The legacy families (noise, occlusion, combined) are kept so the old clients stay exact.
"""
import hashlib
import math

import numpy as np

from realistic_corruption import corrupt as realistic_corrupt, load_donor_bank  # noqa: F401

LEGACY_FAMILIES = ("noise", "occlusion", "combined")
CAMERA_FAMILIES = ("occluder", "lighting")
MOVE_FAMILY = "moved"
RELOCATE_FAMILY = "relocated"
MOVE_FAMILIES = (MOVE_FAMILY, RELOCATE_FAMILY)
VIEW_FAMILY = "viewpoint"
VIEW_DEG = 30.0               # degrees of camera yaw at severity 1.0
ALL_CONDITIONS = ("clean",) + LEGACY_FAMILIES + CAMERA_FAMILIES + MOVE_FAMILIES + (VIEW_FAMILY,)
DESTINATION_JOINT = {"libero_spatial": "plate_1_joint0", "libero_object": "basket_1_joint0"}
# Minimum centre-to-centre distance to any other movable object after the move, by suite.
CLEARANCE = {"libero_spatial": 0.13, "libero_object": 0.16}
LAYOUT_MARGIN = 0.08          # metres the destination may leave the initial object bounding box
FIXTURE_KEYS = ("cabinet", "stove")   # fixed furniture (no free joint); relocation keeps clear of its footprint
FIXTURE_MARGIN = 0.12         # metres added around a fixture footprint (about a plate radius)
RELOCATE_EXTRA = 0.07         # extra metres beyond LAYOUT_MARGIN for the relocation search box
LIFT = 0.12                   # metres the relocated destination is lifted while carried (amendment 2026-10-13)


def stable_seed(*parts):
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:4], "little")


def legacy_corrupt(image, family, severity, seed):
    if family == "clean":
        return image.copy()
    rng = np.random.default_rng(seed)
    x = image.astype(np.float32)
    if family in ("noise", "combined"):
        x += rng.normal(0, 64 * severity, x.shape)
    if family in ("occlusion", "combined"):
        height, width = image.shape[:2]
        side = int(.6 * severity * min(height, width))
        y, z = (height - side) // 2, (width - side) // 2
        x[y:y + side, z:z + side] = 0
    return np.clip(x, 0, 255).astype(np.uint8)


def image_seeds(pair, condition, steps):
    """Per-camera seeds: per step for the legacy families, per episode for the realistic ones."""
    if condition in CAMERA_FAMILIES:
        return [stable_seed(*pair, "corruption", condition, cam) for cam in range(2)]
    return [stable_seed(*pair, "corruption", steps, cam) for cam in range(2)]


def degrade(images, condition, severity, seeds, donors=None, apply_to_wrist=False):
    """Apply the active condition to [agent, wrist] images. `moved` leaves the images alone."""
    if condition == "clean" or condition in MOVE_FAMILIES or condition == VIEW_FAMILY:
        return [v.copy() for v in images]
    if condition in LEGACY_FAMILIES:
        return [legacy_corrupt(v, condition, severity, s) for v, s in zip(images, seeds)]
    if condition in CAMERA_FAMILIES:
        return [realistic_corrupt(v, condition, severity, s, camera=cam, donors=donors, apply_to_wrist=apply_to_wrist)
                for cam, (v, s) in enumerate(zip(images, seeds))]
    raise ValueError(f"unknown condition {condition!r}")


class DestinationMover:
    """Slides the destination object by `distance` metres over `duration` steps from `start`."""

    DIRECTIONS = [(math.cos(k * math.pi / 4), math.sin(k * math.pi / 4)) for k in range(8)]

    def __init__(self, env, suite, seed, distance, start, duration, relocate=False):
        self.relocate = relocate          # True: a new seeded position at least `distance` away
        self.sim = env.sim
        self.joint = DESTINATION_JOINT[suite]
        self.clearance = CLEARANCE[suite]
        self.seed, self.distance, self.start, self.duration = seed, float(distance), int(start), int(duration)
        names = [self.sim.model.joint_id2name(i) for i in range(self.sim.model.njnt)]
        self.others = [n for n in names if n and n.endswith("_joint0") and n != self.joint]
        if self.joint not in names:
            raise ValueError(f"destination joint {self.joint} not in this environment")
        self.origin = None
        self.record = None

    def _fixture_boxes(self):
        """xy bounding boxes of fixed furniture (all geoms of bodies named after a fixture), expanded."""
        m, d = self.sim.model, self.sim.data
        boxes = {}
        for g in range(m.ngeom):
            body = m.body_id2name(m.geom_bodyid[g]) or ""
            key = next((k for k in FIXTURE_KEYS if k in body), None)
            if key is None:
                continue
            prefix = body.split("_" + key)[0] + "_" + key
            lo_, hi_ = d.geom_xpos[g][:2] - m.geom_size[g][:2], d.geom_xpos[g][:2] + m.geom_size[g][:2]
            b = boxes.setdefault(prefix, [lo_.copy(), hi_.copy()])
            b[0], b[1] = np.minimum(b[0], lo_), np.maximum(b[1], hi_)
        return {k: (v[0] - FIXTURE_MARGIN, v[1] + FIXTURE_MARGIN) for k, v in boxes.items()}

    @staticmethod
    def _path_clearance(origin, target, others):
        """Smallest distance from any other object to the straight slide origin->target."""
        seg = target - origin
        t = np.clip(((others - origin) @ seg) / max(seg @ seg, 1e-12), 0.0, 1.0)
        nearest = origin + t[:, None] * seg
        return float(np.linalg.norm(others - nearest, axis=1).min())

    def _plan_relocation(self, q, others, lo, hi, rng):
        """Seeded free spot inside the layout box, >= distance from the origin, clear of every
        movable object along the whole slide (amendment 2026-10-13: the first relocated cohort
        slid the destination straight through the other objects, which fired the success
        predicate mid-slide) and outside every fixture footprint."""
        fixtures = self._fixture_boxes()
        best, best_clear, feasible = None, -1.0, False
        for _ in range(2000):
            target = rng.uniform(lo, hi)
            if np.linalg.norm(target - q[:2]) < self.distance:
                continue
            if any(np.all(target >= flo) and np.all(target <= fhi) for flo, fhi in fixtures.values()):
                continue
            if float(np.linalg.norm(others - target, axis=1).min()) < self.clearance:
                continue                                   # the landing spot itself must be clear
            clear = self._path_clearance(q[:2], target, others)
            if clear >= self.clearance:
                best, best_clear, feasible = target, clear, True
                break
            if clear > best_clear:
                best, best_clear = target, clear
        if best is None:                     # box too small for the distance: farthest corner
            corners = np.array([[lo[0], lo[1]], [lo[0], hi[1]], [hi[0], lo[1]], [hi[0], hi[1]]])
            best = corners[np.argmax(np.linalg.norm(corners - q[:2], axis=1))]
            best_clear = float(np.linalg.norm(others - best, axis=1).min())
        self.origin = q
        self.target_xy = np.asarray(best, dtype=float)
        delta = self.target_xy - q[:2]
        self.direction = delta / max(np.linalg.norm(delta), 1e-9)
        self.distance = float(np.linalg.norm(delta))
        self.record = {"joint": self.joint, "mode": "relocated", "lift_m": LIFT, "path_clearance_m": best_clear, "min_distance_m": self.distance, "distance_m": self.distance,
                       "direction_xy": self.direction.tolist(), "target_xy": self.target_xy.tolist(),
                       "start_step": self.start, "duration_steps": self.duration, "origin_xyz": q[:3].tolist(),
                       "min_clearance_m": best_clear, "clearance_satisfied": feasible, "others": self.others,
                       "fixtures_avoided": sorted(fixtures)}

    def _plan(self):
        """Pick the direction at the moment the move begins, from the objects' current layout."""
        q = self.sim.data.get_joint_qpos(self.joint).copy()
        others = np.array([self.sim.data.get_joint_qpos(n)[:2] for n in self.others])
        lo = np.minimum(others.min(axis=0), q[:2]) - LAYOUT_MARGIN
        hi = np.maximum(others.max(axis=0), q[:2]) + LAYOUT_MARGIN
        rng = np.random.default_rng(self.seed)
        if self.relocate:                         # a wider search box: a person can use the whole table
            self._plan_relocation(q, others, lo - RELOCATE_EXTRA, hi + RELOCATE_EXTRA, rng)
            return
        order = rng.permutation(len(self.DIRECTIONS))
        best, best_clear = None, -1.0
        for k in order:
            d = np.array(self.DIRECTIONS[k])
            target = q[:2] + self.distance * d
            clear = float(np.linalg.norm(others - target, axis=1).min())
            inside = bool(np.all(target >= lo) and np.all(target <= hi))
            if inside and clear >= self.clearance:
                best, best_clear, feasible = int(k), clear, True
                break
            if inside and clear > best_clear:
                best, best_clear, feasible = int(k), clear, False
        if best is None:                                   # nothing inside the layout box: take the clearest
            clears = [float(np.linalg.norm(others - (q[:2] + self.distance * np.array(d)), axis=1).min())
                      for d in self.DIRECTIONS]
            best, best_clear, feasible = int(np.argmax(clears)), max(clears), False
        self.origin = q
        self.direction = np.array(self.DIRECTIONS[best])
        self.record = {"joint": self.joint, "mode": "moved", "distance_m": self.distance, "direction_index": best,
                       "direction_xy": self.direction.tolist(), "start_step": self.start,
                       "duration_steps": self.duration, "origin_xyz": q[:3].tolist(),
                       "min_clearance_m": best_clear, "clearance_satisfied": feasible,
                       "others": self.others}

    def before_step(self, steps):
        """Call before env.step at simulator step `steps`; returns True while the move is active."""
        if steps < self.start or steps >= self.start + self.duration:
            return False
        if self.origin is None:
            self._plan()
        frac = (steps - self.start + 1) / self.duration
        q = self.origin.copy()
        q[:2] = self.origin[:2] + frac * self.distance * self.direction
        if self.relocate:                                  # carried over the table, set down at the end
            q[2] = self.origin[2] + LIFT * math.sin(math.pi * frac)
        self.sim.data.set_joint_qpos(self.joint, q)
        self.sim.data.set_joint_qvel(self.joint, np.zeros(6))
        self.sim.forward()
        if steps == self.start + self.duration - 1:        # position at the end of the slide, exact by construction
            self.record["end_of_slide_xyz"] = q[:3].tolist()
            self.record["end_of_slide_displacement_m"] = float(np.linalg.norm(q[:2] - self.origin[:2]))
        return True

    def final(self):
        if self.record is None:
            return None
        q = self.sim.data.get_joint_qpos(self.joint)
        return dict(self.record, final_xyz=q[:3].tolist(),
                    episode_end_displacement_m=float(np.linalg.norm(q[:2] - np.array(self.record["origin_xyz"][:2]))),
                    note="end_of_slide_* is the intervention; episode_end_* also contains what the robot did to the object afterwards")


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _quat_rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def shift_viewpoint(env, suite, seed, severity, camera="agentview"):
    """Turn the renderer's `camera` about the vertical axis through the point where its optical axis
    meets the table (the destination's height). Call after every reset/set_init_state: robosuite
    rebuilds the model on reset. Returns the record stored in the episode summary."""
    model = env.sim.model
    cid = model.camera_name2id(camera)
    if model.cam_bodyid[cid] != 0:
        raise ValueError(f"{camera} is not a world-fixed camera")
    pos, quat = model.cam_pos[cid].copy(), model.cam_quat[cid].copy()
    forward = _quat_rot(quat) @ np.array([0., 0., -1.])     # MuJoCo cameras look along -z
    table_z = float(env.sim.data.get_joint_qpos(DESTINATION_JOINT[suite])[2])
    pivot = pos + (table_z - pos[2]) / forward[2] * forward
    sign = 1 if seed % 2 == 0 else -1
    yaw = sign * VIEW_DEG * float(severity)
    t = math.radians(yaw)
    c, s = math.cos(t), math.sin(t)
    d = pos - pivot
    model.cam_pos[cid] = pivot + np.array([c * d[0] - s * d[1], s * d[0] + c * d[1], d[2]])
    model.cam_quat[cid] = _quat_mul(np.array([math.cos(t / 2), 0., 0., math.sin(t / 2)]), quat)
    return {"mode": VIEW_FAMILY, "camera": camera, "yaw_deg": yaw, "pivot": pivot.round(4).tolist(),
            "original_pos": pos.round(4).tolist(), "shifted_pos": model.cam_pos[cid].round(4).tolist()}
