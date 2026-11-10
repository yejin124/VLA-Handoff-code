"""Read-only, source-bound feature extraction for nested occlusion controls."""
from __future__ import annotations

import hashlib
import json

import numpy as np
from scipy.spatial.transform import Rotation

from icassp_conditions import stable_seed as corruption_seed
from realistic_corruption import _blob_mask

def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def policy_seed(*parts):
    blob = json.dumps(parts, separators=(',', ':'), ensure_ascii=True).encode()
    return int.from_bytes(hashlib.sha256(blob).digest()[:4], 'big') % (2**31 - 1)


def rotation_matrix(values, convention):
    x = np.asarray(values, dtype=np.float64)
    if x.shape != (3,) or not np.isfinite(x).all():
        raise ValueError('Rotation must have three finite coordinates')
    if convention == 'axisangle':
        return Rotation.from_rotvec(x).as_matrix()
    if convention == 'euler_sxyz':
        return Rotation.from_euler('xyz', x).as_matrix()
    raise ValueError('Unknown rotation convention')


def external40(recorded_baseline, convention):
    b = np.asarray(recorded_baseline, dtype=np.float64)
    if b.shape != (34,) or not np.isfinite(b).all():
        raise ValueError('Expected 34 finite original external coordinates')
    state = b[4:12]
    out = np.concatenate((b[:4], state[:3], rotation_matrix(state[3:6], convention).ravel(),
                          state[6:8], b[12:]))
    if out.shape != (40,):
        raise ValueError('Invalid rebuilt external dimensions')
    return out


def geometry(seed, suite, task, scene, replica, cameras, severity):
    yy, xx = np.mgrid[:256, :256]
    center = np.exp(-((xx - 128) ** 2 + (yy - 128) ** 2) / (2 * 50 ** 2))
    result = []
    for camera in range(cameras):
        s = corruption_seed(seed, suite, task, scene, replica, 'corruption', 'occluder', camera)
        rng = np.random.default_rng(s)
        side, along = int(rng.integers(4)), float(rng.uniform(.15, .85))
        mask = _blob_mask(256, 256, .35 * severity, s, feather_scale=.03, at_edge=True)
        result.extend([float(side == j) for j in range(4)] +
                      [along, float((mask * center).sum() / center.sum()), float(mask.mean())])
    return np.asarray(result, dtype=np.float64)
