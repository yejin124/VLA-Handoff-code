#!/usr/bin/env python3
"""E9: situation decodability (Table 2, column A) with a 999-permutation within-task null.

Same rows and pipeline as code/realistic_probe.py part A: clean vs situation, camera situations at
query 0, goal situations at the first query at or after step 40; StandardScaler -> PCA(<=30,
random_state 0) -> logistic C=1, task-grouped 10-fold out-of-fold AUROC. The paper's null had 20
permutations (minimum Monte Carlo p 1/21). Here the clean/situation labels are exchanged within each
matched initial-state pair (same task and initial state; swap or keep with probability 1/2) 999 times,
which preserves the paired design; rows without a partner keep their label;
p = (1 + #null >= observed) / (1 + 999) (Phipson & Smyth 2010), Holm over the 12 goal-situation tests.
Standardization and PCA are unsupervised, so they are fitted once per fold and reused for every
permutation; this is exact for random_state 0 (the paper's null reseeded PCA per permutation).
Also exposes the row loader and cached probe for E3 (image-embedding baselines).
Run: python code/exp_20261020_e9_probe_null.py
"""
import glob
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import exp_20261020_common as c

CAMERA, GOAL = ("occluder", "lighting"), ("moved", "relocated")
N_PERM = 999


def load_rows(suite, policy, with_handoff=True):
    """realistic_probe.load_rows: main (no dark lighting), main2 lighting 1.25, main3 relocated."""
    rows = []
    for coll in ("main", "main2", "main3"):
        for sdir in sorted(glob.glob(f"{c.DATA}/realistic/{suite}_{coll}/{policy}/shard*/t*_e*_r0_*")):
            summ = json.load(open(Path(sdir) / "summary.json")) if (Path(sdir) / "summary.json").exists() else {}
            if not summ.get("complete"):
                continue
            if (coll == "main" and summ["condition"] == "lighting") or (coll == "main2" and summ["condition"] == "relocated"):
                continue
            queries = [json.loads(l) for l in open(Path(sdir) / "queries.jsonl")]
            picks = {"q0": queries[0]}
            later = [q for q in queries if q["step"] >= 40]
            if later:
                picks["q40"] = later[0]
            for tag, q in picks.items():
                row = {"id": str(Path(sdir).resolve()), "query": q["query"], "task": summ["task"],
                       "episode": summ["episode"], "condition": summ["condition"], "tag": tag,
                       "image_stats": np.array(q["image_stats"], dtype=np.float32)}
                if with_handoff:
                    row["handoff"] = np.load(Path(sdir) / q["artifact"])["handoff"].astype(np.float32)
                rows.append(row)
    return rows


def subset(rows, cond):
    tag = "q0" if cond in CAMERA else "q40"
    sub = [r for r in rows if r["tag"] == tag and r["condition"] in ("clean", cond)]
    y = np.array([r["condition"] == cond for r in sub], dtype=int)
    g = np.array([r["task"] for r in sub])
    pairs = np.array([f"{r['task']}_{r['episode']}" for r in sub])
    return sub, y, g, pairs


class InsufficientSupport(ValueError):
    """The prescribed task folds cannot estimate this test."""


def support_reason(y, groups):
    y, groups = np.asarray(y), np.asarray(groups)
    if y.ndim != 1 or groups.shape != y.shape:
        raise ValueError('Mismatched labels and task identities')
    if len(np.unique(groups)) != 10:
        return 'The prescribed 10 task folds require 10 active tasks'
    if len(np.unique(y)) != 2:
        return 'Both condition labels are required'
    for train, _ in GroupKFold(n_splits=10).split(np.empty((len(y), 1)), groups=groups):
        if len(np.unique(y[train])) != 2:
            return 'A prescribed training fold lacks both condition labels'
    return None


def unavailable(y, reason):
    return dict(status='not_evaluated', reason=reason, n=len(y), p_phipson_smyth=None)


class CachedProbe:
    """Grouped 10-fold PCA<=30 + logistic with the unsupervised steps fitted once per fold."""

    def __init__(self, X, groups):
        if len(np.unique(groups)) != 10:
            raise InsufficientSupport('The prescribed 10 task folds require 10 active tasks')
        self.folds = []
        for tr, te in GroupKFold(n_splits=10).split(X, groups=groups):
            k = min(30, len(tr) - 1, X.shape[1])
            scaler = StandardScaler().fit(X[tr])
            pca = PCA(n_components=k, random_state=0).fit(scaler.transform(X[tr]))
            self.folds.append((tr, te, pca.transform(scaler.transform(X[tr])), pca.transform(scaler.transform(X[te]))))
        self.n = len(X)

    def scores(self, y):
        out = np.zeros(self.n)
        for tr, te, ztr, zte in self.folds:
            if len(np.unique(y[tr])) != 2:
                raise InsufficientSupport('A prescribed training fold lacks both condition labels')
            out[te] = LogisticRegression(C=1.0, max_iter=2000).fit(ztr, y[tr]).decision_function(zte)
        return out

    def auroc(self, y):
        return float(roc_auc_score(y, self.scores(y)))


def permute_pairs(y, pairs, rng):
    """Swap the two labels of each matched clean/situation pair with probability 1/2."""
    y = y.copy()
    for pid in np.unique(pairs):
        idx = np.where(pairs == pid)[0]
        if len(idx) == 2 and rng.random() < 0.5:
            y[idx] = y[idx[::-1]]
    return y


def null_test(X, y, g, pairs, seed, n_perm=N_PERM):
    if n_perm < 1:
        raise ValueError('n_perm must be positive')
    reason = support_reason(y, g)
    if reason:
        return unavailable(y, reason)
    probe = CachedProbe(X, g)
    obs = probe.auroc(y)
    rng = np.random.default_rng(seed)
    try:
        null = np.array([probe.auroc(permute_pairs(y, pairs, rng)) for _ in range(n_perm)])
    except InsufficientSupport as error:
        return unavailable(y, 'A prescribed permutation cannot be fitted: ' + str(error))
    counts = np.unique(pairs, return_counts=True)[1]
    return {"status": "evaluated", "auroc": obs, "n": int(len(y)), "n_perm": n_perm, "matched_pairs": int((counts == 2).sum()),
            "unpaired_rows": int((counts == 1).sum()), "null_95": float(np.quantile(null, .95)),
            "null_max": float(null.max()), "p_phipson_smyth": float((1 + (null >= obs).sum()) / (1 + n_perm))}


def run(cell):
    suite, policy = cell
    rows = load_rows(suite, policy)
    stored_path = c.A / f'realistic_probe_{suite}_{policy}.json'
    stored = json.loads(stored_path.read_text())['A'] if stored_path.exists() else {}
    out = {}
    with threadpool_limits(limits=2):
        for i, cond in enumerate(GOAL):
            sub, y, g, pairs = subset(rows, cond)
            X = np.stack([r["handoff"] for r in sub]) if sub else np.empty((0, 0))
            res = null_test(X, y, g, pairs, seed=c.SEED + i)
            res["paper_auroc"] = stored.get(cond, {}).get("handoff_auroc")
            res["paper_null_max_20perm"] = stored.get(cond, {}).get("null_max")
            out[cond] = res
    return suite, policy, out


def main():
    cells = [(s, p) for s in c.SUITES for p in c.POLICIES]
    with Pool(2, maxtasksperchild=1) as pool:
        results = pool.map(run, cells)
    by_cell = {(s, p): out for s, p, out in results}
    tests = [(s, p, cond, by_cell.get((s, p), {}).get(cond, unavailable([], 'No result for the planned test')))
             for s, p in cells for cond in GOAL]
    ps = [r['p_phipson_smyth'] if r.get('status') == 'evaluated' else 1.0 for *_, r in tests]
    for (_, _, _, r), raw, adj in zip(tests, ps, c.holm(ps)):
        r['holm_input_p'] = raw
        r['holm_p'] = adj if r.get('status') == 'evaluated' else None
    path = c.write("e9_probe_null", {
        "status": "999-permutation null for scene-change decoding",
        "p_value": "Phipson-Smyth (1 + b) / (1 + 999); Holm over the 12 goal-situation tests",
        "permutation": "labels exchanged within matched clean/situation initial-state pairs (swap with probability 1/2)",
        "tests": [{"suite": s, "policy": p, "situation": cond, **r} for s, p, cond, r in tests]},
        [path for s, p in cells if (path := c.A / f"realistic_probe_{s}_{p}.json").exists()])
    print(path)
    for s, p, cond, r in tests:
        if r['status'] != 'evaluated':
            print(f'{s:7s} {p:8s} {cond:9s} not evaluated: {r["reason"]}')
            continue
        prior = '--' if r['paper_auroc'] is None else format(r['paper_auroc'], '.3f')
        print(f"{s:7s} {p:8s} {cond:9s} AUROC {r['auroc']:.3f} (paper {prior}) null95 {r['null_95']:.3f} "
              f"max {r['null_max']:.3f} p={r['p_phipson_smyth']:.4f} holm={r['holm_p']:.4f}")


if __name__ == "__main__":
    main()
