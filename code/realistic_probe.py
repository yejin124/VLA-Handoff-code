#!/usr/bin/env python3
"""Representation-level questions answered on the closed-loop cohorts themselves.

For each policy and suite, from the stored per-query records of the main cohorts:

  A. Is the situation decodable from the handoff?  Out-of-fold AUROC of a logistic probe
     (PCA<=30 fitted on the training tasks, task-grouped 10-fold) for clean vs each situation,
     next to a 4-feature image-statistics baseline and a within-task label-permutation null.
     Camera situations are read at query 0 (step 0: identical robot state, only the image
     differs); goal situations at the first query at or after step 40 (after the move).
  B. Is the native spread decodable from the handoff?  Ridge probe, same split, out-of-fold R^2,
     with a within-task permutation null, pooled over conditions and within each condition.
  C. Does the native spread respond to the situation?  Paired by initial state, spread under the
     situation minus spread under clean at the same query step; two-sided exact-free Wilcoxon
     (scipy) and the median relative change.

Writes data/icassp2027/analysis/realistic_probe_<suite>_<policy>.json and prints verdicts.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data/icassp2027/realistic"
A = ROOT / "data/icassp2027/analysis"
POLICIES = ["pi05", "gr00t", "molmoact"]
CAMERA = ("occluder", "lighting")
GOAL = ("moved", "relocated")
ALPHA = 0.05
N_PERM = 20


def collections(suite):
    """main always; main2 (lighting 1.25) and main3 (relocated, amendment 2026-10-13) only once the
    cohort is complete in the data (realistic_views.cohort_complete; a chain-log line is not trusted
    because the chain writes "done" after a FAILED step too). The relocated episodes of <suite>_main2
    are superseded and never read."""
    from realistic_views import cohort_complete
    return ["main"] + [c for c in ("main2", "main3") if cohort_complete(suite, c)]


def load_rows(suite, policy):
    rows = []
    for coll in collections(suite):
        for sdir in sorted(glob.glob(f"{D}/{suite}_{coll}/{policy}/shard*/t*_e*_r0_*")):
            s = Path(sdir) / "summary.json"
            if not s.exists():
                continue
            summ = json.load(open(s))
            if not summ.get("complete"):
                continue
            if coll == "main" and summ["condition"] == "lighting":   # the dark rungs (1.4/1.5): supplement only, not probed
                continue
            if coll == "main2" and summ["condition"] == "relocated":  # superseded by main3
                continue
            queries = [json.loads(l) for l in open(Path(sdir) / "queries.jsonl")]
            # query 0 and the first query at or after step 40
            picks = {"q0": queries[0]}
            later = [q for q in queries if q["step"] >= 40]
            if later:
                picks["q40"] = later[0]
            for tag, q in picks.items():
                z = np.load(Path(sdir) / q["artifact"])
                rows.append({"task": summ["task"], "episode": summ["episode"], "condition": summ["condition"],
                             "tag": tag, "step": q["step"], "spread": q["spread"], "failure": not summ["success"],
                             "handoff": z["handoff"].astype(np.float32), "image_stats": np.array(q["image_stats"], dtype=np.float32)})
    return rows


def oof_auroc(X, y, groups, seed=0):
    """Out-of-fold AUROC, PCA<=30 + logistic, grouped 10-fold; returns pooled OOF AUROC."""
    scores = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=10).split(X, y, groups):
        k = min(30, len(tr) - 1, X.shape[1])
        model = make_pipeline(StandardScaler(), PCA(n_components=k, random_state=seed), LogisticRegression(C=1.0, max_iter=2000))
        model.fit(X[tr], y[tr])
        scores[te] = model.decision_function(X[te])
    return float(roc_auc_score(y, scores))


def oof_r2(X, y, groups, seed=0):
    pred = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=10).split(X, y, groups):
        k = min(30, len(tr) - 1, X.shape[1])
        model = make_pipeline(StandardScaler(), PCA(n_components=k, random_state=seed), Ridge(alpha=10.0))
        model.fit(X[tr], y[tr])
        pred[te] = model.predict(X[te])
    return float(r2_score(y, pred))


def permute_within_group(y, groups, rng):
    y = y.copy()
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        y[idx] = rng.permutation(y[idx])
    return y


def analyse(suite, policy):
    rows = load_rows(suite, policy)
    if not rows:
        return None
    conditions = sorted({r["condition"] for r in rows} - {"clean"})
    out = {"suite": suite, "policy": policy, "n_rows": len(rows), "A": {}, "B": {}, "C": {}}
    rng = np.random.default_rng(0)
    # ---- A: situation decodable from handoff (clean vs situation)
    for cond in conditions:
        tag = "q0" if cond in CAMERA else "q40"
        sub = [r for r in rows if r["tag"] == tag and r["condition"] in ("clean", cond)]
        if len({r["condition"] for r in sub}) < 2:
            continue
        X = np.stack([r["handoff"] for r in sub]); Xi = np.stack([r["image_stats"] for r in sub])
        y = np.array([r["condition"] == cond for r in sub], dtype=int); g = np.array([r["task"] for r in sub])
        if len(np.unique(g)) < 10:
            out.setdefault('not_evaluated', {})['A/' + cond] = 'The prescribed 10 task folds require 10 active tasks'
            continue
        auc = oof_auroc(X, y, g)
        auc_img = oof_auroc(Xi, y, g)
        null = [oof_auroc(X, permute_within_group(y, g, rng), g, seed=i) for i in range(N_PERM)]
        out["A"][cond] = {"read_at": tag, "n": int(len(y)), "handoff_auroc": auc, "image_stats_auroc": auc_img,
                          "null_95": float(np.percentile(null, 95)), "null_max": float(max(null)),
                          "verdict": "YES decodable" if auc > max(null) else "NO, within the permutation null"}
    # ---- B: spread decodable from handoff
    for name, sel in [("pooled", lambda r: r["tag"] == "q0")] + [(c, (lambda r, c=c: r["condition"] == c and r["tag"] == ("q0" if c in CAMERA else "q40"))) for c in ["clean"] + conditions]:
        sub = [r for r in rows if sel(r)]
        if len(sub) < 40:
            continue
        X = np.stack([r["handoff"] for r in sub]); y = np.array([r["spread"] for r in sub]); g = np.array([r["task"] for r in sub])
        if len(np.unique(g)) < 10:
            out.setdefault('not_evaluated', {})['B/' + name] = 'The prescribed 10 task folds require 10 active tasks'
            continue
        r2 = oof_r2(X, y, g)
        null = [oof_r2(X, permute_within_group(y, g, rng), g, seed=i) for i in range(N_PERM)]
        out["B"][name] = {"n": int(len(y)), "r2": r2, "null_95": float(np.percentile(null, 95)), "null_max": float(max(null)),
                          "verdict": "YES decodable" if r2 > 0 and r2 > max(null) else "NO, not decodable (R2 <= 0 or within the permutation null)"}
    # ---- C: spread responds to the situation (paired by initial state)
    for cond in conditions:
        tag = "q0" if cond in CAMERA else "q40"
        clean = {(r["task"], r["episode"]): r["spread"] for r in rows if r["condition"] == "clean" and r["tag"] == tag}
        pairs = [(clean[(r["task"], r["episode"])], r["spread"]) for r in rows
                 if r["condition"] == cond and r["tag"] == tag and (r["task"], r["episode"]) in clean]
        if len(pairs) < 10:
            continue
        a = np.array([p[0] for p in pairs]); b = np.array([p[1] for p in pairs])
        stat = wilcoxon(b, a, alternative="two-sided")
        rel = np.median((b - a) / np.maximum(a, 1e-9))
        out["C"][cond] = {"read_at": tag, "n_pairs": int(len(pairs)), "median_clean": float(np.median(a)), "median_situation": float(np.median(b)),
                          "median_relative_change": float(rel), "higher_in": int(np.sum(b > a)), "wilcoxon_p": float(stat.pvalue),
                          "verdict": ("YES, spread rises" if stat.pvalue < ALPHA and rel > 0 else "YES, spread falls" if stat.pvalue < ALPHA else "NO change")}
    return out


def main():
    A.mkdir(parents=True, exist_ok=True)
    suite = sys.argv[1] if len(sys.argv) > 1 else "spatial"
    lines = []
    for policy in POLICIES:
        res = analyse(suite, policy)
        if res is None:
            continue
        (A / f"realistic_probe_{suite}_{policy}.json").write_text(json.dumps(res, indent=2) + "\n")
        lines.append(f"\n=== {suite} / {policy} ({res['n_rows']} query rows) ===")
        for c, v in res["A"].items():
            lines.append(f"A situation {c:9s} @{v['read_at']}: handoff AUROC {v['handoff_auroc']:.3f} | image-stats {v['image_stats_auroc']:.3f} | null max {v['null_max']:.3f} -> {v['verdict']}")
        for c, v in res["B"].items():
            lines.append(f"B spread R2 {c:9s}: {v['r2']:+.3f} | null max {v['null_max']:+.3f} (n={v['n']}) -> {v['verdict']}")
        for c, v in res["C"].items():
            lines.append(f"C spread vs clean {c:9s} @{v['read_at']}: median {v['median_clean']:.4f} -> {v['median_situation']:.4f} ({v['median_relative_change']:+.0%}), higher in {v['higher_in']}/{v['n_pairs']}, p={v['wilcoxon_p']:.2e} -> {v['verdict']}")
    text = "\n".join(lines)
    (ROOT / "paper/tables").mkdir(parents=True, exist_ok=True)
    (ROOT / f"paper/tables/probe_{suite}.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
