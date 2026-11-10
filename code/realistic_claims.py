#!/usr/bin/env python3
"""Definite answers to the questions the within-situation tables leave open.

Reads data/icassp2027/analysis/realistic_<suite>_<family>_<policy>.json (per-episode
predictions of every fold) and the episode records, and prints, per policy and situation:

  A. same-sample landmark change: step-40 AUROC recomputed on exactly the episodes that are
     still active at step 80, next to the step-80 AUROC, with a paired Wilcoxon test over tasks
  B. paired readout comparisons at step 80: Handoff-Baseline, Both-Baseline, Handoff-Spread,
     per-task differences with a two-sided sign test
  C. is any readout above chance: one-sided sign test over tasks (AUROC > 0.5)
  D. moved-goal intervention records: clearance satisfied fraction, realised displacement

Verdicts use alpha = 0.05 and are printed as YES / NO. Writes paper/tables/claims_<suite>.txt.
"""
import glob
import json
import gen_realistic_tables as gt
import statistics as st
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
A = ROOT / "data/icassp2027/analysis"
D = ROOT / "data/icassp2027/realistic"
POLICIES = ["pi05", "gr00t", "molmoact"]
FAMILIES = ["occluder", "lighting", "moved", "relocated"]
METHODS = ["native_spread", "baseline", "handoff", "combined"]
ALPHA = 0.05


def auroc(pos, neg):
    if not pos or not neg:
        return None
    return sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))


def sign_test(diffs, alternative="two-sided"):
    d = [x for x in diffs if x != 0]
    n, k = len(d), sum(x > 0 for x in d)
    if n == 0:
        return 1.0
    upper = sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n
    if alternative == "greater":
        return upper
    lower = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * min(upper, lower))


def wilcoxon_p(diffs):
    """Exact paired Wilcoxon signed-rank (two-sided) for small n, via enumeration."""
    d = [x for x in diffs if x != 0]
    n = len(d)
    if n == 0:
        return 1.0
    # midranks for tied |d| (2026-10-13 correction; ranks were assigned by index order before), and the
    # exact null enumerated over the actual rank multiset, so ties change the distribution as they should
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = {}
    j = 0
    while j < n:
        k = j
        while k + 1 < n and abs(d[order[k + 1]]) == abs(d[order[j]]):
            k += 1
        for i in order[j:k + 1]:
            ranks[i] = (j + 1 + k + 1) / 2
        j = k + 1
    w_pos = sum(ranks[i] for i, x in enumerate(d) if x > 0)
    dist = {0.0: 1}
    for r in ranks.values():
        nxt = dict(dist)
        for w, c in dist.items():
            nxt[w + r] = nxt.get(w + r, 0) + c
        dist = nxt
    denom = 2 ** n
    tail = min(sum(c for w, c in dist.items() if w <= w_pos), sum(c for w, c in dist.items() if w >= w_pos)) / denom
    return min(1.0, 2 * tail)


def folds_at(analysis, landmark):
    v = next(x for x in analysis["landmarks"] if x["landmark"] == landmark)
    return {f["test_task"]: f for f in v.get("folds", []) if "methods" in f}


def task_auroc(fold, method, keep=None):
    preds = fold["methods"][method]["predictions"]
    if keep is not None:
        preds = [p for p in preds if p["id"] in keep]
    return auroc([p["score"] for p in preds if p["failure"]], [p["score"] for p in preds if not p["failure"]])


def analyse(suite):
    out = []
    say = out.append
    for policy in POLICIES:
        for fam in FAMILIES:
            path = A / f"realistic_{suite}_{fam}_{policy}.json"
            if not path.exists():
                continue
            a = json.load(open(path))
            f40, f80 = folds_at(a, 40), folds_at(a, 80)
            say(f"\n=== {suite} / {policy} / {fam} ===")
            # prevalence rule of the protocol: 10-80% failures and >= 4 identifiable tasks, else counts only
            nf, n = gt.cohort_counts(suite, policy, fam)
            ident = min(len([t for t in f.values() if task_auroc(t, "handoff") is not None]) for f in (f40, f80))
            if not gt.evaluable(nf, n, ident):
                say(f"RULE failures {nf}/{n}, identifiable tasks {ident}: outside the prevalence rule, counts only, no claim")
                continue
            # A. same-sample landmark change
            for m in METHODS:
                same40, v80, tasks = [], [], []
                for t, fold in f80.items():
                    if t not in f40:
                        continue
                    ids80 = {p["id"] for p in fold["methods"][m]["predictions"]}
                    a40 = task_auroc(f40[t], m, keep=ids80)
                    a80 = task_auroc(fold, m)
                    if a40 is None or a80 is None:
                        continue
                    same40.append(a40); v80.append(a80); tasks.append(t)
                if not same40:
                    continue
                diffs = [b - c for b, c in zip(v80, same40)]
                p = wilcoxon_p(diffs)
                say(f"A {m:14s} step40(same episodes as step80) median {st.median(same40):.2f} -> step80 median {st.median(v80):.2f}; "
                    f"tasks {len(tasks)}, higher at 80 on {sum(x > 0 for x in diffs)}; Wilcoxon p={p:.3f} -> "
                    f"{'YES, the rise holds on the same episodes' if p < ALPHA and st.median(diffs) > 0 else 'NO significant rise on the same episodes'}")
            # B. paired readout comparisons at step 80
            for x, y, label in (("handoff", "baseline", "Handoff-Baseline"), ("combined", "baseline", "Both-Baseline"),
                                ("handoff", "native_spread", "Handoff-Spread"), ("native_spread", "baseline", "Spread-Baseline")):
                diffs = []
                for t, fold in f80.items():
                    ax, ay = task_auroc(fold, x), task_auroc(fold, y)
                    if ax is not None and ay is not None:
                        diffs.append(ax - ay)
                if not diffs:
                    continue
                p = sign_test(diffs)
                verdict = ("YES, first is better" if p < ALPHA and st.median(diffs) > 0 else
                           "YES, second is better" if p < ALPHA and st.median(diffs) < 0 else "NO difference at alpha 0.05")
                say(f"B {label:16s} step80 mean diff {st.mean(diffs):+.2f}, median {st.median(diffs):+.2f}, positive {sum(d > 0 for d in diffs)}/{len(diffs)}, sign p={p:.3f} -> {verdict}")
            # C. above chance
            for L, ff in ((40, f40), (80, f80)):
                for m in METHODS:
                    vals = [task_auroc(fold, m) for fold in ff.values()]
                    vals = [v for v in vals if v is not None]
                    p = sign_test([v - 0.5 for v in vals], "greater")
                    say(f"C step{L} {m:14s} median {st.median(vals):.2f}, >0.5 on {sum(v > 0.5 for v in vals)}/{len(vals)}, p={p:.3f} -> "
                        f"{'YES above chance' if p < ALPHA else 'NO, not above chance'}")
            # D. intervention records
            if fam in ("moved", "relocated"):
                coll = gt.FAMILY_DIR[fam]
                recs = [json.load(open(f))["intervention"] for f in glob.glob(f"{D}/{suite}_{coll}/{policy}/shard*/*_{fam}/summary.json")]
                recs = [r for r in recs if r]
                ok = sum(r["clearance_satisfied"] for r in recs)
                disp = [r.get("end_of_slide_displacement_m", r["distance_m"]) for r in recs]   # pre-amendment records: the slide sets the pose directly, so the end-of-slide displacement is the nominal distance
                what = "path clear (landing clearance by construction)" if fam == "relocated" else "clearance satisfied"
                say(f"D {fam} records {len(recs)}: {what} {ok}/{len(recs)} ({ok/len(recs):.0%}), "
                    f"end-of-slide displacement min {min(disp):.3f} max {max(disp):.3f} m (nominal {recs[0]['distance_m']:.2f})")
    return "\n".join(out)


if __name__ == "__main__":
    suite = sys.argv[1] if len(sys.argv) > 1 else "spatial"
    text = analyse(suite)
    (ROOT / "paper/tables").mkdir(parents=True, exist_ok=True)
    (ROOT / f"paper/tables/claims_{suite}.txt").write_text(text + "\n")
    print(text)
