#!/usr/bin/env python3
"""E3 step 2: generic image embeddings as a baseline for the handoff (CPU).

(a) Situation decodability (Table 2, column A setting; code/exp_20261020_e9_probe_null.py rows and
    pipeline): out-of-fold AUROC for DINOv2-B and ResNet-18 embeddings of the same two camera images,
    with the 999-permutation matched-pair null of E9, and a task-cluster bootstrap CI of the AUROC
    difference handoff minus DINOv2 (both out-of-fold scores, tasks resampled).
(b) Within-situation failure ranking: the embeddings as additional readouts in the paper's split
    (held-out task t, next task excluded as calibration task, readout fitted on the other eight;
    analyze_icassp_trials.fit_scores with the embedding in place of the handoff, so the pipeline is
    StandardScaler -> PCA<=30 -> StandardScaler -> logistic C=1). Every task with both outcomes among
    active episodes is scored (AUROC needs no threshold). Paired per-task comparisons with the
    handoff, baseline and spread, sign test and task-cluster bootstrap CI, per cell and pooled.
Run: python code/exp_20261020_e3_analysis.py
"""
import json
from multiprocessing import Pool

import numpy as np
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

import analyze_icassp_trials as ait
import exp_20261020_common as c
import exp_20261020_e9_probe_null as e9
from artifact_paths import resolve_recorded_path
from exp_20261020_e2_calibration import view_roots

EMB = ("dinov2", "resnet18")
FEAT = c.OUT / "e3_features"
READOUTS = ["native_spread", "baseline", "handoff", "dinov2", "resnet18"]


def features(suite, policy):
    table = {}
    for coll in ("main", "main2", "main3"):
        path = FEAT / f"{suite}_{coll}_{policy}.npz"
        if path.exists():
            with np.load(path) as z:   # read each array once; indexing the lazy archive per row copies it every time
                arrays = {e: z[e] for e in EMB}
                keys = list(zip(z["id"].tolist(), z["query"].tolist()))
            for i, (d, q) in enumerate(keys):
                table[str(resolve_recorded_path(d)), int(q)] = {e: arrays[e][i] for e in EMB}
    return table


def boot_auc_diff(y, s1, s2, groups, n_boot=10000, seed=c.SEED):
    rng = np.random.default_rng(seed)
    tasks = np.unique(groups)
    idx = {t: np.where(groups == t)[0] for t in tasks}
    d = []
    for _ in range(n_boot):
        sel = np.concatenate([idx[t] for t in rng.choice(tasks, len(tasks))])
        if len(np.unique(y[sel])) == 2:
            d.append(roc_auc_score(y[sel], s1[sel]) - roc_auc_score(y[sel], s2[sel]))
    return [float(np.quantile(d, .025)), float(np.quantile(d, .975))]


def decodability(cell):
    suite, policy = cell
    rows = e9.load_rows(suite, policy)
    feat = features(suite, policy)
    out = {}
    with threadpool_limits(limits=2):
        for i, cond in enumerate(("occluder", "lighting", "moved", "relocated")):
            sub, y, g, pairs = e9.subset(rows, cond)
            reason = e9.support_reason(y, g)
            if reason:
                out[cond] = e9.unavailable(y, reason)
                continue
            missing = sum((r["id"], r["query"]) not in feat for r in sub)
            if missing:
                raise ValueError(f"{suite}/{policy}/{cond}: {missing} rows without embeddings")
            res = {"status": "evaluated", "n": int(len(y))}
            scores = {}
            try:
                for name, X in (("handoff", np.stack([r["handoff"] for r in sub])),
                                ("image_stats", np.stack([r["image_stats"] for r in sub])),
                                *((e, np.stack([feat[r["id"], r["query"]][e] for r in sub])) for e in EMB)):
                    probe = e9.CachedProbe(X, g)
                    scores[name] = probe.scores(y)
                    res[f"{name}_auroc"] = float(roc_auc_score(y, scores[name]))
                    if name in EMB and cond in e9.GOAL:
                        rng = np.random.default_rng(c.SEED + i)
                        null = np.array([probe.auroc(e9.permute_pairs(y, pairs, rng)) for _ in range(e9.N_PERM)])
                        res[f"{name}_p_phipson_smyth"] = float((1 + (null >= res[f"{name}_auroc"]).sum()) / (1 + e9.N_PERM))
                for e in EMB:
                    res[f"handoff_minus_{e}_ci95"] = boot_auc_diff(y, scores["handoff"], scores[e], g)
            except e9.InsufficientSupport as error:
                res = e9.unavailable(y, 'A prescribed permutation cannot be fitted: ' + str(error))
            out[cond] = res
    return suite, policy, out


def task_aurocs(cell):
    """Per-task held-out AUROC for every readout, paper split, all identifiable tasks."""
    suite, family, policy, path = cell
    analysis = json.load(open(path))
    trials, _, hashes, _ = ait.load_trials(view_roots(analysis), (80,))
    feat = features(suite, policy)
    tasks = sorted({t["summary"]["task"] for t in trials})
    rows = []
    for t in trials:
        if 80 in t["rows"]:
            r = t["rows"][80]
            r.update(feat[r["id"], r["query"]])
            rows.append(r)
    res = {}
    with threadpool_limits(limits=2):
        for i, t in enumerate(tasks):
            cal = tasks[(i + 1) % len(tasks)]
            test = [r for r in rows if r["task"] == t]
            train = [r for r in rows if r["task"] not in (t, cal)]
            y = [r["failure"] for r in test]
            if len(set(y)) < 2 or len({r["failure"] for r in train}) < 2:
                continue
            cal_rows = [r for r in rows if r["task"] == cal] or test
            res[t] = {"native_spread": float(roc_auc_score(y, [r["spread"] for r in test]))}
            for m in ("baseline", "handoff"):
                res[t][m] = float(roc_auc_score(y, ait.fit_scores(train, cal_rows, test, m)[1]))
            for e in EMB:
                swap = lambda rs: [dict(r, handoff=r[e]) for r in rs]
                res[t][e] = float(roc_auc_score(y, ait.fit_scores(swap(train), swap(cal_rows), swap(test), "handoff")[1]))
    return suite, family, policy, res


def compare(per_task, a, b):
    d = [v[a] - v[b] for v in per_task.values()]
    return {"n_tasks": len(d), "positive": sum(x > 0 for x in d), "mean_diff": float(np.mean(d)),
            "ci95": c.cluster_bootstrap([d]), "sign_p": c.sign_test(d), "diffs": d}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=2, help="worker processes (each holds one cell's handoff matrix)")
    jobs = parser.parse_args().jobs
    if jobs < 1:
        parser.error('--jobs must be positive')
    probe_cells = [(s, p) for s in c.SUITES for p in c.POLICIES]
    within_cells = [(s, f, p, str(path)) for s, f, p, L, path in c.evaluable_cells((80,))]
    with Pool(jobs, maxtasksperchild=1) as pool:
        dec = pool.map(decodability, probe_cells)
        wit = pool.map(task_aurocs, within_cells)
    within = []
    for s, f, p, per_task in wit:
        if not per_task:
            continue
        cell = {"suite": s, "situation": f, "policy": p, "per_task": {str(k): v for k, v in per_task.items()},
                "median_auroc": {m: float(np.median([v[m] for v in per_task.values()])) for m in READOUTS},
                "above_chance": {m: [sum(v[m] > .5 for v in per_task.values()), len(per_task),
                                     c.sign_test([v[m] - .5 for v in per_task.values()], "greater")] for m in READOUTS},
                "comparisons": {f"{a}-{b}": compare(per_task, a, b) for a, b in
                                (("handoff", "dinov2"), ("dinov2", "baseline"), ("dinov2", "native_spread"), ("handoff", "resnet18"))}}
        within.append(cell)
    pooled = {}
    comparisons = ('handoff-dinov2', 'dinov2-baseline', 'dinov2-native_spread', 'handoff-resnet18')
    for comp in comparisons:
        for scope in ("occluder", "moved", "all"):
            strata = [w["comparisons"][comp]["diffs"] for w in within if scope == "all" or w["situation"] == scope]
            if not strata:
                continue
            v = np.concatenate(strata)
            # as in E4: one cluster per suite x task (mean over policies/situations), resampled within suite
            by_task = {}
            for w in within:
                if scope == "all" or w["situation"] == scope:
                    for t, d in zip(w["per_task"], w["comparisons"][comp]["diffs"]):
                        by_task.setdefault((w["suite"], t), []).append(d)
            suite_strata = [g for g in ([np.mean(v_) for (s, _), v_ in by_task.items() if s == su] for su in c.SUITES) if g]
            tmeans = np.concatenate(suite_strata)
            pooled[f"{comp}|{scope}"] = {"cells": len(strata), "n_tasks": len(v), "positive": int((v > 0).sum()),
                                         "mean_diff": float(v.mean()), "ci95": c.cluster_bootstrap(strata),
                                         "sign_p": c.sign_test(list(v)),
                                         "task_clusters": len(tmeans), "task_cluster_positive": int((tmeans > 0).sum()),
                                         "task_cluster_mean": float(tmeans.mean()),
                                         "task_cluster_ci95": c.cluster_bootstrap(suite_strata),
                                         "task_cluster_sign_p": c.sign_test(list(tmeans))}
    family = [f'{comp}|{scope}' for comp in comparisons for scope in ('occluder', 'moved')]
    ps = [pooled[k]['task_cluster_sign_p'] if k in pooled else 1.0 for k in family]
    for k, adj in zip(family, c.holm(ps)):
        if k in pooled:
            pooled[k]['task_cluster_holm_p'] = adj
    manifest = json.loads((FEAT / "manifest.json").read_text())
    path = c.write("e3_image_embedding", {
        "status": "post hoc image-embedding baseline",
        "embeddings": {k: manifest[k] for k in ("dinov2", "resnet18", "precision")},
        "feature_extraction_provenance": manifest["provenance"],
        "decodability": [{"suite": s, "policy": p, "situations": out} for s, p, out in dec],
        "within_situation_step80": within, "pooled_comparisons": pooled},
        [p for *_, p in within_cells] + [FEAT / n for n in manifest["files"]] + [FEAT / "manifest.json"])
    print(path)
    for s, p, out in dec:
        for cond, r in out.items():
            if r.get('status') == 'not_evaluated':
                print(f'DEC {s:7s} {p:8s} {cond:9s} not evaluated: {r["reason"]}')
                continue
            print(f"DEC {s:7s} {p:8s} {cond:9s} handoff {r['handoff_auroc']:.2f} dino {r['dinov2_auroc']:.2f} "
                  f"resnet {r['resnet18_auroc']:.2f} imgstats {r['image_stats_auroc']:.2f} "
                  f"h-dino CI [{r['handoff_minus_dinov2_ci95'][0]:+.2f},{r['handoff_minus_dinov2_ci95'][1]:+.2f}]"
                  + (f" dino p={r['dinov2_p_phipson_smyth']:.3f}" if "dinov2_p_phipson_smyth" in r else ""))
    for w in within:
        med = " ".join(f"{m[:6]} {w['median_auroc'][m]:.2f}" for m in READOUTS)
        hd = w["comparisons"]["handoff-dinov2"]
        print(f"WIT {w['suite']:7s} {w['situation']:9s} {w['policy']:8s} {med} | dino>0.5 {w['above_chance']['dinov2'][0]}/{w['above_chance']['dinov2'][1]} "
              f"p={w['above_chance']['dinov2'][2]:.3f} | h-dino {hd['positive']}/{hd['n_tasks']} {hd['mean_diff']:+.3f} "
              f"[{hd['ci95'][0]:+.3f},{hd['ci95'][1]:+.3f}] p={hd['sign_p']:.3f}")
    for k, v in pooled.items():
        print(f"POOLED {k:30s} {v['positive']}/{v['n_tasks']} {v['mean_diff']:+.3f} [{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}] p={v['sign_p']:.3g} | "
              f"task clusters {v['task_cluster_positive']}/{v['task_clusters']} {v['task_cluster_mean']:+.3f} "
              f"[{v['task_cluster_ci95'][0]:+.3f},{v['task_cluster_ci95'][1]:+.3f}] p={v['task_cluster_sign_p']:.3g} holm={v.get('task_cluster_holm_p', float('nan')):.3g}")


if __name__ == "__main__":
    main()
