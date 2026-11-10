#!/usr/bin/env python3
"""E2: single-task versus pooled-task threshold calibration (post hoc secondary rules).

Primary (preregistered, the paper's rule): the threshold is the upper 90th percentile of the
success scores of ONE calibration task (the next task in sorted order); the readout is fitted on the
remaining eight tasks. Reproduced here and checked against the frozen analysis JSON.

Secondary (post hoc):
  all_ordered_pairs  the single-task rule with every other task as the calibration task, model
                     fitted without both tasks (shows the full spread of single-task calibration)
  pooled_9_task      the upper 90th percentile of the success scores of all nine other tasks; each
                     other task u is scored out of fold by a model fitted on the eight tasks other
                     than t and u. Task t keeps the paper's scores (model without t and its next task).
  shared_3_task      ONE model, fitted on the six tasks other than t and the next three tasks, scores
                     both those three calibration tasks and t; the threshold uses their pooled successes
                     (calibration and test share one scoring function, unlike pooled_9_task).
The single-task rule also shares one model between its calibration task and t. Median and upper
quartile carry task-bootstrap 95% intervals; calibration-success denominators are reported per rule.
All models use code/analyze_icassp_trials.fit_scores unchanged; native spread needs no fitting.
Every evaluable cell, Spatial and Object, steps 40 and 80, all four readouts.
Run: python code/exp_20261020_e2_calibration.py [--jobs 14] [--only suite/family/policy]
"""
import argparse
import itertools
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import analyze_icassp_trials as ait
import exp_20261020_common as c
from artifact_paths import resolve_recorded_path

K = 3   # calibration tasks of the shared-model rule


def view_roots(analysis):
    """Collector roots the frozen analysis read (each shard's config.json parent's parent)."""
    return sorted({str(resolve_recorded_path(k).parent.parent)
                   for k in analysis["source_sha256"] if k.endswith("config.json")})


def rates(y, s, threshold):
    y, s = np.asarray(y), np.asarray(s)
    alarm = s > threshold
    return (float(alarm[y == 0].mean()) if (y == 0).any() else None,
            float(alarm[y == 1].mean()) if (y == 1).any() else None)


def summary(values, boot=True, n_boot=2000):
    """Quantiles over held-out tasks, with task-bootstrap 95% intervals for the median and upper quartile."""
    v = np.array([x for x in values if x is not None])
    if not len(v):
        return None
    out = {"n": len(v), "median": float(np.median(v)), "q1": float(np.quantile(v, .25)),
           "q3": float(np.quantile(v, .75)), "min": float(v.min()), "max": float(v.max())}
    if boot:
        rng = np.random.default_rng(c.SEED)
        draws = v[rng.integers(0, len(v), (n_boot, len(v)))]
        for name, q in (("median", .5), ("q3", .75)):
            stat = np.quantile(draws, q, axis=1)
            out[f"{name}_ci95"] = [float(np.quantile(stat, .025)), float(np.quantile(stat, .975))]
    return out


def gap(a, b):
    return abs((a if a is not None else -1) - (b if b is not None else -1))


def evaluate(analysis, L, tasks, rows):
    """All three calibration rules for every readout at one landmark."""
    by_task = {t: [r for r in rows if r["task"] == t] for t in tasks}
    labels = {t: np.array([r["failure"] for r in by_task[t]]) for t in tasks}
    pair_scores = {}   # (readout, frozenset{t, u}) -> {task: scores}; model fitted without t and u
    for t, u in itertools.combinations(tasks, 2):
        if not by_task[t] or not by_task[u]:
            continue
        train = [r for r in rows if r["task"] not in (t, u)]   # original row order: randomized PCA depends on it
        if len({r["failure"] for r in train}) < 2:
            continue
        for m in ("baseline", "handoff", "combined"):
            su, st = ait.fit_scores(train, by_task[u], by_task[t], m)
            pair_scores[m, frozenset((t, u))] = {u: su, t: st}
    stored = c.folds(analysis, L)
    res = {}
    for m in c.METHODS:
        def scores(task, excluded):
            if m == "native_spread":
                return np.array([r["spread"] for r in by_task[task]])
            return pair_scores.get((m, frozenset((task, excluded))), {}).get(task)
        single, pairs, pooled, shared, check = [], [], [], [], []
        for i, t in enumerate(tasks):
            if not by_task[t]:
                continue
            cal = tasks[(i + 1) % len(tasks)]
            st = scores(t, cal)
            if st is not None and (labels[cal] == 0).any():
                fpr, tpr = rates(labels[t], st, ait.calibrate(labels[cal], scores(cal, t)))
                single.append((t, fpr, tpr, int((labels[cal] == 0).sum())))
                paper = stored.get(t, {}).get("methods", {}).get(m, {}).get("test", {})
                if "fpr" in paper:
                    check.append(max(gap(fpr, paper["fpr"]), gap(tpr, paper["tpr"])))
            for u in tasks:
                if u != t and by_task[u] and (labels[u] == 0).any() and scores(t, u) is not None:
                    pairs.append(rates(labels[t], scores(t, u), ait.calibrate(labels[u], scores(u, t))))
            neg = [x for u in tasks if u != t and by_task[u] and scores(u, t) is not None
                   for x in np.asarray(scores(u, t))[labels[u] == 0]]
            if st is not None and neg:
                thr = float(np.quantile(neg, 0.9, method="higher"))
                fpr, tpr = rates(labels[t], st, thr)
                pooled.append((t, fpr, tpr, float(np.mean(np.array(neg) > thr)), len(neg)))
            # shared rule: ONE model fitted without t and its next K tasks scores both the K calibration tasks and t
            cal_k = [tasks[(i + j) % len(tasks)] for j in range(1, K + 1)]
            cal_rows = [r for r in rows if r["task"] in cal_k]
            train = [r for r in rows if r["task"] != t and r["task"] not in cal_k]
            y_cal = np.array([r["failure"] for r in cal_rows])
            if (y_cal == 0).any() and len({r["failure"] for r in train}) == 2:
                if m == "native_spread":
                    s_cal, s_t = np.array([r["spread"] for r in cal_rows]), np.array([r["spread"] for r in by_task[t]])
                else:
                    s_cal, s_t = ait.fit_scores(train, cal_rows, by_task[t], m)
                fpr, tpr = rates(labels[t], s_t, ait.calibrate(y_cal, s_cal))
                shared.append((t, fpr, tpr, int((y_cal == 0).sum())))
        res[m] = {
            "single_task_rule": {"fpr": summary([x[1] for x in single]), "tpr": summary([x[2] for x in single]),
                                 "calibration_successes": summary([x[3] for x in single], boot=False),
                                 "per_task": [{"task": t, "fpr": f, "tpr": p, "calibration_successes": n} for t, f, p, n in single]},
            f"shared_{K}_task_rule": {"fpr": summary([x[1] for x in shared]), "tpr": summary([x[2] for x in shared]),
                                      "calibration_successes": summary([x[3] for x in shared], boot=False),
                                      "per_task": [{"task": t, "fpr": f, "tpr": p, "calibration_successes": n} for t, f, p, n in shared]},
            "all_ordered_pairs": {"fpr": summary([x[0] for x in pairs], boot=False), "tpr": summary([x[1] for x in pairs], boot=False),
                                  "n_pairs": len(pairs)},
            "pooled_9_task_rule": {"fpr": summary([x[1] for x in pooled]), "tpr": summary([x[2] for x in pooled]),
                                   "calibration_fpr": summary([x[3] for x in pooled]),
                                   "calibration_successes": summary([x[4] for x in pooled], boot=False),
                                   "per_task": [{"task": t, "fpr": f, "tpr": p} for t, f, p, _, _ in pooled]},
            "max_abs_deviation_from_frozen_single_rule": float(max(check)) if check else None}
    return res


def run_cell(cell):
    suite, family, policy, path, landmarks = cell
    analysis = json.load(open(path))
    trials, _, hashes, _ = ait.load_trials(view_roots(analysis), tuple(landmarks))
    tasks = sorted({t["summary"]["task"] for t in trials})
    out = {"suite": suite, "situation": family, "policy": policy, "landmarks": {}}
    with threadpool_limits(limits=2):
        for L in landmarks:
            rows = [t["rows"][L] for t in trials if L in t["rows"]]
            out["landmarks"][L] = evaluate(analysis, L, tasks, rows)
    out["raw_inputs"] = len(hashes)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--only", help="suite/family/policy, for a timing run (output not written)")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    grouped = {}
    for s, f, p, L, path in c.evaluable_cells((40, 80)):
        grouped.setdefault((s, f, p, str(path)), []).append(L)
    cells = [(*k, sorted(v)) for k, v in sorted(grouped.items())]
    if args.only:
        cells = [x for x in cells if "/".join(x[:3]) == args.only]
    results = []
    if cells:
        with Pool(min(args.jobs, len(cells)), maxtasksperchild=1) as pool:
            results = pool.map(run_cell, cells)
    if not args.only:
        path = c.write("e2_calibration", {
            "status": "post hoc secondary calibration rules; the single next-task rule is the original prespecified analysis",
            "threshold": "strict score > upper (method='higher') 90th percentile of calibration successes; nominal FPR 0.10",
            "rules": {"single_task_rule": "paper's rule, next task in sorted order",
                      "all_ordered_pairs": "every other task as calibration task; model fitted without test and calibration task",
                      "pooled_9_task_rule": "successes of all 9 non-test tasks, each scored by a model fitted without it and the test task",
                      "shared_3_task_rule": "one model fitted on the 6 tasks other than the test task and the next 3 tasks scores those 3 calibration tasks and the test task; threshold from the 3 tasks' successes"},
            "uncertainty": "task-bootstrap (2000 resamples of held-out tasks, seed 20261020) 95% intervals for the median and upper-quartile FPR/TPR",
            "cells": results}, [x[3] for x in cells] + [Path(ait.__file__)])
        print(path)
    f = lambda x: f"{x['median']:.2f} ({x['q1']:.2f}-{x['q3']:.2f}) max {x['max']:.2f}" if x else "--"
    for r in results:
        for L, res in r["landmarks"].items():
            for m, v in res.items():
                print(f"{r['suite']:7s} {r['situation']:9s} {r['policy']:8s} {L} {m:13s} dev={v['max_abs_deviation_from_frozen_single_rule']} "
                      f"| single {f(v['single_task_rule']['fpr'])} | shared3 {f(v['shared_3_task_rule']['fpr'])} | pairs {f(v['all_ordered_pairs']['fpr'])} "
                      f"| pooled {f(v['pooled_9_task_rule']['fpr'])} TPR {v['pooled_9_task_rule']['tpr']['median'] if v['pooled_9_task_rule']['tpr'] else float('nan'):.2f}")


if __name__ == "__main__":
    main()
