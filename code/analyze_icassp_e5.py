"""E5 analysis: STAC-style temporal consistency and a backbone probe on overlapping chunks.

Reuses analyze_icassp_trials.py for loading, validation, task-held-out splits,
thresholds and the fixed logistic probe. Adds three methods on the same folds:
`stac_style` (RBF-kernel MMD^2 between the J-draw sets of consecutive queries on
their overlapping steps; bandwidth = median heuristic on the training split only;
higher = more inconsistent), `stac_l2` (L2 between the J-mean chunks on the same
overlap; secondary) and `backbone` (the handoff PCA<=30 + C=1 logistic probe applied
to `backbone_pooled`). "STAC-style"/"SAFE-style" only; no official reproduction.
"""
import argparse
import json
from pathlib import Path
import platform
import sys

import numpy as np
import sklearn
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_icassp_trials as base  # noqa: E402

E5_KEYS = ("planned_samples", "planned_actions", "action_samples", "executed_actions", "backbone")


def load_npz(directory, query, hashes):
    """Hash-checked read of one E5 artifact; validates planned/executed consistency."""
    name = query["artifact"]
    if not isinstance(name, str) or Path(name).name != name or not name.endswith(".npz"):
        raise ValueError("Unsafe artifact path")
    path = (directory / name).resolve()
    if path.parent != directory.resolve():
        raise ValueError("Artifact escapes trial directory")
    base.record(path, hashes)
    if hashes[str(path)] != query["artifact_sha256"]:
        raise ValueError(f"Artifact digest mismatch: {path}")
    with np.load(path, allow_pickle=False) as source:
        data = {key: base.finite(source[key], key) for key in E5_KEYS}
    planned, samples = data["planned_actions"], data["planned_samples"]
    executed = int(query["executed_steps"])
    if planned.ndim != 2 or planned.shape[1] != 7 or samples.shape != (query["draws"], *planned.shape):
        raise ValueError("Invalid planned chunk dimensions")
    if not 1 <= executed <= len(planned) or query.get("planned_steps") != len(planned):
        raise ValueError("Executed prefix inconsistent with planned chunk")
    if (not np.array_equal(planned, samples[0])
            or not np.array_equal(data["executed_actions"], planned[:executed])
            or not np.array_equal(data["action_samples"], samples[:, :executed])):
        raise ValueError("Executed prefix disagrees with planned draws")
    if data["backbone"].ndim != 1 or len(data["backbone"]) < 2:
        raise ValueError("Invalid backbone vector")
    return data


def stac_pair(directory, queries, landmark, hashes):
    """Last consecutive pair (q, q+1) whose overlap window ends at or before the landmark."""
    chosen = None
    for q in range(len(queries) - 1):
        end = queries[q]["step"] + queries[q]["planned_steps"] - 1
        if end <= landmark:
            chosen = q
    if chosen is None:
        raise ValueError(f"No consecutive query pair completes before landmark {landmark}")
    first, second = queries[chosen], queries[chosen + 1]
    executed = int(first["executed_steps"])
    overlap = int(first["planned_steps"]) - executed
    if overlap < 1 or second["planned_steps"] < overlap:
        raise ValueError("Consecutive queries do not overlap")
    if second["step"] != first["step"] + executed:
        raise ValueError("Query steps do not match the executed prefix")
    a = load_npz(directory, first, hashes)["planned_samples"][:, executed:]
    b = load_npz(directory, second, hashes)["planned_samples"][:, :overlap]
    a, b = a.reshape(len(a), -1), b.reshape(len(b), -1)
    return {"pair": [first["query"], second["query"]], "overlap_steps": overlap,
            "window": [second["step"], first["step"] + first["planned_steps"] - 1],
            "a": a, "b": b, "l2": float(np.linalg.norm(a.mean(0) - b.mean(0)))}


def pairwise_sq(x, y):
    return np.maximum(np.sum(x * x, 1)[:, None] + np.sum(y * y, 1)[None, :] - 2 * x @ y.T, 0.)


def median_bandwidth(rows):
    """Median off-diagonal distance within each training pair's pooled draws."""
    distances = []
    for row in rows:
        z = np.concatenate((row["stac"]["a"], row["stac"]["b"])).astype(np.float64)
        d = np.sqrt(pairwise_sq(z, z))
        distances.append(d[~np.eye(len(z), dtype=bool)])
    sigma = float(np.median(np.concatenate(distances)))
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Degenerate median bandwidth on the training split")
    return sigma


def mmd2(a, b, sigma):
    """Biased (V-statistic) RBF MMD^2; nonnegative, exactly zero for identical sets."""
    a, b = a.astype(np.float64), b.astype(np.float64)
    k = lambda x, y: np.exp(-pairwise_sq(x, y) / (2 * sigma ** 2))
    return float(k(a, a).mean() + k(b, b).mean() - 2 * k(a, b).mean())


def extend_rows(trials, landmarks, hashes):
    for trial in trials:
        rows = trial["rows"]
        if not rows:
            continue
        directory = Path(next(iter(rows.values()))["id"])
        queries = [json.loads(line) for line in (directory / "queries.jsonl").read_text().splitlines() if line.strip()]
        trial["latencies"] = [float(q["latency_seconds"]) for q in queries]
        for landmark in landmarks:
            if landmark not in rows:
                continue
            row = rows[landmark]
            row["backbone"] = load_npz(directory, queries[row["query"]], hashes)["backbone"]
            row["stac"] = stac_pair(directory, queries, landmark, hashes)


def swap_handoff(rows):
    return [dict(r, handoff=r["backbone"]) for r in rows]


def add_e5_methods(result, rows):
    """Extend every evaluated fold of a base landmark result with the E5 methods."""
    pooled = {name: [] for name in ("stac_style", "stac_l2", "backbone")}
    for fold in result["folds"]:
        if fold.get("status") != "evaluated":
            continue
        test_task, cal_task = fold["test_task"], fold["calibration_task"]
        train = [r for r in rows if r["task"] not in (test_task, cal_task)]
        cal = [r for r in rows if r["task"] == cal_task]
        test = [r for r in rows if r["task"] == test_task]
        cy, ty = [r["failure"] for r in cal], [r["failure"] for r in test]
        sigma = median_bandwidth(train)
        fold["stac_bandwidth"] = sigma
        scores = {"stac_style": ([mmd2(r["stac"]["a"], r["stac"]["b"], sigma) for r in cal],
                                 [mmd2(r["stac"]["a"], r["stac"]["b"], sigma) for r in test]),
                  "stac_l2": ([r["stac"]["l2"] for r in cal], [r["stac"]["l2"] for r in test]),
                  "backbone": base.fit_scores(swap_handoff(train), swap_handoff(cal), swap_handoff(test), "handoff")
                  if cal else ([], base.fit_scores(swap_handoff(train), swap_handoff(test), swap_handoff(test), "handoff")[1])}
        for method, (cs, ts) in scores.items():
            threshold = base.calibrate(cy, cs) if fold["threshold_available"] else None
            fold["methods"][method] = {"test": base.metrics(ty, ts, threshold),
                "calibration": base.metrics(cy, cs, threshold) if cal else None,
                "predictions": [{k: r[k] for k in ("id", "group", "condition", "replicate",
                    "query_step", "query", "failure")} | {"score": float(score),
                    "alarm": None if threshold is None else bool(score > threshold),
                    "stac_pair": r["stac"]["pair"], "stac_window": r["stac"]["window"]}
                    for r, score in zip(test, ts)]}
            pooled[method].extend((r["failure"], float(score), None if threshold is None
                                   else bool(score > threshold)) for r, score in zip(test, ts))
        two_class = len(set(ty)) == 2
        fold["backbone_minus_handoff_auroc"] = (fold["methods"]["backbone"]["test"]["auroc"] -
            fold["methods"]["handoff"]["test"]["auroc"] if two_class else None)
        fold["stac_minus_native_spread_auroc"] = (fold["methods"]["stac_style"]["test"]["auroc"] -
            fold["methods"]["native_spread"]["test"]["auroc"] if two_class else None)
    for method, values in pooled.items():
        if not values:
            continue
        y, score, alarm = map(np.asarray, zip(*values))
        report = base.metrics(y, score)
        has = alarm != None  # noqa: E711
        ya, aa = y[has], alarm[has].astype(bool)
        report.update(tpr=float(aa[ya == 1].mean()) if np.any(ya == 1) else None,
                      fpr=float(aa[ya == 0].mean()) if np.any(ya == 0) else None)
        result.setdefault("pooled_oof_descriptive", {})[method] = report


def evaluate_landmark(trials, landmark):
    result = base.evaluate_landmark(trials, landmark)
    rows = [t["rows"][landmark] for t in trials if landmark in t["rows"]]
    if rows:
        result["stac_overlap_steps"] = sorted({r["stac"]["overlap_steps"] for r in rows})
    if any(f.get("status") == "evaluated" for f in result["folds"]):
        add_e5_methods(result, rows)
    return result


def cost(trials):
    latencies = [q for t in trials for q in t.get("latencies", [])]
    return {"completed_episodes": len(trials),
            "total_queries": int(sum(t["summary"]["queries"] for t in trials)),
            "total_wall_seconds": float(sum(t["summary"]["wall_seconds"] for t in trials)),
            "mean_query_latency_seconds": float(np.mean(latencies)) if latencies else None,
            "median_query_latency_seconds": float(np.median(latencies)) if latencies else None}


def analyze(roots, landmarks=(40, 80)):
    if not landmarks or any(type(v) is not int or v < 0 for v in landmarks):
        raise ValueError("Landmarks must be nonnegative simulator steps")
    trials, configs, hashes, pending = base.load_trials(roots, landmarks)
    for cfg in configs:
        if type(cfg.get("execute_steps")) is not int or cfg["execute_steps"] < 1:
            raise ValueError("E5 campaigns must record execute_steps")
    extend_rows(trials, landmarks, hashes)
    with threadpool_limits(limits=2):
        results = [evaluate_landmark(trials, v) for v in landmarks]
    return {"schema_version": "e5-1", "analysis_sha256": base.digest(Path(__file__)),
            "base_analysis_sha256": base.digest(Path(base.__file__)),
            "environment": {"python": platform.python_version(), "numpy": np.__version__,
                            "sklearn": sklearn.__version__},
            "endpoint": "eventual_failure_at_fixed_prefix_conditioned_on_still_active",
            "status": "analysis_of_completed_trials_only", "pending_trial_directories": pending,
            "collection_warning": "Completed subset may be outcome-dependent while collection is active; freeze a complete predeclared campaign before confirmatory interpretation.",
            "configs": configs, "source_sha256": hashes, "cost": cost(trials), "landmarks": results,
            "protocol": {"split": "leave_one_task_out; next sorted completed task is calibration",
                "supervised": "fixed C=1 logistic; train-only standardization; PCA<=30 for handoff and backbone",
                "baseline": "image mean/std, state, executed prefix mean/std/first action/norm",
                "spread": "mean raw std (ddof=0) over the executed prefix draws; larger predicts failure",
                "stac_style": "biased RBF MMD^2 between the J-draw sets of consecutive queries on their overlapping steps; last pair whose overlap window ends at or before the landmark; bandwidth = median off-diagonal distance within pooled draws of training pairs; larger predicts failure",
                "stac_l2": "L2 between the J-mean chunks of the same pair on the same overlap; secondary, no fitting",
                "backbone": "identical pipeline to handoff applied to backbone_pooled (mean-token and last-token final backbone hidden state)",
                "threshold": "strict score > calibration-success 90th percentile (higher)",
                "nominal_fpr": .1, "threshold_warning": "finite-sample calibration is not a guarantee",
                "claim_scope": "STAC-style and SAFE-style probes only; no official STAC/SAFE or conformal calibration reproduction; no onset, lead-time or causal-use claim"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--landmarks", type=int, nargs="+", default=[40, 80])
    args = parser.parse_args()
    result = analyze(args.input, tuple(args.landmarks))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "cost": result["cost"], "landmarks": [
        {k: r[k] for k in ("landmark", "status", "active_episodes")} for r in result["landmarks"]]}))


if __name__ == "__main__":
    main()
