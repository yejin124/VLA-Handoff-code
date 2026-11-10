"""Fixed-prefix eventual-failure analysis of source-bound closed-loop rollout records (LIBERO, SimplerEnv-Bridge).
artifact() is the digest-checked loader of one recorded query used by ral_confirm_analysis.py and ral_nested_features.py.

No onset or lead-time claims. Supervised models are fixed in advance; test tasks
and their adjacent calibration tasks never enter fitting or preprocessing.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import warnings

import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(path, hashes):
    path = path.resolve()
    hashes[str(path)] = digest(path)
    return path


def read_json(path, hashes):
    return json.loads(record(path, hashes).read_text())


def finite(value, name):
    array = np.asarray(value, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"Nonfinite {name}")
    return array


def artifact(directory, query, hashes):
    name = query["artifact"]
    if not isinstance(name, str) or Path(name).name != name or not name.endswith(".npz"):
        raise ValueError("Unsafe artifact path")
    path = (directory / name).resolve()
    if path.parent != directory.resolve():
        raise ValueError("Artifact escapes trial directory")
    record(path, hashes)
    if hashes[str(path)] != query["artifact_sha256"]:
        raise ValueError(f"Artifact digest mismatch: {path}")
    with np.load(path, allow_pickle=False) as source:
        data = {key: finite(source[key], key) for key in
                ("action_samples", "executed_actions", "handoff", "state")}
    samples, actions = data["action_samples"], data["executed_actions"]
    if actions.ndim != 2 or actions.shape[1] != 7 or not len(actions):
        raise ValueError("Invalid executed action dimensions")
    if samples.shape != (query["draws"], *actions.shape) or query["draws"] < 2:
        raise ValueError("Spread analysis requires at least two recorded draws")
    if not np.array_equal(samples[0], actions):
        raise ValueError("Executed actions differ from draw zero")
    spread = float(samples.std(axis=0, ddof=0).mean())
    if not np.isclose(spread, query["spread"], rtol=1e-4, atol=1e-7):
        raise ValueError("Recorded spread disagrees with raw draws")
    if data["handoff"].ndim != 1 or len(data["handoff"]) < 2:
        raise ValueError("Invalid handoff vector")
    state = data["state"]
    if state.shape != (8,) or not np.array_equal(state, finite(query["state"], "query state")):
        raise ValueError("Invalid or inconsistent state")
    image = finite(query["image_stats"], "image stats")
    if image.shape != (4,):
        raise ValueError("Expected two image means and two image standard deviations")
    # Every statistic concerns planned actions at this query, before execution.
    baseline = np.concatenate((image, state, actions.mean(axis=0),
                               actions.std(axis=0), actions[0],
                               [np.linalg.norm(actions)]))
    return spread, baseline, data["handoff"]


def load_trials(roots, landmarks):
    hashes, configs, trials, seen, pending = {}, [], [], set(), []
    config_paths = set()
    for root in roots:
        root = Path(root).resolve()
        config_paths.update([root / "config.json"] if (root / "config.json").exists()
                            else root.glob("shard*/config.json"))
    if not config_paths:
        raise ValueError("No collector config.json files found")
    fingerprint = None
    for path in sorted(config_paths):
        cfg = read_json(path, hashes)
        expected = cfg.copy()
        stated = expected.pop("config_sha256")
        actual = hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()
        if actual != stated:
            raise ValueError("Collector configuration digest mismatch")
        # Collections that differ only in which conditions they carry (and their per-condition
        # severities) may be pooled: suite, tasks, episodes, seed, draws and horizon must still agree.
        # apply_to_wrist changes the occluder only, and the client hash changed when the relocated
        # family was added (2026-10-12); each shard's own digest is still verified above.
        scientific = {k: v for k, v in expected.items()
                      if k not in ("shard", "nshards", "port", "conditions", "severities", "severity", "apply_to_wrist", "collector_sha256")}
        if fingerprint is not None and scientific != fingerprint:
            raise ValueError("Incompatible campaigns: analyze model/configuration separately")
        fingerprint = scientific
        configs.append(cfg)
        for directory in sorted(path.parent.glob("t*_e*_r*_*")):
            if not directory.is_dir() or ".incomplete-" in directory.name:
                continue
            summary_path = directory / "summary.json"
            if not summary_path.exists():
                pending.append(str(directory))
                continue
            summary = read_json(summary_path, hashes)
            if summary.get("complete") is not True:
                raise ValueError("Incomplete summary must not be scored")
            if summary["config_sha256"] != stated or type(summary["success"]) is not bool:
                raise ValueError("Summary configuration or outcome is invalid")
            if (type(summary["steps"]) is not int or not 0 < summary["steps"] <= cfg["max_steps"]
                    or summary["terminal_reason"] not in ("success", "terminated", "timeout")
                    or (summary["terminal_reason"] == "success") != summary["success"]
                    or (summary["terminal_reason"] == "timeout" and summary["steps"] != cfg["max_steps"])):
                raise ValueError("Invalid terminal outcome")
            key = (summary["task"], summary["initial_state_sha256"],
                   summary["replicate"], summary["condition"])
            if key in seen:
                raise ValueError("Duplicate task/initial-state/replicate/condition")
            seen.add(key)
            trace_path = record(directory / "queries.jsonl", hashes)
            queries = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
            if len(queries) != summary["queries"] or not queries:
                raise ValueError("Trace count disagrees with summary")
            if queries[0]["step"] != 0:
                raise ValueError("First query must precede the first action")
            previous = -1
            for index, query in enumerate(queries):
                if (query["query"] != index or type(query["step"]) is not int
                        or not previous < query["step"] < summary["steps"]):
                    raise ValueError("Queries must have strictly increasing valid steps")
                if query["draws"] != cfg["draws"]:
                    raise ValueError("Query draw count differs from configuration")
                previous = query["step"]
            rows = {}
            for landmark in landmarks:
                if summary["steps"] <= landmark:
                    continue
                available = [q for q in queries if q["step"] <= landmark]
                if not available:
                    raise ValueError("No pre-landmark query in an active episode")
                query = available[-1]
                spread, baseline, handoff = artifact(directory, query, hashes)
                rows[landmark] = {"id": str(directory.resolve()), "task": summary["task"],
                    "group": [summary["task"], summary["initial_state_sha256"]],
                    "condition": summary["condition"], "replicate": summary["replicate"],
                    "query_step": query["step"], "query": query["query"],
                    "failure": int(not summary["success"]), "spread": spread,
                    "baseline": baseline, "handoff": handoff,
                    "latency_seconds": float(query["latency_seconds"])}
            trials.append({"summary": summary, "rows": rows})
    return trials, configs, hashes, pending


def metrics(labels, scores, threshold=None):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if not len(labels) or len(labels) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("Invalid labels/scores")
    positives, negatives = labels == 1, labels == 0
    result = {"n": len(labels), "failures": int(positives.sum()),
              "prevalence": float(positives.mean()),
              "auroc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
              "auprc": float(average_precision_score(labels, scores)) if positives.any() else None}
    if threshold is not None:
        alarms = scores > threshold
        result.update(threshold=float(threshold),
                      tpr=float(alarms[positives].mean()) if positives.any() else None,
                      fpr=float(alarms[negatives].mean()) if negatives.any() else None)
    return result


def calibrate(labels, scores, fpr=.1):
    negatives = np.asarray(scores)[np.asarray(labels) == 0]
    if not len(negatives):
        raise ValueError("Calibration task has no success episodes")
    return float(np.quantile(negatives, 1 - fpr, method="higher"))


def fit_scores(train, calibration, test, kind):
    """All fitted transformations see train only. No test-driven model selection."""
    matrices = []
    for key in (["baseline", "handoff"] if kind == "combined" else [kind]):
        x = [np.stack([r[key] for r in rows]) for rows in (train, calibration, test)]
        scaler = StandardScaler().fit(x[0])
        x = [scaler.transform(a) for a in x]
        if key == "handoff":
            dims = min(30, x[0].shape[0] - 1, x[0].shape[1])
            pca = PCA(n_components=dims, svd_solver="randomized", random_state=0).fit(x[0])
            x = [pca.transform(a) for a in x]
            # Equalize retained-PC scales for the fixed logistic regularizer.
            scaler = StandardScaler().fit(x[0])
            x = [scaler.transform(a) for a in x]
        matrices.append(x)
    x = [np.concatenate([m[i] for m in matrices], axis=1) for i in range(3)]
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model = LogisticRegression(C=1., solver="liblinear", max_iter=2000, random_state=0)
        model.fit(x[0], [r["failure"] for r in train])
    return model.predict_proba(x[1])[:, 1], model.predict_proba(x[2])[:, 1]


def evaluate_landmark(trials, landmark):
    rows = [t["rows"][landmark] for t in trials if landmark in t["rows"]]
    tasks = sorted({t["summary"]["task"] for t in trials})
    active_tasks = sorted({r["task"] for r in rows})
    result = {"landmark": landmark, "completed_episodes": len(trials),
              "active_episodes": len(rows), "already_terminated": len(trials) - len(rows),
              "active_tasks": active_tasks, "folds": [], "descriptive_by_task": []}
    for task in tasks:
        all_task = [t for t in trials if t["summary"]["task"] == task]
        subset = [r for r in rows if r["task"] == task]
        conditions = []
        for condition in sorted({r["condition"] for r in subset}):
            group = [r for r in subset if r["condition"] == condition]
            conditions.append({"condition": condition, "n": len(group),
                "failures": sum(r["failure"] for r in group),
                "mean_spread": float(np.mean([r["spread"] for r in group]))})
        result["descriptive_by_task"].append({"task": task, "completed": len(all_task),
            "active": len(subset), "already_terminated": len(all_task) - len(subset),
            "conditions": conditions})
    if len(active_tasks) < 4:
        result["status"] = "descriptive_only_fewer_than_four_active_tasks"
        return result
    result["status"] = "task_heldout_evaluation"
    pooled = {name: [] for name in ("native_spread", "baseline", "handoff", "combined")}
    for index, test_task in enumerate(tasks):
        cal_task = tasks[(index + 1) % len(tasks)]
        train = [r for r in rows if r["task"] not in (test_task, cal_task)]
        cal = [r for r in rows if r["task"] == cal_task]
        test = [r for r in rows if r["task"] == test_task]
        fold = {"test_task": test_task, "calibration_task": cal_task,
                "train_tasks": sorted({r["task"] for r in train}),
                "train_n": len(train), "calibration_n": len(cal), "test_n": len(test)}
        result["folds"].append(fold)
        if not test:
            fold["status"] = "skipped_missing_test"
            continue
        if len({r["failure"] for r in train}) < 2:
            fold["status"] = "skipped_single_class_training"
            continue
        # Ranking (AUROC) needs no threshold: a calibration task without successes only
        # removes the alarm operating point for this fold, not the fold itself.
        has_threshold = any(r["failure"] == 0 for r in cal)
        fold.update(status="evaluated", methods={}, threshold_available=has_threshold)
        cy, ty = [r["failure"] for r in cal], [r["failure"] for r in test]
        for method in pooled:
            if method == "native_spread":
                cs, ts = [r["spread"] for r in cal], [r["spread"] for r in test]
            elif cal:
                cs, ts = fit_scores(train, cal, test, method)
            else:
                ts = fit_scores(train, test, test, method)[1]
            threshold = calibrate(cy, cs) if has_threshold else None
            fold["methods"][method] = {"test": metrics(ty, ts, threshold),
                "calibration": metrics(cy, cs, threshold) if cal else None,
                "predictions": [{k: r[k] for k in ("id", "group", "condition", "replicate",
                    "query_step", "query", "failure")} | {"score": float(score),
                    "alarm": None if threshold is None else bool(score > threshold)}
                    for r, score in zip(test, ts)]}
            pooled[method].extend((r["failure"], float(score), None if threshold is None
                                   else bool(score > threshold)) for r, score in zip(test, ts))
        fold["combined_minus_baseline_auroc"] = (
            fold["methods"]["combined"]["test"]["auroc"] -
            fold["methods"]["baseline"]["test"]["auroc"]
            if len(set(ty)) == 2 else None)
    if not any(f.get("status") == "evaluated" for f in result["folds"]):
        result["status"] = "no_evaluable_folds"
    result["pooled_oof_descriptive"] = {}
    for method, values in pooled.items():
        if not values:
            continue
        y, score, alarm = map(np.asarray, zip(*values))
        report = metrics(y, score)
        has = alarm != None  # noqa: E711 (object array; folds without a threshold carry None)
        ya, aa = y[has], alarm[has].astype(bool)
        report.update(tpr=float(aa[ya == 1].mean()) if np.any(ya == 1) else None,
                      fpr=float(aa[ya == 0].mean()) if np.any(ya == 0) else None)
        result["pooled_oof_descriptive"][method] = report
    result["uncertainty_note"] = (
        "Pooled OOF is descriptive: models/calibration tasks differ, training sets overlap, "
        "conditions and replicates are dependent, and active populations differ by landmark. "
        "No IID bootstrap CI, population-level significance, or early-warning lead time is claimed.")
    return result


def analyze(roots, landmarks=(40, 80)):
    if not landmarks or any(type(v) is not int or v < 0 for v in landmarks):
        raise ValueError("Landmarks must be nonnegative simulator steps")
    trials, configs, hashes, pending = load_trials(roots, landmarks)
    with threadpool_limits(limits=2):
        results = [evaluate_landmark(trials, v) for v in landmarks]
    return {"schema_version": 1, "analysis_sha256": digest(Path(__file__)),
            "environment": {"python": platform.python_version(), "numpy": np.__version__,
                            "sklearn": sklearn.__version__},
            "endpoint": "eventual_failure_at_fixed_prefix_conditioned_on_still_active",
            "status": "analysis_of_completed_trials_only", "pending_trial_directories": pending,
            "collection_warning": "Completed subset may be outcome-dependent while collection is active; freeze a complete predeclared campaign before confirmatory interpretation.",
            "configs": configs, "source_sha256": hashes, "landmarks": results,
            "protocol": {"split": "leave_one_task_out; next sorted completed task is calibration",
                "supervised": "fixed C=1 logistic; train-only standardization; handoff PCA<=30",
                "baseline": "image mean/std, state, executed chunk mean/std/first action/norm",
                "spread": "mean raw executable-coordinate std with ddof=0; larger predicts failure",
                "threshold": "strict score > calibration-success 90th percentile (higher)",
                "nominal_fpr": .1, "threshold_warning": "finite-sample calibration is not a guarantee",
                "claim_scope": "No STAC/SAFE reproduction; no handoff causal-use or onset claim"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--landmarks", type=int, nargs="+", default=[40, 80])
    args = parser.parse_args()
    result = analyze(args.input, tuple(args.landmarks))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "landmarks": [
        {k: r[k] for k in ("landmark", "status", "active_episodes")} for r in result["landmarks"]]}))


if __name__ == "__main__":
    main()
