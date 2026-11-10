"""Shared helpers for the within-condition reanalyses (E1-E4).

Reads the frozen within-situation analyses (data/icassp2027/analysis/realistic_<suite>_<family>_<policy>.json,
written by code/analyze_icassp_trials.py) and applies the paper's evaluability rule
(code/gen_realistic_tables.py: failures 10-80% and >= 4 identifiable tasks). Data live in the main
checkout because data/ is not in git; outputs go to data/icassp2027/exp_20261020/.
"""
import glob
import hashlib
import json
import subprocess
from math import comb
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parent
DATA = CODE.parent / "data/icassp2027"
A = DATA / "analysis"
OUT = DATA / "exp_20261020"
POLICIES = ["pi05", "gr00t", "molmoact"]
FAMILIES = ["occluder", "lighting", "moved", "relocated"]
FAMILY_DIR = {"occluder": "main", "moved": "main", "lighting": "main2", "relocated": "main3"}
METHODS = ["native_spread", "baseline", "handoff", "combined"]
SUITES = ["spatial", "object"]
LO, HI = 0.10, 0.80
SEED = 20261020


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance(inputs):
    """Input digests plus the exact code state of this worktree."""
    git = lambda *a: subprocess.run(["git", "-C", str(CODE), *a], capture_output=True, text=True).stdout.strip()
    return {"git_commit": git("rev-parse", "HEAD"), "git_dirty_files": git("status", "--porcelain").splitlines(),
            "inputs_sha256": {str(p): sha256(p) for p in sorted(set(map(str, inputs)))}}


def write(name, result, inputs):
    OUT.mkdir(parents=True, exist_ok=True)
    result = {"provenance": provenance(inputs), **result}
    path = OUT / f"{name}.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return path


def analysis_path(suite, family, policy):
    return A / f"realistic_{suite}_{family}_{policy}.json"


def folds(analysis, landmark):
    v = next(x for x in analysis["landmarks"] if x["landmark"] == landmark)
    return {f["test_task"]: f for f in v.get("folds", []) if "methods" in f}


def task_aurocs(analysis, landmark, method):
    """{test task: held-out AUROC} for tasks with both outcomes."""
    return {t: f["methods"][method]["test"]["auroc"] for t, f in folds(analysis, landmark).items()
            if f["methods"][method]["test"].get("auroc") is not None}


def cohort_counts(suite, policy, family):
    rows = [json.load(open(f)) for f in glob.glob(
        f"{DATA}/realistic/{suite}_{FAMILY_DIR[family]}/{policy}/shard*/*_{family}/summary.json")]
    rows = [r for r in rows if r.get("complete")]
    return sum(not r["success"] for r in rows), len(rows)


def evaluable_cells(landmarks=(80,)):
    """(suite, family, policy, landmark, analysis path) for every cell the paper evaluates."""
    cells = []
    for suite in SUITES:
        for family in FAMILIES:
            for policy in POLICIES:
                path = analysis_path(suite, family, policy)
                if not path.exists():
                    continue
                a = json.load(open(path))
                nf, n = cohort_counts(suite, policy, family)
                for L in landmarks:
                    ident = max(len(task_aurocs(a, L, m)) for m in METHODS)
                    if n and LO <= nf / n <= HI and ident >= 4:
                        cells.append((suite, family, policy, L, path))
    return cells


def sign_test(diffs, alternative="two-sided"):
    """Exact sign test, zeros dropped (identical to code/realistic_claims.py)."""
    d = [x for x in diffs if x != 0]
    n, k = len(d), sum(x > 0 for x in d)
    if n == 0:
        return 1.0
    upper = sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n
    if alternative == "greater":
        return upper
    lower = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * min(upper, lower))


def holm(pvalues):
    order = np.argsort(pvalues)
    m, adjusted, running = len(pvalues), [None] * len(pvalues), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def cluster_bootstrap(groups, stat=np.mean, n_boot=10000, seed=SEED):
    """Percentile 95% CI resampling whole clusters. groups: list of per-stratum arrays of cluster
    values; strata are resampled independently (one stratum = plain task-cluster bootstrap)."""
    rng = np.random.default_rng(seed)
    groups = [np.asarray(g, float) for g in groups]
    draws = np.empty(n_boot)
    for b in range(n_boot):
        draws[b] = stat(np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups]))
    return [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]
