"""Analysis of the view-restore branches (plan: docs/PROTOCOLS.md).

Gate = the paper's prespecified step-80 alarm (held-out score above the calibration task's upper
90th-percentile success score), read from the frozen within-situation analyses. Per task, the gate's
success gain over continue is compared with a random gate at the same realized rate within that
task. Cells are pooled over suite-task clusters (policy mean), as in the paper's post hoc pooling.

    python code/exp_20261029_view_restore_analysis.py [--suites spatial,object]
"""
import argparse
import json
from math import comb
from pathlib import Path

import numpy as np
from gate_covariance import gate_covariance, gate_covariance_fraction

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/icassp2027"
POLICIES = ["pi05", "gr00t", "molmoact"]
METHODS = ["handoff", "native_spread", "baseline", "combined"]
SEED = 20261029


def sign_test(x):
    x = [v for v in x if v != 0]
    n, k = len(x), sum(v > 0 for v in x)
    if not n:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def holm(ps):
    order = np.argsort(ps)
    out, run = [0.0] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, (len(ps) - r) * ps[i])
        out[i] = min(1.0, run)
    return out


def boot(x, n=10000):
    rng = np.random.default_rng(SEED)
    x = np.asarray(x)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(axis=1)
    return [float(np.quantile(m, .025)), float(np.quantile(m, .975))]


def source_key(value):
    """Join retained records without depending on the collector's checkout path."""
    parts = Path(value).parts
    if ".." in parts:
        raise ValueError("Source identities must not contain parent traversal")
    marker = ("data", "icassp2027", "realistic")
    for index in range(len(parts) - len(marker)):
        if parts[index:index + len(marker)] == marker:
            return "/".join(parts[index + len(marker):])
    raise ValueError(f"Unrecognized source identity: {value}")


def load_cell(suite, policy):
    recs = {}
    for f in (DATA / "exp_20261029" / suite / policy).glob("*.json"):
        r = json.loads(f.read_text())
        if set(r["branches"]) == {"continue", "restore"}:
            key = source_key(r["source"])
            if key in recs:
                raise ValueError(f"Duplicate branch record for {key}")
            recs[key] = r
    a = json.loads((DATA / "analysis" / f"realistic_{suite}_occluder_{policy}.json").read_text())
    lm = next(x for x in a["landmarks"] if x["landmark"] == 80)
    rows = []
    for fold in lm["folds"]:
        if "methods" not in fold:
            continue
        for i, pred in enumerate(fold["methods"]["handoff"]["predictions"]):
            key = source_key(pred["id"])
            if key not in recs:
                continue
            r = recs[key]
            row = {"task": fold["test_task"], "source_failure": pred["failure"],
                   "cont": int(r["branches"]["continue"]["success"]), "rest": int(r["branches"]["restore"]["success"]),
                   "replay_error": r["replay_max_state_error"],
                   "image_match": r["branches"]["continue"].get("first_image_matches_source"),
                   "action_match": r["branches"]["continue"].get("first_actions_match_source")}
            for m in METHODS:
                p = fold["methods"][m]["predictions"][i]
                assert p["id"] == pred["id"]
                row[m] = int(p["alarm"])
            rows.append(row)
    return rows, lm["active_episodes"]


def task_effects(rows, method, exact=False):
    """{task: gate gain - rate-matched random gain} in success probability per active prefix."""
    out = {}
    for t in sorted({r["task"] for r in rows}):
        rs = [r for r in rows if r["task"] == t]
        d = np.array([r["rest"] - r["cont"] for r in rs])
        g = np.array([r[method] for r in rs])
        out[t] = (gate_covariance_fraction if exact else gate_covariance)(g, d)
    return out


def summarize(rows):
    d = np.array([r["rest"] - r["cont"] for r in rows])
    res = {"n": len(rows), "continue_success": int(sum(r["cont"] for r in rows)),
           "restore_success": int(sum(r["rest"] for r in rows)),
           "rescued": int(sum(r["cont"] == 0 and r["rest"] == 1 for r in rows)),
           "lost": int(sum(r["cont"] == 1 and r["rest"] == 0 for r in rows)),
           "continue_matches_source": int(sum(r["cont"] == 1 - r["source_failure"] for r in rows)),
           "first_action_matches_source": int(sum(bool(r["action_match"]) for r in rows)),
           "max_replay_error": max(r["replay_error"] for r in rows), "gates": {}}
    for m in METHODS:
        g = np.array([r[m] for r in rows])
        fails = [r for r in rows if r["cont"] == 0]
        res["gates"][m] = {
            "rate": float(g.mean()), "alarm_count": int(g.sum()), "gate_success": int(sum(r["rest"] if r[m] else r["cont"] for r in rows)),
            "random_expected_success": float(sum(r["cont"] for r in rows) + sum(
                np.mean([r[m] for r in rows if r["task"] == t]) *
                sum(r["rest"] - r["cont"] for r in rows if r["task"] == t)
                for t in {r["task"] for r in rows})),
            "alarmed_continue_failures": int(sum(r[m] for r in fails)), "continue_failures": len(fails),
            "rescued_alarmed": int(sum(r[m] and r["rest"] for r in fails)),
            "lost_alarmed": int(sum(r[m] and r["cont"] and not r["rest"] for r in rows)),
            "task_effects": task_effects(rows, m)}
    return res


NAMES = {"pi05": "$\\pi_{0.5}$", "gr00t": "GR00T-N1.7", "molmoact": "MolmoAct2"}


def write_table(cells):
    """paper/tables/view_restore.tex: per-cell outcomes of continue, always-restore, and the handoff/spread gates."""
    lines = ["% generated by code/exp_20261029_view_restore_analysis.py; do not edit",
             "\\begin{tabular}{llrrrrrr}", "\\toprule",
             "Suite & Policy & $n$ & Cont. & Restore & Resc./lost & Handoff gate ($q$) & Spread gate ($q$) \\\\",
             "\\midrule"]
    for key, c in cells.items():
        suite, policy = key.split("|")
        gate = lambda m: (f"{c['gates'][m]['gate_success']} vs.\\ {c['gates'][m]['random_expected_success']:.1f} "
                          f"(${c['gates'][m]['rate']:.2f}$)")
        lines.append(f"{suite.capitalize()} & {NAMES[policy]} & {c['n']} & {c['continue_success']} & {c['restore_success']} & "
                     f"{c['rescued']}/{c['lost']} & {gate('handoff')} & {gate('native_spread')} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (ROOT / "paper/tables/view_restore.tex").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suites", default="spatial,object")
    ap.add_argument("--output", type=Path, default=DATA / "exp_20261029/view_restore_analysis.json")
    ap.add_argument("--no-table", action="store_true", help="Write only the selected JSON output")
    a = ap.parse_args()
    cells, clusters = {}, {m: {} for m in METHODS}
    for suite in a.suites.split(","):
        for policy in POLICIES:
            rows, active = load_cell(suite, policy)
            if not rows:
                continue
            cells[f"{suite}|{policy}"] = res = summarize(rows)
            res["active_at_80"], res["complete"] = active, len(rows) == active
            for m in METHODS:
                for t, v in task_effects(rows, m, exact=True).items():
                    clusters[m].setdefault((suite, t), []).append(v)
    pooled = {}
    for m in METHODS:
        x = [float(np.mean(v)) for v in clusters[m].values()]
        pooled[m] = {"clusters": len(x), "mean": float(np.mean(x)) if x else None, "ci95": boot(x) if x else None,
                     "positive": int(sum(v > 0 for v in x)), "negative": int(sum(v < 0 for v in x)), "sign_p": sign_test(x)}
    for m, h in zip(METHODS, holm([pooled[m]["sign_p"] for m in METHODS])):
        pooled[m]["holm_p"] = h
    if not a.no_table:
        write_table(cells)
    out = a.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"cells": cells, "pooled": pooled}, indent=2) + "\n")
    for k, c in cells.items():
        print(k, {x: c[x] for x in ("n", "active_at_80", "continue_success", "restore_success", "rescued", "lost",
                                    "continue_matches_source")},
              {m: (round(g["rate"], 2), g["gate_success"], round(g["random_expected_success"], 1)) for m, g in c["gates"].items()})
    print(json.dumps(pooled, indent=1))


if __name__ == "__main__":
    main()
