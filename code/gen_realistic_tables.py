#!/usr/bin/env python3
"""LaTeX tables for the realistic-situation campaign from data/icassp2027/analysis/realistic_*.json.

Data layout (one client run per policy and suite collects all conditions on the same initial
states): data/icassp2027/realistic/<suite>_main/<policy>/shardNN/t*_e*_r0_<condition>/.
Within-situation analyses read the per-condition views that code/realistic_views.py builds;
the pooled analysis reads the mixed directory. Re-query cohorts are one family per directory.

Never hand-edit the generated files; change this script. Writes to paper/tables/.
  severities.tex        chosen severity, failures/N and identifiable tasks per policy and situation
  within_realistic.tex  within-situation task-held-out AUROC (median, IQR) per policy, situation, step
  pooled_realistic.tex  four situations pooled: macro AUROC per readout, plus label-only baselines
  requery_realistic.tex re-query regime: prefix spread, baseline, handoff, backbone, STAC-style, L2
  object_realistic.tex  Object suite within-situation AUROC (or counts where the rule fails)
  lightingdark_supp.tex the dark lighting rungs (1.4/1.5), supplement only
  libero_pro.tex        LIBERO-PRO reset perturbation outcomes per task (from the frozen analysis JSON)
"""
import argparse
import collections
import glob
import json
import math
import statistics as st
import sys
from pathlib import Path

from table_layout import note, panel, scientific_p, signed

ROOT = Path(__file__).resolve().parent.parent
A = ROOT / "data/icassp2027/analysis"
D = ROOT / "data/icassp2027/realistic"
OUT = ROOT / "paper/tables"
PROTOCOL = json.load(open(ROOT / "config/icassp2027/realistic_protocol.json"))
POLICIES = [("pi05", "$\\pi_{0.5}$"), ("gr00t", "GR00T-N1.7"), ("molmoact", "MolmoAct2")]
FAMILIES = [("occluder", "Occluder"), ("lighting", "Dim light"), ("moved", "Moved goal"), ("relocated", "Relocated goal")]
FAMILY_DIR = {"occluder": "main", "moved": "main", "lighting": "main2", "relocated": "main3"}   # main3: amendment 2026-10-13
METHODS = [("native_spread", "Spread"), ("baseline", "External"), ("handoff", "Handoff"), ("combined", "Combined")]
LO, HI = 0.10, 0.80


def load(name):
    p = A / name
    return json.load(open(p)) if p.exists() else None


def folds(analysis, landmark):
    for v in analysis["landmarks"]:
        if v["landmark"] == landmark:
            return [f for f in v.get("folds", []) if "methods" in f]
    return []


def aurocs(fs, key):
    return [f["methods"][key]["test"]["auroc"] for f in fs if key in f["methods"] and f["methods"][key]["test"].get("auroc") is not None]


def med_iqr(values):
    if not values:
        return "--"
    s = sorted(values)
    return f"{st.median(s):.2f} ({s[len(s)//4]:.2f}--{s[(3*len(s))//4]:.2f})"


def counts(summaries):
    rows = [json.load(open(f)) for f in summaries]
    rows = [r for r in rows if r.get("complete")]
    return sum(not r["success"] for r in rows), len(rows)


def cohort_counts(suite, policy, family):
    return counts(glob.glob(f"{D}/{suite}_{FAMILY_DIR[family]}/{policy}/shard*/*_{family}/summary.json"))


def evaluable(nf, n, ident):
    return n and LO <= nf / n <= HI and ident >= 4


def severities():
    chosen = PROTOCOL["severities_in_force"]
    lines = ["\\begin{tabular}{@{}llcccc@{}}", "\\toprule",
             "Policy & Situation & Setting & Failures/$N$ & Tasks & Evaluable\\\\", "\\midrule"]
    for pk, pname in POLICIES:
        for fk, fname in FAMILIES:
            sev = chosen.get(pk, {}).get(fk, "--")
            nf, n = cohort_counts("spatial", pk, fk)
            a = load(f"realistic_spatial_{fk}_{pk}.json")
            ident = [len(aurocs(folds(a, L), "handoff")) if a else 0 for L in (40, 80)]
            if ident[0] != ident[1]:
                raise ValueError("Task counts differ across landmarks; restore separate columns")
            if fk == "occluder":
                sev = str(sev).removesuffix("+wrist")
            elif fk == "moved":
                sev = f"{100 * float(sev):.0f} cm"
            elif fk == "relocated":
                sev = rf"$\ge{100 * float(sev):g}$ cm"
            ok = "yes" if a and evaluable(nf, n, min(ident)) else "no"
            lines.append(f"{pname} & {fname} & {sev} & {nf}/{n} & {ident[0]} & {ok}\\\\")
    return "\n".join(lines + ["\\bottomrule", "\\end{tabular}", note(
        "Camera settings are protocol severity levels; the occluder affects both camera views. "
        "Goal settings are displacement distances. Task counts are identical at steps 40 and 80 "
        "and count evaluated held-out folds with defined AUROC, after excluding single-class training folds.")])


def excluded_note(excluded):
    """Retain counts once per excluded cell instead of empty metric rows."""
    groups = {}
    for policy, situation, step, failed, total, tasks in excluded:
        groups.setdefault((policy, situation, failed, total, tasks), []).append(step)
    descriptions = []
    for (policy, situation, failed, total, tasks), steps in groups.items():
        when = "both landmarks" if steps == [40, 80] else "step " + "/".join(map(str, steps))
        descriptions.append(f"{policy}, {situation.lower()}: {failed}/{total} failures, "
                            f"{tasks} evaluated tasks ({when})")
    return note("Only cells meeting the prespecified reporting rule are tabulated "
                r"($10$--$80\%$ failures and at least four evaluated tasks with defined AUROC). "
                "Counts retained for excluded cells: " + "; ".join(descriptions) + ". "
                "Tasks count evaluated held-out folds with defined AUROC; single-class training folds are excluded.")


def within(suite="spatial"):
    lines = ["\\begin{tabular}{@{}lllc" + "c" * len(METHODS) + "@{}}", "\\toprule",
             "Policy & Situation & Step & Fail/$N$ (tasks) & " + " & ".join(m[1] for m in METHODS) + "\\\\", "\\midrule"]
    excluded = []
    for pk, pname in POLICIES:
        for fk, fname in FAMILIES:
            a = load(f"realistic_{suite}_{fk}_{pk}.json")
            if a is None:
                continue
            nf, n = cohort_counts(suite, pk, fk)
            for L in (40, 80):
                fs = folds(a, L)
                ident = max((len(aurocs(fs, k)) for k, _ in METHODS), default=0)
                if evaluable(nf, n, ident):
                    cells = [med_iqr(aurocs(fs, k)) for k, _ in METHODS]
                else:
                    excluded.append((pname, fname, L, nf, n, ident))
                    continue
                lines.append(f"{pname} & {fname} & {L} & {nf}/{n} ({ident}) & " + " & ".join(cells) + "\\\\")
    return "\n".join(lines + ["\\bottomrule", "\\end{tabular}", excluded_note(excluded)])


def auroc(pos, neg):
    if not pos or not neg:
        return None
    return sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))


def label_baselines(policy, landmark):
    """Macro AUROC of label-only scores on the pooled Spatial cohort: degraded-vs-clean, and an
    ordinal label whose family order comes from the pilot prevalence (protocol pooled_analysis)."""
    order = {"clean": 0}
    for i, fam in enumerate(PROTOCOL["pilot_prevalence_order"][policy], 1):
        order[fam] = i
    # the same episodes as the pooled analysis: the gated pooled view (realistic_views.py builds it only
    # from collections whose chain step has finished)
    rows = [json.load(open(f)) for f in glob.glob(f"{ROOT}/data/icassp2027/realistic_views/spatial_pooled/{policy}/shard*/*/summary.json")]
    rows = [r for r in rows if r.get("complete") and r["steps"] > landmark]
    if not rows:
        raise ValueError("Missing pooled views; run python code/realistic_views.py --prepare")
    out = []
    for score in (lambda r: int(r["condition"] != "clean"), lambda r: order[r["condition"]]):
        per = []
        for t in range(10):
            sub = [r for r in rows if r["task"] == t]
            a = auroc([score(r) for r in sub if not r["success"]], [score(r) for r in sub if r["success"]])
            if a is not None:
                per.append(a)
        out.append(sum(per) / len(per) if per else None)
    return out, len(rows)


def pooled():
    lines = ["\\begin{tabular}{@{}lcc" + "c" * len(METHODS) + "cc@{}}", "\\toprule",
             "Policy & Step & $N$ & " + " & ".join(m[1] for m in METHODS) + " & Perturbed & Ordinal\\\\", "\\midrule"]
    for pk, pname in POLICIES:
        a = load(f"realistic_spatial_pooled_{pk}.json")
        if a is None:
            continue
        for L in (40, 80):
            fs = folds(a, L)
            macro = []
            for k, _ in METHODS:
                v = aurocs(fs, k)
                macro.append(f"{st.mean(v):.2f}" if v else "--")
            (deg, ordinal), n = label_baselines(pk, L)
            fmt = lambda v: "--" if v is None else f"{v:.2f}"
            lines.append(f"{pname} & {L} & {n} & " + " & ".join(macro) + f" & {fmt(deg)} & {fmt(ordinal)}\\\\")
    return "\n".join(lines + ["\\bottomrule", "\\end{tabular}"])


def requery():
    meths = [("native_spread", "Prefix spread"), ("baseline", "External"), ("handoff", "Handoff"),
             ("backbone", "Backbone probe"), ("stac_style", "STAC-style MMD"), ("stac_l2", "Chunk L2")]
    lines = [r'\begin{tabular}{@{}lllccccccc@{}}', r'\toprule',
             'Policy & Situation & Step & Fail/$N$ (tasks) & ' +
             ' & '.join(label for _, label in meths) + r'\\', r'\midrule']
    for pk, pname in POLICIES:
        if pk not in ('pi05', 'molmoact'):
            continue
        for fk, fname in FAMILIES:
            if fk not in ('occluder', 'moved', 'relocated'):
                continue
            a = load(f"realistic_requery_{fk}_{pk}.json")
            if a is None:
                continue
            nf, n = counts(glob.glob(f"{D}/requery_{fk}/{pk}/shard*/*/summary.json"))
            for L in (40, 80):
                fs = folds(a, L)
                ident = max((len(aurocs(fs, k)) for k, _ in meths), default=0)
                values = ([med_iqr(aurocs(fs, k)) for k, _ in meths]
                          if evaluable(nf, n, ident) else ['--'] * len(meths))
                lines.append(f"{pname} & {fname} & {L} & {nf}/{n} ({ident}) & " +
                             ' & '.join(values) + r'\\')
    return '\n'.join(lines + [r'\bottomrule', r'\end{tabular}'])


def lightingdark():
    """Supplement table: the dark lighting rungs (1.4/1.5) of both main suites and the Spatial re-query cohort,
    reported separately from the severity-1.25 cohorts. Median (IQR) AUROC
    over identifiable tasks and the number of tasks whose AUROC exceeds 0.5 for the native spread."""
    chosen = PROTOCOL["chosen_severities"]          # the dark rungs as chosen on 2026-10-08
    panels, excluded = [], []
    for coll, label, src in (("spatial_main", "Spatial main (full chunk)", "spatial"),
                             ("object_main", "Object main (full chunk)", "object"),
                             ("requery_lightingdark", "Spatial re-query", "requery")):
        meths = METHODS + ([("backbone", "Backbone"), ("stac_style", "STAC-style")] if src == "requery" else [])
        rows = []
        for pk, pname in POLICIES:
            a = load(f"realistic_{src}_lightingdark_{pk}.json")
            if a is None:
                continue
            sub = "*_lighting" if src in ("spatial", "object") else "*"
            nf, n = counts(glob.glob(f"{D}/{coll}/{pk}/shard*/{sub}/summary.json"))
            for L in (40, 80):
                fs = folds(a, L)
                ident = max((len(aurocs(fs, k)) for k, _ in meths), default=0)
                if not evaluable(nf, n, ident):
                    excluded.append((pname + " (" + label + ")", "Dim light", L, nf, n, ident))
                    continue
                cells = []
                for k, _ in meths:
                    v = aurocs(fs, k)
                    if not v:
                        raise ValueError(f"Missing dark-cohort readout: {src}/{pk}/{k}")
                    cell = med_iqr(v)
                    if " (" in cell:
                        center, interval = cell.split(" (", 1)
                        cell = r"\shortstack{" + center + r"\\(" + interval + "}"
                    cells.append(cell)
                rows.append(f"{pname} & {chosen[pk]['lighting']} & {L} & {nf}/{n} ({ident}) & " + " & ".join(cells))
        panels.append(panel(label, "lclc" + "c" * len(meths),
                            "Policy & Sev. & Step & Fail/$N$ (tasks) & " + " & ".join(m[1] for m in meths), rows))
    return "\n".join(panels + [note("Backbone probes were collected only in the re-query regime; STAC-style consistency requires overlapping predictions."),
                                 excluded_note(excluded)])


def libero_pro():
    """Reset-time LIBERO-PRO perturbation: failures and tasks with both outcomes, from the frozen
    analysis JSON (descriptive_by_task at the first landmark, where every episode is still active;
    the raw records of this cohort were deleted on 2026-10-12 with the centered-mask cohorts)."""
    lines = ["\\begin{tabular}{@{}llcc@{}}", "\\toprule", "Policy & Condition & Failures/$N$ & Tasks with both outcomes\\\\", "\\midrule"]
    for name, model in (("$\\pi_{0.5}$", "pi05"), ("MolmoAct2", "molmoact")):
        for cond in ("original", "position"):
            a = load(f"{model}_libero_pro_{cond}.json")
            first = min(a["landmarks"], key=lambda v: v["landmark"])
            assert first["already_terminated"] == 0, "per-task counts need every episode active"
            per_task = [(t["conditions"][0]["n"], t["conditions"][0]["failures"]) for t in first["descriptive_by_task"]]
            nf, n = sum(f for _, f in per_task), sum(k for k, _ in per_task)
            both = sum(1 for k, f in per_task if 0 < f < k)
            lines.append(f"{name} & {cond} & {nf}/{n} & {both}/{len(per_task)}\\\\")
    return "\n".join(lines + ["\\bottomrule", "\\end{tabular}"])


def transfer(suite="spatial"):
    """Threshold fixed on the calibration task (90th percentile of its successes, strict >):
    realized calibration FPR, held-out FPR median (Q1--Q3) and held-out TPR median."""
    lines = ["\\begin{tabular}{@{}lllcccc@{}}", "\\toprule",
             "Policy & Situation & Step & Score & Cal.\\ FPR & Test FPR median (Q1--Q3) & Test TPR median\\\\", "\\midrule"]
    for pk, pname in POLICIES:
        for fk, fname in FAMILIES:
            a = load(f"realistic_{suite}_{fk}_{pk}.json")
            if a is None:
                continue
            nf, n = cohort_counts(suite, pk, fk)
            for L in (40, 80):
                fs = folds(a, L)
                ident = max((len(aurocs(fs, k)) for k, _ in METHODS), default=0)
                if not evaluable(nf, n, ident):
                    continue
                for key, label in (("native_spread", "Spread"), ("handoff", "Handoff")):
                    tf = [f["methods"][key]["test"]["fpr"] for f in fs if f["methods"][key]["test"].get("fpr") is not None]
                    tt = [f["methods"][key]["test"]["tpr"] for f in fs if f["methods"][key]["test"].get("tpr") is not None]
                    cf = [f["methods"][key]["calibration"]["fpr"] for f in fs if f["methods"][key]["calibration"].get("fpr") is not None]
                    if not tf:
                        continue
                    q = sorted(tf)
                    lines.append(f"{pname} & {fname} & {L} & {label} & {st.median(cf):.2f} & {st.median(q):.2f} ({q[len(q)//4]:.2f}--{q[(3*len(q))//4]:.2f}) & {st.median(tt):.2f}\\\\")
    return "\n".join(lines + ["\\bottomrule", "\\end{tabular}"])


def sign_tests(suite="spatial"):
    """Tasks whose held-out AUROC exceeds 0.5, per readout, with a one-sided sign-test p value."""
    from math import comb
    rows = []
    for pk, pname in POLICIES:
        for fk, fname in FAMILIES:
            a = load(f"realistic_{suite}_{fk}_{pk}.json")
            if a is None:
                continue
            for L in (40, 80):
                fs = folds(a, L)
                cells = []
                for key, _ in METHODS:
                    v = aurocs(fs, key)
                    g = sum(x > 0.5 for x in v)
                    pval = sum(comb(len(v), i) for i in range(g, len(v) + 1)) / 2 ** len(v) if v else 1.0
                    cells.append(f"{g}/{len(v)} (p={pval:.3f})")
                rows.append(f"{pk:9s} {fk:9s} {L:3d}  " + "  ".join(cells))
    return "\n".join(rows)


def probe(suites=("spatial", "object")):
    """Keep the planned condition grid even when a readout has no eligible folds."""
    e9 = json.loads((A.parent / "exp_20261020/e9_probe_null.json").read_text())
    holm = {(t["suite"], t["policy"], t["situation"]): t.get("holm_p") for t in e9["tests"]}
    valid = lambda x: isinstance(x, (int, float)) and math.isfinite(x)
    decimal = lambda x: f"{x:.2f}" if valid(x) else "--"
    signed_value = lambda x: signed(x) if valid(x) else "--"
    rows, pooled_rows, unavailable = [], [], []
    for suite in suites:
        pooled_values = []
        for pk, pname in POLICIES:
            r = json.loads((A / f"realistic_probe_{suite}_{pk}.json").read_text())
            for fk, fname in FAMILIES:
                a, b, c = (r.get(section, {}).get(fk, {}) for section in ("A", "B", "C"))
                p = holm.get((suite, pk, fk))
                dag = "$^\\dagger$" if valid(p) and p > 0.05 else ""
                score = decimal(a.get('handoff_auroc'))
                row = f"{suite.capitalize()} & {pname} & {fname} & {score}{dag} / {decimal(a.get('image_stats_auroc'))} & {signed_value(b.get('r2'))}"
                change = c.get('median_relative_change')
                change_text = signed(change * 100, 0) + r"\%" if valid(change) else "--"
                count, higher, wp = c.get('n_pairs'), c.get('higher_in'), c.get('wilcoxon_p')
                pair = f"{higher}/{count}, {scientific_p(wp) if valid(wp) else '--'}" if count is not None and higher is not None else "--"
                rows.append(row + f" & {change_text} & {pair}")
                missing = [name for name, value in (("scene decoding", a.get('handoff_auroc')),
                           ("spread readout", b.get('r2')), ("paired change", change)) if not valid(value)]
                if missing:
                    reasons = r.get('not_evaluated', {})
                    detail = '; '.join(str(reasons[k]) for k in (f'A/{fk}', f'B/{fk}') if k in reasons)
                    unavailable.append(f"{suite.capitalize()} {pk} {fname}: {', '.join(missing)} unavailable" + (f" ({detail})" if detail else ""))
            pooled_values.append(signed_value(r.get('B', {}).get('pooled', {}).get('r2')))
        pooled_rows.append(suite.capitalize() + " & " + " & ".join(pooled_values))
    result = [panel("A. Condition-specific measurements", "lllcccc",
                    "Suite & Policy & Situation & Scene AUROC (handoff / image) & Spread $R^2$ & Spread change & Higher in, $p$", rows),
              panel(r"B. Spread $R^2$ pooled over all five conditions", "lccc",
                    "Suite & " + " & ".join(name for _, name in POLICIES), pooled_rows)]
    if unavailable:
        result.append(note(("Dashes denote unavailable estimates; pair counts are retained independently. " +
                            "; ".join(unavailable) + ".").replace("_", r"\_")))
    return "\n".join(result)


def claims(suite="spatial"):
    """Tests behind the within-situation claims, from realistic_claims.py's functions."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import realistic_claims as rc
    rows, comparisons = [], []
    for pk, pname in POLICIES:
        for fk, fname in FAMILIES:
            path = A / f"realistic_{suite}_{fk}_{pk}.json"
            if not path.exists():
                continue
            a = json.load(open(path))
            nf, n = cohort_counts(suite, pk, fk)
            f40, f80 = rc.folds_at(a, 40), rc.folds_at(a, 80)
            ident = max((len(aurocs(folds(a, 80), k)) for k, _ in METHODS), default=0)
            if not evaluable(nf, n, ident):
                continue
            hb = [rc.task_auroc(f, "handoff") - rc.task_auroc(f, "baseline") for f in f80.values()
                  if rc.task_auroc(f, "handoff") is not None and rc.task_auroc(f, "baseline") is not None]
            cb = [rc.task_auroc(f, "combined") - rc.task_auroc(f, "baseline") for f in f80.values()
                  if rc.task_auroc(f, "combined") is not None and rc.task_auroc(f, "baseline") is not None]
            p_hb, p_cb = rc.sign_test(hb), rc.sign_test(cb)
            if not hb or not cb:
                raise ValueError("Missing paired readout comparison")
            comparisons.append(f"{pname} & {fname} & {signed(st.mean(hb))} ({p_hb:.3f}) & {signed(st.mean(cb))} ({p_cb:.3f})")
            for m, label in METHODS:
                vals = [rc.task_auroc(f, m) for f in f80.values()]
                vals = [v for v in vals if v is not None]
                p_c = rc.sign_test([v - 0.5 for v in vals], "greater")
                same40, v80 = [], []
                for t, fold in f80.items():
                    if t in f40:
                        ids = {q["id"] for q in fold["methods"][m]["predictions"]}
                        a40, a80 = rc.task_auroc(f40[t], m, keep=ids), rc.task_auroc(fold, m)
                        if a40 is not None and a80 is not None:
                            same40.append(a40); v80.append(a80)
                p_rise = rc.wilcoxon_p([b - c for b, c in zip(v80, same40)]) if same40 else 1.0
                if not same40:
                    raise ValueError("Missing same-episode landmark comparison")
                rise = f"{st.median(same40):.2f}$\\to${st.median(v80):.2f} ({p_rise:.3f})"
                rows.append(f"{pname} & {fname} & {label} & {sum(v > 0.5 for v in vals)}/{len(vals)} ({p_c:.3f}) & {rise}")
    return "\n".join([
        panel("A. Ranking and landmark comparisons", "lllcc",
              "Policy & Situation & Readout & Above chance (tasks, $p$) & Step 40$\\to$80 ($p$)", rows),
        panel("B. Paired readout comparisons at step 80", "llcc",
              "Policy & Situation & Handoff$-$External ($p$) & Combined$-$External ($p$)", comparisons)])


def required_inputs(selected):
    """Preflight JSON dependencies before any table is rendered or written."""
    required = set()
    spatial = {"severities", "within_realistic", "transfer_realistic", "claims_realistic"}
    for suite in ("spatial", "object"):
        if (suite == "spatial" and spatial.intersection(selected)) or (suite == "object" and "object_realistic" in selected):
            required.update(A / f"realistic_{suite}_{family}_{policy}.json"
                            for policy, _ in POLICIES for family, _ in FAMILIES)
    if "probe_realistic" in selected:
        required.update(A / f"realistic_probe_{suite}_{policy}.json"
                        for suite in ("spatial", "object") for policy, _ in POLICIES)
        required.add(A.parent / "exp_20261020/e9_probe_null.json")
    if "pooled_realistic" in selected:
        required.update(A / f"realistic_spatial_pooled_{policy}.json" for policy, _ in POLICIES)
    if "requery_realistic" in selected:
        required.update(A / f"realistic_requery_{family}_{policy}.json"
                        for policy in ("pi05", "molmoact") for family in ("occluder", "moved", "relocated"))
    if "libero_pro" in selected:
        required.update(A / f"{policy}_libero_pro_{condition}.json"
                        for policy in ("pi05", "molmoact") for condition in ("original", "position"))
    if "lightingdark_supp" in selected:
        required.update(A / f"realistic_{suite}_lightingdark_{policy}.json"
                        for suite in ("spatial", "object") for policy, _ in POLICIES)
    return required


def main():
    jobs = dict((name, fn) for name, fn in
                (("severities", severities), ("within_realistic", within), ("pooled_realistic", pooled),
                 ("requery_realistic", requery), ("object_realistic", lambda: within("object")),
                 ("transfer_realistic", transfer), ("probe_realistic", probe), ("claims_realistic", claims),
                 ("lightingdark_supp", lightingdark), ("libero_pro", libero_pro)))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tables', nargs='+', choices=tuple(jobs),
                        default=['within_realistic', 'object_realistic', 'probe_realistic'],
                        help='Retained tables by default; historical extras need their own input cohorts')
    args = parser.parse_args()
    missing = sorted(p for p in required_inputs(args.tables) if not p.is_file())
    if missing:
        raise FileNotFoundError("Required table inputs are missing:\n" + "\n".join(map(str, missing)))
    # Rendering can fail on schema/class-support errors: finish it all before writing.
    rendered = {f"{name}.tex": "% generated by code/gen_realistic_tables.py; do not edit\n" + jobs[name]() + "\n"
                for name in args.tables}
    if set(args.tables).intersection({'severities', 'within_realistic', 'transfer_realistic', 'claims_realistic'}):
        rendered['sign_tests.txt'] = sign_tests() + "\n"
    OUT.mkdir(parents=True, exist_ok=True)
    for name, content in rendered.items():
        (OUT / name).write_text(content)
        print("wrote", OUT / name)


if __name__ == "__main__":
    main()
