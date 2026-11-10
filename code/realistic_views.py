#!/usr/bin/env python3
"""Per-condition views of a mixed-condition cohort, for the within-situation analyses.

The main realistic cohorts collect clean, occluder, lighting and moved on the same initial
states in one client run (one config.json per shard). The analyzer pools everything under a
root, so for a within-situation analysis this script builds
data/icassp2027/realistic_views/<cohort>/<policy>/<condition>/shardNN/ with symlinks to that
condition's trial directories and a copy of the shard's config.json (the analyzer verifies
its digest, so it is copied unchanged).

    python code/realistic_views.py data/icassp2027/realistic/spatial_main
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "icassp2027" / "realistic_views"


EXPECTED = {"main2": ("lighting",), "main3": ("relocated",)}
POLICIES = ("pi05", "gr00t", "molmoact")


def cohort_complete(suite, coll, per_cell=200):
    """True only when every policy has `per_cell` complete episodes of every condition that the
    collection is defined to carry. A chain-log "done" line is not used: the chain also writes
    "done" after a step that reported FAILED, so completion is read from the data."""
    import json
    root = ROOT / "data" / "icassp2027" / "realistic" / f"{suite}_{coll}"
    for policy in POLICIES:
        for cond in EXPECTED[coll]:
            n = 0
            for f in root.glob(f"{policy}/shard*/t*_e*_r*_{cond}/summary.json"):
                if json.load(open(f)).get("complete"):
                    n += 1
            if n != per_cell:
                return False
    return True


def build(cohort):
    cohort = Path(cohort).resolve()
    made = 0
    for policy_dir in sorted(p for p in cohort.iterdir() if p.is_dir()):
        for shard in sorted(policy_dir.glob("shard*")):
            for trial in sorted(shard.glob("t*_e*_r*_*")):
                if not (trial / "summary.json").exists():
                    continue
                condition = trial.name.rsplit("_", 1)[1]
                if cohort.name.endswith("_main2") and condition == "relocated":   # superseded by *_main3 (2026-10-13)
                    continue
                view = OUT / cohort.name / policy_dir.name / condition / shard.name
                view.mkdir(parents=True, exist_ok=True)
                if not (view / "config.json").exists():
                    shutil.copy(shard / "config.json", view / "config.json")
                link = view / trial.name
                if not link.exists():
                    link.symlink_to(trial)
                    made += 1
    print(f"{cohort.name}: {made} new links under {OUT / cohort.name}")


def build_pooled(name, parts):
    """realistic_views/<name>/<policy>/shardNN_<k>/ with the listed conditions of each collection.

    parts: list of (collection_dir, [conditions]). Each source shard keeps its own config.json copy
    in its own view shard directory, so the analyzer's digest check still passes."""
    for k, (cohort, conditions) in enumerate(parts):
        cohort = Path(cohort).resolve()
        for policy_dir in sorted(p for p in cohort.iterdir() if p.is_dir()):
            for shard in sorted(policy_dir.glob("shard*")):
                view = OUT / name / policy_dir.name / f"{shard.name}_{k}"
                view.mkdir(parents=True, exist_ok=True)
                if not (view / "config.json").exists():
                    shutil.copy(shard / "config.json", view / "config.json")
                for trial in sorted(shard.glob("t*_e*_r*_*")):
                    if trial.name.rsplit("_", 1)[1] in conditions and (trial / "summary.json").exists():
                        link = view / trial.name
                        if not link.exists():
                            link.symlink_to(trial)
    print(f"pooled view {OUT / name}")


def prepare():
    """Rebuild all views required by the complete retained manuscript campaign."""
    data = ROOT / "data/icassp2027/realistic"
    for suite in ("spatial", "object"):
        parts = []
        for collection, conditions in (("main", ["clean", "occluder", "moved"]),
                                       ("main2", ["lighting"]), ("main3", ["relocated"])):
            directory = data / f"{suite}_{collection}"
            if not directory.is_dir() or (collection != "main" and not cohort_complete(suite, collection)):
                raise ValueError(f"Missing or incomplete retained collection: {directory}")
            build(directory)
            parts.append((directory, conditions))
        build_pooled(f"{suite}_pooled", parts)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        prepare()
        sys.exit(0)
    if sys.argv[1:] == ["--prepare"]:
        prepare()
        sys.exit(0)
    if sys.argv[1:2] == ["--complete"]:      # --complete SUITE COLL -> exit 0 iff the cohort is complete
        sys.exit(0 if cohort_complete(sys.argv[2], sys.argv[3]) else 1)
    if sys.argv[1:2] == ["--pooled"]:        # --pooled NAME DIR:cond,cond DIR:cond ...
        parts = [(spec.split(":")[0], spec.split(":")[1].split(",")) for spec in sys.argv[3:]]
        build_pooled(sys.argv[2], parts)
    else:
        for c in sys.argv[1:]:
            build(c)
