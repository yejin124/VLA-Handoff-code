# Reproduction protocols

This document specifies collection and analysis settings for a new replication. It is not a preregistration. The manuscript describes the original ordering of confirmatory, secondary, and exploratory analyses. Results and internal development records are not part of this distribution.

## Collection and state conventions

`config/reproduction/cohorts.json` defines the complete task, initialization, seed, replicate, severity, condition, and output-directory grid. Run every listed configuration for a complete replication. The main LIBERO campaign uses the preserved wrapped-axis-angle collector; the follow-up LIBERO cohorts use the official axis-angle convention. These are distinct experimental conditions. Bridge uses its native state convention. Do not exchange collectors between cohorts.

Collect eight action draws for each observation and execute draw zero. Spread is the mean population standard deviation (`ddof=0`) over the executed action coordinates and time steps. Preserve input images, states, seeds, checkpoint metadata, and query artifacts. Complete-grid, source, and input-digest checks must pass before analysis. A successful episode that ends before a landmark has no active query at that landmark.

## Outcome readouts and inference

The primary occluder cohorts are GR00T Bridge (landmark 20) and GR00T LIBERO-Spatial (landmark 80). LIBERO also has a secondary landmark at 40. The primary scene-geometry feature vector is G; the mean-pooled handoff is H. Fit G, H, and GH on the same task-held-out folds. Reserve the next task in cyclic order and fit on the remaining tasks. Fit standardization and at most 30 PCA components on training data only. The logistic readout uses L2 regularization, C=1, and `liblinear`. Scores are evaluated by AUROC within each held-out task and averaged with equal task weight.

The supervised action-spread features S concatenate mean log standard deviation for each action coordinate, mean log standard deviation for each executed time step, and log scalar spread, with epsilon `1e-8`. Fit S and GS using the same folds and logistic procedure; S has no PCA. G_RBF is the nonlinear scene baseline. DINOv2 and ResNet-18 provide image controls without policy training.

Eligibility rules in the loaders are fixed: report class counts and an ineligible status when support is insufficient. Do not change the rules to obtain a favorable estimate. Original LIBERO within-condition analyses require a failure prevalence between 10% and 80% and at least four tasks with both outcomes at the landmark.

Confirmatory inference uses 2,000 full-refit bootstrap attempts with seed `20261106`, resampling scene or initialization IDs within each observed task. Keep all replicas and collection sources for each selected scene together. Refit preprocessing and readouts in each attempt. Invalid class-support draws are omitted jointly for paired comparisons and counted; require at least 100 valid draws for intervals. Software and convergence errors must be fixed, not excluded as invalid draws. Intervals are conditional on the observed tasks.

One-sided bootstrap p-values are `(1 + count(delta <= 0)) / (1 + valid_draws)`. Apply the analysis module's declared Holm families, including ineligible comparisons conservatively with p=1. Percentile intervals are individual 95% intervals. Keep secondary cells and per-task effects; do not select a favorable landmark, seed, or condition after analysis.

## Input-level comparison

Pair clean and perturbed episodes by policy, suite, task, initialization, and replicate. Occlusion uses the first query; moved-goal comparisons use the first query after movement, with a matched clean query. Episodes without that query are unavailable pairs and must be counted. Fit the handoff risk readout using perturbed observations from other tasks, withholding the test task and next cyclic task. Compare the change in risk score on the logit scale with the change in log spread. Table means use delta_r = mean(paired change) / population_SD(clean values), separately for each signal; these standardized means differ from the per-pair changes used in the correlations. Report task-averaged Spearman correlation, its task-bootstrap interval, and the one-sided sign test with Holm correction over the 12 policy–suite–condition comparisons. Initial-image mismatches limit attribution to goal movement.

## GR00T pre/post interface

Capture mean-pooled backbone tokens before `action_head.vlln` (PRE), the LayerNorm output (MID), and the context consumed by the DiT (POST). Replay the same primary query, checkpoint, state, images, instruction, seed, and GPU model. The original action samples must match bitwise. POST must agree with the stored float16 handoff within relative maximum error `2**-9` (denominator floor `1e-12`). Count failed parity queries; if more than 1% of a cell fails, do not fit that cell.

Use the outcome-readout and full-refit bootstrap procedures above. PRE minus POST is the primary contrast, corrected over the two primary cells. Report MID minus POST, PRE minus MID, GPRE minus GPOST, PRE minus G, and PRE minus Spread as secondary contrasts. Run all PCA 10/50 and C 0.1/10 sensitivities. A nonsignificant contrast does not establish an information-preserving interface.

## Direct handoff intervention

Use GR00T Bridge primary queries. Fit risk and log-spread directions once on all selected handoffs; these directions are interventions rather than held-out predictions. Add the same vector to each context token. Include the baseline, risk direction at scales 1/2/4, spread direction at scales 1/2/4, and four random directions at scales 1/2/4: 19 conditions per query. Random directions use seeds 1–4: draw vectors in the raw handoff's top-30 PCA span, then project them orthogonal to the risk direction and normalize. The final directions need not remain in the original PCA span. Scale each unit direction by its projected handoff standard deviation.

Route queries to their recorded GPU model. Baseline action parity requires maximum absolute error at most `1e-5`. The manipulation check allows at most 5% relative pooled-handoff error because the context arithmetic is bfloat16. Reject incomplete, nonfinite, or invalidly normalized outputs. The positive control requires the scale-2 spread direction to increase log spread with its interval lower bound above `log(1.10)`. Without that control, the intervention cannot identify the cause of a readout–spread gap. Conditional on that control passing, compare the scale-2 risk-minus-mean-random log-spread contrast with log(1.10): an upper 95% bound below the margin supports a response below it, a lower bound above the margin supports a response above it, and otherwise the test is inconclusive. Action movement is the episode-mean L2 displacement of the mean sampled action chunk divided by the median baseline chunk norm, reported as a percentage. Hold that original full-cohort median fixed during bootstrap resampling.

## Sampling-count sensitivity

Use every active primary query from the validated Bridge and SmolVLA cohorts. Query counts come from collection, not from published counts. Retain recorded outcomes; no additional closed-loop rollout is performed for this analysis. For each query, collect 64 sequential chunks under its recorded seed and an independent deterministic seed. Prefixes of length 8/16/32/64 define J; do not select chunks.

Independent seeds use the first four big-endian bytes of a SHA-256 digest, modulo `2**31`. The digest input is `bridge-J64-20261110|` or `smol-J64-20261110|` followed by the raw-root-relative POSIX episode directory. These strings define the original randomization algorithm. Check collisions and equality to the paired original seed.

Prepare immutable receipts binding code, this protocol, input hashes, datasets, checkpoint, and runtime. Original-seed prefixes must reproduce all eight recorded float32 chunks bitwise; both handoffs must pass the `2**-9` relative tolerance. Require finite action arrays of shape `(64,4,7)` for Bridge or `(64,10,7)` for SmolVLA. Any failure stops collection. Resume only with identical receipts and valid completed files.

Recompute J8 baselines from raw records. An external result file is an optional comparison, never an input needed to fit the baseline. Refit G/H/GH/S/GS at every J and seed, with shared bootstrap multiplicities across all views. Validate cached versus uncached fits. Bridge corrects four primary contrasts: GH minus raw Spread and H minus learned S at J64, each under two seeds. SmolVLA corrects eight: those contrasts across two seeds and two suites. Report all other views, seed repeatability, and paired J64-minus-J8/J32 changes. Two seeds do not establish convergence or characterize all Monte Carlo variation.

The public GR00T replication uses the unmodified official source revision in `config/reproduction/gr00t_source.json`. The original runs used a local revision with disabled extensions. Thus source fingerprints differ, and exact replay of original archived trajectories is not guaranteed by this package. A new collection and its replays must use one consistent runtime and pass all parity checks. Raw original inputs and their original runtime are needed for archival verification.

## Ancillary analyses

View restoration branches from recorded simulator states and explicitly chooses the historical or official state convention. Require matched images/actions before attributing a difference to restored visibility. The replay-prefix and simulator-state limitations in the supplement apply. Re-query analyses use the corresponding E5 collector and policy wrapper. Single-trajectory DROID probes and old sampling-repeatability inputs are not distributed; they cannot be regenerated from the fresh cohort grid alone.
