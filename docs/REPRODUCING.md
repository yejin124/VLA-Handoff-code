# Reproducing the experiments

Run from the repository root. The public package has no result JSON, raw episode archive, generated tables, or weights. Output paths below are created by the corresponding collection or analysis stage. A fresh replication can change outcomes; never substitute a paper number when a new estimate differs.

## 1. Select and collect cohorts

Complete [environment setup](ENVIRONMENTS.md), start one matching policy server, and use `code/reproduce.py collect`. `--dry-run` prints the complete command without running a simulator. `--python` chooses the simulator's interpreter; the helper itself requires only the standard library. `--raw-root` defaults to this checkout. Keep that default for the historical table scripts, or place a symlink named `data` pointing to an external output directory.

```bash
python code/reproduce.py list
python code/reproduce.py collect spatial_confirm_gr00t --python "$LIBERO_PYTHON" --port 5662
python code/reproduce.py collect bridge_new --python "$SIMPLER_PYTHON" --port 5612
```

These examples collect 600 LIBERO and 864 Bridge episodes respectively. Other cohorts have the same interface. `config/reproduction/cohorts.json` gives every task, initialization, seed, replicate count, severity, and condition. It contains parameters only.

| Cohort name pattern | Episodes | Collector / convention |
|---|---:|---|
| `{spatial,object}_main_{gr00t,pi05,molmoact}` | 3,600 | Original collector: clean, occluder, moved goal |
| `{spatial,object}_main2_{gr00t,pi05,molmoact}` | 1,200 | Original collector: retained lighting severity |
| `{spatial,object}_main3_{gr00t,pi05,molmoact}` | 1,200 | Original collector: retained relocated goal |
| `{spatial,object}_confirm_{gr00t,pi05}` | 2,400 | Corrected official axis-angle convention |
| `{spatial,object}_corrected_{molmoact,smolvla}` | 800 | Corrected official convention |
| `{spatial,object}_tau0_tau0` | 400 | Corrected official convention |
| `bridge_reused`, `bridge_new` | 288 + 864 | Bridge state convention; same protocol, separate seeds |

`bridge_reused` is the cohort's historical name. Collecting it now produces a new replication, not a new preregistered experiment. The manuscript describes the original chronology; [PROTOCOLS.md](PROTOCOLS.md) specifies the replication settings.

The original LIBERO campaign used wrapped axis angles, which differ from the official GR00T/MolmoAct2 state convention. `recorded_failure_rollout_20261103.py` preserves that archived implementation; `icassp_rollout.py` uses the corrected convention for the follow-up cohorts. Do not silently swap them. The main campaign's superseded lighting/relocation pilot cells are excluded from this 10,752-episode grid.

For parallel collection, use `--shard 0 --nshards 2` and `--shard 1 --nshards 2` on separate matching servers. The complete set of shards must use the same `nshards`. Do not run two writers on one shard. Collectors preserve completed episodes and reject a changed resume configuration.

Each shard writes `config.json`, `server.json`, and episode directories with `summary.json`, `queries.jsonl`, and per-query NPZ artifacts. NPZ files contain sampled actions, executed draw 0, the pooled handoff, state, and stored observations at analysis landmarks. The analyst checks digests, draw-zero identity, seeds, complete grids, and within-task class support. A failed check is an error to investigate, not an exclusion rule to adjust after seeing results.

## 2. Primary and additional occluder analyses

Activate the CPU environment. Run only after all requested cohorts are complete.

```bash
export RAL_ROOT="$PWD" RAL_RAW_ROOT="$PWD"
# The primary family plus its raw-spread contrasts.
python code/spread_contrast.py 8 data/confirmatory/primary \
  gr00t_bridge_confirm,gr00t_spatial_confirm
# Supervision-matched spread control.
python code/learned_spread_contrast.py 8 data/learned_spread
python code/primary_predictions.py data/confirmatory/primary_predictions.json
```

`spread_contrast.py` defaults to the two primary cohorts when the third argument is omitted; to fit **all** additional cohorts, pass the full comma-separated cohort list from `ral_confirm_analysis.COHORTS`:

```bash
python code/spread_contrast.py 8 data/confirmatory/all \
  gr00t_bridge_confirm,gr00t_bridge_confirm_new,gr00t_spatial_confirm,gr00t_object_confirm,pi05_spatial_confirm,pi05_object_confirm,molmoact_spatial,molmoact_object,smolvla_spatial,smolvla_object,tau0_spatial,tau0_object
```

The explicit command populates the full table set using 2,000 full-refit bootstrap attempts. Output is `<directory>/confirm_results.json`. Use a different directory for a different analysis specification.

Fit the DINOv2 control with a GPU environment that also has the analysis packages:

```bash
python code/ral_confirm_robustness.py --workers 8 --output data/confirmatory/image
```

This extracts image embeddings from the scored queries and refits the primary predictors. It also runs the PCA/regularization sensitivity settings. The pre/post-interface sensitivity is a separate analysis below. The bootstrap intervals are conditional on observed tasks; they do not estimate generalization to a new population of tasks.

## 3. Original LIBERO conditions and scene-change decoding

Create within-condition views and the analysis folder:

```bash
python code/realistic_views.py
mkdir -p data/icassp2027/analysis paper/tables
```

Analyze all 24 suite–policy–condition combinations. Occluder/moved use `main`, lighting uses `main2`, and relocated uses `main3`:

```bash
for suite in spatial object; do
  for policy in gr00t pi05 molmoact; do
    for condition in occluder lighting moved relocated; do
      case "$condition" in
        lighting) campaign=main2 ;;
        relocated) campaign=main3 ;;
        *) campaign=main ;;
      esac
      python code/analyze_icassp_trials.py \
        --input "data/icassp2027/realistic_views/${suite}_${campaign}/${policy}/${condition}" \
        --output "data/icassp2027/analysis/realistic_${suite}_${condition}_${policy}.json" \
        --landmarks 40 80 || exit 1
    done
  done
done
python code/realistic_probe.py spatial
python code/realistic_probe.py object
python code/exp_20261020_e9_probe_null.py
```

For the image-encoder controls, run `exp_20261020_e3_extract.py` in the GPU environment, then `exp_20261020_e3_analysis.py` in the CPU environment. They use the same recorded frames, training-only transforms, and task folds. Their output is `data/icassp2027/exp_20261020/`.

Input-level uncertainty changes (T6) use:

```bash
python code/injection_input_level.py data/icassp2027/realistic data/injection/exp1_input_level.json
```

## 4. Interface and handoff intervention replays

These operate on the recorded primary-cohort inputs and outcomes. In the GR00T policy environment, set the source and matching checkpoint explicitly. For Bridge:

```bash
export GR00T_ROOT=/path/to/Isaac-GR00T
export GR00T_MODEL_PATH=/path/to/GR00T-N1.7-SimplerEnv-Bridge
PREPOST_MODE=bridge POLICY_SERVER_PORT=5952 CUDA_VISIBLE_DEVICES=0 python code/serve_prepost_gr00t.py
```

For Spatial, use `GR00T_MODEL_PATH=/path/to/GR00T-N1.7-LIBERO/libero_spatial` and `PREPOST_MODE=libero`. Start a server for each GPU model present in that cohort, and pass all their ports to the driver. Collect each primary cohort in the analysis environment:

```bash
python code/prepost_driver.py run data/prepost gr00t_bridge_confirm 5952
python code/prepost_driver.py collect data/prepost gr00t_bridge_confirm
# Repeat the two commands for gr00t_spatial_confirm with its LIBERO server.
python code/prepost_contrast.py 8 data/prepost data/prepost/main
```

PCA 10/50 and C 0.1/10 sensitivities use the environment variables described in `prepost_contrast.py`; save each setting separately. For direct handoff intervention, start a server in the GR00T environment:

```bash
GR00T_ROOT=/path/to/Isaac-GR00T GR00T_MODEL_PATH=/path/to/GR00T-N1.7-SimplerEnv-Bridge \
POLICY_SERVER_PORT=5950 CUDA_VISIBLE_DEVICES=0 python code/serve_injection_gr00t.py
```

Pass ports for every recorded GPU model if the collection used multiple models. In the analysis environment, run:

```bash
python code/injection_handoff_driver.py directions data/injection/exp2_bridge
python code/injection_handoff_driver.py run data/injection/exp2_bridge 5950
python code/injection_handoff_analysis.py data/injection/exp2_bridge
```

Use a fresh output directory and a GPU matching the recorded inputs. The analysis checks all 19 conditions, prefix parity, successful manipulation, finite values, and positive normalization denominators. Do not interpret a failed positive control as causal evidence.

## 5. J64 sampling follow-up

These protocols resample the active queries from a completed, validated collection without changing its episode outcomes. New collections may have different active-query counts; the drivers use the validated loader's identities and counts. Keep the same checkpoint/runtime and GPU model used for that collection. All prefix and provenance checks remain required. To verify the paper's archived trajectories, obtain the original raw inputs and matching runtime from the authors; the unmodified official GR00T source is not certified as bitwise equivalent to that original local source.

Start a sampling server in each matching policy environment. Keep the checkpoint, device model, and runtime used for the base collection:

```bash
# GR00T environment; repeat on another device/port if that cohort requires it.
GR00T_ROOT=/path/to/Isaac-GR00T GR00T_MODEL_PATH=/path/to/GR00T-N1.7-SimplerEnv-Bridge \
POLICY_SERVER_PORT=5980 POLICY_EXECUTE_STEPS=4 CUDA_VISIBLE_DEVICES=0 python code/serve_bridge_sampling.py

# SmolVLA environment.
SMOLVLA_CHECKPOINT=/path/to/smolvla_libero HF_HUB_OFFLINE=1 \
POLICY_SERVER_PORT=6000 POLICY_EXECUTE_STEPS=10 CUDA_VISIBLE_DEVICES=0 python code/serve_smol_sampling.py
```

In the CPU analysis environment, use new output directories and the ports of running compatible servers (one port suffices when the collection used one GPU model):

```bash
python code/bridge_sampling_replay.py prepare /path/to/new_bridge_replay
python code/bridge_sampling_replay.py run /path/to/new_bridge_replay 5980
python code/bridge_sampling_replay.py verify /path/to/new_bridge_replay
python code/bridge_sampling_analysis.py /path/to/new_bridge_replay --workers 8

python code/smol_sampling_replay.py prepare /path/to/new_smol_replay
python code/smol_sampling_analysis.py baseline /path/to/new_smol_replay
python code/smol_sampling_replay.py run /path/to/new_smol_replay 6000
python code/smol_sampling_replay.py verify /path/to/new_smol_replay
python code/smol_sampling_analysis.py analyze /path/to/new_smol_replay --workers 8
```

Both analyses refit from raw features without archived result JSON. Optionally pass `--reference-results /path/to/original/confirm_results.json` to the Bridge analysis or SmolVLA `baseline` command to additionally check an external historical analysis. Baseline recomputation, original-prefix identity, dataset integrity, and cached/uncached fit checks remain mandatory.

Both replay drivers read source code and `docs/PROTOCOLS.md` from their code checkout; `RAL_RAW_ROOT` selects the episode-record root. Use an output directory outside that raw root for SmolVLA. The checks deliberately reject source/runtime changes within a prepared run; after changing code, prepare a new output directory.

```bash
python code/curate_sampling_results.py /path/to/new_bridge_replay /path/to/new_smol_replay data/sampling_summaries
```

## 6. Ancillary experiments

These collections are additional to the 10,752-episode grid. Run from the repository root after environment setup; use fresh output directories, or resume only an identical configuration. The commands preserve the original experimental parameters, while fresh outcomes and eligible sample counts can differ.

### Supplement S5 / Table S4: overlapping-chunk re-query experiment

Collect Spatial only, with π0.5 and MolmoAct2, for occluder, moved goal, and relocated goal. Each policy–condition cell has 10 tasks × 20 initialization IDs × one replicate = 200 episodes, split into two shards (1,200 episodes total). The initialization order below is intentional because it determines shard assignment. Each query samples J=8 full 10-step plans, executes their first five steps, and re-queries; consecutive plans overlap by five steps. The episode horizon is 280 steps. The E5 collector retains the original wrapped-axis-angle state convention; it has no `--state-convention` option.

Start the matching E5 server in its policy environment, one policy at a time. Replace the local installation paths and choose an available GPU. The clients below use ports 5661 and 5660, respectively; port numbers do not define the experimental condition.

```bash
# π0.5 environment:
OPENPI_ROOT=/path/to/openpi PI05_CHECKPOINT=/path/to/pi05_libero_pytorch \
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=0 POLICY_SERVER_PORT=5661 POLICY_SERVER_J=8 \
  python code/serve_icassp_pi05_e5.py

# MolmoAct2 environment:
MOLMO_MODEL_PATH=/path/to/MolmoAct2-LIBERO CUDA_VISIBLE_DEVICES=0 \
POLICY_SERVER_PORT=5660 POLICY_SERVER_J=8 \
  python code/serve_icassp_molmo_e5.py
```

In a separate terminal with the LIBERO client configuration, define this Bash function and invoke it for the running server. Both shards are executed sequentially, so one matching server is sufficient.

```bash
collect_requery() {
  local policy="$1" model="$2" port="$3" moved_severity="$4"
  local condition severity shard
  local -a wrist
  for condition in occluder moved relocated; do
    wrist=()
    case "$condition" in
      occluder) severity=1.0; wrist=(--apply-to-wrist) ;;
      moved) severity="$moved_severity" ;;
      relocated) severity=0.25 ;;
    esac
    for shard in 0 1; do
      "$LIBERO_PYTHON" code/icassp_rollout_e5.py \
        --out "data/icassp2027/realistic/requery_${condition}/${policy}" \
        --port "$port" --model "$model" --suite libero_spatial \
        --tasks 0,1,2,3,4,5,6,7,8,9 \
        --episodes 35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,8,9,10,11,12 \
        --replicates 1 --conditions "$condition" --shard "$shard" --nshards 2 \
        --draws 8 --execute-steps 5 --max-steps 280 --onset 0 \
        --severity "$severity" --move-step 30 --move-duration 10 \
        --seed 20261011 "${wrist[@]}" || return 1
    done
  done
}
# Run each call while its matching E5 server is active:
collect_requery pi05 pi0.5 5661 0.20
collect_requery molmoact MolmoAct2 5660 0.15
```

After all six cells are complete, run the E5 analyzer in the CPU analysis environment. The ordinary `analyze_icassp_trials.py` alone does not compute the backbone or temporal-consistency readouts. `analyze_icassp_e5.py` adds them to the same task-held-out fits; its MMD bandwidth uses training tasks only. These are STAC-style and backbone probes, not reproductions of an official STAC or SAFE implementation.

```bash
for policy in pi05 molmoact; do
  for condition in occluder moved relocated; do
    python code/analyze_icassp_e5.py \
      --input "data/icassp2027/realistic/requery_${condition}/${policy}" \
      --output "data/icassp2027/analysis/realistic_requery_${condition}_${policy}.json" \
      --landmarks 40 80 || exit 1
  done
done
python code/gen_realistic_tables.py --tables requery_realistic
cp paper/tables/requery_realistic.tex paper/tables/supp_requery.tex
```

The detailed re-query export retains six readouts and both landmarks in one flat table. Ineligible cells keep their episode/task counts and dashes. The compact Supplement Table S4 selects step 80 and its three relevant readouts in section 7. It reads generated analysis JSON and raw episode counts. The historical lighting re-query pilot is not needed for this retained table.

### Supplement S6: paired view restoration

First finish the six `{spatial,object}_main_{pi05,gr00t,molmoact}` cohorts and the within-condition analyses in sections 1 and 3. The documented main-cohort grid uses `recorded_failure_rollout_20261103.py`, including for a fresh replication. Its prefixes therefore require `--state-convention recorded`. Use `official` only when the actual source prefixes were collected with the corrected official state convention; do not interchange them.

Use each main cohort's matching standard server from `ENVIRONMENTS.md`, with the same checkpoint, source, numerical runtime, and GPU model used to collect that cohort. For GR00T select the corresponding `libero_spatial` or `libero_object` checkpoint and eight executed steps; the π0.5 and MolmoAct2 standard servers return ten-step chunks. Collection below runs one suite–policy cell at a time, so start only the needed server. The port choices match `ENVIRONMENTS.md`.

```bash
collect_restoration() {
  local suite="$1" policy="$2" model="$3" port="$4"
  "$LIBERO_PYTHON" code/exp_20261029_view_restore.py \
    --source "data/icassp2027/realistic/${suite}_main/${policy}" \
    --out "data/icassp2027/exp_20261029/${suite}/${policy}" \
    --port "$port" --model "$model" --shard 0 --nshards 1 \
    --state-convention recorded
}
# Invoke each line with its matching server active:
collect_restoration spatial pi05 pi0.5 5672
collect_restoration spatial gr00t GR00T 5662
collect_restoration spatial molmoact MolmoAct2 5670
collect_restoration object pi05 pi0.5 5672
# Restart the GR00T server with the Object checkpoint before this call:
collect_restoration object gr00t GR00T 5662
collect_restoration object molmoact MolmoAct2 5670
```

The collector allocates both continue and restore branches to every source occluder episode with `steps > 80`; it does not randomly choose which episodes receive a branch. It sorts the source paths and partitions them by `sources[shard::nshards]`; the commands above use one shard. To split a cell, run all shard indices with one common `nshards` and no duplicate writers. Outputs remain directly under the cell directory, rather than in new shard subdirectories. Never use `--replay-only` in that same output directory: replay-only records do not contain branch outcomes.

Each branch starts in a fresh simulator and replays the recorded actions through the end of the chunk queried at step 80. State-coordinate errors above `1e-5` produce an explicit replay-mismatch record. At the next query, continue retains the occluder on both cameras and restore clears it. Subsequent queries request **J=1**, reuse the source's policy seeds, and run to the source's 280-step horizon. Source task/initialization/seed parameters are inherited from the corresponding `config.json`; there is no new allocation seed. In the main grid the Spatial seed is 20261009, Object seed 20261010, initialization IDs are 15–34, tasks are 0–9, and there is one replicate. Original source queries used J=8, so matching the seed alone does not establish equality of their first actions with J=1 continuations.

In the CPU environment, analyze and render:

```bash
python code/exp_20261029_view_restore_analysis.py --suites spatial,object \
  --output data/view_restore/view_restore_analysis.json --no-table
python - <<'PY' || exit 1
import json
from pathlib import Path
p = Path('data/view_restore/view_restore_analysis.json')
r = json.loads(p.read_text())
expected = {f'{s}|{p}' for s in ('spatial', 'object')
            for p in ('pi05', 'gr00t', 'molmoact')}
if set(r['cells']) != expected:
    raise SystemExit('Missing restoration cells; inspect branch records and eligible folds.')
for key, c in r['cells'].items():
    print(key, {k: c[k] for k in ('n', 'active_at_80', 'complete',
        'max_replay_error', 'continue_matches_source', 'first_action_matches_source')})
    if not c['complete'] or c['max_replay_error'] > 1e-5:
        raise SystemExit('Incomplete restoration analysis; inspect recorded mismatches and missing prefixes.')
PY
python code/make_view_restore_table.py \
  data/view_restore/view_restore_analysis.json paper/tables/supp_view_restore.tex
```

Do not force the original active-prefix count or replace new outcomes. Inspect recorded `replay_mismatch`, `first_image_matches_source`, and `first_actions_match_source` fields alongside the analysis; coordinate replay checks do not certify the full simulator state. Attribute differences specifically to restored visibility only where the required image/action parity holds. If fresh source folds lack usable calibrated alarms, the current restoration analyzer cannot evaluate those gates; retain that limitation instead of treating a missing alarm as no intervention.

Restoration allocation is evaluated after collecting both outcomes. Each gate uses its source step-80 held-out alarm, based on the calibration task's upper 90th-percentile success score. Its random comparator has the **same realized intervention rate within each task**, using the expected success count analytically; it is not a new random rollout or a global-rate comparison. Selective effects first average policies within each suite–task cluster, then weight those clusters equally. The analysis uses 10,000 cluster-bootstrap resamples with seed 20261029, a tie-excluding sign test, and Holm correction across the four gate methods (handoff, native spread, baseline, combined). The displayed table includes the handoff and native-spread gates.

The earlier single-trajectory DROID study in Supplement S8 and the historical independent-seed sampling-repeatability inputs for Section S4 are outside this fresh-collection package. These commands do not regenerate those studies.

## 7. Generate tables and figures

Run these commands after the corresponding analyses above. The formatters read your generated estimates; they do not contain the paper's values. Missing required inputs cause an error before writing a partial table set. Ineligible estimates appear as dashes with counts and status retained. The TeX layouts include replication status information; CSV files preserve unrounded values.

```bash
mkdir -p paper/tables paper/figures

# Original-campaign tables, including the probe table consumed by Figure 4.
python code/gen_realistic_tables.py

# Main Tables 2, 3, 5 and detailed image-control export (TeX and CSV).
# The image export needs the completed image extraction and E3 analysis from section 3.
python code/make_current_tables.py \
  --confirm-results data/confirmatory/all/confirm_results.json --output paper/tables

# Main Table 4, after the input-level analysis in section 3.
python code/make_injection_table.py data/injection/exp1_input_level.json paper/tables/input_injection.tex

# Detailed confirmatory, image-control and policy exports for Supplement Tables S2–S3.
python code/make_supp_confirm_tables.py data/confirmatory paper/tables \
  --confirm-results data/confirmatory/all/confirm_results.json \
  --robustness-results data/confirmatory/image/robustness_results.json \
  --policy-results data/confirmatory/all/confirm_results.json \
  --spread-results data/confirmatory/all/confirm_results.json

# Detailed spread/interface exports supporting the text in Supplement Section S3.
python code/make_learned_spread_table.py data/learned_spread/confirm_results.json paper/tables
python code/make_prepost_table.py data/prepost/main/confirm_results.json paper/tables

# Figures 3 and 4. Figure 3 uses a stored clean Spatial task-3 input frame.
python code/plot_perturbation_examples.py data
python code/plot_figures.py data/confirmatory/all/confirm_results.json \
  data/confirmatory/primary_predictions.json paper/figures paper/tables/probe_realistic.tex

# Conceptual Figures 1 and 2; no experimental results required.
python code/draw_uncertainty_pipeline.py
python code/draw_model_architectures.py

# Detailed J/seed exports, after the J64 analysis and curation in section 5.
python code/make_sampling_tables.py data/sampling_summaries paper/tables
```

`make_current_tables.py` writes `main_confirm`, `main_goal`, `main_occluder`, and `supp_image_encoders` as `.tex` and `.csv` files. To generate a subset, add `--tables main2 main3 main5` or `--tables supp2`. Counts for the original LIBERO tables come from episode summaries; `--episodes-root` selects a different input location.

For the interface sensitivity results summarized in Supplement Section S3, run all four predefined settings and pass every result to the formatter:

```bash
PREPOST_PCA=10 python code/prepost_contrast.py 8 data/prepost data/prepost/pca10
PREPOST_PCA=50 python code/prepost_contrast.py 8 data/prepost data/prepost/pca50
PREPOST_C=0.1 python code/prepost_contrast.py 8 data/prepost data/prepost/c0p1
PREPOST_C=10 python code/prepost_contrast.py 8 data/prepost data/prepost/c10
python code/make_prepost_table.py data/prepost/main/confirm_results.json paper/tables \
  'PCA 10=data/prepost/pca10/confirm_results.json' 'PCA 50=data/prepost/pca50/confirm_results.json' \
  'C 0.1=data/prepost/c0p1/confirm_results.json' 'C 10=data/prepost/c10/confirm_results.json'
```

Section 6 gives the separate re-query and restoration collection/analysis commands. Supplement Section S4's earlier repeatability study needs the historical inputs described there. Generate the intervention table and the five compact tables after all detailed exports above are available:

```bash
python code/make_handoff_injection_table.py \
  data/injection/exp2_bridge/analysis.json paper/tables/supp_handoff_injection.tex
python code/make_compact_supp_tables.py paper/tables paper/tables
```

The compact formatter copies numerical cells from your generated estimates. It keeps the complete planned subsets, including negative effects and ineligible cells, and checks all inputs before writing. It does not overwrite the detailed exports.

| Supplement table | Generated file | Content |
| --- | --- | --- |
| S1 | `supp_goal_summary.tex` | Both goal perturbations, all policies and suites |
| S2 | `supp_confirm_summary.tex` | GR00T scene and image controls, all landmarks |
| S3 | `supp_policy_summary.tex` | Policy contrasts at both landmarks |
| S4 | `supp_requery_summary.tex` | Step-80 temporal consistency, all six cells |
| S5 | `supp_handoff_injection.tex` | All intervention directions and scales |
| S6 | `supp_sampling_summary.tex` | All J64 contrasts and both seed streams |

Protocol settings, supervised-spread and interface sensitivity, earlier repeatability, restoration and intermediate J results are described in the supplement text. Their detailed analysis exports remain available through the commands above rather than as additional manuscript tables.

## 8. Protocols and third-party inputs

[PROTOCOLS.md](PROTOCOLS.md) specifies the analysis settings, validity checks, and interpretation limits. It contains no experimental results or internal development history. See [third-party inputs and software](THIRD_PARTY.md) for provenance and licensing.
