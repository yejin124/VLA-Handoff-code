# VLA-Handoff

Code for *Outcome Uncertainty in Dual-System VLAs: Comparing Handoff Readouts and Action Spread*, by Yejin Jo and Minsu Jang.

![Overview of the handoff readout and sampled-action spread](docs/assets/fig1_pipeline.svg)

## The paper in brief

A dual-system vision-language-action (VLA) policy passes a representation, the *handoff*, from a vision-language model (System 2) to a flow- or diffusion-based action expert (System 1). The paper asks whether the outcome uncertainty that a supervised readout recovers from the handoff is also expressed in the spread of the action expert's sampled actions.

Five released policies (GR00T-N1.7, MolmoAct2, π0.5, SmolVLA, τ0) are run frozen for 10,752 closed-loop episodes on LIBERO and SimplerEnv-Bridge under controlled camera and goal perturbations. Within each policy, suite, and condition, the failure-ranking AUROC of a task-held-out handoff readout is compared with that of the spread of eight sampled action chunks. Controls include a scene-geometry predictor, a supervised readout of spread fitted on the same folds, readouts fitted before and after GR00T's interface, a direct intervention on the handoff, and replays with 64 action samples.

The handoff adds failure-ranking information beyond scene geometry, and in several settings (GR00T on Bridge, SmolVLA on LIBERO) action spread ranks failures clearly worse than the handoff readout; in others the two are close. Action spread can therefore underrepresent outcome uncertainty that is accessible at the action expert's input. The direct intervention failed its positive control, so the cause of the gap is not established.

This repository holds the collection and analysis code, the cohort configurations, and the fixed occluder textures. Saved results, episode records, and model weights are not included; the original episode records are available from the authors on request.

## Reproduce

1. CPU environment (Linux, Python 3.11):

   ```bash
   python3.11 -m venv .venv && source .venv/bin/activate
   python -m pip install -r requirements.txt
   python -m unittest discover -s tests
   ```

2. Collect episodes. Install LIBERO or SimplerEnv and start the matching policy server as described in [docs/ENVIRONMENTS.md](docs/ENVIRONMENTS.md), then run the collector with the simulator's Python. The 30 configurations listed by `reproduce.py list` cover the 10,752 episodes of the paper.

   ```bash
   python code/reproduce.py list
   python code/reproduce.py configure-libero "$LIBERO_ROOT"
   export LIBERO_CONFIG_PATH="$PWD/config/local/libero" MUJOCO_GL=egl
   python code/reproduce.py collect spatial_confirm_gr00t --python "$LIBERO_PYTHON" --port 5662
   ```

3. Analyze the collected cohorts in the CPU environment:

   ```bash
   export RAL_ROOT="$PWD" RAL_RAW_ROOT="$PWD"
   python code/spread_contrast.py 8 data/confirmatory/primary gr00t_bridge_confirm,gr00t_spatial_confirm
   python code/learned_spread_contrast.py 8 data/learned_spread
   ```

   The remaining analyses (original LIBERO conditions, pre/post-interface replay, handoff intervention, J = 64 replays) are in [docs/REPRODUCING.md](docs/REPRODUCING.md).

A new simulator run is a replication: it does not reproduce the archived trajectories bitwise. The collection and analysis settings are in [docs/PROTOCOLS.md](docs/PROTOCOLS.md), and [VALIDATION.md](VALIDATION.md) states what was checked.

## License

Original code is licensed under [Apache 2.0](LICENSE); see [NOTICE](NOTICE) and [docs/THIRD_PARTY.md](docs/THIRD_PARTY.md) for third-party material. Citation metadata is in [CITATION.cff](CITATION.cff).
