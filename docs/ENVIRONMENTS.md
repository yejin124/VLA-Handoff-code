# Environments and model inputs

Run CPU analysis, simulator clients, and policy servers in separate environments. Install each upstream project's dependencies at the indicated revision; do not install the CPU analysis requirements over a GPU environment. All commands below are run from the VLA-Handoff repository root unless explicitly stated otherwise. Paths such as `/path/to/...` must be replaced with local installations.

## CPU analysis

Use Linux, Python 3.11, and `requirements.txt`. The bootstrap uses `multiprocessing` with `fork`. Cairo and fonts (Liberation Sans and Nimbus Sans) are needed only for figure rendering. A CPU-only installation can list every cohort, run the tests, and fit readouts from locally generated records.

## LIBERO client

Install [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) and its simulator dependencies following its instructions. Keep the supplied task BDDL and initialization files. Set:

```bash
export LIBERO_ROOT=/path/to/LIBERO
export LIBERO_PYTHON=/path/to/libero-environment/bin/python
python code/reproduce.py configure-libero "$LIBERO_ROOT"
export LIBERO_CONFIG_PATH="$PWD/config/local/libero"
export PYTHONPATH="$LIBERO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl
```

`configure-libero` validates the checkout and writes its local paths into `config/local/libero/config.yaml`. Collectors render 256 × 256 agent and wrist views, apply the protocol's rotation, record all J=8 draws, and execute draw 0. Set `MUJOCO_EGL_DEVICE_ID` if the simulator needs an explicit visible device. Ten original donor textures are included under `assets/occluder`; set `VLA_DONOR_BANK` only to deliberately use a different input bank.

## Policy servers

The table lists the recorded versions, not a claim that every other version is incompatible. Exact checkpoint-file and source fingerprints are in `config/reproduction/<policy>.json`. Download checkpoints from their official repositories; compare their files with these fingerprints. If a fingerprint differs, record the new version and treat the run as a replication with different software.

| Policy | Official source / checkpoint | Recorded environment | Executed steps |
|---|---|---|---:|
| GR00T-N1.7 LIBERO | [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T), [LIBERO checkpoints](https://huggingface.co/nvidia/GR00T-N1.7-LIBERO) | See `gr00t_spatial.json` and `gr00t_object.json` | 8 |
| GR00T-N1.7 Bridge | [Bridge checkpoint](https://huggingface.co/nvidia/GR00T-N1.7-SimplerEnv-Bridge), revision `940134b3c2948ccfdf8e7393f2d2ca869dc42833` | Python 3.12.13, torch 2.9.0+cu128, transformers 4.57.3 | 4 |
| MolmoAct2 | [MolmoAct2-LIBERO](https://huggingface.co/allenai/MolmoAct2-LIBERO) | Python 3.11.15, torch 2.5.1+cu124, transformers 5.3.0 | 10 |
| π0.5 | [OpenPI](https://github.com/Physical-Intelligence/openpi), `pi05_libero` PyTorch checkpoint | OpenPI `215abfb217dbac7d5f1273282331b9b1866c0479`, torch 2.7.1+cu126, NumPy 1.26.4 | 10 |
| SmolVLA | [lerobot/smolvla_libero](https://huggingface.co/lerobot/smolvla_libero), revision `a24f1f380461b665158d52acd140152c79e8b36d` | LeRobot 0.6.1, torch 2.11.0+cu130, transformers 5.5.4, NumPy 2.2.6 | 10 |
| τ0 low-level policy | [tau-0-vla](https://github.com/sii-research/tau-0-vla), official LIBERO checkpoint | source `d804047f7b2d61c45077daada3183703aeed8cf5`, Python 3.11, torch 2.7.1+cu128, transformers 5.5.4 | 8 |

For a fresh GR00T replication, use the unmodified official source at `376ba890cff8c9de64d71d982772a9c36185fdd7` (also recorded in `config/reproduction/gr00t_source.json`). The public servers attach measurement hooks directly; no model-source patch is required.

```bash
git clone https://github.com/NVIDIA/Isaac-GR00T.git /path/to/Isaac-GR00T
git -C /path/to/Isaac-GR00T checkout 376ba890cff8c9de64d71d982772a9c36185fdd7
```

Follow the upstream installation instructions at that revision. Original recorded runs used a local source revision with disabled extensions. This package does not distribute those extensions or claim that the official source reproduces original trajectories bitwise. Keep one source/runtime for a new collection and its replay, and retain the automatically recorded fingerprints. GPU inference on this clean official-source configuration has not been rerun during packaging.

Start only the server needed for the current cohort. The examples use `python` from that policy's environment and GPU 0; choose an available GPU and port locally.

```bash
# GR00T LIBERO: choose libero_spatial or libero_object to match the cohort.
export GR00T_ROOT=/path/to/Isaac-GR00T
export GR00T_MODEL_PATH=/path/to/GR00T-N1.7-LIBERO/libero_spatial
CUDA_VISIBLE_DEVICES=0 POLICY_SERVER_PORT=5662 POLICY_EXECUTE_STEPS=8 \
  python code/serve_icassp_gr00t.py

# GR00T Bridge: use the Bridge checkpoint and the SimplerEnv environment.
GR00T_MODEL_PATH=/path/to/GR00T-N1.7-SimplerEnv-Bridge \
CUDA_VISIBLE_DEVICES=0 POLICY_SERVER_PORT=5612 POLICY_EXECUTE_STEPS=4 \
  python code/serve_simpler_gr00t.py

# MolmoAct2 uses the checkpoint's local custom model code and normalizers.
MOLMO_MODEL_PATH=/path/to/MolmoAct2-LIBERO CUDA_VISIBLE_DEVICES=0 \
POLICY_SERVER_PORT=5670 python code/serve_icassp_molmo.py

# π0.5: convert/download the PyTorch checkpoint using OpenPI's instructions.
OPENPI_ROOT=/path/to/openpi PI05_CHECKPOINT=/path/to/pi05_libero_pytorch \
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=0 POLICY_SERVER_PORT=5672 \
  python code/serve_icassp_pi05.py

# SmolVLA: cache HuggingFaceTB/SmolVLM2-500M-Video-Instruct beforehand.
SMOLVLA_CHECKPOINT=/path/to/smolvla_libero HF_HUB_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=0 POLICY_SERVER_PORT=5910 POLICY_EXECUTE_STEPS=10 \
  python code/serve_icassp_smolvla.py

# τ0: install the upstream deployment environment, including its FLA dependency.
TAU0_ROOT=/path/to/tau-0-vla TAU0_CHECKPOINT=/path/to/tau-0-vla-libero \
CUDA_VISIBLE_DEVICES=0 POLICY_SERVER_PORT=5920 POLICY_EXECUTE_STEPS=8 \
  python code/serve_icassp_tau0.py
```

The SmolVLA sampling environment additionally used `num2words==0.5.14` and `docopt==0.6.2`. The τ0 environment used `flash-linear-attention==0.5.0`; see [PROTOCOLS.md](PROTOCOLS.md) for parity requirements. Every server binds to localhost and publishes model, checkpoint, sampling, and source metadata on ping. Collectors save this metadata and reject conflicting resume configurations.

## SimplerEnv-Bridge client

Install the [SimplerEnv integration shipped with Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T/tree/main/gr00t/eval/sim/SimplerEnv), including its simulator assets, in a separate environment. Set the source path and simulator interpreter, then use `--python "$SIMPLER_PYTHON"` for `bridge_reused` and `bridge_new`:

```bash
export GR00T_ROOT=/path/to/Isaac-GR00T
export SIMPLER_PYTHON=/path/to/simpler-environment/bin/python
```
 Those cohorts use four canonical tasks, scenes 0–23, one camera, and 120-step episodes.

## Image-encoder controls

Use an environment with torch, torchvision, and transformers. Predownload `facebook/dinov2-base` revision `f9e44c814b77203eaa57a6bdbbd535f21ede1415` and torchvision ResNet-18 `IMAGENET1K_V1`. The extractor respects the Hugging Face cache configuration and `torch.hub.get_dir()` (including `TORCH_HOME`). It records weight checksums. It needs recorded input images, not fitted result summaries.

The recorded hardware was A100 80 GB and RTX 6000 Ada. J64 replay verifies the GPU model as well as the checkpoint and recorded prefix. Cross-hardware bitwise equivalence is not assumed. Full GPU environment installation and all closed-loop episodes were not repeated during the submission packaging check.
