#!/usr/bin/env python3
"""E3 step 1: frozen generic image embeddings of the stored policy inputs (GPU).

For every complete main-campaign episode (realistic/<suite>_<main|main2|main3>/<policy>/shard*/...),
every query that stored its transmitted images (steps 0, 40, 80, ...; these are the landmark and
probe queries), embeds the agent and wrist images with
  dinov2  facebook/dinov2-base at a pinned revision: [CLS, mean patch token] per camera -> 3,072 values
  resnet  torchvision ResNet-18 ImageNet-1k V1: global-average-pooled features per camera -> 1,024 values
No fine-tuning; features only. Writes data/icassp2027/exp_20261020/e3_features/<suite>_<coll>_<policy>.npz
with the resolved episode directory and query index of each row.
Run (gr00t env): CUDA_VISIBLE_DEVICES=0 python code/exp_20261020_e3_extract.py
"""
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from transformers import AutoImageProcessor, AutoModel
from huggingface_hub import hf_hub_download

import exp_20261020_common as c

DINO = "facebook/dinov2-base"
DINO_REVISION = "f9e44c814b77203eaa57a6bdbbd535f21ede1415"
STEPS = {0, 40, 80}
BATCH = 128


def episodes(suite, coll, policy):
    for d in sorted(glob.glob(f"{c.DATA}/realistic/{suite}_{coll}/{policy}/shard*/t*_e*_r*_*")):
        s = Path(d) / "summary.json"
        if s.exists() and json.load(open(s)).get("complete"):
            yield Path(d).resolve()


def main():
    dev = "cuda"
    proc = AutoImageProcessor.from_pretrained(DINO, revision=DINO_REVISION, local_files_only=True)
    dino = AutoModel.from_pretrained(DINO, revision=DINO_REVISION, local_files_only=True).to(dev).eval().half()
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    resnet = torchvision.models.resnet18(weights=weights)
    resnet.fc = torch.nn.Identity()
    resnet = resnet.to(dev).eval().half()
    rtf = weights.transforms()
    out_dir = c.OUT / "e3_features"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"dinov2": {"repo": DINO, "revision": DINO_REVISION,
                           "safetensors_sha256": c.sha256(hf_hub_download(DINO, "model.safetensors", revision=DINO_REVISION, local_files_only=True))},
                "resnet18": {"weights": "IMAGENET1K_V1", "file_sha256": c.sha256(
                    Path(torch.hub.get_dir()) / "checkpoints/resnet18-f37072fd.pth")},
                "precision": "fp16 inference", "files": {}}
    for suite in c.SUITES:
        for coll in ("main", "main2", "main3"):
            for policy in c.POLICIES:
                ids, qidx, steps, images = [], [], [], []
                t0 = time.time()
                for d in episodes(suite, coll, policy):
                    for q in (json.loads(l) for l in open(d / "queries.jsonl")):
                        if q["step"] not in STEPS:
                            continue
                        with np.load(d / q["artifact"]) as z:
                            if "agent_image" not in z.files:
                                continue
                            images.append((z["agent_image"], z["wrist_image"]))
                        ids.append(str(d)); qidx.append(q["query"]); steps.append(q["step"])
                if not ids:
                    continue
                dfeat, rfeat = [], []
                with torch.no_grad():
                    for i in range(0, len(images), BATCH):
                        batch = images[i:i + BATCH]
                        per_cam_d, per_cam_r = [], []
                        for cam in (0, 1):
                            imgs = [b[cam] for b in batch]
                            px = proc(images=imgs, return_tensors="pt")["pixel_values"].to(dev).half()
                            h = dino(pixel_values=px).last_hidden_state.float()
                            per_cam_d.append(torch.cat([h[:, 0], h[:, 1:].mean(1)], 1).cpu())
                            t = torch.stack([rtf(torch.from_numpy(x).permute(2, 0, 1)) for x in imgs]).to(dev).half()
                            per_cam_r.append(resnet(t).float().cpu())
                        dfeat.append(torch.cat(per_cam_d, 1)); rfeat.append(torch.cat(per_cam_r, 1))
                name = f"{suite}_{coll}_{policy}.npz"
                np.savez(out_dir / name, id=np.array(ids), query=np.array(qidx), step=np.array(steps),
                         dinov2=torch.cat(dfeat).numpy(), resnet18=torch.cat(rfeat).numpy())
                manifest["files"][name] = {"rows": len(ids), "seconds": round(time.time() - t0, 1)}
                print(name, len(ids), f"{time.time() - t0:.0f}s", flush=True)
    manifest["provenance"] = c.provenance([])
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
