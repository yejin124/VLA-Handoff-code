"""Prespecified robustness analyses for the two primary cells (docs/PROTOCOLS.md):
image-content control I (DINOv2-base CLS of the observed images of the query H is read from) and readout sensitivity.
Uses the primary run's data loader, bootstrap streams (seed, 2,000 draws) and fitting code.
    RAL_RAW_ROOT=$PWD CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 \
        python code/ral_confirm_robustness.py --workers 20
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
import os
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import ral_confirm_analysis as rc
from ral_nested_fit import evaluate
from recheck_restoration_allocation import Reader

CELLS = ('gr00t_bridge_confirm', 'gr00t_spatial_confirm')
# later cells fixed by amendments (pi0.5: plan amendment c; tau-0: tau-0 plan amendment); select with --cells
IMAGE = ({'G': ('G',), 'GI': ('G', 'I'), 'GIH': ('G', 'I', 'H'), 'GEI': ('G', 'E', 'I'), 'GEIH': ('G', 'E', 'I', 'H')},
         {'GIH-GI': ('GIH', 'GI'), 'GEIH-GEI': ('GEIH', 'GEI')})
GH = ({'G': ('G',), 'GH': ('G', 'H')}, {'GH-G': ('GH', 'G')})
VARIANTS = {'image': dict(spec=IMAGE), 'pca10': dict(spec=GH, max_components=10), 'pca50': dict(spec=GH, max_components=50),
            'C0.1': dict(spec=GH, C=.1), 'C10': dict(spec=GH, C=10)}


def read_embedding_cache(cache: Path) -> dict[str, np.ndarray]:
    """Read numeric arrays only; reject legacy pickle-backed caches."""
    if not cache.exists():
        return {}
    with np.load(cache, allow_pickle=False) as stored:
        if 'ids' not in stored.files:
            raise ValueError('Legacy image cache: use a fresh output directory')
        ids = stored['ids'].tolist()
        if not isinstance(ids, list) or not all(isinstance(k, str) for k in ids) or len(set(ids)) != len(ids):
            raise ValueError('Invalid image cache identifiers')
        values = {key: stored[f'feature_{i}'] for i, key in enumerate(ids)}
        if any(v.ndim != 1 or not np.isfinite(v).all() for v in values.values()):
            raise ValueError('Invalid image cache features')
        return values


def embed_images(datasets, cache):
    """DINOv2-base CLS token per camera for the exact images of each scored query."""
    keys = sorted({(i, int(q)) for d in datasets.values() for i, q in zip(d['ids'], d['query_step'])})
    stored = read_embedding_cache(cache)
    todo = [k for k in keys if f'{k[0]}|{k[1]}' not in stored]
    if todo:
        import torch
        from transformers import AutoImageProcessor, AutoModel
        proc = AutoImageProcessor.from_pretrained('facebook/dinov2-base', revision='f9e44c814b77203eaa57a6bdbbd535f21ede1415')
        model = AutoModel.from_pretrained('facebook/dinov2-base', revision='f9e44c814b77203eaa57a6bdbbd535f21ede1415').cuda().eval()
        images, owner = [], []          # one row per camera image; owner maps it back to its query
        for directory, step in todo:
            trace = [json.loads(x) for x in (Path(directory) / 'queries.jsonl').read_text().splitlines() if x.strip()]
            q = [t for t in trace if t['step'] == step]
            if len(q) != 1:
                raise ValueError(f'No unique query at step {step} in {directory}')
            z = np.load(Path(directory) / q[0]['artifact'])
            cams = [z[k] for k in ('agent_image', 'wrist_image') if k in z.files]
            if not cams:
                raise ValueError(f'No stored image for step {step} in {directory}')
            images += cams
            owner += [f'{directory}|{step}'] * len(cams)
        rows = []
        with torch.no_grad():
            for start in range(0, len(images), 128):
                out = model(**{k: v.cuda() for k, v in proc(images=images[start:start + 128], return_tensors='pt').items()})
                rows.append(out.last_hidden_state[:, 0].float().cpu().numpy())
        rows = np.concatenate(rows)
        parts = {}
        for r, o in zip(rows, owner):                # agent then wrist, in stored order
            parts.setdefault(o, []).append(r)
        stored.update({k: np.concatenate(v) for k, v in parts.items()})
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, ids=np.asarray(list(stored)),
                            **{f'feature_{i}': v for i, v in enumerate(stored.values())})
    for d in datasets.values():
        d['I'] = np.stack([stored[f'{i}|{int(q)}'] for i, q in zip(d['ids'], d['query_step'])])


# Holm families of one-sided GIH-GI tests fixed before their cohorts' outcomes were analysed
IMAGE_FAMILIES = {'pi05': ('pi05_spatial_confirm|40', 'pi05_spatial_confirm|80', 'pi05_object_confirm|40',
                           'pi05_object_confirm|80')}                   # plan amendment c (descriptive family)


def image_families(image):
    out = {}
    for family, keys in IMAGE_FAMILIES.items():
        present = [k in image for k in keys]
        if not any(present):
            continue
        if not all(present):
            raise ValueError(f'Image family {family} needs all of {keys} in one run')
        ps = [image[k]['contrasts'].get('GIH-GI', {}).get('one_sided_p', 1.0) for k in keys]
        out[family] = {k: {'GIH-GI': image[k]['contrasts'].get('GIH-GI'), 'holm_p': h}
                       for k, h in zip(keys, rc.holm(ps))}
    return out


def init(datasets):
    global DATASETS
    DATASETS = datasets
    threadpool_limits(limits=1)


def run(variant, counts=None):
    v = VARIANTS[variant]
    blocks, contrasts = v['spec']
    kw = {k: v[k] for k in ('C', 'max_components') if k in v}
    out = {}
    for key, d in DATASETS.items():
        r = evaluate(d, None if counts is None else counts[key], blocks=blocks, contrasts=contrasts, **kw)
        out[key] = r if r['status'] != 'evaluated' else {
            'status': 'evaluated', 'contrasts': {c: r['contrasts'][c]['mean'] for c in contrasts},
            'methods': {m: r['methods'][m]['mean_auroc'] for m in blocks}}
    return out


def attempt(index):
    counts = rc.counts_for(index, DATASETS)
    return {name: run(name, counts) for name in VARIANTS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--output', type=Path, default=Path('data/ral_confirm_20261105/robustness'))
    ap.add_argument('--cells', default=','.join(CELLS))
    a = ap.parse_args()
    root = Path(os.environ.get('RAL_RAW_ROOT', Path(__file__).resolve().parents[1]))
    reader, datasets = Reader(), {}
    for cell in a.cells.split(','):
        datasets.update(rc.load(root, cell, reader)[0])
    embed_images(datasets, a.output / 'dinov2_cls.npz')
    init(datasets)
    point = {name: run(name) for name in VARIANTS}
    with Pool(a.workers, initializer=init, initargs=(datasets,)) as pool:
        draws = pool.map(attempt, range(rc.ATTEMPTS), chunksize=10)
    res = {'plan': 'docs/PROTOCOLS.md', 'attempts': rc.ATTEMPTS, 'seed': rc.SEED,
           'image_dim': int(next(iter(datasets.values()))['I'].shape[1]), 'variants': {}}
    for name in VARIANTS:
        res['variants'][name] = {}
        for (cell, step) in datasets:
            key = (cell, step)
            est = point[name][key]
            valid = [d[name][key] for d in draws if d[name][key]['status'] == 'evaluated']
            entry = {'valid': len(valid), 'methods': est.get('methods'), 'contrasts': {}}
            for c, value in est.get('contrasts', {}).items():
                arr = np.asarray([x['contrasts'][c] for x in valid])
                entry['contrasts'][c] = {'estimate': value, 'ci95': np.quantile(arr, [.025, .975]).tolist(),
                                         'one_sided_p': float((1 + np.sum(arr <= 0)) / (1 + len(arr)))} \
                    if len(valid) >= rc.MIN_VALID else {'estimate': value}
            res['variants'][name][f'{cell}|{step}'] = entry
    res['image_families'] = image_families(res['variants']['image'])
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / 'robustness_results.json').write_text(json.dumps(res, indent=2) + '\n')
    print(json.dumps({n: {k: v['contrasts'] for k, v in r.items()} for n, r in res['variants'].items()}, indent=1))


if __name__ == '__main__':
    main()
