"""Fig. 3: the evaluated camera conditions, without tiny severity-sweep panels.

Same input frame, seeds, and corruption function as the original 4x4 montage.
The intermediate severities are omitted from this illustration, not the data.
Usage: python code/plot_perturbation_examples.py [data-root]
"""
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import realistic_corruption as rc

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'data'
frame = sorted((DATA / 'icassp2027/realistic/spatial_main/pi05').glob('shard*/t3_*_clean/q0000.npz'))[0]
z = np.load(frame)
donors = rc.load_donor_bank(rc.DONOR_BANK)
plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 10, 'pdf.fonttype': 42})
# Typeset at the final single-column width; no font-size loss from shrinking.
# Rows are conditions, columns are cameras, so each frame remains visible.
fig, axes = plt.subplots(3, 2, figsize=(3.42, 5.2))
record = {'input_relative_to_data': str(frame.relative_to(DATA)),
          'input_sha256': hashlib.sha256(frame.read_bytes()).hexdigest(),
          'corruption_code_sha256': hashlib.sha256(Path(rc.__file__).read_bytes()).hexdigest(),
          'panels': [], 'omitted_display_only': 'intermediate severity examples'}
for camera, key in enumerate(['agent_image', 'wrist_image']):
    source = z[key]
    settings = [('clean', 0, 20261008 + 1000 * camera),
                ('occluder', 1.0, 20261008 + 1000 * camera),
                ('lighting', 1.25, 20261008 + 1000 * (camera + 2))]
    for j, (family, severity, seed) in enumerate(settings):
        pixels = rc.corrupt(source, family, severity, seed, camera=camera,
                            donors=donors, apply_to_wrist=True)
        ax = axes[j, camera]
        ax.imshow(pixels)
        ax.set(xticks=[], yticks=[])
        for spine in ax.spines.values():
            spine.set_visible(False)
        record['panels'].append({'camera': camera, 'family': family, 'severity': severity,
                                 'seed': seed, 'pixels_sha256': hashlib.sha256(pixels.tobytes()).hexdigest()})
fig.subplots_adjust(left=.02, right=.98, top=.885, bottom=.015, wspace=.10, hspace=.24)
fig.canvas.draw()
for column, label in enumerate(['Agent view', 'Wrist view']):
    position = axes[0, column].get_position()
    fig.text(position.x0 + position.width / 2, .992, label,
             ha='center', va='top', fontsize=10.5, fontweight='bold')
for row, label in enumerate(['Clean', 'Occluder (s = 1.0)', 'Dim light (s = 1.25)']):
    position = axes[row, 0].get_position()
    fig.text(.5, position.y1 + .012, label, ha='center', va='bottom', fontsize=10)
(ROOT / 'paper/figures').mkdir(parents=True, exist_ok=True)
fig.savefig(ROOT / 'paper/figures/fig3_perturbations.pdf', dpi=320)
out = ROOT / 'paper/figures/src/fig3_perturbation_sources.json'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(record, indent=2) + '\n')
print(out)
