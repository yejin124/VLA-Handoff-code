"""Fig. 2: five aligned conditioning paths and readout sites.

Every row shares the same model, pipeline, action-output, and pooled-dimension
columns. Architecture descriptions and model references are in paper/main.tex, Section IV-A and the Fig. 2 caption.
"""
from html import escape
from pathlib import Path

import cairosvg

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'paper/figures'
W, H = 968, 498
INK, MUTED, ORANGE, BLUE = '#2d3142', '#4f5d75', '#bf4520', '#2b6cb0'
FONT = "Helvetica, Arial, 'Liberation Sans', sans-serif"
BOX_W, BOX_H, ROW_H = 112, 44, 84
FIRST_X, LAST_X, ACTION_X, POOLED_X = 178, 658, 800, 942
parts = []
put = parts.append


def text(x, y, label, size=15, color=INK, bold=False, anchor='middle', role=None):
    role_attr = f' data-role="{role}"' if role else ''
    put(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
        f'font-weight="{600 if bold else 400}" text-anchor="{anchor}"{role_attr}>{escape(label)}</text>')


def box(x, y, lines, kind):
    color, fill = {'plain': (MUTED, 'white'), 'vlm': (ORANGE, '#fbefe9'),
                   'expert': (BLUE, '#edf3fa')}[kind]
    put('<g data-role="node">')
    put(f'<rect x="{x}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="4" '
        f'fill="{fill}" stroke="{color}" stroke-width="1.2"/>')
    if lines[1]:
        text(x + BOX_W / 2, y + 18, lines[0], 15.5, color, True)
        text(x + BOX_W / 2, y + 36, lines[1], 14.5, MUTED)
    else:
        text(x + BOX_W / 2, y + 27, lines[0], 15.5, color, True)
    put('</g>')


def arrow(x1, y1, x2, y2, dashed=False):
    dash = ' stroke-dasharray="4,3"' if dashed else ''
    put(f'<path d="M{x1},{y1} L{x2},{y2}" fill="none" stroke="{MUTED}" '
        f'stroke-width="1.2"{dash} marker-end="url(#fig2-arrow)"/>')


def panel(row, name, pooled, nodes, taps, enclosure=False):
    y0 = 32 + row * ROW_H
    by, mid, label_y = y0 + 12, y0 + 34, y0 + 76
    put(f'<g data-role="model-row" data-model="{escape(name)}">')
    text(12, mid + 5, name, 16, INK, True, 'start', 'model-name')
    text(POOLED_X, mid + 5, f'{pooled:,}', 16, MUTED, anchor='end', role='pooled-dimension')
    if enclosure:
        # The implementation boundary includes LayerNorm, self-attention and DiT.
        put(f'<rect x="332" y="{y0 + 6}" width="444" height="56" rx="4" '
            f'fill="none" stroke="{MUTED}" stroke-width=".9" stroke-dasharray="5,4"/>')
    for left, right in zip(nodes, nodes[1:]):
        arrow(left[0] + BOX_W, mid, right[0], mid)
    arrow(LAST_X + BOX_W, mid, ACTION_X - 8, mid)
    for x, lines, kind in nodes:
        box(x, by, lines, kind)
    for x, label, hollow in taps:
        arrow(x, mid + 4, x, label_y - 14, True)
        put(f'<circle cx="{x}" cy="{mid}" r="3.5" fill="{"white" if hollow else INK}" '
            f'stroke="{INK}" stroke-width="1.3"/>')
        text(x, label_y, label, 13.5, MUTED, True)
    text(ACTION_X, mid + 5, 'Actions', 15, BLUE, True, 'start', 'action-output')
    put('</g>')


put(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
    f'font-family="{FONT}" role="img" aria-labelledby="fig2-title fig2-desc">')
put('<title id="fig2-title">Conditioning paths and readout sites</title>')
put('<desc id="fig2-desc">Five aligned policy rows show the recorded conditioning paths and readout taps, '
    'with a proxy before rotary embedding for pi0.5 and pooled feature dimensions in the right column.</desc>')
put('<defs><marker id="fig2-arrow" markerWidth="7" markerHeight="6" refX="6" refY="3" orient="auto">'
    f'<path d="M0,0 L7,3 L0,6 Z" fill="{MUTED}"/></marker></defs>')
put(f'<rect width="{W}" height="{H}" fill="white"/>')
text(12, 19, 'Model', 14, MUTED, anchor='start')
text(474, 19, 'Conditioning path', 14, MUTED)
text(POOLED_X, 19, 'Pooled', 14, MUTED, anchor='end')

panel(0, '(a) GR00T-N1.7', 2048, [
    (FIRST_X, ['VLM', 'tokens'], 'vlm'), (338, ['LayerNorm', ''], 'plain'),
    (498, ['Self-attn.', '4 layers'], 'plain'), (LAST_X, ['DiT', 'expert'], 'expert'),
], [(314, 'PRE', False), (474, 'MID', False), (634, 'POST', False)], enclosure=True)
panel(1, '(b) MolmoAct2', 73728, [
    (FIRST_X, ['Molmo2-ER', '36 layers'], 'vlm'), (418, ['K/V cache', '36 pairs'], 'plain'),
    (LAST_X, ['Expert', 'cross-attn.'], 'expert'),
], [(594, 'READOUT', False)])
panel(2, '(c) π0.5', 9216, [
    (FIRST_X, ['PaliGemma', 'VLM prefix'], 'vlm'), (338, ['K/V proj.', 'pre-RoPE'], 'plain'),
    (498, ['Cache', 'RoPE keys'], 'plain'), (LAST_X, ['Expert', 'prefix attn.'], 'expert'),
], [(474, 'PROXY', True)])
panel(3, '(d) SmolVLA', 10240, [
    (FIRST_X, ['SmolVLM2', '16 layers'], 'vlm'), (418, ['K/V cache', '16 pairs'], 'plain'),
    (LAST_X, ['Expert', 'SA / CA'], 'expert'),
], [(594, 'READOUT', False)])
panel(4, '(e) τ0', 6144, [
    (FIRST_X, ['Qwen3.5', 'hybrid VLM'], 'vlm'), (418, ['K/V cache', '6 full-attn.'], 'plain'),
    (LAST_X, ['Expert', '6 joint-attn.'], 'expert'),
], [(594, 'READOUT', False)])

# Short legend; model-specific qualifications are in the caption.
arrow(226, 478, 258, 478)
text(272, 483, 'Policy', 14, MUTED, anchor='start')
arrow(382, 478, 414, 478, True)
put(f'<circle cx="382" cy="478" r="3.5" fill="{INK}"/>')
text(428, 483, 'Offline readout', 14, MUTED, anchor='start')
put(f'<circle cx="624" cy="478" r="3.5" fill="white" stroke="{INK}" stroke-width="1.3"/>')
text(638, 483, 'Proxy', 14, MUTED, anchor='start')
put('</svg>')
svg = '\n'.join(parts) + '\n'
src = OUT / 'src'
src.mkdir(parents=True, exist_ok=True)
(src / 'fig2_architectures.svg').write_text(svg)
(src / 'fig2_architectures.html').write_text(
    '<!doctype html><html lang="en"><head><meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width,initial-scale=1">'
    '<title>Conditioning paths and readout sites</title><style>'
    'body{margin:0;background:white;font-family:Helvetica,Arial,sans-serif}'
    'main{padding:16px}h1{font-size:20px}.diagram{overflow-x:auto;max-width:100%}'
    'svg{display:block;min-width:968px}</style></head><body><main>'
    '<h1>Conditioning paths and readout sites</h1><div class="diagram">' + svg +
    '</div></main></body></html>\n')
cairosvg.svg2pdf(bytestring=svg.encode(), write_to=str(OUT / 'fig2_architectures.pdf'))
print(OUT / 'fig2_architectures.pdf')
