"""Fig. 1: a compact schematic of policy flow and two uncertainty measurements."""
from html import escape
from pathlib import Path

import cairosvg

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'paper/figures'
W, H = 840, 322
INK, MUTED, ORANGE, BLUE = '#2d3142', '#4f5d75', '#bf4520', '#2b6cb0'
FONT = "Helvetica, Arial, 'Liberation Sans', sans-serif"
parts = []


def text(x, y, label, size=16, color=INK, bold=False):
    parts.append(f'<text x="{x}" y="{y}" text-anchor="middle" font-size="{size}" '
                 f'font-weight="{600 if bold else 400}" fill="{color}">{escape(label)}</text>')


def path(d, color=MUTED, dashed=False):
    dash = ' stroke-dasharray="5,4"' if dashed else ''
    marker = {MUTED: 'gray', ORANGE: 'orange', BLUE: 'blue'}[color]
    parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.6"{dash} '
                 f'marker-end="url(#fig1-{marker})"/>')


def node(x, y, w, h, color=MUTED, fill='white'):
    parts.append('<g data-role="node">')
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" '
                 f'fill="{fill}" stroke="{color}" stroke-width="1.4"/>')


def end_node():
    parts.append('</g>')


def icon_path(d, color=MUTED, width=2):
    parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" '
                 'stroke-linecap="round" stroke-linejoin="round"/>')


parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
             f'font-family="{FONT}" role="img" aria-labelledby="fig1-title fig1-desc">')
parts.append('<title id="fig1-title">Uncertainty in a dual-system VLA</title>')
parts.append('<desc id="fig1-desc">The policy passes a handoff from System 2 to System 1 and executes one of eight draws, '
             'while an offline readout and action spread are compared against episode outcomes.</desc><defs>')
for key, color in [('gray', MUTED), ('orange', ORANGE), ('blue', BLUE)]:
    parts.append(f'<marker id="fig1-{key}" markerWidth="7" markerHeight="6" refX="6" refY="3" orient="auto">'
                 f'<path d="M0,0 L7,3 L0,6 Z" fill="{color}"/></marker>')
parts.append(f'</defs><rect width="{W}" height="{H}" fill="white"/>')

# Computation is horizontal. The measurement branches never feed back into it.
for x1, x2 in [(120, 160), (292, 432), (564, 612), (716, 752)]:
    path(f'M{x1},76 H{x2}')
path('M362,76 V160', ORANGE, True)
path('M664,120 V160', BLUE, True)
path('M362,216 V248', ORANGE, True)
path('M664,216 V248', BLUE, True)
path('M790,120 V257 Q790,265 782,265 H770', MUTED, True)
text(362, 57, 'Handoff', 17)
parts.append(f'<circle cx="362" cy="76" r="3" fill="{ORANGE}"/>')

# Observation: a camera image and a short instruction, represented schematically.
node(12, 28, 108, 92)
parts.append(f'<rect x="32" y="44" width="48" height="31" rx="3" fill="none" stroke="{MUTED}" stroke-width="1.8"/>')
parts.append(f'<circle cx="67" cy="52" r="3" fill="{MUTED}"/>')
icon_path('M37,69 L48,56 L58,67 L64,61 L75,71')
parts.append(f'<rect x="66" y="68" width="32" height="18" rx="3" fill="white" stroke="{MUTED}" stroke-width="1.6"/>')
icon_path('M72,74 H92 M72,80 H86', width=1.4)
text(66, 106, 'Observation', 16, MUTED, True)
end_node()

node(160, 28, 132, 92, ORANGE, '#fbefe9')
# A compact token grid distinguishes the representation-producing module.
for row in range(2):
    for col in range(4):
        parts.append(f'<rect x="{205 + col * 11}" y="{41 + row * 10}" width="7" height="6" rx="1" fill="{ORANGE}"/>')
text(226, 85, 'System 2', 20, ORANGE, True)
text(226, 107, 'VLM', 16, MUTED)
end_node()

node(432, 28, 132, 92, BLUE, '#edf3fa')
icon_path('M475,59 H484 L491,42 L500,61 L507,49 H521', BLUE)
text(498, 85, 'System 1', 20, BLUE, True)
text(498, 107, 'Action expert', 16, MUTED)
end_node()

node(612, 28, 104, 92, BLUE, '#edf3fa')
# Eight tiles denote sampled chunks, not measured trajectories or distributions.
for row in range(4):
    for col in range(2):
        parts.append(f'<rect x="{640 + col * 26}" y="{41 + row * 11}" width="20" height="7" rx="2" '
                     f'fill="white" stroke="{BLUE}" stroke-width="1.2"/>')
text(664, 106, '8 draws', 17, BLUE, True)
end_node()

node(752, 28, 76, 92)
# A gripper icon identifies the executed robot action.
icon_path('M768,82 H805 M781,80 L775,63 L790,46 L806,61 M806,61 V70 M801,70 V76 M811,70 V76')
for x, y in [(775, 63), (790, 46)]:
    parts.append(f'<circle cx="{x}" cy="{y}" r="3" fill="white" stroke="{MUTED}" stroke-width="1.8"/>')
text(790, 106, 'Execute', 16, MUTED, True)
end_node()

for x, center, title, subtitle, color, fill in [
    (256, 362, 'Outcome uncertainty', 'Offline readout', ORANGE, '#fbefe9'),
    (558, 664, 'Action uncertainty', 'Action spread', BLUE, '#edf3fa'),
]:
    node(x, 160, 212, 56, color, fill)
    text(center, 183, title, 17, color, True)
    text(center, 205, subtitle, 15.5, MUTED)
    end_node()
node(256, 248, 514, 34)
text(513, 271, 'Compare against episode outcomes', 17, MUTED, True)
end_node()
path('M290,306 H320')
text(355, 311, 'Policy', 15)
path('M438,306 H468', dashed=True)
text(508, 311, 'Offline', 15)
parts.append('</svg>')

svg = '\n'.join(parts) + '\n'
src = OUT / 'src'
src.mkdir(parents=True, exist_ok=True)
(src / 'fig1_pipeline.svg').write_text(svg)
(src / 'fig1_pipeline.html').write_text(
    '<!doctype html><html lang="en"><head><meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width,initial-scale=1">'
    '<title>Uncertainty in a dual-system VLA</title><style>'
    'body{margin:0;background:white;font-family:Helvetica,Arial,sans-serif}'
    'main{padding:16px}h1{font-size:20px}.diagram{overflow-x:auto;max-width:100%}'
    'svg{display:block;min-width:840px}</style></head><body><main>'
    '<h1>Uncertainty in a dual-system VLA</h1><div class="diagram">' + svg +
    '</div></main></body></html>\n')
cairosvg.svg2pdf(bytestring=svg.encode(), write_to=str(OUT / 'fig1_pipeline.pdf'))
print(OUT / 'fig1_pipeline.pdf')
