"""Render the complete J64 follow-up summaries; no raw GPU records required.

python code/make_sampling_tables.py data/sampling_sensitivity_20261110 paper/tables
"""
import argparse
import hashlib
import json
from pathlib import Path


def number(value, signed=False):
    return f'{value:+.3f}' if signed else f'{value:.3f}'


def interval(estimate, limits, signed=False):
    if estimate is None:
        return r'\textemdash'
    if limits is None:
        return '$' + number(estimate, signed) + r'\;[\text{--}]$'
    return '$' + number(estimate, signed) + r'\;[' + ', '.join(number(x, signed) for x in limits) + ']$'


def probability(value):
    return f'${value:.4f}$' if value < .001 else f'${value:.3f}$'


def query_count(views):
    """Read active-query counts from point-estimate held-out folds, not paper constants."""
    reference = None
    for name, view in views.items():
        folds = view.get('folds', [])
        counts = {}
        for fold in folds:
            task, count = fold['test_task'], fold['test_rows_with_multiplicity']
            if task in counts or type(count) is not int or count <= 0:
                raise ValueError(f'Invalid point-estimate fold count in {name}')
            counts[task] = count
        if not counts or (reference is not None and counts != reference):
            raise ValueError(f'Missing or inconsistent active-query counts in {name}')
        reference = counts
    if reference is None:
        raise ValueError('No sampling views supplied')
    return sum(reference.values())


def load(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    for name, expected in manifest['files_sha256'].items():
        path = directory / name
        if Path(name).name != name or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Frozen summary hash mismatch: {name}')
    bridge = json.loads((directory / 'bridge_results.json').read_text())
    smol = json.loads((directory / 'smol_results.json').read_text())
    output = [('GR00T / Bridge', bridge['scopes']['full'], bridge['primary_holm'],
               bridge['repeatability'], bridge['valid_joint'], query_count(bridge['scopes']['full']), '')]
    for cell, label in [('smolvla_spatial', 'SmolVLA / Spatial'),
                        ('smolvla_object', 'SmolVLA / Object')]:
        value = smol['cells'][cell]
        output.append((label, value['views'], smol['primary_holm'], value['repeatability'],
                       value['valid_joint'], query_count(value['views']), cell + '/'))
    return output


def header(columns, labels):
    return [r'\begin{tabular}{' + columns + '}', r'\toprule', labels + r' \\', r'\midrule']


def finish(directory, name, lines):
    lines += [r'\bottomrule', r'\end{tabular}']
    (directory / name).write_text('\n'.join(lines) + '\n')


def render(source, destination):
    groups = load(source)
    destination.mkdir(parents=True, exist_ok=True)
    primary = header('llccc', r'Seed & Comparison & Difference [95\% CI] & One-sided $p$ & Holm $p$')
    ranking = header('lccc', r'Seed & $J$ & Raw spread AUROC [95\% CI] & Learned spread AUROC [95\% CI]')
    repeat = header('ccc', r'$J$ & Pooled $\rho$ [95\% CI] & Mean within-task $\rho$ [95\% CI]')
    for index, (label, views, tests, reliability, valid, count, prefix) in enumerate(groups):
        if index:
            for lines in (primary, ranking, repeat):
                lines.append(r'\midrule')
        ref = views['original_64']['methods']
        primary.append(r'\multicolumn{5}{l}{\textbf{' + label + r'} ($n=' + str(count) +
                       '$; valid bootstrap draws: ' + str(valid) + r')} \\')
        ranking.append(r'\multicolumn{4}{l}{\textbf{' + label + r'}: handoff $' +
                       number(ref['H']['mean_auroc']) + '$; scene + handoff $' +
                       number(ref['GH']['mean_auroc']) + r'$} \\')
        repeat.append(r'\multicolumn{3}{l}{\textbf{' + label + r'}} \\')
        for seed, seed_label in [('original', 'Original'), ('independent', 'Independent')]:
            for contrast, contrast_label in [('GH-Spread', 'Scene + handoff minus raw spread'),
                                              ('H-S', 'Handoff minus learned spread')]:
                value = tests[prefix + seed + '_64/' + contrast]
                primary.append(' & '.join([seed_label, contrast_label,
                                           interval(value['mean'], value['ci95'], True),
                                           probability(value['one_sided_p']), probability(value['holm_p'])]) + r' \\')
            for j in (8, 16, 32, 64):
                methods = views[f'{seed}_{j}']['methods']
                ranking.append(' & '.join([seed_label, str(j)] +
                               [interval(methods[name]['mean_auroc'], methods[name]['ci95'])
                                for name in ('Spread', 'S')]) + r' \\')
        for j in (8, 16, 32, 64):
            value = reliability[str(j)]
            def limits(field):
                return value[field + '_ci95'] if not prefix else value[field + '_interval']['ci95']
            repeat.append(' & '.join([str(j), interval(value['pooled'], limits('pooled')),
                                      interval(value['mean_task'], limits('mean_task'))]) + r' \\')
    finish(destination, 'supp_sampling_primary.tex', primary)
    finish(destination, 'supp_sampling_ranking.tex', ranking)
    finish(destination, 'supp_sampling_seeds.tex', repeat)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    render(args.source, args.destination)
