"""Render main Tables 2/3/5 and Supplement Table S2 from newly generated analyses.

Run from the repository root; use --help for input paths. No published numerical
results are embedded. TeX and CSV outputs retain missing/ineligible estimates as
dashes/empty cells with status fields, rather than substituting paper values.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import tempfile
from typing import Any

POLICIES = ('pi05', 'gr00t', 'molmoact')
SUITES = ('spatial', 'object')
CONDITIONS = ('occluder', 'lighting', 'moved', 'relocated')
METHODS = ('native_spread', 'baseline', 'handoff', 'combined')
LABELS = {'pi05': r'$\pi_{0.5}$', 'gr00t': 'GR00T', 'molmoact': 'MolmoAct2'}
SITUATIONS = {'occluder': 'Occluder', 'lighting': 'Dim light',
              'moved': 'Moved goal', 'relocated': 'Relocated goal'}
Record = dict[str, Any]


def read(path: Path) -> Record:
    return json.loads(path.read_text())


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def fmt(value: Any, digits: int = 2, signed: bool = False) -> str:
    if not finite(value):
        return '--'
    value = 0.0 if round(value, digits) == 0 else value
    return f'{value:+.{digits}f}' if signed else f'{value:.{digits}f}'


def prob(value: Any) -> str:
    if finite(value) and 0 < value < .0001:
        return f'{value:.1e}'
    return fmt(value, 4 if finite(value) and value < .001 else 3)


def interval(value: Any, limits: Any, digits: int = 3) -> str:
    if not finite(value):
        return '--'
    if not isinstance(limits, list) or len(limits) != 2 or not all(map(finite, limits)):
        return fmt(value, digits, True) + ' [--]'
    return fmt(value, digits, True) + ' [' + ', '.join(fmt(x, digits, True) for x in limits) + ']'


def bold(value: str, enabled: bool) -> str:
    return r'\textbf{' + value + '}' if enabled and value != '--' else value


def holm(values: list[float]) -> list[float]:
    adjusted, maximum = [1.0] * len(values), 0.0
    for rank, index in enumerate(sorted(range(len(values)), key=values.__getitem__)):
        maximum = max(maximum, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, maximum)
    return adjusted


def sign_above_chance(values: list[float]) -> float:
    non_ties = [x for x in values if x != .5]
    n, positive = len(non_ties), sum(x > .5 for x in non_ties)
    return sum(math.comb(n, k) for k in range(positive, n + 1)) / 2**n if n else 1.0


def aucs(analysis: Record, step: int, method: str) -> list[float]:
    landmark = next((x for x in analysis.get('landmarks', []) if x['landmark'] == step), {})
    values = [f.get('methods', {}).get(method, {}).get('test', {}).get('auroc')
              for f in landmark.get('folds', [])]
    return [float(x) for x in values if finite(x)]


def med(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def med_iqr(values: list[float]) -> str:
    if not values:
        return '--'
    ordered = sorted(values)
    return f'{fmt(med(ordered))} ({fmt(ordered[len(ordered)//4])}--{fmt(ordered[3*len(ordered)//4])})'


def episode_counts(root: Path, suite: str, policy: str, condition: str) -> tuple[int, int]:
    files = sorted((root / f'{suite}_main' / policy).glob(f'shard*/t*_e*_r*_{condition}/summary.json'))
    if not files:
        raise FileNotFoundError(f'No {suite}/{policy}/{condition} source episode summaries under {root}')
    records = [read(p) for p in files]
    if any(r.get('complete') is not True or type(r.get('success')) is not bool for r in records):
        raise ValueError(f'Incomplete or invalid episode summaries: {suite}/{policy}/{condition}')
    ids = [(r['task'], r['episode'], r['replicate']) for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f'Duplicate episode identities: {suite}/{policy}/{condition}')
    return sum(not r['success'] for r in records), len(records)


def eligible(analysis: Record, failures: int, count: int) -> bool:
    support = min(len(aucs(analysis, step, 'handoff')) for step in (40, 80))
    return count > 0 and .1 <= failures / count <= .8 and support >= 4


def write_tables(destination: Path, name: str, spec: str, header: str,
                 lines: list[str], records: list[Record], note: str) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    table = [r'\begin{tabular}{@{}' + spec + r'@{}}', r'\toprule', header + r'\\', r'\midrule',
             *lines, r'\bottomrule', r'\end{tabular}', r'\par\smallskip\noindent ' + note]
    (destination / f'{name}.tex').write_text('\n'.join(table) + '\n')
    fields = list(dict.fromkeys(k for r in records for k in r))
    with (destination / f'{name}.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in r.items()}
                         for r in records)


def main_confirm(source: Path, destination: Path) -> None:
    data = read(source)
    cells = [('gr00t_bridge_confirm', 20, 'Bridge'), ('gr00t_spatial_confirm', 80, 'LIBERO-Spatial')]
    estimates = [data['cells'].get(f'{key}|{step}', {}) for key, step, _ in cells]
    raw = [c.get('contrasts', {}).get('GH-G', {}).get('one_sided_p') for c in estimates]
    adjusted = holm([x if finite(x) else 1.0 for x in raw])
    lines, records = [], []
    for (key, step, label), cell, p_holm in zip(cells, estimates, adjusted):
        counts = data.get('cohorts', {}).get(key, {}).get('landmarks', {}).get(str(step), {})
        n = counts.get('active')
        failures = sum(v[1] for v in counts['task_counts'].values()) if 'task_counts' in counts else None
        methods, contrasts = cell.get('methods', {}), cell.get('contrasts', {})
        added, gap = contrasts.get('GH-G', {}), contrasts.get('GH-Spread', {})
        supported = cell.get('status') == 'evaluated'
        status = cell.get('status', 'not_evaluated')
        significant = supported and finite(gap.get('one_sided_p')) and gap['one_sided_p'] < .05
        scores = {m: methods.get(m, {}).get('estimate') if supported else None for m in ('G', 'GH', 'Spread')}
        records.append(dict(cohort=key, step=step, status=status, active=n, failures=failures,
                            valid_draws=cell.get('valid'), **scores, handoff_increment=added.get('estimate'),
                            handoff_ci95=added.get('ci95'), handoff_raw_p=added.get('one_sided_p'),
                            handoff_holm_p=p_holm if finite(added.get('one_sided_p')) else None,
                            gap=gap.get('estimate'), gap_ci95=gap.get('ci95'), gap_raw_p=gap.get('one_sided_p')))
        count_text = f'{n} ({failures})' if n is not None and failures is not None else '--'
        values = [f'{label}, {step}', count_text, fmt(scores['G'], 3), fmt(scores['GH'], 3),
                  bold(fmt(scores['Spread'], 3), significant), interval(added.get('estimate'), added.get('ci95')),
                  prob(p_holm) if supported and finite(added.get('one_sided_p')) else '--',
                  bold(interval(gap.get('estimate'), gap.get('ci95')), significant), prob(gap.get('one_sided_p'))]
        lines.append(' & '.join(values) + r'\\')
    write_tables(destination, 'main_confirm', 'lcccccccc',
                 r'Cohort, step & $N$ (fail) & Scene & Scene + handoff & Spread & $\Delta_{\rm handoff}$ [95\% CI] & Holm $p$ & $\Delta_{\rm gap}$ [95\% CI] & $p$',
                 lines, records, 'Dashes denote unavailable or ineligible estimates; CSV records status and valid bootstrap counts. T2 uses Holm correction over the two primary cells; the secondary gap uses an unadjusted one-sided tail probability.')


def original_tables(analysis_dir: Path, episodes_root: Path, destination: Path, selected: set[str]) -> None:
    data, probes, counts = {}, {}, {}
    for suite in SUITES:
        for policy in POLICIES:
            probes[suite, policy] = read(analysis_dir / f'realistic_probe_{suite}_{policy}.json')
            for condition in [c for key, c in (('main3', 'moved'), ('main5', 'occluder')) if key in selected]:
                data[suite, policy, condition] = read(analysis_dir / f'realistic_{suite}_{condition}_{policy}.json')
                counts[suite, policy, condition] = episode_counts(episodes_root, suite, policy, condition)
    corrected = {}
    if 'main3' in selected:
        for suite in SUITES:
            family = [(policy, step, method) for policy in POLICIES for step in (40, 80) for method in METHODS]
            ps = []
            for policy, step, method in family:
                analysis = data[suite, policy, 'moved']
                ok = eligible(analysis, *counts[suite, policy, 'moved'])
                values = aucs(analysis, step, method)
                ps.append(sign_above_chance(values) if ok and values else 1.0)
            corrected.update({(suite, *key): p for key, p in zip(family, holm(ps))})
    goal_lines, goal_rows, occ_lines, occ_rows = [], [], [], []
    for suite in SUITES:
        for policy in POLICIES:
            probe = probes[suite, policy]
            if 'main3' in selected:
                a, c = probe.get('A', {}).get('moved', {}), probe.get('C', {}).get('moved', {})
                analysis = data[suite, policy, 'moved']
                fail, n = counts[suite, policy, 'moved']
                ok = eligible(analysis, fail, n)
                values = aucs(analysis, 80, 'native_spread') if ok else []
                failure_auc, adjusted = med(values), corrected[suite, policy, 80, 'native_spread']
                pair_count, higher, change = c.get('n_pairs'), c.get('higher_in'), c.get('median_relative_change')
                response = f'{higher}/{pair_count} ({fmt(100*change, 0, True)}\\%)' if pair_count and finite(change) else '--'
                p = c.get('wilcoxon_p')
                status = 'evaluated' if a and c and values else 'partly_or_not_evaluated'
                goal_rows.append(dict(suite=suite, policy=policy, status=status, failures=fail, episodes=n,
                                      scene_auroc=a.get('handoff_auroc'), pairs=pair_count, higher=higher,
                                      relative_change=change, wilcoxon_p=p, failure_auroc=failure_auc,
                                      evaluable_tasks=len(values), failure_sign_p=sign_above_chance(values) if values else None,
                                      failure_holm_p=adjusted if values else None, reasons=probe.get('not_evaluated', {})))
                goal_lines.append(' & '.join([suite.capitalize(), LABELS[policy], fmt(a.get('handoff_auroc')), response,
                                             bold(prob(p), finite(p) and p >= .05),
                                             bold(fmt(failure_auc), bool(values) and adjusted >= .05)]) + r'\\')
            if 'main5' in selected:
                analysis = data[suite, policy, 'occluder']
                fail, n = counts[suite, policy, 'occluder']
                ok = eligible(analysis, fail, n)
                spread = aucs(analysis, 80, 'native_spread') if ok else []
                handoff = aucs(analysis, 80, 'handoff') if ok else []
                r2 = probe.get('B', {}).get('occluder', {}).get('r2')
                occ_rows.append(dict(suite=suite, policy=policy, status='evaluated' if ok else 'ineligible',
                                     failures=fail, episodes=n, spread_task_aurocs=spread, handoff_task_aurocs=handoff,
                                     spread_median=med(spread), handoff_median=med(handoff), r2=r2,
                                     reasons=probe.get('not_evaluated', {})))
                occ_lines.append(' & '.join([suite.capitalize(), LABELS[policy], f'{fail}/{n}',
                                            med_iqr(spread), med_iqr(handoff), bold(fmt(r2), finite(r2) and r2 <= 0)]) + r'\\')
    if 'main3' in selected:
        write_tables(destination, 'main_goal', 'llcccc',
                     r'Suite & Policy & Handoff & Spread up (change) & $p$ & Failure AUROC', goal_lines, goal_rows,
                     'Spread-up counts use the available matched pairs. Bold marks no detected paired change or no above-chance ranking after the prescribed 24-test Holm family. Dashes and CSV status retain unavailable analyses.')
    if 'main5' in selected:
        write_tables(destination, 'main_occluder', 'llcccc',
                     r'Suite & Policy & Fail/$N$ & Spread & Handoff & $R^2$', occ_lines, occ_rows,
                     'AUROC entries show median and order-statistic IQR over evaluable tasks. Dashes retain ineligible failure-ranking cells or unavailable readouts; CSV retains counts, per-task AUROCs and status.')


def supplementary_images(source: Path, destination: Path) -> None:
    data = read(source)
    dec = {(v['suite'], v['policy']): v.get('situations', {}) for v in data.get('decodability', [])}
    ranking = {(v['suite'], v['policy'], v['situation']): v for v in data.get('within_situation_step80', [])}
    lines, records = [], []
    for suite in SUITES:
        for policy in POLICIES:
            for condition in CONDITIONS:
                value = dec.get((suite, policy), {}).get(condition, {})
                dino = fmt(value.get('dinov2_auroc'))
                if finite(value.get('dinov2_p_phipson_smyth')):
                    dino += ' (' + prob(value['dinov2_p_phipson_smyth']) + ')'
                limits = value.get('handoff_minus_dinov2_ci95')
                ci = '[' + ', '.join(fmt(x, 2, True) for x in limits) + ']' if limits else '--'
                lines.append(' & '.join(['A', suite.capitalize(), SITUATIONS[condition], LABELS[policy],
                                        fmt(value.get('handoff_auroc')), fmt(value.get('image_stats_auroc')), dino,
                                        fmt(value.get('resnet18_auroc')), ci]) + r'\\')
                records.append(dict(value, part='A', suite=suite, policy=policy, condition=condition,
                                    status=value.get('status', 'evaluated' if value else 'not_evaluated_or_missing')))
    lines.append(r'\midrule')
    for suite in SUITES:
        for condition in CONDITIONS:
            for policy in POLICIES:
                value = ranking.get((suite, policy, condition), {})
                m, comp = value.get('median_auroc', {}), value.get('comparisons', {}).get('handoff-dinov2', {})
                test = value.get('above_chance', {}).get('dinov2')
                dino = fmt(m.get('dinov2'))
                if test:
                    dino += f' ({test[0]}/{test[1]}, {prob(test[2])})'
                lines.append(' & '.join(['B', suite.capitalize(), SITUATIONS[condition], LABELS[policy],
                                        fmt(m.get('handoff')), fmt(m.get('baseline')), dino, fmt(m.get('resnet18')),
                                        interval(comp.get('mean_diff'), comp.get('ci95'), 2)]) + r'\\')
                records.append(dict(part='B', suite=suite, policy=policy, condition=condition,
                                    status=value.get('status', 'evaluated' if value else 'not_evaluated_or_missing'),
                                    medians=m, handoff_minus_dino=comp, dino_sign_test=test))
    lines.append(r'\midrule')
    for condition in ('occluder', 'moved'):
        for comparison in ('handoff-dinov2', 'handoff-resnet18', 'dinov2-baseline', 'dinov2-native_spread'):
            value = data.get('pooled_comparisons', {}).get(f'{comparison}|{condition}', {})
            text = interval(value.get('task_cluster_mean'), value.get('task_cluster_ci95'))
            if value:
                text += f" ({value.get('task_cluster_positive', '--')}/{value.get('task_clusters', '--')} clusters; sign $p$ {prob(value.get('task_cluster_sign_p'))}; Holm {prob(value.get('task_cluster_holm_p'))})"
            lines.append(' & '.join(['C', 'Pooled', SITUATIONS[condition], comparison,
                                    r'\multicolumn{5}{l}{' + text + '}']) + r'\\')
            records.append(dict(value, part='C', condition=condition, comparison=comparison,
                                status=value.get('status', 'evaluated' if value else 'not_evaluated_or_missing')))
    write_tables(destination, 'supp_image_encoders', 'lllllcccc',
                 r'Part & Suite & Situation & Policy & Handoff & Image stats / Base. & DINOv2 & ResNet-18 & Handoff$-$DINOv2 [95\% CI]',
                 lines, records, 'All prescribed cells are retained, including non-ceiling camera results and unavailable estimates. Part A reads the first query for camera conditions and the first query after displacement for goals. CSV retains source statistics and status.')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tables', nargs='+', choices=('main2', 'main3', 'main5', 'supp2'),
                        default=['main2', 'main3', 'main5', 'supp2'])
    parser.add_argument('--confirm-results', type=Path, default=Path('data/confirmatory/all/confirm_results.json'))
    parser.add_argument('--analysis-dir', type=Path, default=Path('data/icassp2027/analysis'))
    parser.add_argument('--episodes-root', type=Path, default=Path('data/icassp2027/realistic'))
    parser.add_argument('--image-results', type=Path, default=Path('data/icassp2027/exp_20261020/e3_image_embedding.json'))
    parser.add_argument('--output', type=Path, default=Path('paper/tables'))
    args = parser.parse_args()
    selected = set(args.tables)
    # Complete every selected render before touching the requested destination.
    with tempfile.TemporaryDirectory(prefix='vla-table-render-') as staging:
        target = Path(staging)
        if 'main2' in selected:
            main_confirm(args.confirm_results, target)
        if selected & {'main3', 'main5'}:
            original_tables(args.analysis_dir, args.episodes_root, target, selected)
        if 'supp2' in selected:
            supplementary_images(args.image_results, target)
        rendered = {p.name: p.read_text() for p in target.iterdir()}
    args.output.mkdir(parents=True, exist_ok=True)
    for name, text in rendered.items():
        (args.output / name).write_text(text)


if __name__ == '__main__':
    main()
