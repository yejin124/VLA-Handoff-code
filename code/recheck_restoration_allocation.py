"""Independent CPU reanalysis of recorded restoration pairs; no rollout execution.

Only Reader is used by the current analyses. main() is the 2026-11-05 recheck of the view-restoration pairs and needs
records and a plan file that are not distributed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import io
import itertools
import json
from math import comb
from pathlib import Path

import numpy as np

METHODS = ['handoff', 'native_spread', 'baseline', 'combined']
SUITES = ['spatial', 'object']
POLICIES = ['pi05', 'gr00t', 'molmoact']


class Reader:
    def __init__(self):
        self.hashes = {}

    def read(self, path):
        blob = path.read_bytes()
        digest = hashlib.sha256(blob).hexdigest()
        if str(path) in self.hashes and self.hashes[str(path)] != digest:
            raise ValueError(f'Input changed during audit: {path}')
        self.hashes[str(path)] = digest
        return blob

    def json(self, path):
        return json.loads(self.read(path))


def source_key(path):
    parts = Path(path).parts
    if '..' in parts:
        raise ValueError('Parent traversal in source identity')
    marker = ('data', 'icassp2027', 'realistic')
    for i in range(len(parts)-len(marker)):
        if parts[i:i+3] == marker:
            return '/'.join(parts[i+3:])
    raise ValueError(f'Unrecognized source identity: {path}')


def sign_p(values):
    untied = [v for v in values if v != 0]
    n, positive = len(untied), sum(v > 0 for v in untied)
    return min(1., 2*sum(comb(n, k) for k in range(min(positive, n-positive)+1))/2**n) if n else 1.


def holm(values):
    result, running = [0.]*len(values), 0.
    for rank, index in enumerate(sorted(range(len(values)), key=values.__getitem__)):
        running = max(running, (len(values)-rank)*values[index])
        result[index] = min(running, 1.)
    return result


def exact_effect(g, d):
    n = len(g)
    return Fraction(n*sum(a*b for a,b in zip(g,d))-sum(g)*sum(d), n*n)


def load_rows(reader, root):
    data, rows, checks = root/'data/icassp2027', [], Counter()
    for suite in SUITES:
        for policy in POLICIES:
            branch_records = {}
            for path in sorted((data/'exp_20261029'/suite/policy).glob('*.json')):
                r = reader.json(path)
                if set(r['branches']) != {'continue', 'restore'}:
                    continue
                key = source_key(r['source'])
                if key in branch_records:
                    raise ValueError(f'Duplicate branch identity: {key}')
                branch_records[key] = r
            analysis = reader.json(data/'analysis'/f'realistic_{suite}_occluder_{policy}.json')
            landmark = next(x for x in analysis['landmarks'] if x['landmark']==80)
            matched = set()
            for fold in landmark['folds']:
                if 'methods' not in fold:
                    continue
                predictions = {}
                for method in METHODS:
                    ps = fold['methods'][method]['predictions']
                    predictions[method] = {source_key(p['id']):p for p in ps}
                    if len(predictions[method]) != len(ps):
                        raise ValueError('Duplicate prediction ID')
                for key, pred in predictions['handoff'].items():
                    if key not in branch_records:
                        continue
                    if key in matched:
                        raise ValueError('Branch appears in multiple test folds')
                    matched.add(key)
                    r = branch_records[key]
                    c, b = r['branches']['continue'], r['branches']['restore']
                    source = data/'realistic'/key
                    summary = reader.json(source/'summary.json')
                    if pred['failure'] != (not summary['success']):
                        raise ValueError('Failure label disagrees with source summary')
                    if r['task'] != fold['test_task']:
                        raise ValueError('Task identity mismatch')
                    checks['source_labels_verified'] += 1
                    if r['replay_max_state_error'] > 1e-5:
                        raise ValueError('Retained branch exceeds recorded robot-state tolerance')
                    checks['recorded_eight_value_state_checks_within_tolerance'] += 1
                    qpath = source/f"q{r['branch_query']:04d}.npz"
                    raw_match = None
                    if qpath.exists():
                        with np.load(io.BytesIO(reader.read(qpath)),allow_pickle=False) as z:
                            actions = z['executed_actions']
                        raw_match = bool(np.array_equal(actions, np.asarray(c['first_actions'], dtype=np.float32)))
                        if raw_match != c.get('first_actions_match_source'):
                            raise ValueError(f'Incorrect first-action flag: {key}')
                        checks['first_action_flags_verified'] += 1
                    elif c.get('first_actions') is not None:
                        raise ValueError('Missing source query for a reported continue action')
                    no_query = (not qpath.exists() and c.get('first_actions') is None and b.get('first_actions') is None
                                and c['queries_after_branch']==0 and b['queries_after_branch']==0)
                    checks['no_next_query'] += int(no_query)
                    image_match = c.get('first_image_matches_source') is True
                    if qpath.exists():
                        traces = []
                        for path in source.glob('*.jsonl'):
                            if path.name == 'steps.jsonl':
                                continue
                            candidates = [json.loads(line) for line in reader.read(path).splitlines()]
                            traces += [x for x in candidates if 'image_sha256' in x and x.get('query')==r['branch_query']]
                        if len(traces)!=1:
                            raise ValueError('No unique query trace')
                        actual_image_match = c['first_image_sha256'] == traces[0]['image_sha256']
                        if actual_image_match != c.get('first_image_matches_source'):
                            raise ValueError('Incorrect image match flag')
                        checks['first_image_flags_verified'] += 1
                    row = {'id':key, 'suite':suite, 'policy':policy, 'task':fold['test_task'],
                           'continue':int(c['success']), 'restore':int(b['success']),
                           'action_match':raw_match is True, 'image_match':image_match, 'no_query':no_query}
                    for method in METHODS:
                        p = predictions[method][key]
                        threshold = fold['methods'][method]['calibration']['threshold']
                        if bool(p['alarm']) != (p['score'] > threshold):
                            raise ValueError('Alarm differs from strict stored threshold')
                        row[method] = int(p['alarm'])
                        checks['frozen_alarms_verified'] += 1
                    rows.append(row)
            if matched != set(branch_records):
                raise ValueError(f'Unmatched complete branches for {suite}/{policy}')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Repeated source across cells')
    return rows, dict(checks)


def summarize(rows, method, draws):
    strata = defaultdict(list)
    for row in rows:
        strata[(row['suite'], row['task'], row['policy'])].append(row)
    effects, cluster_members = {}, defaultdict(list)
    for (suite,task,policy), rs in strata.items():
        g = [r[method] for r in rs]
        d = [r['restore']-r['continue'] for r in rs]
        effects[(suite,task,policy)] = exact_effect(g,d)
        cluster_members[(suite,task)].append(policy)
    # Preserve first-seen suite/policy/task order for deterministic reproduction
    # of the archived bootstrap. Spatial task 4 enters with MolmoAct2 only.
    clusters = list(cluster_members)
    values = [sum((effects[(s,t,p)] for p in cluster_members[(s,t)]),Fraction())/len(cluster_members[(s,t)]) for s,t in clusters]
    observed = sum(values,Fraction())/len(values)
    x = np.array([float(v) for v in values])
    boot_rng = np.random.default_rng(20261029)
    boot = x[boot_rng.integers(0,len(x),(10000,len(x)))].mean(axis=1)
    # Independent exact-count random gates; sample category counts, not Bernoulli gates.
    rng, null, expected_variance = np.random.default_rng(20261105), np.zeros(draws), 0.
    for (s,t,p), rs in strata.items():
        n, k = len(rs), sum(r[method] for r in rs)
        d = [r['restore']-r['continue'] for r in rs]
        weight = 1/(len(values)*len(cluster_members[(s,t)]))
        colors = [d.count(1), d.count(-1), d.count(0)]
        selected = rng.multivariate_hypergeometric(colors,k,size=draws)
        null += weight*((selected[:,0]-selected[:,1])/n-k*sum(d)/(n*n))
        if n>1:
            expected_variance += weight**2 * k*(n-k)/(n-1) * float(np.var(d))/n**2
    p = float((1+np.count_nonzero(np.abs(null)>=abs(float(observed))-1e-14))/(draws+1))
    mc_se = float(np.sqrt(expected_variance/draws))
    if mc_se and abs(null.mean()) > 6*mc_se:
        raise ValueError('Random-allocation simulation mean violates its zero expectation')
    loo = [(sum(x)-v)/(len(x)-1) for v in x] if len(x)>1 else []
    return {'prefixes':len(rows), 'strata':len(strata), 'clusters':len(values),
        'mean':float(observed), 'exact_mean':str(observed), 'ci95':[float(v) for v in np.quantile(boot,[.025,.975])],
        'positive':sum(v>0 for v in values),'negative':sum(v<0 for v in values),'tied':sum(v==0 for v in values),
        'sign_p':sign_p(values),'conditional_allocation_p':p,
        'conditional_allocation_draws':draws, 'null_mean':float(null.mean()),
        'null_mean_mc_se':mc_se, 'analytic_null_variance':expected_variance,
        'sampled_null_variance':float(null.var()),
        'leave_one_cluster_out_mean_range':[min(loo),max(loo)] if loo else None,
        'gate_count':sum(r[method] for r in rows),
        'gate_successes':sum(r['restore'] if r[method] else r['continue'] for r in rows),
        'continue_successes':sum(r['continue'] for r in rows),
        'restore_successes':sum(r['restore'] for r in rows),
        'random_expected_successes':sum(r['continue'] for r in rows)+sum(sum(r[method] for r in rs)/len(rs)*sum(r['restore']-r['continue'] for r in rs) for rs in strata.values()),
        'cluster_effects':{f'{s}|{t}':str(v) for (s,t),v in zip(clusters,values)}}


def self_test():
    cases = 0
    for n in range(1,6):
        for d in itertools.product([-1,0,1],repeat=n):
            for k in range(n+1):
                values = []
                for selected in itertools.combinations(range(n),k):
                    g = [int(i in selected) for i in range(n)]
                    effect = exact_effect(g,d)
                    direct = sum(Fraction(a*b,n) for a,b in zip(g,d))-Fraction(sum(g)*sum(d),n*n)
                    assert effect == direct
                    values.append(effect)
                assert sum(values,Fraction()) == 0
                counts = Counter(sum(d[i] for i in selected) for selected in itertools.combinations(range(n),k))
                plus, minus, zero = d.count(1),d.count(-1),d.count(0)
                analytic = Counter()
                for a in range(plus+1):
                    for b in range(minus+1):
                        c = k-a-b
                        if 0<=c<=zero:
                            analytic[a-b] += comb(plus,a)*comb(minus,b)*comb(zero,c)
                assert counts == analytic
                cases += 1
    return cases


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--draws',type=int,default=100000)
    args = ap.parse_args()
    workspace, output = Path(__file__).resolve().parents[1], args.output.resolve()
    if workspace not in output.parents:
        raise ValueError('Output must be inside the repository')
    reader = Reader()
    reader.read(workspace/'docs/PROTOCOLS.md')
    tests = self_test()
    rows, checks = load_rows(reader,args.root.resolve())
    selectors = {'full':lambda r:True, 'action_concordant':lambda r:r['action_match'] or r['no_query'],
        'image_and_action_concordant':lambda r:(r['action_match'] and r['image_match']) or r['no_query']}
    result = {}
    for name, rule in selectors.items():
        selected = [r for r in rows if rule(r)]
        result[name] = {m:summarize(selected,m,args.draws) for m in METHODS}
    for key in ['sign_p','conditional_allocation_p']:
        for method,p in zip(METHODS,holm([result['full'][m][key] for m in METHODS])):
            result['full'][method][key+'_holm4'] = p
        pairs = [(f,m) for f in ['action_concordant','image_and_action_concordant'] for m in METHODS]
        for (f,m),p in zip(pairs,holm([result[f][m][key] for f,m in pairs])):
            result[f][m][key+'_holm8'] = p
    expected = reader.json(args.root/'data/icassp2027/failure_submission/view_restore_analysis.json')['pooled']
    differences = {}
    for method in METHODS:
        actual = result['full'][method]
        differences[method] = {'mean_error':actual['mean']-expected[method]['mean'],
            'ci95_errors':[a-b for a,b in zip(actual['ci95'],expected[method]['ci95'])]}
        for key in ['positive','negative','clusters','sign_p']:
            if actual[key] != expected[method][key]:
                raise ValueError(f'Primary {method} {key} failed to reproduce')
        if abs(differences[method]['mean_error'])>1e-12:
            raise ValueError('Primary mean failed to reproduce')
        if max(abs(v) for v in differences[method]['ci95_errors'])>1e-12:
            raise ValueError('Primary bootstrap interval failed to reproduce')
    payload = {'schema_version':1,'time_utc':datetime.now(timezone.utc).isoformat(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'Independent validation and post hoc sensitivity of retained pairs; no new rollouts, full-state causal validation, or physical robot evidence.',
        'self_test_exhaustive_cases':tests,'checks':checks,'primary_reproduction':differences,'results':result,
        'filters':{'action_concordant':'Verified exact first continue/source action or no next query in source and both branches.',
            'image_and_action_concordant':'Verified first image and action concordance or no next query in source and both branches.'},
        'limitations':['Selection changes the population and does not establish full-state replay fidelity.',
            'Intervals condition on frozen fits; shared training sets limit population-level sign-test interpretation.',
            'Random-allocation p values condition on these retained outcome pairs and are not new-task or causal source-policy tests.',
            'Neither allocation gain nor these filters establishes rescue selection among failed continuations.'],
        'source_files_sha256':reader.hashes}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps({'checks':checks,'self_test_cases':tests,'primary_reproduction':differences,
        'handoff':{f:{k:v for k,v in x['handoff'].items() if k!='cluster_effects'} for f,x in result.items()}},indent=2))


if __name__ == '__main__':
    main()
