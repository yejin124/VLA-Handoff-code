"""Validate the complete 19-condition Bridge replay before computing statistics."""
import re

import numpy as np

CONDITIONS = {('baseline', 0.0)} | {
    (name, alpha)
    for name in ('risk', 'spread', 'rand1', 'rand2', 'rand3', 'rand4')
    for alpha in (1.0, 2.0, 4.0)
}


def finite_number(value, name, minimum=0.0, strictly_positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    if value < minimum or (strictly_positive and value == minimum):
        raise ValueError(f'{name} is outside its valid range')


def validate_episode(episode):
    """Reject missing checks, duplicate conditions and non-finite action statistics."""
    eid = episode.get('id', '')
    match = re.fullmatch(r'.+/t(\d+)_e\d+_r\d+_occluder', eid)
    if not match or episode.get('task') != int(match[1]) or episode.get('failure') not in (0, 1):
        raise ValueError(f'Invalid episode identity or outcome: {eid!r}')
    results = episode.get('results', [])
    keys = [(row.get('dir'), row.get('alpha')) for row in results]
    if len(keys) != len(CONDITIONS) or set(keys) != CONDITIONS:
        raise ValueError(f'{eid}: expected exactly one result for each of the 19 conditions')
    for row in results:
        label = f"{eid}/{row['dir']}/{row['alpha']}"
        finite_number(row.get('spread'), label + '/spread', strictly_positive=True)
        action = np.asarray(row.get('mean_action', []), dtype=float)
        if action.shape != (28,) or not np.isfinite(action).all():
            raise ValueError(f'{label}: expected 28 finite action means (4 steps x 7 dimensions)')
        check = 'parity_max_abs' if row['dir'] == 'baseline' else 'manip_rel_err'
        finite_number(row.get(check), label + '/' + check)
        if row[check] > (1e-5 if row['dir'] == 'baseline' else 0.05):
            raise ValueError(f'{label}: failed {check}')
