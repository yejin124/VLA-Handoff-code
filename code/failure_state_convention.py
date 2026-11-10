"""Explicit state-contract selection for reproducing historical restoration."""
import hashlib
from pathlib import Path

import icassp_rollout as current

ARCHIVE_SHA256 = '07141b7ee7002cf4296708cdc3984652c2ddd030b0b2ac1854c5f2cfe951fd24'


def select_state_convention(choice):
    if choice == 'official':
        if current.ROBOT_STATE_CONVENTION != 'libero_official_axis_angle_no_wrap':
            raise ValueError('Current collector does not declare the official state convention')
        module, name = current, current.ROBOT_STATE_CONVENTION
    elif choice == 'recorded':
        import recorded_failure_rollout_20261103 as archived
        if hashlib.sha256(Path(archived.__file__).read_bytes()).hexdigest() != ARCHIVE_SHA256:
            raise ValueError('Historical collector source changed')
        module, name = archived, 'legacy_wrapped_axis_angle'
    else:
        raise ValueError('Unknown state convention: ' + str(choice))
    path = Path(module.__file__)
    return module.robot_state, {
        'choice': choice, 'name': name, 'source_file': path.name,
        'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'scope': 'Numerical robot-state input; does not certify full simulator or model parity',
    }


def require_same_state_convention(record, receipt, *, historical_reference=False):
    saved = record.get('state_convention')
    if saved is None and historical_reference and receipt['choice'] == 'recorded':
        # These original continuation outcomes predate explicit provenance fields.
        # The caller must deliberately request the archived numerical convention.
        return
    if saved != receipt:
        raise ValueError('State convention is absent or differs; use a new output directory '
                         'and --state-convention recorded for historical reference outcomes')
