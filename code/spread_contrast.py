"""Secondary contrasts H-Spread and GH-Spread on the frozen confirmatory bootstrap.

Reruns code/ral_confirm_analysis.py (2,000 full-refit draws, seed 20261106) for the two primary cells with
two contrasts added; GH-G must reproduce the frozen primary estimates exactly. The analysis code is in this
repository (code/ral_confirm_analysis.py, code/ral_nested_fit.py); the raw cohorts are not distributed.

Usage: RAL_ROOT=<repo root with the raw records> python code/spread_contrast.py <workers> <out_dir> [cohort,cohort,...]
The default cohorts are the two GR00T primary cells; other policies' confirmatory cohorts can be passed explicitly.
"""
import os
import sys

ROOT = os.environ.get('RAL_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('RAL_RAW_ROOT', ROOT)
import ral_nested_fit as f  # noqa: E402

f.CONTRASTS.update({'H-Spread': ('H', 'Spread'), 'GH-Spread': ('GH', 'Spread')})
import ral_confirm_analysis as a  # noqa: E402

a.CONTRASTS = f.CONTRASTS
cells = sys.argv[3] if len(sys.argv) > 3 else 'gr00t_bridge_confirm,gr00t_spatial_confirm'
sys.argv = ['spread_contrast', '--workers', sys.argv[1], '--output', sys.argv[2], '--cells', cells]
a.main()
