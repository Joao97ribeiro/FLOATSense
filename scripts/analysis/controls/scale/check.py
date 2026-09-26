"""Checks that re-run inference reproduces the stored zero-shot damage.

Compares <controls>/a4_scale/infer/<src>/seed<k>/
damage_comparison_<model>_fa_check_<tgt>.csv (infer_designed.py ... check)
with the same simulations of the stored zero-shot file
<runs>/<src>/seed<k>/damage_comparison_<model>_fa_zs_<tgt>.csv. Prob-TCN
samples its variance head, so only its true damage has to match. Paths: see
layout.py.

Usage: python scripts/analysis/controls/scale/check.py <src> <tgt> <model> <seed>
"""
import os
import sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import layout  # noqa: E402  pylint: disable=wrong-import-position
src, tgt, model, seed = sys.argv[1:5]
new = pd.read_csv(layout.controls("a4_scale", "infer", src, f"seed{seed}", f"damage_comparison_{model}_fa_check_{tgt}.csv")).set_index("sim_id")
old = pd.read_csv(layout.damage_csv(src, int(seed), model, f"zs_{tgt}")).set_index("sim_id").loc[new.index]
cols = [c for c in new.columns if c.startswith("damage_")]
rel = (new[cols] / old[cols] - 1).abs()
print(src, tgt, model, seed, "max|rel| true %.2e rec %.2e" % (rel[[c for c in cols if "true" in c]].values.max(), rel[[c for c in cols if "rec" in c]].values.max()))
