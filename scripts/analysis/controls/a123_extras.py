"""Analyses 1-3 metrics for protocol variants (seed 0): two-axis, SCADA-only,
top-only training. Same definitions and bootstrap as a123_cpu.py.

Reads the seed-0 runs of the sensor ablations (ablation/twoaxis,
ablation/scada) and of the top-only diagnostic (toponly); writes
a123_extras_seed0.csv and tables_extras.md to the controls folder. Paths: see
layout.py.

Usage: python scripts/analysis/controls/a123_extras.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import layout  # noqa: E402  pylint: disable=wrong-import-position
from a123_cpu import B, cluster_sums, meta, point  # noqa: E402  pylint: disable=wrong-import-position

EXTRA = {  # label -> (experiment, model)
    "TCN, two axes": ("twoaxis", "tcn"),
    "Prob-TCN, two axes": ("twoaxis", "prob_tcn"),
    "TCN, SCADA only": ("scada", "tcn"),
    "Prob-TCN, SCADA only": ("scada", "prob_tcn"),
    "TCN, top only": ("toponly", "tcn"),
    "Prob-TCN, top only": ("toponly", "prob_tcn"),
}
rng = np.random.default_rng(1)
rows = []
for t in layout.TOWERS:
    m = meta(t)
    op, w = m.op.values, m.damage_weight.values
    labels = np.unique(op)
    counts = rng.multinomial(len(labels), np.ones(len(labels)) / len(labels), size=B)
    for lab, (kind, model) in EXTRA.items():
        path = os.path.join(layout.experiment_dir(kind, t),
                            f"damage_comparison_{model}_fa.csv")
        d = pd.read_csv(path).set_index("sim_id").loc[m.index]
        for h in ("tower_bottom", "tower_top"):
            tc, rc = f"damage_true_{h}", f"damage_rec_{h}"
            if "top only" in lab:
                if h != "tower_top":
                    continue
                # The research runs named the single-height columns by the
                # direction ('_fa'); the release names them by the gauge.
                if "damage_true_fa" in d.columns:
                    tc, rc = "damage_true_fa", "damage_rec_fa"
            true, rec = d[tc].values, d[rc].values
            pt, dt, dr = point(true, rec, op, w)
            cs = cluster_sums(op, dt, dr, true, rec, w).loc[labels]
            sb = counts @ cs.xy.values / (counts @ cs.xx.values)
            lb = counts @ cs.wr.values / (counts @ cs.wt.values)
            rows.append({"model": lab, "tower": t, "height": h, "seed": 0, **pt,
                         "slope_lo": np.percentile(sb, 2.5), "slope_hi": np.percentile(sb, 97.5),
                         "lifetime_ratio_lo": np.percentile(lb, 2.5),
                         "lifetime_ratio_hi": np.percentile(lb, 97.5)})
df = pd.DataFrame(rows)
df.to_csv(layout.controls("a123_extras_seed0.csv"), index=False)
df["height"] = df.height.map({"tower_bottom": "base", "tower_top": "top"})
df["cell"] = df.apply(lambda r: f"{r.r2:.2f} / {r.within2:.2f} / {r.rho_wc:.2f} / {r.slope:.2f} [{r.slope_lo:.2f},{r.slope_hi:.2f}] / {r.lifetime_ratio:.2f}", axis=1)
p = df.pivot_table(index="model", columns=["tower", "height"], values="cell", aggfunc="first")
p = p[[c for c in [(t, h) for t in layout.TOWERS for h in ("base", "top")] if c in p.columns]]
p.columns = [f"{a} {b}" for a, b in p.columns]
with open(layout.controls("tables_extras.md"), "w", encoding="utf-8") as f:
    f.write(p.to_markdown())
print(p.to_markdown())
