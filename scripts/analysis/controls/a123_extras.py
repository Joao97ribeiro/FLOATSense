"""Analyses 1-3 metrics for protocol variants (seed 0): two-axis, SCADA-only,
top-only training, XGBoost condition-only surrogate. Same definitions and
bootstrap as a123_cpu.py."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a123_cpu import B, ROOT, cluster_sums, meta, point  # noqa: E402

EXTRA = {
    "TCN, two axes": "heights/ablation/twoaxis/{t}/damage_comparison_tcn_fa.csv",
    "Prob-TCN, two axes": "heights/ablation/twoaxis/{t}/damage_comparison_prob_tcn_fa.csv",
    "TCN, SCADA only": "heights/ablation/scada/{t}/damage_comparison_tcn_fa.csv",
    "Prob-TCN, SCADA only": "heights/ablation/scada/{t}/damage_comparison_prob_tcn_fa.csv",
    "TCN, top only": "diag/toponly/{t}/damage_comparison_tcn_fa.csv",
    "Prob-TCN, top only": "diag/toponly/{t}/damage_comparison_prob_tcn_fa.csv",
}
rng = np.random.default_rng(1)
rows = []
for t in ("ref", "opt1", "opt2"):
    m = meta(t)
    op, w = m.op.values, m.damage_weight.values
    labels = np.unique(op)
    counts = rng.multinomial(len(labels), np.ones(len(labels)) / len(labels), size=B)
    for lab, tpl in EXTRA.items():
        d = pd.read_csv(f"{ROOT}/outputs/" + tpl.format(t=t)).set_index("sim_id").loc[m.index]
        for h in ("tower_bottom", "tower_top"):
            tc, rc = f"damage_true_{h}", f"damage_rec_{h}"
            if "top only" in lab:
                if h != "tower_top":
                    continue
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
df.to_csv("a123_extras_seed0.csv", index=False)
df["height"] = df.height.map({"tower_bottom": "base", "tower_top": "top"})
df["cell"] = df.apply(lambda r: f"{r.r2:.2f} / {r.within2:.2f} / {r.rho_wc:.2f} / {r.slope:.2f} [{r.slope_lo:.2f},{r.slope_hi:.2f}] / {r.lifetime_ratio:.2f}", axis=1)
p = df.pivot_table(index="model", columns=["tower", "height"], values="cell", aggfunc="first")
p = p[[c for c in [(t, h) for t in ("ref", "opt1", "opt2") for h in ("base", "top")] if c in p.columns]]
p.columns = [f"{a} {b}" for a, b in p.columns]
open("tables_extras.md", "w").write(p.to_markdown())
print(p.to_markdown())
