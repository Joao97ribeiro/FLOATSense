"""Analysis 1b: attainable condition-only surrogates (trained on the train split).

Inputs (wind_speed, wave_hs, wave_tp) only; target log10 D per simulation
from the released damage (damage.parquet), train split (1,728 sims). Scored
on the test split against damage_true_* of the seed-0 TCN run, with the
same metrics as a123_cpu.py (rho_wc = 0 by construction).
Model: XGBoost (500 trees, depth 4, lr 0.05). A Gaussian process was also
tried and dropped: its length-scale fit lands on degenerate optima that
break extrapolation cells arbitrarily, and the paper reports XGBoost only.
Needs xgboost. Writes a1b_surrogate.csv to the controls
folder. Paths: see layout.py.

Usage: python scripts/analysis/controls/a1b_surrogate.py
"""
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import layout  # noqa: E402  pylint: disable=wrong-import-position
from floatsense import load_tower  # noqa: E402  pylint: disable=wrong-import-position
from floatsense.data import HEIGHT_TARGETS  # noqa: E402  pylint: disable=wrong-import-position
from floatsense.metrics import summarize_damage  # noqa: E402  pylint: disable=wrong-import-position

X = ["wind_speed", "wave_hs", "wave_tp"]
rows = []
for t in layout.TOWERS:
    release = load_tower(layout.DATA, t)
    m = release.metadata
    # damage.parquet columns are the section ids, base to top, in the order
    # of the 11 gauges.
    lab = release.damage()
    lab.columns = [f"D_{stem}" for stem, _, _ in HEIGHT_TARGETS]
    tr = release.split_ids("train")
    te = pd.read_csv(layout.damage_csv(t, 0, "tcn")).set_index("sim_id")
    reg = (m.loc[te.index, "wind_group"] + "/" + m.loc[te.index, "wave_group"]).values
    for h in ("tower_bottom", "tower_top"):
        y = np.log10(lab.loc[tr, f"D_{h}"].values)
        xtr, xte = m.loc[tr, X].values, m.loc[te.index, X].values
        true = te[f"damage_true_{h}"].values
        preds = {}
        g = xgb.XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.05,
                             subsample=0.8, random_state=0, n_jobs=8)
        g.fit(xtr, y)
        preds["XGBoost"] = 10**g.predict(xte)
        for name, rec in preds.items():
            for grp in ["all"] + sorted(set(reg)):
                sel = np.ones(len(reg), bool) if grp == "all" else reg == grp
                s = summarize_damage(true[sel], rec[sel])
                rows.append({"model": name, "tower": t, "height": h,
                             "regime": grp, "n": int(sel.sum()),
                             "r2": s["r2_log_damage"],
                             "within2": s["fraction_within_factor2"],
                             "median_ratio": s["median_damage_ratio"],
                             "rho_wc": 0.0})
d = pd.DataFrame(rows)
d.to_csv(layout.controls("a1b_surrogate.csv"), index=False)
a = d[d.regime == "all"]
print(a.round(3).to_string())
