import os
"""Analysis 1b: attainable condition-only surrogates (trained on E2 train).

Inputs (wind_speed, wave_hs, wave_tp) only; target log10 D per simulation
from data/<tower>/release/damage.csv, E2 train split (1,728 sims). Scored
on the E2 test split against damage_true_* of the benchmark CSVs, with the
same metrics as a123_cpu.py (rho_wc = 0 by construction).
Models: XGBoost (500 trees, depth 4, lr 0.05) and a GP (RBF-ARD + white
noise, standardized inputs, fit on op-point means of log10 D).
"""
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

ROOT = os.environ.get("FLOATSENSE_ROOT", ".")
sys.path.insert(0, ROOT)
from floatsense.metrics import summarize_damage  # noqa: E402

X = ["wind_speed", "wave_hs", "wave_tp"]
rows = []
for t in ("ref", "opt1", "opt2"):
    m = pd.read_csv(f"{ROOT}/data/{t}/release/metadata.csv").set_index("sim_id")
    lab = pd.read_csv(f"{ROOT}/data/{t}/release/damage.csv").set_index("sim_id")
    tr = pd.read_csv(f"{ROOT}/data/{t}/splits/E2/train.csv").sim_id.values
    te = pd.read_csv(f"{ROOT}/outputs/heights/{t}/seed0/damage_comparison_tcn_fa.csv").set_index("sim_id")
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
        mu, sd = xtr.mean(0), xtr.std(0)
        opm = pd.DataFrame(xtr).assign(y=y).groupby([0, 1, 2]).y.mean()
        xo = (np.array(opm.index.tolist()) - mu) / sd
        k = ConstantKernel() * RBF([1.0] * 3) + WhiteKernel(1e-3)
        gp = GaussianProcessRegressor(k, normalize_y=True, random_state=0,
                                      n_restarts_optimizer=2).fit(xo, opm.values)
        preds["GP"] = 10**gp.predict((xte - mu) / sd)
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
d.to_csv("a1b_surrogate.csv", index=False)
a = d[d.regime == "all"]
print(a.round(3).to_string())
