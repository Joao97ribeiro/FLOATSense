import os
"""Cluster-bootstrap 95% CIs (B=2000, operating points resampled, scale held
fixed at its calibration value) for seed 0, TCN and Prob-TCN."""
import sys
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.environ.get("FLOATSENSE_ROOT", "."))
from floatsense.metrics import cluster_bootstrap
R = os.environ.get("FLOATSENSE_ROOT", ".")
A = f"{R}/outputs/controls/a4_scale"
S = pd.read_csv(f"{A}/scale_recal_all.csv")
S = S[(S.seed == 0) & S.model.isin(["tcn", "prob_tcn"]) & S.calib.isin(["none", "rand5", "des18"])]

def cond(t):
    m = pd.read_csv(f"{R}/data/{t}/release/metadata.csv").set_index("sim_id")
    return m[["wind_speed_id", "wave_hs_id", "wave_tp_id"]].astype(str).agg("_".join, axis=1)

def job(d):
    r = pd.Series(d)
    zs = pd.read_csv(f"{R}/outputs/heights/{r.src}/seed0/damage_comparison_{r.model}_fa_zs_{r.tgt}.csv")
    c = cond(r.tgt).loc[zs.sim_id].values
    t, p = zs[f"damage_true_{r.height}"].values, r.scale * zs[f"damage_rec_{r.height}"].values
    ci = cluster_bootstrap(t, p, c, 2000, 0)
    return {**d, "r2_lo": ci["r2_log_damage"][0], "r2_hi": ci["r2_log_damage"][1],
            "f2_lo": ci["fraction_within_factor2"][0], "f2_hi": ci["fraction_within_factor2"][1]}

with ProcessPoolExecutor(24) as ex:
    out = pd.DataFrame(list(ex.map(job, S.to_dict('records'))))
out.to_csv(f"{A}/scale_recal_seed0_ci.csv", index=False)
out["pair"] = out.src + "->" + out.tgt
out["R2"] = out.apply(lambda r: f"{r.r2_log_damage:.2f} [{r.r2_lo:.2f},{r.r2_hi:.2f}]", axis=1)
out["F2"] = out.apply(lambda r: f"{r.fraction_within_factor2:.2f} [{r.f2_lo:.2f},{r.f2_hi:.2f}]", axis=1)
print(out.pivot_table(index=["model", "height", "pair"], columns="calib", values="R2", aggfunc="first").to_string())
print(out.pivot_table(index=["model", "height", "pair"], columns="calib", values="F2", aggfunc="first").to_string())
