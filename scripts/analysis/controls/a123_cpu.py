"""Reviewer analyses 1-3 (CPU) for FLOATSense, 2026-09-26.

1. Condition-only oracle (mean of TRUE damage over the 6 realizations of
   each operating point, in-sample and leave-one-out) vs models.
2. Within-condition calibration slope: OLS slope of predicted deviation
   dhat on true deviation d, d_i = log10 D_i - mean_{c_i} log10 D.
3. Lifetime-weighted damage ratio sum(w*Dhat)/sum(w*D) over the test split.
Cluster bootstrap over operating points, B=2000, percentile 95%.
Metric definitions reuse floatsense.metrics.summarize_damage and the
rho_wc code of scripts/benchmark/build_master.py.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.environ.get("FLOATSENSE_ROOT", ".")
sys.path.insert(0, ROOT)
from floatsense.metrics import summarize_damage  # noqa: E402

OUT = f"{ROOT}/outputs/controls"
TOWERS = ("ref", "opt1", "opt2")
HEIGHTS = ("tower_bottom", "tower_top")
B = 2000
MODELS = {  # label -> path template (tower, seed)
    "Floor": "heights/{t}/seed{s}/damage_comparison_naive_fa.csv",
    "TCN": "heights/{t}/seed{s}/damage_comparison_tcn_fa.csv",
    "Prob-TCN (sample)": "heights/{t}/seed{s}/damage_comparison_prob_tcn_fa.csv",
    "Prob-TCN (mean head)": "heights/noise/{t}/damage_comparison_prob_mean_fa.csv",
    "PatchTST": "heights/{t}/seed{s}/damage_comparison_transformer_fa.csv",
    "Mamba": "heights/{t}/seed{s}/damage_comparison_mamba_fa.csv",
    "Physics": "physics_heights/{t}_E2/damage_heights.csv",
}
SEED0_ONLY = {"Prob-TCN (mean head)", "Physics"}


def meta(tower):
    m = pd.read_csv(f"{ROOT}/data/{tower}/release/metadata.csv")
    m = m[m.split == "test"].set_index("sim_id")
    m["op"] = m[["wind_speed_id", "wave_hs_id", "wave_tp_id"]].astype(
        str).agg("_".join, axis=1)
    return m


def load(path, m):
    d = pd.read_csv(path).set_index("sim_id").loc[m.index]
    return d


def point(true, rec, op, w):
    s = summarize_damage(true, rec)
    lt, lr = np.log10(true), np.log10(rec)
    g = pd.Series(op)
    dt = lt - pd.Series(lt).groupby(g).transform("mean").values
    dr = lr - pd.Series(lr).groupby(g).transform("mean").values
    vr = np.sum(dr**2)
    rho = 0.0 if vr < 1e-24 else float(np.corrcoef(dt, dr)[0, 1])
    return {
        "r2": s["r2_log_damage"], "within2": s["fraction_within_factor2"],
        "median_ratio": s["median_damage_ratio"], "rho_wc": rho,
        "slope": float(np.sum(dt * dr) / np.sum(dt**2)),
        "amp_ratio": float(np.sqrt(vr / np.sum(dt**2))),
        "lifetime_ratio": float(np.sum(w * rec) / np.sum(w * true)),
    }, dt, dr


def cluster_sums(op, dt, dr, true, rec, w):
    g = pd.DataFrame({"op": op, "xy": dt * dr, "xx": dt**2, "wt": w * true,
                      "wr": w * rec}).groupby("op").sum()
    return g


def main():
    rng = np.random.default_rng(0)
    rows, boots = [], []
    for t in TOWERS:
        m = meta(t)
        op, w = m.op.values, m.damage_weight.values
        labels = np.unique(op)
        k = len(labels)
        counts = rng.multinomial(k, np.ones(k) / k, size=B)  # B x K
        ref = load(f"{ROOT}/outputs/heights/{t}/seed0/damage_comparison_tcn_fa.csv", m)
        preds = {}
        for h in HEIGHTS:
            true = ref[f"damage_true_{h}"].values
            lt = np.log10(true)
            gm = pd.Series(lt).groupby(op)
            n = gm.transform("count").values
            s_log = gm.transform("sum").values
            s_lin = pd.Series(true).groupby(op).transform("sum").values
            preds[("Oracle, arith. mean", h, 0)] = (true, s_lin / n)
            preds[("Oracle, geo. mean", h, 0)] = (true, 10**(s_log / n))
            preds[("Oracle LOO, arith.", h, 0)] = (true, (s_lin - true) / (n - 1))
            preds[("Oracle LOO, geo.", h, 0)] = (true, 10**((s_log - lt) / (n - 1)))
        for lab, tpl in MODELS.items():
            for s in ((0,) if lab in SEED0_ONLY else (0, 1, 2)):
                p = f"{ROOT}/outputs/" + tpl.format(t=t, s=s)
                if not os.path.exists(p):
                    print("missing", p)
                    continue
                d = load(p, m)
                for h in HEIGHTS:
                    tr = d[f"damage_true_{h}"].values
                    if lab != "Physics":  # physics has its own true pipeline
                        assert np.allclose(tr, ref[f"damage_true_{h}"].values,
                                           rtol=1e-6), (p, h)
                    preds[(lab, h, s)] = (tr, d[f"damage_rec_{h}"].values)
        for (lab, h, s), (true, rec) in preds.items():
            ok = (true > 0) & (rec > 0)
            assert ok.all(), (lab, t, h, s, (~ok).sum())
            pt, dt, dr = point(true, rec, op, w)
            rows.append({"model": lab, "tower": t, "height": h, "seed": s,
                         **pt})
            cs = cluster_sums(op, dt, dr, true, rec, w).loc[labels]
            slope_b = counts @ cs.xy.values / (counts @ cs.xx.values)
            life_b = counts @ cs.wr.values / (counts @ cs.wt.values)
            boots.append(pd.DataFrame({"model": lab, "tower": t, "height": h,
                                       "seed": s, "b": np.arange(B),
                                       "slope": slope_b,
                                       "lifetime_ratio": life_b}))
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(f"{OUT}/a123_per_seed.csv", index=False)
    bt = pd.concat(boots)
    keys = ["model", "tower", "height"]
    # seed-median point; interval = percentiles of the per-resample seed
    # median (same resample of operating points for every seed)
    med = per_seed.groupby(keys).median(numeric_only=True).drop(columns="seed")
    med["n_seeds"] = per_seed.groupby(keys).size()
    bm = bt.groupby(keys + ["b"])[["slope", "lifetime_ratio"]].median()
    q = bm.groupby(keys).quantile([0.025, 0.975]).unstack()
    for c in ("slope", "lifetime_ratio"):
        med[f"{c}_lo"] = q[(c, 0.025)]
        med[f"{c}_hi"] = q[(c, 0.975)]
    med = med.reset_index()
    med.to_csv(f"{OUT}/a123_seed_median.csv", index=False)
    s0 = per_seed[per_seed.seed == 0].set_index(keys)
    b0 = bt[bt.seed == 0].groupby(keys)[["slope", "lifetime_ratio"]].quantile(
        [0.025, 0.975]).unstack()
    for c in ("slope", "lifetime_ratio"):
        s0[f"{c}_lo"] = b0[(c, 0.025)]
        s0[f"{c}_hi"] = b0[(c, 0.975)]
    s0.reset_index().to_csv(f"{OUT}/a123_seed0.csv", index=False)
    pd.set_option("display.width", 250)
    print(med.round(3).to_string())


if __name__ == "__main__":
    main()
