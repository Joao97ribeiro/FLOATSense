"""Scale-only cross-tower recalibration of zero-shot predictions.

For source->target, model, seed: s_h = median_i(D_true_i / D_pred_i) over the
calibration sims (target training sims, re-inferred with the source
checkpoint by infer_designed.py into <controls>/a4_scale/infer), per height
h. Applied to the STORED zero-shot test predictions (<runs>/<src>/seed<k>/
damage_comparison_<m>_fa_zs_<tgt>.csv): D_rec' = s_h * D_rec. Metrics with
floatsense.metrics.summarize_damage (log10 R^2, fraction in [0.5, 2],
median ratio). Calibration sets:
  des5   first 5 rows of fewshot/train_designed.csv (wind 4.5-8.5 m/s)
  rand5  fewshot/train_5_draw0.csv (the paper's N=5 set)
  des18  all 18 of fewshot/train_designed.csv (the paper's '18 designed')
  oracle s_h = median ratio on the test set itself (upper bound of any
         single-scale correction)
The paper's fine-tuning numbers (columns paper_FT_*) are read from the LaTeX
tables curve_top.tex and curve_base.tex of the folder in
FLOATSENSE_PAPER_TABLES; the columns stay empty when it is not set.
Writes scale_recal_all.csv, scale_recal_seed0.csv and
scale_recal_seedmedian.csv to <controls>/a4_scale. Paths: see layout.py.

Usage: python scripts/analysis/controls/scale/analyze.py
"""
import os
import re
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import layout  # noqa: E402  pylint: disable=wrong-import-position
from floatsense.metrics import summarize_damage  # noqa: E402  pylint: disable=wrong-import-position

A = layout.controls("a4_scale")
TW = list(layout.TOWERS)


def load(src, tgt, m, seed, tag):
    """Damage of a source checkpoint re-inferred on target simulations."""
    return pd.read_csv(f"{A}/infer/{src}/seed{seed}/damage_comparison_{m}_fa_{tag}_{tgt}.csv")


rows = []
for seed in (0, 1, 2):
    for src in TW:
        for tgt in TW:
            if src == tgt:
                continue
            for m in ("tcn", "prob_tcn", "naive"):
                try:
                    zs = pd.read_csv(layout.damage_csv(src, seed, m, f"zs_{tgt}"))
                    cal = {"des5": load(src, tgt, m, seed, "designed").iloc[:5],
                           "des18": load(src, tgt, m, seed, "designed")}
                except FileNotFoundError:
                    continue
                try:
                    cal["rand5"] = load(src, tgt, m, seed, "rand5")
                except FileNotFoundError:
                    pass
                for h in ("tower_bottom", "tower_top"):
                    t, r = zs[f"damage_true_{h}"].values, zs[f"damage_rec_{h}"].values
                    base = dict(seed=seed, src=src, tgt=tgt, model=m, height=h)
                    rows.append({**base, "calib": "none", "scale": 1.0, **summarize_damage(t, r)})
                    scales = {k: float(np.median(c[f"damage_true_{h}"] / c[f"damage_rec_{h}"])) for k, c in cal.items()}
                    scales["oracle"] = 1.0 / float(np.median(r / t))
                    for k, s in scales.items():
                        rows.append({**base, "calib": k, "scale": s, **summarize_damage(t, s * r)})
df = pd.DataFrame(rows)
df.to_csv(f"{A}/scale_recal_all.csv", index=False)

# paper fine-tuning numbers (seed 0, draw 0)
def paper(table):
    """(pair, model) -> (N = 5, 18 designed) cells of a paper LaTeX table."""
    out, pair = {}, None
    names = {"twref": "ref", "twopta": "opt1", "twoptb": "opt2"}
    folder = os.environ.get("FLOATSENSE_PAPER_TABLES")
    if not folder:
        return out
    for line in open(os.path.join(folder, f"{table}.tex"), encoding="utf-8"):
        mm = re.search(r"\\(tw\w+)\{\} \$\\to\$ \\(tw\w+)\{\}", line)
        if mm:
            pair = (names[mm.group(1)], names[mm.group(2)])
            continue
        mm = re.match(r"\s*\\quad (TCN|Prob-TCN) &(.*)\\\\", line)
        if mm and pair:
            vals = [v.strip() for v in mm.group(2).split("&")]
            out[(pair, mm.group(1))] = (vals[1], vals[6])  # N=5, 18 designed
    return out
P = {"tower_top": paper("curve_top"), "tower_bottom": paper("curve_base")}
MN = {"tcn": "TCN", "prob_tcn": "Prob-TCN"}

s0 = df[df.seed == 0]
lines = []
for m in ("tcn", "prob_tcn", "naive"):
    for h in ("tower_bottom", "tower_top"):
        d = s0[(s0.model == m) & (s0.height == h)]
        if d.empty:
            continue
        piv = d.pivot_table(index=["src", "tgt"], columns="calib", values=["r2_log_damage", "fraction_within_factor2"])
        sc = d.pivot_table(index=["src", "tgt"], columns="calib", values="scale")
        for (src, tgt), row in piv.iterrows():
            ft = P[h].get(((src, tgt), MN.get(m, "")), ("", ""))
            lines.append(dict(model=m, height=h, pair=f"{src}->{tgt}",
                **{f"R2_{c}": row[("r2_log_damage", c)] for c in ["none", "des5", "rand5", "des18", "oracle"] if ("r2_log_damage", c) in row},
                **{f"f2_{c}": row[("fraction_within_factor2", c)] for c in ["none", "des5", "rand5", "des18", "oracle"] if ("fraction_within_factor2", c) in row},
                s_des5=sc.loc[(src, tgt), "des5"], s_rand5=sc.loc[(src, tgt)].get("rand5", np.nan), s_des18=sc.loc[(src, tgt), "des18"],
                paper_FT_N5=ft[0], paper_FT_des18=ft[1]))
out = pd.DataFrame(lines)
out.to_csv(f"{A}/scale_recal_seed0.csv", index=False)
pd.set_option("display.width", 250)
print(out.round(2).to_string(index=False))
# seed medians
med = df.groupby(["model", "height", "src", "tgt", "calib"])[["r2_log_damage", "fraction_within_factor2"]].median().reset_index()
med.to_csv(f"{A}/scale_recal_seedmedian.csv", index=False)
print(df.groupby(["model"]).seed.unique())
