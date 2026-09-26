"""Markdown tables for summary.md from a123_seed_median.csv."""
import pandas as pd

d = pd.read_csv("a123_seed_median.csv")
ORDER = ["Oracle, arith. mean", "Oracle LOO, arith.", "Oracle, geo. mean",
         "Oracle LOO, geo.", "Floor", "Physics", "TCN", "Prob-TCN (sample)",
         "Prob-TCN (mean head)", "PatchTST", "Mamba"]
H = {"tower_bottom": "base", "tower_top": "top"}
d["height"] = d.height.map(H)
d["model"] = pd.Categorical(d.model, ORDER, ordered=True)


def piv(col, fmt, models=ORDER, ci=False):
    x = d[d.model.isin(models)].copy()
    if ci:
        x["v"] = x.apply(lambda r: f"{r[col]:{fmt}} [{r[col+'_lo']:{fmt}}, {r[col+'_hi']:{fmt}}]", axis=1)
    else:
        x["v"] = x[col].map(lambda v: f"{v:{fmt}}")
    p = x.pivot_table(index="model", columns=["tower", "height"], values="v",
                      aggfunc="first", observed=True)
    p = p[[(t, h) for t in ("ref", "opt1", "opt2") for h in ("base", "top")]]
    p.columns = [f"{t} {h}" for t, h in p.columns]
    return p.to_markdown()


out = []
for col, name, fmt in [("r2", "log-damage R^2", ".3f"),
                       ("within2", "fraction within factor 2", ".2f"),
                       ("median_ratio", "median ratio Dhat/D", ".2f"),
                       ("rho_wc", "rho_wc", ".2f")]:
    out.append(f"#### {name}\n\n" + piv(col, fmt) + "\n")
M = ORDER[4:]
out.append("#### A2: within-condition calibration slope [95% cluster CI]\n\n"
           + piv("slope", ".2f", M, ci=True) + "\n")
out.append("#### A2b: amplitude ratio sd(dhat)/sd(d)\n\n"
           + piv("amp_ratio", ".2f", M) + "\n")
out.append("#### A3: lifetime-weighted damage ratio [95% cluster CI]\n\n"
           + piv("lifetime_ratio", ".2f", M, ci=True) + "\n")
open("tables_a123.md", "w").write("\n".join(out))
print("\n".join(out))
