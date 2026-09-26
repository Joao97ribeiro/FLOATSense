import os
import glob, pandas as pd
d = pd.concat([pd.read_csv(f) for f in glob.glob(os.environ.get("FLOATSENSE_ROOT", ".") + "/outputs/controls/a5_waveform/waveform_*_seed*.csv")])
d.to_csv(os.environ.get("FLOATSENSE_ROOT", ".") + "/outputs/controls/a5_waveform/waveform_all.csv", index=False)
# per tower-seed median over sims, then median over seeds
g = d.groupby(["variant", "height", "tower", "seed"])[["pearson", "nrmse", "var_ratio"]].median().groupby(["variant", "height", "tower"]).median()
t = g.unstack("tower")
order = ["tower_bottom", "tower_5", "tower_8", "tower_top"]
t = t.reindex(order, level="height")
t.to_csv(os.environ.get("FLOATSENSE_ROOT", ".") + "/outputs/controls/a5_waveform/waveform_summary.csv")
pd.set_option("display.width", 250)
print(t.round(3).to_string())
print(d.groupby("variant").seed.unique())
# fraction of sims with pearson > 0.9 at top
print(d[d.height=="tower_top"].groupby(["variant","tower"]).pearson.apply(lambda x:(x>0.9).mean()).unstack().round(2))
