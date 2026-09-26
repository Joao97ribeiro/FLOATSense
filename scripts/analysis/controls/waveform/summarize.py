"""Summary of the waveform metrics of every tower and seed (waveform.py).

Reads <controls>/a5_waveform/waveform_<tower>_seed<k>.csv and writes
waveform_all.csv and waveform_summary.csv next to them. Paths: see layout.py.

Usage: python scripts/analysis/controls/waveform/summarize.py
"""
import glob
import os
import sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import layout  # noqa: E402  pylint: disable=wrong-import-position
A = layout.controls("a5_waveform")
d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{A}/waveform_*_seed*.csv"))])
d.to_csv(f"{A}/waveform_all.csv", index=False)
# per tower-seed median over sims, then median over seeds
g = d.groupby(["variant", "height", "tower", "seed"])[["pearson", "nrmse", "var_ratio"]].median().groupby(["variant", "height", "tower"]).median()
t = g.unstack("tower")
order = ["tower_bottom", "tower_5", "tower_8", "tower_top"]
t = t.reindex(order, level="height")
t.to_csv(f"{A}/waveform_summary.csv")
pd.set_option("display.width", 250)
print(t.round(3).to_string())
print(d.groupby("variant").seed.unique())
# fraction of sims with pearson > 0.9 at top
print(d[d.height=="tower_top"].groupby(["variant","tower"]).pearson.apply(lambda x:(x>0.9).mean()).unstack().round(2))
