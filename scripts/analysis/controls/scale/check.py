import os
import sys, pandas as pd, numpy as np
src, tgt, model, seed = sys.argv[1:5]
R = os.path.join(os.environ.get("FLOATSENSE_ROOT", "."), "outputs")
new = pd.read_csv(f"{R}/controls/a4_scale/infer/{src}/seed{seed}/damage_comparison_{model}_fa_check_{tgt}.csv").set_index("sim_id")
old = pd.read_csv(f"{R}/heights/{src}/seed{seed}/damage_comparison_{model}_fa_zs_{tgt}.csv").set_index("sim_id").loc[new.index]
cols = [c for c in new.columns if c.startswith("damage_")]
rel = (new[cols] / old[cols] - 1).abs()
print(src, tgt, model, seed, "max|rel| true %.2e rec %.2e" % (rel[[c for c in cols if "true" in c]].values.max(), rel[[c for c in cols if "rec" in c]].values.max()))
