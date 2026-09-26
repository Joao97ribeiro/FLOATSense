"""Signed waveform metrics, within tower, on 200 fixed test sims per tower.

Checkpoint <runs>/<tower>/seed<k>/<model>_fa.pt, the unchanged
trainer data path (400-1000 s window, source normalization, gravity-corrected
inputs). Both series denormalized (kN.m, zero-mean target) and low-passed at
3 Hz (as the damage score). Per sim and height: Pearson r(pred, true) and
NRMSE = RMS(pred - true) / std(true). Variants: tcn; prob_tcn_mean (mean
head, no sampling); prob_tcn_sample (mean + sigma * N(0,1), torch seed = seed).
Heights: bottom (z/H 0), tower_5 (0.48), tower_8 (0.78), top (1.0).
Check: variance ratio var(pred)/var(true) vs the stored var_ratio column
(tcn and prob_tcn draw columns of damage_comparison_<m>_fa.csv).
The sample is drawn once (seed 0) from the test split and written to
<controls>/a5_waveform/sample_ids_<tower>.csv; num_sims < 200 scores only
its first num_sims simulations (a quick check). Runs on the GPU when one is
visible (CUDA_VISIBLE_DEVICES="" for the CPU). Paths: see layout.py.

Usage: python scripts/analysis/controls/waveform/waveform.py <tower> <seed> [models] [num_sims]
  models: comma-separated, default tcn,prob_tcn; num_sims: default 200
"""
import os, sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import layout  # noqa: E402  pylint: disable=wrong-import-position
from floatsense import SequenceModelTrainer, load_tower  # noqa: E402  pylint: disable=wrong-import-position
from floatsense.heights import calibrate_profile  # noqa: E402  pylint: disable=wrong-import-position
from floatsense.physics import lowpass  # noqa: E402  pylint: disable=wrong-import-position
from floatsense.data import HEIGHT_TARGETS  # noqa: E402  pylint: disable=wrong-import-position

tower, seed = sys.argv[1], int(sys.argv[2])
NUM_SIMS = int(sys.argv[4]) if len(sys.argv) > 4 else 200
release = load_tower(layout.DATA, tower)
OUT = layout.controls("a5_waveform")
ids = sorted(np.random.default_rng(0).choice(release.split_ids("test"), 200, replace=False).tolist())
pd.Series(ids, name="sim_id").to_csv(f"{OUT}/sample_ids_{tower}.csv", index=False)
ids = ids[:NUM_SIMS]
SECTIONS = [0, 5, 8, 10]
rows = []
MODELS = sys.argv[3].split(",") if len(sys.argv) > 3 else ["tcn", "prob_tcn"]
for model in MODELS:
    tr = SequenceModelTrainer(
        release=release, output_dir=layout.run_dir(tower, seed),
        direction="fa", model_name=model,
        min_time=400.0, max_time=1000.0, lowpass_hz=3.0, crop_length=4096, batch_size=16,
        learning_rate=1e-3, num_epochs=50, val_every=0, early_stopping_patience=0,
        num_workers=4, sn_intercepts_log10=[12.010, 15.350], sn_slopes=[3.0, 5.0],
        loss_name="mse", damage_loss_weight=1.0, init_checkpoint=None, calibration_path=None,
        condition_bound=0.5, target_channel=None, damage_section=0, input_channels=None,
        height_targets=True, height_factors=calibrate_profile(layout.DATA, tower)["factors"], seed=seed)
    tr.load_checkpoint(); tr.model.eval()
    ds = tr._make_dataset(ids, None)
    torch.manual_seed(seed)
    fs = ds.sampling_frequency
    with torch.no_grad():
        for i in range(len(ds)):
            for s in SECTIONS:
                ds.section = s
                item = ds[i]
                x = item["inputs"][None].to(tr.device); c = item["condition"][None].to(tr.device)
                out = tr.model(x, c)[0]
                variants = {}
                if getattr(tr.model, "predicts_variance", False):
                    variants["prob_tcn_mean"] = out[0].cpu().numpy()
                    sig = torch.exp(0.5 * out[1])
                    variants["prob_tcn_sample"] = (out[0] + sig * torch.randn_like(sig)).cpu().numpy()
                else:
                    variants[model] = out[0].cpu().numpy()
                true = lowpass(ds.denormalize_target(item["target"][0].numpy()), fs, 3.0)
                for v, p in variants.items():
                    p = lowpass(ds.denormalize_target(p), fs, 3.0)
                    rows.append(dict(tower=tower, seed=seed, variant=v, sim_id=ds.sim_ids[i],
                                     height=HEIGHT_TARGETS[s][0], z_h=HEIGHT_TARGETS[s][2],
                                     pearson=float(np.corrcoef(p, true)[0, 1]),
                                     nrmse=float(np.sqrt(np.mean((p - true) ** 2)) / np.std(true)),
                                     var_ratio=float(np.var(p) / np.var(true))))
df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/waveform_{tower}_seed{seed}.csv", index=False)
for m, v in [(m, m) for m in MODELS if m != "prob_tcn"] + ([("prob_tcn", "prob_tcn_sample")] if "prob_tcn" in MODELS else []):
    st = pd.read_csv(layout.damage_csv(tower, seed, m)).set_index("sim_id")
    d = df[df.variant == v]
    for h in d.height.unique():
        e = d[d.height == h].set_index("sim_id")
        rel = (e.var_ratio / st.loc[e.index, f"var_ratio_{h}"] - 1).abs()
        print(f"check {tower} s{seed} {v} {h}: median|rel| var_ratio {rel.median():.2e} max {rel.max():.2e}")
print(df.groupby(["variant", "height"])[["pearson", "nrmse", "var_ratio"]].median().round(3))
