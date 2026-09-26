"""Re-infer source-tower checkpoints on target-tower sims (no training).

Builds SequenceModelTrainer as scripts/train/run.py does for the height task
(source tower, --height_targets, config.cfg defaults), loads
<runs>/<src>/seed<k>/<model>_fa.pt, and calls the unchanged
trainer.evaluate() on the target tower (source normalization stats,
400-1000 s window, 3 Hz low-pass, rainflow + bilinear SN at the sections of
the target tower). Writes damage_comparison_<model>_fa_<which>_<tgt>.csv to
<controls>/a4_scale/infer/<src>/seed<k>; the run folder is never written.
Runs on the GPU when one is visible (CUDA_VISIBLE_DEVICES="" for the CPU).
Paths: see layout.py.

Usage: python scripts/analysis/controls/scale/infer_designed.py <src> <tgt> <model> <seed> <which>
  which = designed (the 18 sims of fewshot/train_designed)
        | rand5 (fewshot/train_5_draw0)
        | check (first 3 test sims; compare with check.py)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import layout  # noqa: E402  pylint: disable=wrong-import-position
from floatsense import SequenceModelTrainer  # noqa: E402  pylint: disable=wrong-import-position
from floatsense import load_tower  # noqa: E402  pylint: disable=wrong-import-position
from floatsense.heights import calibrate_profile  # noqa: E402  pylint: disable=wrong-import-position

SPLITS = {"designed": "fewshot/train_designed",
          "rand5": "fewshot/train_5_draw0"}

src, tgt, model, seed, which = sys.argv[1:6]
seed = int(seed)
out = layout.controls("a4_scale", "infer", src, f"seed{seed}")
source = load_tower(layout.DATA, src)
trainer = SequenceModelTrainer(
    release=source,
    output_dir=layout.run_dir(src, seed),
    direction="fa", model_name=model, min_time=400.0, max_time=1000.0,
    lowpass_hz=3.0, crop_length=4096, batch_size=16, learning_rate=1e-3,
    num_epochs=50, val_every=0, early_stopping_patience=0, num_workers=4,
    sn_intercepts_log10=[12.010, 15.350], sn_slopes=[3.0, 5.0],
    loss_name="mse", damage_loss_weight=1.0, init_checkpoint=None,
    calibration_path=None, condition_bound=0.5, target_channel=None,
    damage_section=0, input_channels=None, height_targets=True,
    height_factors=calibrate_profile(layout.DATA, src)["factors"], seed=seed)
trainer.load_checkpoint()
trainer.output_dir = out  # never write into the original folders
target = load_tower(layout.DATA, tgt)
if which in SPLITS:
    ids = target.split_ids(SPLITS[which])
elif which == "check":
    ids = target.split_ids("test")[:3]
else:
    raise ValueError(f"which must be designed, rand5 or check: {which}")
# evaluate() seeds the variance-head noise with the trainer seed itself.
print(trainer.evaluate(ids, release=target, tag=f"{which}_{tgt}"))
