"""Re-infer source-tower checkpoints on target-tower sims (no training).

Builds SequenceModelTrainer exactly as scripts/train/run.py does for the
height task (source tower, --height_targets, config.cfg defaults), loads
outputs/heights/<src>/seed<k>/<model>_fa.pt, and calls the unchanged
trainer.evaluate() on the target tower's time series (source normalization
stats, 400-1000 s window, 3 Hz low-pass, rainflow + bilinear SN of the
target tower). Writes damage_comparison_<model>_fa_<tag>.csv here.

Usage: python infer_designed.py <src> <tgt> <model> <seed> <which>
  which = designed (18 sims of fewshot/train_designed) | rand5 (fewshot/train_5_draw0) | check (first 3 test sims)
"""
import os
import sys

ROOT = os.environ.get("FLOATSENSE_ROOT", ".")
sys.path.insert(0, ROOT)
sys.path.insert(0, f"{ROOT}/scripts/train")
os.chdir(ROOT)

from floatsense.heights import calibrate_profile  # noqa: E402
from floatsense.splits import read_split, split_ids  # noqa: E402
from floatsense.tower import Tower  # noqa: E402
from floatsense.trainer import SequenceModelTrainer  # noqa: E402

src, tgt, model, seed, which = sys.argv[1:6]
seed = int(seed)
out = f"{ROOT}/outputs/controls/a4_scale/infer/{src}/seed{seed}"
os.makedirs(out, exist_ok=True)
source = f"data/{src}"
trainer = SequenceModelTrainer(
    packed_dir=os.path.join(source, "timeseries"),
    output_dir=f"outputs/heights/{src}/seed{seed}",
    tower=Tower(json_path=os.path.join(source, "tower.json")),
    direction="fa", model_name=model, min_time=400.0, max_time=1000.0,
    lowpass_hz=3.0, crop_length=4096, batch_size=16, learning_rate=1e-3,
    num_epochs=50, val_every=0, early_stopping_patience=0, num_workers=4,
    sn_intercepts_log10=[12.010, 15.350], sn_slopes=[3.0, 5.0],
    loss_name="mse", damage_loss_weight=1.0, init_checkpoint=None,
    calibration_path=None, condition_bound=0.5, target_channel=None,
    damage_section=0, input_channels=None, height_targets=True,
    height_factors=calibrate_profile(source)["factors"], seed=seed)
trainer.load_checkpoint()
trainer.output_dir = out  # never write into the original folders
tdir = f"data/{tgt}"
if which == "designed":
    ids = [int(i) for i in read_split(tdir, "fewshot/train_designed").sim_id]
elif which == "rand5":
    ids = [int(i) for i in read_split(tdir, "fewshot/train_5_draw0").sim_id]
else:
    ids = split_ids(tdir, "E2/test")[:3]
import torch  # noqa: E402
torch.manual_seed(seed)
print(trainer.evaluate(ids, packed_dir=os.path.join(tdir, "timeseries"),
                       tower=Tower(json_path=os.path.join(tdir, "tower.json")),
                       tag=f"{which}_{tgt}"))
