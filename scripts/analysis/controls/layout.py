"""Where the control scripts read the dataset and the runs, and write.

Environment variables (relative values are taken from FLOATSENSE_ROOT):

  FLOATSENSE_ROOT      repository root (default: current directory)
  FLOATSENSE_DATA      released dataset (default: data/FLOATSense)
  FLOATSENSE_RUNS      within-tower runs, <runs>/<tower>/seed<k>/
                       (default: outputs/within, scripts/train/config.cfg)
  FLOATSENSE_OUTPUTS   root of the other experiments: ablation/, toponly/,
                       physics/, noise/ (default: outputs)
  FLOATSENSE_CONTROLS  where the controls write (default: outputs/controls)
  FLOATSENSE_LAYOUT    'release' (default: the folders scripts/train/run.py
                       and scripts/physics/run.py write, see the README) or
                       'paper' (the folders of the paper's research runs:
                       set FLOATSENSE_RUNS=<outputs>/heights with it)
"""

import os
import sys

ROOT = os.path.abspath(os.environ.get("FLOATSENSE_ROOT", "."))
sys.path.insert(0, ROOT)


def _path(variable: str, default: str) -> str:
    return os.path.join(ROOT, os.environ.get(variable, default))


DATA = _path("FLOATSENSE_DATA", "data/FLOATSense")
RUNS = _path("FLOATSENSE_RUNS", "outputs/within")
OUTPUTS = _path("FLOATSENSE_OUTPUTS", "outputs")
CONTROLS = _path("FLOATSENSE_CONTROLS", "outputs/controls")
LAYOUT = os.environ.get("FLOATSENSE_LAYOUT", "release")
if LAYOUT not in ("release", "paper"):
    raise ValueError(f"FLOATSENSE_LAYOUT must be release or paper: {LAYOUT}")

TOWERS = ("ref", "opt1", "opt2")

# Folder of one run of the other experiments (seed 0), relative to OUTPUTS.
_FOLDERS = {
    "release": {
        "twoaxis": "ablation/twoaxis/{t}/seed0",
        "scada": "ablation/scada/{t}/seed0",
        "toponly": "toponly/{t}/seed0",
        "noise": "noise/{t}",
        "physics": "physics/{t}",
    },
    "paper": {
        "twoaxis": "heights/ablation/twoaxis/{t}",
        "scada": "heights/ablation/scada/{t}",
        "toponly": "diag/toponly/{t}",
        "noise": "heights/noise/{t}",
        "physics": "physics_heights/{t}_E2",
    },
}


def run_dir(tower: str, seed: int) -> str:
    """Folder of a within-tower run: checkpoints and damage CSVs."""
    return os.path.join(RUNS, tower, f"seed{seed}")


def damage_csv(tower: str, seed: int, model: str, tag: str = "") -> str:
    """Per-simulation damage of a within-tower run (tag: e.g. 'zs_ref')."""
    return os.path.join(run_dir(tower, seed), f"damage_comparison_{model}_fa"
                        + (f"_{tag}" if tag else "") + ".csv")


def experiment_dir(kind: str, tower: str) -> str:
    """Folder of a seed-0 run of another experiment (see _FOLDERS)."""
    return os.path.join(OUTPUTS, _FOLDERS[LAYOUT][kind].format(t=tower))


def controls(*parts: str) -> str:
    """Path under the controls output folder (the folder is created)."""
    path = os.path.join(CONTROLS, *parts)
    os.makedirs(os.path.dirname(path) if os.path.splitext(path)[1] else path,
                exist_ok=True)
    return path
