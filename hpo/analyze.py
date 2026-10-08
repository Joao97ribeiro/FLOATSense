# pylint: disable=wrong-import-position
# pylint: disable=too-many-locals
"""Leaderboard of the validation-tuned track.

Scores the sealed test outputs of phase 3 (hpo/final.py) with the scorer of
the benchmark (scripts/benchmark/run.py: R^2 of log10 damage, median damage
ratio, fraction within a factor of two with cluster-bootstrap intervals,
mean relative error and within-condition correlation, per gauge and regime
cell) and builds the leaderboard of the track, for the headline variant
(last epoch) and the secondary one (median best validation epoch):

  scores.csv          per model, tower, seed, gauge and group (the
                      benchmark table of the sealed runs)
  leaderboard.csv     per model: each metric at the base, z/H 0.78, the top
                      and the mean of the 11 gauges, median over the seeds
                      per tower, then mean over the towers (group 'all')
  per_tower.csv       the same before the mean over towers
  by_group.csv        the leaderboard per regime cell
  top3.csv            the three best models per criterion (R^2 at the base,
                      mean of 11, top)
  families.csv        the best model of each family per criterion

and, once for the track, configs.csv (the selected configuration of each
model and tower, from phase 2) and curves.csv (best validation score so far
against the trial number, per study).

    python hpo/analyze.py --root=outputs/hpo --dataset_dir=data/FLOATSense
    python hpo/analyze.py --dry_run --out=/tmp/leaderboard   # synthetic
"""

import argparse
import glob
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hpo import common
from hpo import confirm
from hpo import constants as C
from hpo import search
from hpo import search_space as S

BENCHMARK = os.path.join(common.REPO, "scripts", "benchmark", "run.py")
GAUGES = ["tower_bottom"] + [f"tower_{i}" for i in range(1, 10)] + ["tower_top"]
# Reported positions: the base, z/H 0.78 (tower_8), the top, mean of 11.
POSITIONS = {"base": "tower_bottom", "z078": "tower_8", "top": "tower_top"}
METRICS = ("r2_log_damage", "fraction_within_factor2", "median_damage_ratio",
           "mean_relative_error", "within_condition_correlation")
CRITERIA = ("base", "mean11", "top")  # R^2 of log10 damage
VARIANTS = ("last", "best")
FLOOR = "naive"


def score(sealed_root: str, dataset_dir: str, out: str,
          num_resamples: int) -> pd.DataFrame:
    """Runs the benchmark scorer on a sealed tree; returns the 'fa' rows
    of the learned models (and the floor), with tower and seed columns."""
    path = os.path.join(out, "benchmark.csv")
    subprocess.run([
        sys.executable, BENCHMARK, f"--dataset_dir={dataset_dir}",
        f"--output_root={sealed_root}", f"--results_csv={path}",
        f"--num_resamples={num_resamples}"
    ],
                   check=True,
                   cwd=common.REPO)
    table = pd.read_csv(path)
    table = table[(table["direction"] == "fa") &
                  table["model"].isin(list(S.LEARNED) + [FLOOR])].copy()
    parts = table["run"].str.split("/", expand=True)
    table["tower"] = parts[0]
    table["seed"] = parts[1].str.replace("seed", "").astype(int)
    return table


def per_tower(table: pd.DataFrame) -> pd.DataFrame:
    """Median over the seeds per model, tower, group and position."""
    mean11 = table.groupby(["model", "tower", "seed", "group"],
                           as_index=False)[list(METRICS)].mean()
    mean11["position"] = "mean11"
    rows = [mean11]
    for position, gauge in POSITIONS.items():
        rows.append(
            table.loc[table["gauge"] == gauge,
                      ["model", "tower", "seed", "group", *METRICS]].assign(
                          position=position))
    long = pd.concat(rows, ignore_index=True)
    return long.groupby(["model", "tower", "group", "position"],
                        as_index=False)[list(METRICS)].median()


def leaderboard(towers: pd.DataFrame) -> pd.DataFrame:
    """Mean over the towers of the per-tower medians, one row per model and
    group, one column per metric and position."""
    mean = towers.groupby(["model", "group", "position"],
                          as_index=False)[list(METRICS)].mean()
    wide = mean.pivot(index=["model", "group"], columns="position")
    wide.columns = [f"{metric}_{position}" for metric, position in wide.columns]
    wide = wide.reset_index()
    wide["family"] = wide["model"].map(S.FAMILIES).fillna("floor")
    return wide.sort_values("r2_log_damage_mean11",
                            ascending=False).reset_index(drop=True)


def rankings(board: pd.DataFrame) -> tuple:
    """Top-3 per criterion and best model of each family per criterion."""
    learned = board[board["model"].isin(S.LEARNED)]
    top3, families = [], []
    for criterion in CRITERIA:
        column = f"r2_log_damage_{criterion}"
        ranked = learned.sort_values(column, ascending=False)
        for rank, (_, row) in enumerate(ranked.head(3).iterrows(), 1):
            top3.append({
                "criterion": criterion,
                "rank": rank,
                "model": row["model"],
                "r2": row[column]
            })
        for family, group in ranked.groupby("family", sort=False):
            families.append({
                "criterion": criterion,
                "family": family,
                "model": group.iloc[0]["model"],
                "r2": group.iloc[0][column]
            })
    return pd.DataFrame(top3), pd.DataFrame(families)


def configs(root: str) -> pd.DataFrame:
    """Selected configuration of every model and tower (phase 2)."""
    rows = []
    for model in S.LEARNED:
        for tower in C.TOWERS_SEARCHED:
            record = common.read_json(confirm.winner_path(root, model, tower))
            if record is None:
                continue
            rows.append({
                "model": model,
                "tower": tower,
                "config": json.dumps(record["winner_config"]),
                "val_median": record["winner_median"],
                "margin_to_second": record["margin_to_second"],
                "best_epoch": record["best_epoch"]
            })
    return pd.DataFrame(rows)


def curves(root: str) -> pd.DataFrame:
    """Best validation score so far against the trial number, per study."""
    rows = []
    for model in S.LEARNED:
        for tower in C.TOWERS_SEARCHED:
            study = search.open_study(root, model, tower, create=False)
            if study is None:
                continue
            best = -np.inf
            for trial in sorted(study.get_trials(deepcopy=False),
                                key=lambda t: t.number):
                if search.eligible(trial):
                    best = max(best, trial.value)
                rows.append({
                    "model": model,
                    "tower": tower,
                    "trial": trial.number,
                    "state": trial.state.name,
                    "value": trial.value,
                    "best_so_far": best if np.isfinite(best) else None
                })
    return pd.DataFrame(rows)


def build(args: argparse.Namespace) -> Dict[str, pd.DataFrame]:
    """Writes the leaderboard files of every sealed variant."""
    os.makedirs(args.out, exist_ok=True)
    boards = {}
    for variant in VARIANTS:
        sealed = os.path.join(args.root, "sealed", variant)
        if not glob.glob(os.path.join(sealed, "*", "seed*", "*.csv")):
            print(f"no sealed '{variant}' runs under {sealed}")
            continue
        out = os.path.join(args.out, variant)
        os.makedirs(out, exist_ok=True)
        table = score(sealed, args.dataset_dir, out, args.num_resamples)
        table.to_csv(os.path.join(out, "scores.csv"), index=False)
        towers = per_tower(table)
        towers[towers["group"] == "all"].to_csv(os.path.join(
            out, "per_tower.csv"),
                                                index=False)
        board = leaderboard(towers)
        board[board["group"] != "all"].to_csv(os.path.join(out, "by_group.csv"),
                                              index=False)
        board = board[board["group"] == "all"].drop(columns="group")
        board.to_csv(os.path.join(out, "leaderboard.csv"), index=False)
        top3, families = rankings(board)
        top3.to_csv(os.path.join(out, "top3.csv"), index=False)
        families.to_csv(os.path.join(out, "families.csv"), index=False)
        boards[variant] = board
        print(f"== {variant}: R^2 of log10 damage (median over seeds, mean "
              "over towers)")
        print(
            board[["model", "family"] +
                  [f"r2_log_damage_{c}" for c in CRITERIA]].round(3).to_string(
                      index=False))
    configs(args.root).to_csv(os.path.join(args.out, "configs.csv"),
                              index=False)
    curves(args.root).to_csv(os.path.join(args.out, "curves.csv"), index=False)
    return boards


def write_synthetic(root: str,
                    rng: np.random.Generator,
                    n_points: int = 20) -> str:
    """A synthetic dataset (labels and a one-sample series shard per tower)
    and sealed runs of every learned model, for a dry run. Returns the
    dataset directory."""
    sims = np.arange(1, 6 * n_points + 1)
    point = (sims - 1) // 6
    dataset_dir = os.path.join(root, "data")
    heights = np.linspace(1.0, 150.0, len(GAUGES))
    quality = dict(zip(S.LEARNED, np.linspace(0.05, 0.6, len(S.LEARNED))))
    quality[FLOOR] = 0.8
    for tower in C.TOWERS_SEARCHED:
        folder = os.path.join(dataset_dir, tower)
        os.makedirs(folder, exist_ok=True)
        pd.DataFrame({
            "sim_id": sims,
            "split": "test",
            "wind_speed": 4.0 + point % 5,
            "wave_hs": 1.0 + point // 5,
            "wave_tp": 8.0,
            "wind_group": np.where(point % 5 < 3, "In-train", "Extrapolate"),
            "wave_group": "In-train"
        }).to_parquet(os.path.join(folder, "metadata.parquet"))
        pd.DataFrame({
            "section_id": np.arange(1, 3 * len(GAUGES), 3),
            "channel": GAUGES,
            "gauge_height_m": heights,
            "z_over_h": heights / heights[-1],
            "gauge_radius_m": np.linspace(5.0, 3.0, len(GAUGES)),
            "gauge_thickness_m": np.linspace(0.066, 0.038, len(GAUGES))
        }).to_parquet(os.path.join(folder, "sections.parquet"))
        with pq.ParquetWriter(
                os.path.join(folder, "series-00000-of-00001.parquet"),
                pa.schema([("sim_id", pa.int64()), ("time_s", pa.float32()),
                           ("tower_top_afa_mod", pa.float32())])) as writer:
            for sim in sims:
                writer.write_table(
                    pa.table({
                        "sim_id": np.array([sim]),
                        "time_s": np.zeros(1, np.float32),
                        "tower_top_afa_mod": np.zeros(1, np.float32)
                    }))
        level = rng.normal(0, 1, n_points)[point]
        true = {
            g: 10**(level + 0.3 * rng.normal(0, 1, len(sims)) - 6 + 0.1 * i)
            for i, g in enumerate(GAUGES)
        }
        for variant in VARIANTS:
            for seed in range(C.N_SEEDS):
                runs = os.path.join(root, "sealed", variant, tower,
                                    f"seed{seed}")
                os.makedirs(runs, exist_ok=True)
                for model, noise in quality.items():
                    frame = {"sim_id": sims}
                    for gauge in GAUGES:
                        frame[f"damage_true_{gauge}"] = true[gauge]
                        frame[f"damage_rec_{gauge}"] = true[gauge] * 10**(
                            rng.normal(0, noise, len(sims)))
                    pd.DataFrame(frame).to_csv(os.path.join(
                        runs, f"damage_comparison_{model}_fa.csv"),
                                               index=False)
    for model in S.LEARNED:
        for tower in C.TOWERS_SEARCHED:
            common.write_json(
                confirm.winner_path(root, model, tower), {
                    "winner_config": S.FIXED_RECIPE,
                    "winner_median": 0.5,
                    "margin_to_second": 0.01,
                    "best_epoch": 250
                })
    return dataset_dir


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root",
                        default="outputs/hpo",
                        help="Root of the track (sealed runs, phase 2, "
                        "studies).")
    parser.add_argument("--dataset_dir", default="data/FLOATSense")
    parser.add_argument("--out", default="outputs/hpo/leaderboard")
    parser.add_argument("--num_resamples",
                        type=int,
                        default=1000,
                        help="Bootstrap resamples of the benchmark scorer.")
    parser.add_argument("--dry_run",
                        action="store_true",
                        help="Synthetic sealed runs instead of --root.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    """Builds the leaderboard (of synthetic runs with --dry_run)."""
    args = parse_args(argv)
    if args.dry_run:
        args.root = os.path.join(args.out, "synthetic")
        args.dataset_dir = write_synthetic(args.root, np.random.default_rng(0))
    return build(args)


if __name__ == "__main__":
    main()
