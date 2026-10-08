# pylint: disable=wrong-import-position
# pylint: disable=use-dict-literal
# pylint: disable=too-many-locals
"""Leaderboard of the validation-tuned track.

Scores the sealed test outputs of phase 3 (hpo/final.py; only the runs with
a SEALED_<model>.json marker that is not 'skipped') with the scorer of
the benchmark (scripts/benchmark/run.py: R^2 of log10 damage, median damage
ratio, fraction within a factor of two with cluster-bootstrap intervals,
mean relative error and within-condition correlation, per gauge and regime
cell) and builds the leaderboard of the track for each test variant, in
its own folder: primary_last/ (PRIMARY, the headline: the last epoch) and
secondary_best/ (SECONDARY: the median best validation epoch of phase 2):

  scores.csv          per model, tower, seed, gauge and group (the
                      benchmark table of the sealed runs)
  leaderboard.csv     per model: each metric at the base, z/H 0.78, the top
                      and the mean of the 11 gauges, median over the seeds
                      per tower, then mean over the scored towers (group
                      'all'); n_towers counts the towers scored with all
                      N_SEEDS final seeds, missing_towers names the others
                      (a shelved (model, tower) has no winner, a pair whose
                      final seeds all diverged has no test score, and a
                      tower with a diverged seed has fewer seeds: its score
                      stays visible), n_seeds_<tower> the seeds scored per
                      tower. Only the models with every tower complete are
                      ranked (ranked = True, first, by the mean of 11);
                      the others follow, unranked, by name (a model with no
                      scored tower has an empty row)
  per_tower.csv       the same before the mean over towers (with n_seeds)
  by_group.csv        the leaderboard per regime cell
  top3.csv            the three best ranked models per criterion (R^2 at
                      the base, mean of 11, top; ties broken by the model
                      name), with n_towers
  families.csv        the best ranked model of each family per criterion,
                      with n_towers

and, once for the track:

  configs.csv         the selected configuration of each model and tower
                      (phase 2) with its status: 'winner', 'shelved' (no
                      configuration) or 'all_seeds_diverged' (every final
                      seed diverged: no test score)
  curves.csv          best validation score so far against the counted
                      position of each counted trial (1..N), per study
  gpu_hours.csv       GPU-hours per model, tower and phase (search,
                      confirm, final), summed over every attempt of every
                      run (pruned, crashed and preempted ones included, and
                      hard-killed ones from their last heartbeat: 'killed'
                      counts them; from the attempts.json of the runs), and
                      the test inference runs (phase 'test'), with the
                      totals per model ('all' tower and phase) and overall
                      ('all' model)

The tuned leaderboard ranks only the 20 learned models: the naive floor
and the physics baseline are not tuned, and their scores are those of the
fixed-recipe benchmark results (scripts/benchmark/run.py).

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
from hpo import final
from hpo import search
from hpo import search_space as S

BENCHMARK = os.path.join(common.REPO, "scripts", "benchmark", "run.py")
GAUGES = ["tower_bottom"] + [f"tower_{i}" for i in range(1, 10)] + ["tower_top"]
# Reported positions: the base, z/H 0.78 (tower_8), the top, mean of 11.
POSITIONS = {"base": "tower_bottom", "z078": "tower_8", "top": "tower_top"}
METRICS = ("r2_log_damage", "fraction_within_factor2", "median_damage_ratio",
           "mean_relative_error", "within_condition_correlation")
CRITERIA = ("base", "mean11", "top")  # R^2 of log10 damage
VARIANTS = final.VARIANTS  # primary first
PHASE_DIRS = {
    "search": "trials",
    "confirm": "phase2",
    "final": "final",
    "test": "test_runs"
}


def score(sealed_root: str, dataset_dir: str, out: str,
          num_resamples: int) -> pd.DataFrame:
    """Runs the benchmark scorer on a sealed tree; returns the 'fa' rows
    of the learned models, with tower and seed columns."""
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
                  table["model"].isin(S.LEARNED)].copy()
    markers = {
        key:
            common.read_json(
                os.path.join(sealed_root, key[1], f"SEALED_{key[0]}.json"))
        for key in set(zip(table["model"], table["run"]))
    }
    table = table[[
        bool(markers[key]) and not markers[key].get("skipped")
        for key in zip(table["model"], table["run"])
    ]].copy()
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
    keys = ["model", "tower", "group", "position"]
    out = long.groupby(keys, as_index=False)[list(METRICS)].median()
    seeds = long.groupby(keys, as_index=False)["seed"].nunique()
    return out.merge(seeds.rename(columns={"seed": "n_seeds"}), on=keys)


def leaderboard(towers: pd.DataFrame) -> pd.DataFrame:
    """Mean over the scored towers of the per-tower medians, one row per
    model and group, one column per metric and position; n_towers counts
    the towers with all N_SEEDS seeds and missing_towers names the others
    (scored with fewer seeds, or not at all)."""
    mean = towers.groupby(["model", "group", "position"],
                          as_index=False)[list(METRICS)].mean()
    wide = mean.pivot(index=["model", "group"], columns="position")
    wide.columns = [f"{metric}_{position}" for metric, position in wide.columns]
    wide = wide.reset_index()
    wide["family"] = wide["model"].map(S.FAMILIES)
    seeds = towers.groupby(["model", "tower"])["n_seeds"].max()
    complete = {
        m: [t for t in C.TOWERS_SEARCHED if seeds.get((m, t), 0) == C.N_SEEDS]
        for m in wide["model"]
    }
    wide["n_towers"] = wide["model"].map(lambda m: len(complete[m]))
    wide["missing_towers"] = wide["model"].map(lambda m: ",".join(
        t for t in C.TOWERS_SEARCHED if t not in complete[m]))
    for tower in C.TOWERS_SEARCHED:
        wide[f"n_seeds_{tower}"] = wide["model"].map(
            lambda m, t=tower: int(seeds.get((m, t), 0)))
    return ordered(wide)


def ordered(board: pd.DataFrame) -> pd.DataFrame:
    """Ranked models (every tower scored with N_SEEDS seeds) first, by the
    mean of 11 (ties by name); then the unranked ones by name."""
    board = board.assign(ranked=board["n_towers"] == len(C.TOWERS_SEARCHED))
    ranked = board[board["ranked"]].sort_values(
        ["r2_log_damage_mean11", "model"],
        ascending=[False, True],
        kind="mergesort")
    rest = board[~board["ranked"]].sort_values("model", kind="mergesort")
    return pd.concat([ranked, rest], ignore_index=True)


def with_missing(board: pd.DataFrame) -> pd.DataFrame:
    """Adds an empty row for each learned model without any scored tower
    (every tower parked)."""
    absent = [m for m in S.LEARNED if m not in set(board["model"])]
    if not absent:
        return board
    rows = pd.DataFrame({
        "model": absent,
        "family": [S.FAMILIES[m] for m in absent],
        "n_towers": 0,
        "missing_towers": ",".join(C.TOWERS_SEARCHED),
        "ranked": False
    })
    return ordered(pd.concat([board, rows], ignore_index=True))


def rankings(board: pd.DataFrame) -> tuple:
    """Top-3 per criterion and best model of each family per criterion,
    among the ranked models."""
    learned = board[board["model"].isin(S.LEARNED) & board["ranked"]]
    top3, families = [], []
    for criterion in CRITERIA:
        column = f"r2_log_damage_{criterion}"
        # Stable, ties broken by the model name.
        ranked = learned.dropna(subset=[column]).sort_values(
            [column, "model"], ascending=[False, True], kind="mergesort")
        for rank, (_, row) in enumerate(ranked.head(3).iterrows(), 1):
            top3.append({
                "criterion": criterion,
                "rank": rank,
                "model": row["model"],
                "r2": row[column],
                "n_towers": row["n_towers"]
            })
        for family, group in ranked.groupby("family", sort=False):
            families.append({
                "criterion": criterion,
                "family": family,
                "model": group.iloc[0]["model"],
                "r2": group.iloc[0][column],
                "n_towers": group.iloc[0]["n_towers"]
            })
    return pd.DataFrame(top3), pd.DataFrame(families)


def configs(root: str) -> pd.DataFrame:
    """Selected configuration of every model and tower (phase 2); a shelved
    (model, tower) (parked marker, checked first: a pair can be shelved
    after its winner) has a row with status 'shelved' and no configuration;
    a winner whose final seeds all diverged has status
    'all_seeds_diverged'."""
    rows = []
    diverged = set(final.skipped(root)["all_seeds_diverged"])
    for model in S.LEARNED:
        for tower in C.TOWERS_SEARCHED:
            parked = common.read_json(confirm.parked_path(root, model, tower))
            if parked is not None:
                rows.append({
                    "model": model,
                    "tower": tower,
                    "status": "shelved",
                    "reason": parked.get("reason")
                })
                continue
            record = common.read_json(confirm.winner_path(root, model, tower))
            if record is None:
                continue
            rows.append({
                "model": model,
                "tower": tower,
                "status": ("all_seeds_diverged"
                           if f"{model}/{tower}" in diverged else "winner"),
                "config": json.dumps(record["winner_config"]),
                "val_median": record["winner_median"],
                "margin_to_second": record["margin_to_second"],
                "best_epoch": record["best_epoch"]
            })
    return pd.DataFrame(rows)


def curves(root: str) -> pd.DataFrame:
    """Best validation score so far against the counted position (1..N) of
    the counted trials, per study (failed, over-cap and requeued trials are
    left out; 'trial' keeps the Optuna number)."""
    rows = []
    for model in S.LEARNED:
        for tower in C.TOWERS_SEARCHED:
            study = search.open_study(root, model, tower, create=False)
            if study is None:
                continue
            best = -np.inf
            trials = [
                t for t in sorted(study.get_trials(deepcopy=False),
                                  key=lambda t: t.number) if search.counted(t)
            ]
            for position, trial in enumerate(trials, 1):
                if search.eligible(trial):
                    best = max(best, trial.value)
                rows.append({
                    "model": model,
                    "tower": tower,
                    "position": position,
                    "trial": trial.number,
                    "state": trial.state.name,
                    "value": trial.value,
                    "best_so_far": best if np.isfinite(best) else None
                })
    return pd.DataFrame(rows)


def gpu_hours(root: str) -> pd.DataFrame:
    """GPU-hours per model, tower and phase over every attempt of every run
    (attempts.json; one GPU per run), with totals per model and overall."""
    rows = []
    for phase, folder in PHASE_DIRS.items():
        for path in sorted(
                glob.glob(os.path.join(root, folder, "*", "*",
                                       "attempts.json"))):
            study = os.path.basename(os.path.dirname(os.path.dirname(path)))
            model, tower = study.rsplit("_", 1)
            events = (common.read_json(path) or {}).get("events", [])
            rows.append(
                dict(model=model,
                     tower=tower,
                     phase=phase,
                     runs=1,
                     attempts=len(events),
                     killed=sum(
                         common.event_status(e) == "killed" for e in events),
                     gpu_hours=sum(e.get("seconds") or 0.0 for e in events) /
                     3600))
    columns = [
        "model", "tower", "phase", "runs", "attempts", "killed", "gpu_hours"
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    table = pd.DataFrame(rows)
    keys = ["model", "tower", "phase"]
    parts = [table.groupby(keys, as_index=False)[columns[3:]].sum()]
    parts.append(
        table.groupby("model",
                      as_index=False)[columns[3:]].sum().assign(tower="all",
                                                                phase="all"))
    parts.append(
        table.groupby("phase",
                      as_index=False)[columns[3:]].sum().assign(model="all",
                                                                tower="all"))
    parts.append(
        pd.DataFrame([table[columns[3:]].sum()]).assign(model="all",
                                                        tower="all",
                                                        phase="all"))
    out = pd.concat(parts, ignore_index=True)[columns]
    return out.astype({"runs": int, "attempts": int, "killed": int})


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
        board = with_missing(board[board["group"] == "all"].drop(
            columns="group")).assign(variant=variant)
        board.to_csv(os.path.join(out, "leaderboard.csv"), index=False)
        top3, families = rankings(board)
        top3.to_csv(os.path.join(out, "top3.csv"), index=False)
        families.to_csv(os.path.join(out, "families.csv"), index=False)
        boards[variant] = board
        print(f"== {variant} ({final.VARIANT_LABELS[variant]}): R^2 of "
              "log10 damage (median over seeds, mean over towers)")
        print(
            board[["model", "family"] +
                  [f"r2_log_damage_{c}" for c in CRITERIA] +
                  ["ranked", "missing_towers"]].round(3).to_string(index=False))
    configs(args.root).to_csv(os.path.join(args.out, "configs.csv"),
                              index=False)
    curves(args.root).to_csv(os.path.join(args.out, "curves.csv"), index=False)
    hours = gpu_hours(args.root)
    hours.to_csv(os.path.join(args.out, "gpu_hours.csv"), index=False)
    total = hours[(hours["model"] == "all") & (hours["phase"] == "all")]
    if len(total):
        print(f"GPU-hours (every attempt): {total['gpu_hours'].iloc[0]:.1f}")
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
                    common.write_json(
                        os.path.join(runs, f"SEALED_{model}.json"),
                        {"variant": variant})
    for model in S.LEARNED:
        for tower in C.TOWERS_SEARCHED:
            common.write_json(
                confirm.winner_path(root, model, tower), {
                    "winner_config": S.FIXED_RECIPE,
                    "winner_median": 0.5,
                    "margin_to_second": 0.01,
                    "best_epoch": 250
                })
    write_synthetic_attempts(root, rng)
    return dataset_dir


def write_synthetic_attempts(root: str, rng: np.random.Generator) -> None:
    """attempts.json of synthetic runs of every phase (a few search trials,
    the confirmation and final units, the test inference runs), for
    gpu_hours.csv of a dry run."""
    statuses = ("ok", "pruned", "crash", "preempted")
    for model in S.LEARNED:
        for tower in C.TOWERS_SEARCHED:
            study = f"{model}_{tower}"
            dirs = [("trials", f"t{n:03d}") for n in range(4)]
            dirs += [("phase2", f"c{r}_s{k}")
                     for r in range(C.N_TOP)
                     for k in range(C.N_SEEDS)]
            dirs += [("final", f"s{k}") for k in range(C.N_SEEDS)]
            dirs += [("test_runs", f"{v}_s{k}")
                     for v in VARIANTS
                     for k in range(C.N_SEEDS)]
            for folder, name in dirs:
                events = []
                for status in rng.choice(statuses, rng.integers(1, 3)):
                    seconds = float(rng.uniform(600, 3600))
                    events.append({
                        "start": common.now(),
                        "end": common.now(),
                        "seconds": round(seconds, 1),
                        "host": "synthetic",
                        "gpu": "none",
                        "status": str(status)
                    })
                common.write_json(
                    os.path.join(root, folder, study, name, "attempts.json"), {
                        "crashes": 0,
                        "free": 0,
                        "events": events
                    })


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
