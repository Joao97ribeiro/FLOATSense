# pylint: disable=too-many-locals
"""Scores every run under an output tree at the 11 gauges.

Reads the per-simulation damage CSVs written by scripts/train/run.py
(damage_comparison_<model>_fa[_zs_<target>].csv) and scripts/physics/run.py
(damage_heights.csv), and writes one long table with, per file, gauge and
regime cell (plus 'all'): R^2 of log10 damage, median damage ratio, fraction
within a factor of two with cluster-bootstrap 95% intervals, the mean
relative error and the within-condition correlation.

The scored tower of a file is the tower named in its path, or <target> for a
zero-shot file (_zs_<target>) and for a <source>_to_<target> directory.

Usage: python scripts/benchmark/run.py --flagfile=scripts/benchmark/config.cfg
"""

import glob
import multiprocessing
import os
import re
import sys

from absl import app
from absl import flags
from absl import logging
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from floatsense import load_tower  # noqa: E402  pylint: disable=wrong-import-position
from floatsense import summarize_by_group  # noqa: E402  pylint: disable=wrong-import-position
from floatsense.data import HEIGHT_TARGETS  # noqa: E402  pylint: disable=wrong-import-position
from floatsense.metrics import condition_key  # noqa: E402  pylint: disable=wrong-import-position
from floatsense.metrics import mean_relative_error  # noqa: E402  pylint: disable=wrong-import-position
from floatsense.metrics import within_condition_correlation  # noqa: E402  pylint: disable=wrong-import-position

FLAGS = flags.FLAGS
flags.DEFINE_string("dataset_dir", None, "Released FLOATSense dataset.")
flags.DEFINE_string("output_root", "outputs", "Root of the run outputs.")
flags.DEFINE_string("results_csv", "outputs/tables/results.csv",
                    "Where the table goes.")
flags.DEFINE_string("test_split", "test", "Split providing the regime cells.")
flags.DEFINE_integer("num_resamples", 1000, "Bootstrap resamples (0 = none).")
flags.DEFINE_integer("num_workers", 16, "Parallel scoring processes.")

TOWERS = ("ref", "opt1", "opt2")
GAUGES = [(stem, z_over_h) for stem, _, z_over_h in HEIGHT_TARGETS]
_CONTEXTS = {}


def tower_context(tower: str) -> dict:
    """Regime cells and operating conditions of a tower's test split."""
    release = load_tower(FLAGS.dataset_dir, tower)
    meta = release.metadata.loc[release.split_ids(FLAGS.test_split)]
    return {"cells": release.regime_cells(FLAGS.test_split),
            "clusters": condition_key(meta)}


def describe(path: str) -> dict:
    """Model, protocol, source and scored tower of one damage CSV."""
    parts = os.path.relpath(path, FLAGS.output_root).split(os.sep)
    name = os.path.basename(path)
    if name == "damage_heights.csv":
        run = parts[-2]
        tower = run.split("_")[0]
        zero_shot = re.search(r"_zs_(\w+)$", run)
        return {"model": "physics", "run": os.path.dirname(
            os.path.relpath(path, FLAGS.output_root)),
                "source": zero_shot.group(1) if zero_shot else tower,
                "target": tower}
    stem = name[len("damage_comparison_"):-len(".csv")]
    model, target = re.match(r"(.+?)_fa(?:_zs_(\w+))?$", stem).groups()
    pair = next((re.match(r"(\w+)_to_(\w+)$", p) for p in parts
                 if re.match(r"\w+_to_\w+$", p)), None)
    tower = next((p for p in parts if p in TOWERS), None)
    source = pair.group(1) if pair else tower
    return {"model": model,
            "run": os.path.dirname(os.path.relpath(path, FLAGS.output_root)),
            "source": source,
            "target": target or (pair.group(2) if pair else tower)}


def _init(contexts):
    _CONTEXTS.update(contexts)


def score_file(path: str) -> pd.DataFrame:
    """Scores one CSV at every gauge (runs in a worker)."""
    meta = describe(path)
    context = _CONTEXTS[meta["target"]]
    df = pd.read_csv(path).set_index("sim_id")
    cells = context["cells"].loc[context["cells"].index.intersection(df.index)]
    tables = []
    for stem, z_over_h in GAUGES:
        true_col, rec_col = f"damage_true_{stem}", f"damage_rec_{stem}"
        if true_col not in df.columns:
            continue
        table = summarize_by_group(df, stem, cells, context["clusters"],
                                   FLAGS.num_resamples)
        groups = {"all": df.index}
        groups.update(cells.groupby(cells).groups)
        mre, corr = [], []
        for label in table["group"]:
            subset = df.loc[df.index.intersection(groups[label])]
            mre.append(mean_relative_error(subset[true_col], subset[rec_col]))
            corr.append(within_condition_correlation(
                subset[true_col], subset[rec_col],
                context["clusters"].loc[subset.index].values))
        table["mean_relative_error"] = mre
        table["within_condition_correlation"] = corr
        table["gauge"], table["z_over_h"] = stem, z_over_h
        for key, value in meta.items():
            table[key] = value
        tables.append(table)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def main(_):
    """Writes the results table."""
    paths = sorted(
        glob.glob(os.path.join(FLAGS.output_root, "**",
                               "damage_comparison_*.csv"), recursive=True) +
        glob.glob(os.path.join(FLAGS.output_root, "**", "damage_heights.csv"),
                  recursive=True))
    contexts = {tower: tower_context(tower) for tower in TOWERS}
    logging.info("%d damage CSVs under %s", len(paths), FLAGS.output_root)
    with multiprocessing.Pool(FLAGS.num_workers, _init, (contexts,)) as pool:
        tables = pool.map(score_file, paths)
    results = pd.concat([t for t in tables if len(t)], ignore_index=True)
    first = ["run", "model", "source", "target", "gauge", "z_over_h", "group"]
    results = results[first + [c for c in results.columns if c not in first]]
    os.makedirs(os.path.dirname(FLAGS.results_csv) or ".", exist_ok=True)
    results.to_csv(FLAGS.results_csv, index=False)
    logging.info("%d rows -> %s", len(results), FLAGS.results_csv)


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    flags.mark_flag_as_required("dataset_dir")
    app.run(main)
