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
flags.DEFINE_enum("damage_radius", "gauge", ["gauge", "section"],
                  "Radius of the damage: at the gauge height (default) or "
                  "the mean of the nearest FLOATBench section (release v1.0).")
flags.DEFINE_string("results_csv", None,
                    "Where the table goes (default: <output_root>/tables/results.csv).")
flags.DEFINE_string("test_split", "test", "Split providing the regime cells.")
flags.DEFINE_integer("num_resamples", 1000, "Bootstrap resamples (0 = none).")
flags.DEFINE_integer("num_workers", 16, "Parallel scoring processes.")

TOWERS = ("ref", "opt1", "opt2")
GAUGES = [(stem, z_over_h) for stem, _, z_over_h in HEIGHT_TARGETS]
_CONTEXTS = {}


def tower_context(tower: str) -> dict:
    """Regime cells of the test split and operating conditions of every
    simulation (so runs scored on other splits, e.g. val/val, work too)."""
    release = load_tower(FLAGS.dataset_dir, tower, FLAGS.damage_radius)
    return {"cells": release.regime_cells(FLAGS.test_split),
            "clusters": condition_key(release.metadata)}


def describe(path: str) -> dict:
    """Model, protocol, source and scored tower of one damage CSV."""
    parts = os.path.relpath(path, FLAGS.output_root).split(os.sep)
    name = os.path.basename(path)
    if name == "damage_heights.csv":
        run = parts[-2]
        tower = run.split("_")[0]
        zero_shot = re.search(r"_zs_(ref|opt1|opt2)", run)
        return {"model": "physics", "run": os.path.dirname(
            os.path.relpath(path, FLAGS.output_root)),
                "direction": "ss" if run.endswith("_ss") else "fa",
                "source": zero_shot.group(1) if zero_shot else tower,
                "target": tower}
    stem = name[len("damage_comparison_"):-len(".csv")]
    match = re.match(r"(.+?)_(fa|ss)(?:_zs_(\w+))?$", stem)
    if not match:
        return {"model": stem, "run": "", "direction": "", "source": None,
                "target": None}
    model, direction, target = match.groups()
    pair = next((re.match(r"(\w+)_to_(\w+)$", p) for p in parts
                 if re.match(r"\w+_to_\w+$", p)), None)
    tower = next((p for p in parts if p in TOWERS), None)
    # A zero-shot file inside <src>_to_<tgt> was adapted on <tgt>.
    source = ((pair.group(2) if target else pair.group(1)) if pair
              else tower)
    return {"model": model,
            "run": os.path.dirname(os.path.relpath(path, FLAGS.output_root)),
            "direction": direction,
            "source": source,
            "target": target or (pair.group(2) if pair else tower)}


def _init(contexts):
    _CONTEXTS.update(contexts)


def score_file(path: str) -> pd.DataFrame:
    """Scores one CSV at every gauge (runs in a worker)."""
    meta = describe(path)
    if meta["target"] not in _CONTEXTS:
        logging.warning("Skipping %s: no tower (ref, opt1, opt2) in its path "
                        "or an unknown file name.", path)
        return pd.DataFrame()
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
        mre, corr, dropped = [], [], []
        for label in table["group"]:
            subset = df.loc[df.index.intersection(groups[label])]
            dropped.append(len(subset) - int(
                ((subset[true_col] > 0) & (subset[rec_col] > 0)).sum()))
            mre.append(mean_relative_error(subset[true_col], subset[rec_col]))
            corr.append(within_condition_correlation(
                subset[true_col], subset[rec_col],
                context["clusters"].loc[subset.index].values))
        table["mean_relative_error"] = mre
        table["within_condition_correlation"] = corr
        # Simulations with a non-positive or missing damage are left out
        # of every metric, as in the paper; count them.
        table["num_dropped"] = dropped
        if dropped[0]:
            logging.warning("%s, %s: %d simulations without a positive "
                            "damage left out of the metrics.", path, stem,
                            dropped[0])
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
    tables = [t for t in tables if len(t)]
    if not tables:
        logging.error("Nothing to score under %s.", FLAGS.output_root)
        return
    results = pd.concat(tables, ignore_index=True)
    first = ["run", "model", "direction", "source", "target", "gauge",
             "z_over_h", "group"]
    results = results[first + [c for c in results.columns if c not in first]]
    results_csv = FLAGS.results_csv or os.path.join(FLAGS.output_root, "tables",
                                                   "results.csv")
    os.makedirs(os.path.dirname(results_csv) or ".", exist_ok=True)
    results.to_csv(results_csv, index=False)
    logging.info("%d rows -> %s", len(results), results_csv)


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    flags.mark_flag_as_required("dataset_dir")
    app.run(main)
