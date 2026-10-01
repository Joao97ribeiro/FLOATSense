"""Writes the label and metadata files of the released dataset.

Reads the released series of each tower and the FLOATBench dataset (the
train/test files give the split), and writes, next to the series
(<dataset_dir>/<tower>/):
  metadata.parquet  one row per simulation, FLOATBench column order:
                    sim_id, wind_speed_id, wind_speed, mean_wind_speed,
                    std_wind_speed, wave_hs_id, wave_hs, wave_tp_id, wave_tp,
                    wind_seed_id, split, wind_group, wave_group, damage_weight
  sections.parquet  the 11 scored FLOATBench sections: section_id,
                    section_height_m, section_radius_m, section_thickness_m
                    (as in FLOATBench), channel, gauge_height_m, z_over_h,
                    gauge_radius_m, gauge_thickness_m (at the gauge height)
  damage.parquet    sim_id, section_id, damage (11 rows per simulation)

The damage is computed as in the evaluation: the fore-aft moment of each
gauge is cut to the scored window (400-1000 s, an even 6,000 samples as in
the evaluation), its mean removed,
low-passed at 3 Hz, rainflow-counted and passed through the S-N curve with
the outer radius at the gauge height and the wall thickness of the section
that contains it (release v1.1; v1.0 used the mean radius of that section,
which underestimates the top damage of the redesigns).

Usage: python scripts/data/build_labels.py --dataset_dir=data/FLOATSense \
           --floatbench_dir=data/FLOATBench --towers=ref,opt1,opt2 \
           --tower_geometry=ref:<ref.json>,opt1:<opt1.json>,opt2:<opt2.json>
"""

import glob
import multiprocessing
import os
import sys
import types

from absl import app
from absl import flags
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from floatsense.data import HEIGHT_TARGETS  # noqa: E402  pylint: disable=wrong-import-position
from floatsense.fatigue import compute_base_damage  # noqa: E402  pylint: disable=wrong-import-position
from floatsense.physics import lowpass  # noqa: E402  pylint: disable=wrong-import-position
from floatsense.tower import Tower  # noqa: E402  pylint: disable=wrong-import-position

FLAGS = flags.FLAGS
flags.DEFINE_string("dataset_dir", None, "Released dataset, one folder per tower.")
flags.DEFINE_string("floatbench_dir", None, "FLOATBench dataset directory.")
flags.DEFINE_list("towers", ["ref", "opt1", "opt2"], "Towers to process.")
flags.DEFINE_integer("workers", 8, "Worker processes for the damage.")
flags.DEFINE_list("tower_geometry", None,
                  "<tower>:<path> of the tower geometry JSON of the OpenFAST "
                  "campaign, one per tower (diameter and z at the section "
                  "transitions, section thickness); the outer radius at each "
                  "gauge height is interpolated linearly from it.")

FS, MIN_TIME, MAX_TIME, LOWPASS_HZ = 10.0, 400.0, 1000.0, 3.0
SN_INTERCEPTS, SN_SLOPES = [12.010, 15.350], [3.0, 5.0]
HEIGHT = 149.386
METADATA_COLUMNS = [
    "sim_id", "wind_speed_id", "wind_speed", "mean_wind_speed",
    "std_wind_speed", "wave_hs_id", "wave_hs", "wave_tp_id", "wave_tp",
    "wind_seed_id", "split", "wind_group", "wave_group", "damage_weight"
]
GEOMETRY_COLUMNS = [
    "section_id", "section_height_m", "section_radius_m", "section_thickness_m"
]


def build_metadata(floatbench: pd.DataFrame) -> pd.DataFrame:
    """Operating point, split, regime labels and weight per simulation.

    Args:
        floatbench (pd.DataFrame): FLOATBench train and test rows of the
          tower, with a 'split' column naming the file each row came from.

    Returns:
        pd.DataFrame: One row per simulation, sorted by sim_id.
    """
    per_sim = floatbench.drop_duplicates("sim_id")
    return per_sim[METADATA_COLUMNS].sort_values("sim_id").reset_index(
        drop=True)


def build_sections(floatbench: pd.DataFrame,
                   geometry_path: str) -> pd.DataFrame:
    """Geometry of the 11 scored FLOATBench sections and their gauges.

    Args:
        floatbench (pd.DataFrame): FLOATBench rows of the tower.
        geometry_path (str): Tower geometry JSON (see --tower_geometry).

    Returns:
        pd.DataFrame: One row per scored section, base to top.
    """
    geometry = floatbench.drop_duplicates("section_id").set_index("section_id")
    rows = []
    for name, index, z_over_h in HEIGHT_TARGETS:
        row = {"section_id": index + 1}
        row.update(geometry.loc[index + 1, GEOMETRY_COLUMNS[1:]].to_dict())
        row.update({"channel": name, "gauge_height_m": z_over_h * HEIGHT,
                    "z_over_h": z_over_h})
        rows.append(row)
    sections = pd.DataFrame(rows)
    radius, thickness = Tower(json_path=geometry_path).gauge_properties(
        sections["gauge_height_m"].to_numpy())
    sections["gauge_radius_m"] = radius
    sections["gauge_thickness_m"] = thickness
    return sections


def _shard_damage(args):
    """Damage at the 11 sections of every simulation of one shard."""
    path, sections = args
    geometry = types.SimpleNamespace(
        mean_radius_sections=sections["gauge_radius_m"].to_numpy(),
        thickness_sections=sections["gauge_thickness_m"].to_numpy())
    # The evaluation keeps an even number of samples of the inclusive
    # 400-1000 s window (floatsense.data), i.e. 400.0 to 999.9 s.
    start = int(round(MIN_TIME * FS))
    stop = start + 2 * ((int(round(MAX_TIME * FS)) + 1 - start) // 2)
    shard = pq.ParquetFile(path)
    columns = ["sim_id"] + [f"{c}_mfa" for c in sections["channel"]]
    rows = []
    for group in range(shard.num_row_groups):
        table = shard.read_row_group(group, columns=columns).to_pandas()
        sim_id = int(table["sim_id"].iloc[0])
        for position, (section_id, channel) in enumerate(
                zip(sections["section_id"], sections["channel"])):
            moment = table[f"{channel}_mfa"].to_numpy(float)[start:stop]
            moment = lowpass(moment - moment.mean(), FS, LOWPASS_HZ)
            damage = compute_base_damage(moment, geometry, SN_INTERCEPTS,
                                         SN_SLOPES, section=position)
            rows.append((sim_id, int(section_id), damage))
    return rows


def build_damage(tower_dir: str, sections: pd.DataFrame) -> pd.DataFrame:
    """Reference fore-aft damage at the 11 sections of every simulation.

    Args:
        tower_dir (str): Released tower directory with the series shards.
        sections (pd.DataFrame): Output of build_sections.

    Returns:
        pd.DataFrame: sim_id, section_id, damage, sorted by both.
    """
    shards = sorted(glob.glob(os.path.join(tower_dir, "series-*.parquet")))
    with multiprocessing.Pool(FLAGS.workers) as pool:
        parts = pool.map(_shard_damage, [(s, sections) for s in shards])
    damage = pd.DataFrame([r for part in parts for r in part],
                          columns=["sim_id", "section_id", "damage"])
    return damage.sort_values(["sim_id", "section_id"]).reset_index(drop=True)


def main(_) -> None:
    """Writes metadata, sections and damage for every requested tower."""
    geometry_paths = dict(
        item.split(":", 1) for item in (FLAGS.tower_geometry or []))
    missing = [t for t in FLAGS.towers if t not in geometry_paths]
    if missing:
        raise ValueError(f"--tower_geometry has no path for {missing}.")
    for tower in FLAGS.towers:
        tower_dir = os.path.join(FLAGS.dataset_dir, tower)
        floatbench = pd.concat([
            pd.read_csv(os.path.join(FLAGS.floatbench_dir, tower,
                                     f"{split}_damage.csv")).assign(split=split)
            for split in ("train", "test")
        ])
        metadata = build_metadata(floatbench)
        sections = build_sections(floatbench, geometry_paths[tower])
        damage = build_damage(tower_dir, sections)
        metadata.to_parquet(os.path.join(tower_dir, "metadata.parquet"),
                            index=False)
        sections.to_parquet(os.path.join(tower_dir, "sections.parquet"),
                            index=False)
        damage.to_parquet(os.path.join(tower_dir, "damage.parquet"),
                          index=False)
        print(f"{tower}: {len(metadata)} simulations, {len(sections)} "
              f"sections, {len(damage)} damage rows", flush=True)


if __name__ == "__main__":
    flags.mark_flags_as_required(["dataset_dir", "floatbench_dir"])
    app.run(main)
