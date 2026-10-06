"""Writes the label and metadata files of the released dataset.

Maintainers only: the released files already contain the labels. The gauge
geometry (--gauge_profile) is the gauge profile written by the fatigue flow
of the OpenFAST campaign (tower_weighted_damage_profile_gauges.csv).

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
gauge is cut to the scored window (400.0-1000.0 s inclusive, 6,001 samples,
as in the evaluation and FLOATBench), its mean removed, low-passed at 3 Hz
with a zero-phase Butterworth filter (see --lowpass), rainflow-counted and
passed through the S-N curve with the outer radius at the gauge height and
the wall thickness of the section that contains it.

Usage: python scripts/data/build_labels.py --dataset_dir=data/FLOATSense \
           --floatbench_dir=data/FLOATBench --towers=ref,opt1,opt2 \
           --gauge_profile=ref:<ref.csv>,opt1:<opt1.csv>,opt2:<opt2.csv>
"""

import glob
import multiprocessing
import os
import sys

from absl import app
from absl import flags
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from floatsense.constants import LOWPASS_HZ  # noqa: E402  pylint: disable=wrong-import-position
from floatsense.constants import LOWPASS_ORDER  # noqa: E402  pylint: disable=wrong-import-position
from floatsense.constants import MAX_TIME  # noqa: E402  pylint: disable=wrong-import-position
from floatsense.constants import MIN_TIME  # noqa: E402  pylint: disable=wrong-import-position
from floatsense.constants import SAMPLING_FREQUENCY  # noqa: E402  pylint: disable=wrong-import-position
from floatsense.release import TowerGauges  # noqa: E402  pylint: disable=wrong-import-position
from floatsense.fatigue import damage_filter  # noqa: E402  pylint: disable=wrong-import-position

FLAGS = flags.FLAGS
flags.DEFINE_string("dataset_dir", None, "Released dataset, one folder per tower.")
flags.DEFINE_string("floatbench_dir", None, "FLOATBench dataset directory.")
flags.DEFINE_list("towers", ["ref", "opt1", "opt2"], "Towers to process.")
flags.DEFINE_integer("workers", 8, "Worker processes for the damage.")
flags.DEFINE_list("gauge_profile", None,
                  "<tower>:<path> of the gauge profile of the OpenFAST "
                  "campaign, one per tower (CSV with gauge_id, gauge, "
                  "section_id, z [m], radius [m], thickness [m], bottom to "
                  "top: the outer "
                  "radius at each gauge height and the thickness of the "
                  "section containing it).")
flags.DEFINE_bool("lowpass", True,
                  "Zero-phase Butterworth low-pass of the moment before the "
                  "damage, as in the evaluation.")
flags.DEFINE_float("lowpass_hz", LOWPASS_HZ, "Cutoff of --lowpass [Hz].")
flags.DEFINE_integer("lowpass_order", LOWPASS_ORDER,
                     "Butterworth order of one pass (sosfiltfilt runs two).")

FS = SAMPLING_FREQUENCY
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
                   gauge_profile: pd.DataFrame) -> pd.DataFrame:
    """Geometry of the 11 scored FLOATBench sections and their gauges.

    Args:
        floatbench (pd.DataFrame): FLOATBench rows of the tower.
        gauge_profile (pd.DataFrame): Gauge profile (see --gauge_profile).

    Returns:
        pd.DataFrame: One row per scored section, base to top.
    """
    # Every gauge property comes from the gauge profile of the OpenFAST
    # campaign: name, FLOATBench section, height, radius and thickness.
    missing = {"gauge", "section_id"} - set(gauge_profile.columns)
    if missing:
        raise ValueError(f"The gauge profile has no {sorted(missing)} column; "
                         "regenerate it with the current fatigue flow.")
    gauges = gauge_profile.sort_values("gauge_id")
    heights = gauges["z [m]"].to_numpy() - gauges["z [m]"].iloc[0]
    geometry = floatbench.drop_duplicates("section_id").set_index("section_id")
    rows = []
    for name, section_id, height in zip(gauges["gauge"],
                                        gauges["section_id"].astype(int),
                                        heights):
        row = {"section_id": section_id}
        row.update(geometry.loc[section_id, GEOMETRY_COLUMNS[1:]].to_dict())
        row.update({"channel": name, "gauge_height_m": float(height),
                    "z_over_h": float(height / heights[-1])})
        rows.append(row)
    sections = pd.DataFrame(rows)
    sections["gauge_radius_m"] = gauges["radius [m]"].to_numpy()
    sections["gauge_thickness_m"] = gauges["thickness [m]"].to_numpy()
    return sections


def _shard_damage(args):
    """Damage at the 11 sections of every simulation of one shard."""
    path, sections, apply_lowpass, lowpass_hz, lowpass_order = args
    geometry = TowerGauges(sections)
    # The scored window, inclusive: 400.0 to 1000.0 s (6,001 samples).
    start = int(round(MIN_TIME * FS))
    stop = int(round(MAX_TIME * FS)) + 1
    shard = pq.ParquetFile(path)
    columns = ["sim_id"] + [f"{c}_mfa" for c in sections["channel"]]
    rows = []
    for group in range(shard.num_row_groups):
        table = shard.read_row_group(group, columns=columns).to_pandas()
        sim_id = int(table["sim_id"].iloc[0])
        for gauge, (section_id, channel) in enumerate(
                zip(sections["section_id"], sections["channel"])):
            moment = table[f"{channel}_mfa"].to_numpy(float)[start:stop]
            moment = damage_filter(moment - moment.mean(), FS, apply_lowpass,
                                   lowpass_hz, lowpass_order)
            damage = geometry.damage(moment, gauge)
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
        parts = pool.map(_shard_damage,
                         [(s, sections, FLAGS.lowpass, FLAGS.lowpass_hz,
                           FLAGS.lowpass_order) for s in shards])
    damage = pd.DataFrame([r for part in parts for r in part],
                          columns=["sim_id", "section_id", "damage"])
    return damage.sort_values(["sim_id", "section_id"]).reset_index(drop=True)


def main(_) -> None:
    """Writes metadata, sections and damage for every requested tower."""
    profile_paths = dict(
        item.split(":", 1) for item in (FLAGS.gauge_profile or []))
    missing = [t for t in FLAGS.towers if t not in profile_paths]
    if missing:
        raise ValueError(f"--gauge_profile has no path for {missing}.")
    for tower in FLAGS.towers:
        tower_dir = os.path.join(FLAGS.dataset_dir, tower)
        floatbench = pd.concat([
            pd.read_csv(os.path.join(FLAGS.floatbench_dir, tower,
                                     f"{split}_damage.csv")).assign(split=split)
            for split in ("train", "test")
        ])
        metadata = build_metadata(floatbench)
        sections = build_sections(floatbench,
                                  pd.read_csv(profile_paths[tower]))
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
