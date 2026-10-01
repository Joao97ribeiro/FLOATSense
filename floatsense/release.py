# pylint: disable=too-many-instance-attributes
"""Reader of the released FLOATSense dataset.

The dataset has one directory per tower and one shared parked file:

  <dataset_dir>/<tower>/series-<k>-of-00017.parquet  sim_id, time_s and 37
                                    channels; one row group per simulation
  <dataset_dir>/<tower>/series_stats.parquet  per-channel statistics over the
                                    scored window
  <dataset_dir>/<tower>/metadata.parquet  operating point, split, regime
                                    labels and lifetime weight
  <dataset_dir>/<tower>/sections.parquet  the 11 scored FLOATBench sections
  <dataset_dir>/<tower>/damage.parquet  reference damage (sim_id, section_id;
                                    damage = section radius, damage_gauge =
                                    radius at the gauge, from v1.1)
  <dataset_dir>/parked.parquet      the 22 parked runs of every tower

Split names: `train` and `test` (the regime-aware partition, from
metadata.parquet), `val/train` and `val/val` (model selection) and
`fewshot/train_<n>_draw<k>` (adaptation sets); the last two are lists of
sim_id shipped with the code in `splits/`, the same on the three towers.
"""

import glob
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SAMPLING_FREQUENCY = 10.0
TOWER_HEIGHT = 149.386
NUM_SECTIONS = 30
SPLITS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                          "splits")
RADIUS_RULES = ("gauge", "section")


def _ordered(values: dict) -> np.ndarray:
    """Values of a d0.., h0.. or t1.. dict in index order."""
    return np.array([
        values[key] for key in sorted(values, key=lambda k: int(k[1:]))
    ], dtype=float)


def gauge_properties(geometry_path: str,
                     gauge_heights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Outer radius and wall thickness of a tower at the gauge heights.

    The radius is interpolated linearly between the section transitions
    (the tower tapers linearly within a section); the thickness is that of
    the section that contains the gauge.

    Args:
        geometry_path (str): Tower geometry JSON of the OpenFAST campaign
          (diameter, z and thickness transitions).
        gauge_heights (np.ndarray): Gauge heights above the tower base [m].

    Returns:
        Tuple[np.ndarray, np.ndarray]: Radius [m] and thickness [m].
    """
    with open(geometry_path, encoding="utf-8") as file:
        geometry = json.load(file)
    z = _ordered(geometry["z_transitions (m)"])
    radius = _ordered(geometry["diameter_transitions (m)"]) / 2
    thickness = _ordered(geometry["thickness_sections (m)"])
    heights = np.asarray(gauge_heights, dtype=float) + z[0]
    section = np.clip(np.searchsorted(z, heights, side="right") - 1, 0,
                      len(thickness) - 1)
    return np.interp(heights, z, radius), thickness[section]


class TowerSections:
    """Radius and wall thickness used for the damage at the scored sections.

    Arrays are indexed by the zero-based FLOATBench section (section_id - 1);
    only the scored sections are filled, the others are NaN.

    Two rules set the radius and thickness of a scored section:
      * 'gauge' (default): the outer radius at the gauge height and the
        thickness of the section that contains the gauge, so that the
        stress is taken where the moment is recorded.
      * 'section': the mean outer radius and thickness of the nearest
        FLOATBench section (release v1.0). At the top of the redesigns this
        radius is 2.5 to 4% larger than at the gauge and lowers the top
        damage by 20 to 35%; it reproduces the v1.0 numbers.

    Attributes:
        mean_radius_sections (np.ndarray): Radius used for the damage [m].
        thickness_sections (np.ndarray): Thickness used for the damage [m].
        height (float): Tower height [m].
        radius_rule (str): 'gauge' or 'section'.
    """

    def __init__(self, sections: pd.DataFrame, radius_rule: str = "gauge"):
        """Initializes the geometry from a sections.parquet table.

        Args:
            sections (pd.DataFrame): Rows with section_id, section_radius_m,
              section_thickness_m and, for the 'gauge' rule, gauge_radius_m
              and gauge_thickness_m (release v1.1).
            radius_rule (str): 'gauge' or 'section'.
        """
        if radius_rule not in RADIUS_RULES:
            raise ValueError(f"radius_rule must be one of {RADIUS_RULES}, "
                             f"got '{radius_rule}'.")
        self.radius_rule = radius_rule
        self.mean_radius_sections = np.full(NUM_SECTIONS, np.nan)
        self.thickness_sections = np.full(NUM_SECTIONS, np.nan)
        index = sections["section_id"].to_numpy(int) - 1
        if radius_rule == "section":
            radius = sections["section_radius_m"].to_numpy(float)
            thickness = sections["section_thickness_m"].to_numpy(float)
        elif "gauge_radius_m" in sections:
            radius = sections["gauge_radius_m"].to_numpy(float)
            thickness = sections["gauge_thickness_m"].to_numpy(float)
        else:
            raise KeyError("gauge_radius_m is not in sections.parquet (release "
                           "v1.0 has the 'section' rule only); rebuild it with "
                           "scripts/data/build_labels.py or use "
                           "--damage_radius=section.")
        self.mean_radius_sections[index] = radius
        self.thickness_sections[index] = thickness
        self.height = TOWER_HEIGHT


class ReleasedTower:
    """One tower of the released dataset.

    Attributes:
        tower_dir (str): Directory of the tower.
        name (str): Tower name (ref, opt1 or opt2).
        channels (List[str]): The 37 channel names, in file order.
        sampling_frequency (float): Sampling rate [Hz].
        metadata (pd.DataFrame): metadata.parquet indexed by sim_id.
        sections (pd.DataFrame): sections.parquet, base to top.
        geometry (TowerSections): Section properties for the damage.
    """

    def __init__(self, tower_dir: str, radius_rule: str = "gauge"):
        """Reads the metadata and indexes the series shards.

        Args:
            tower_dir (str): <dataset_dir>/<tower>.
            radius_rule (str): Radius of the damage, 'gauge' or 'section'
              (see TowerSections).
        """
        self.tower_dir = tower_dir
        self.name = os.path.basename(os.path.normpath(tower_dir))
        self.sampling_frequency = SAMPLING_FREQUENCY
        self.metadata = pd.read_parquet(
            os.path.join(tower_dir, "metadata.parquet")).set_index("sim_id")
        self.sections = pd.read_parquet(
            os.path.join(tower_dir, "sections.parquet"))
        self.geometry = TowerSections(self.sections, radius_rule)
        self._shards = sorted(
            glob.glob(os.path.join(tower_dir, "series-*.parquet")))
        if not self._shards:
            raise FileNotFoundError(f"No series shards in {tower_dir}")
        schema = pq.read_schema(self._shards[0])
        self.channels = [c for c in schema.names if c not in ("sim_id",
                                                              "time_s")]
        self._row_groups = self._index_row_groups()
        self._files: Dict[str, pq.ParquetFile] = {}
        self._series_stats: Optional[pd.DataFrame] = None

    def _index_row_groups(self) -> Dict[int, Tuple[str, int]]:
        """Maps sim_id to (shard, row group) from the Parquet statistics."""
        index = {}
        for path in self._shards:
            metadata = pq.ParquetFile(path).metadata
            column = metadata.schema.names.index("sim_id")
            for group in range(metadata.num_row_groups):
                stats = metadata.row_group(group).column(column).statistics
                index[int(stats.min)] = (path, group)
        return index

    def __getstate__(self):
        # Open Parquet handles are not picklable (DataLoader workers).
        state = self.__dict__.copy()
        state["_files"] = {}
        return state

    @property
    def sim_ids(self) -> List[int]:
        """Sorted identifiers of the simulations present in the local shards."""
        return sorted(self._row_groups)

    def load(self, sim_id: int) -> np.ndarray:
        """Full (10001, 37) float32 array of one simulation."""
        if int(sim_id) not in self._row_groups:
            raise KeyError(
                f"sim_id {sim_id} is not in the series shards of {self.tower_dir} "
                f"({len(self._row_groups)} simulations present). The review subset "
                "holds only the simulations of splits/review/test; training and the "
                "physics calibration need the full dataset.")
        path, group = self._row_groups[int(sim_id)]
        if path not in self._files:
            self._files[path] = pq.ParquetFile(path)
        table = self._files[path].read_row_group(group, columns=self.channels)
        return np.stack([table.column(c).to_numpy() for c in self.channels],
                        axis=1)

    @property
    def series_stats(self) -> pd.DataFrame:
        """series_stats.parquet indexed by sim_id (read on first use)."""
        if self._series_stats is None:
            self._series_stats = pd.read_parquet(
                os.path.join(self.tower_dir,
                             "series_stats.parquet")).set_index("sim_id")
        return self._series_stats

    def split_ids(self, name: str) -> List[int]:
        """Sorted simulation IDs of a split (see the module docstring)."""
        if name in ("train", "test"):
            ids = self.metadata.index[self.metadata["split"] == name]
        else:
            ids = pd.read_csv(os.path.join(SPLITS_DIR, f"{name}.csv"))["sim_id"]
        return sorted(int(i) for i in ids)

    def regime_cells(self, name: str = "test") -> pd.Series:
        """Regime cell label ('<wind_group>/<wave_group>') per sim_id."""
        meta = self.metadata.loc[self.split_ids(name)]
        return meta["wind_group"] + "/" + meta["wave_group"]

    def damage(self) -> pd.DataFrame:
        """damage.parquet as a (sim_id x section_id) table, with the radius
        rule of this tower ('damage_gauge' for 'gauge', 'damage' for
        'section')."""
        damage = pd.read_parquet(os.path.join(self.tower_dir, "damage.parquet"))
        column = "damage_gauge" if self.geometry.radius_rule == "gauge" else (
            "damage")
        if column not in damage:
            raise KeyError(f"{column} is not in damage.parquet (release v1.0 "
                           "has the 'section' rule only); rebuild it with "
                           "scripts/data/build_labels.py or use "
                           "--damage_radius=section.")
        return damage.pivot(index="sim_id", columns="section_id",
                            values=column)


def split_tag(name: str) -> str:
    """Output tag of a calibration split: '' for train, 'val' for val/train,
    'fs<n>_draw<k>' for fewshot/train_<n>_draw<k>, else the name with '_'."""
    if name == "train":
        return ""
    if name == "val/train":
        return "val"
    if name.startswith("fewshot/train_"):
        return "fs" + name[len("fewshot/train_"):]
    return name.replace("/", "_")


def load_tower(dataset_dir: str,
               name: str,
               radius_rule: str = "gauge") -> ReleasedTower:
    """Opens <dataset_dir>/<name> with the given damage radius rule."""
    return ReleasedTower(os.path.join(dataset_dir, name), radius_rule)


def load_parked(dataset_dir: str, tower: str) -> Dict[str, np.ndarray]:
    """The parked runs of one tower as {run: (10001, 28) array}, plus channels.

    Returns:
        dict: run name -> array, and '_channels' -> list of channel names.
    """
    parked = pd.read_parquet(os.path.join(dataset_dir, "parked.parquet"),
                             filters=[("tower", "==", tower)])
    channels = [c for c in parked.columns
                if c not in ("tower", "run", "time_s")]
    runs = {run: group.sort_values("time_s")[channels].to_numpy(np.float32)
            for run, group in parked.groupby("run", sort=True)}
    runs["_channels"] = channels
    return runs
