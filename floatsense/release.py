# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
"""Reader of the released FLOATSense dataset.

The dataset has one directory per tower and one shared parked file:

  <dataset_dir>/<tower>/series-<k>-of-00017.parquet  sim_id, time_s and 37
                                    channels; one row group per simulation
  <dataset_dir>/<tower>/series_stats.parquet  per-channel statistics over the
                                    scored window
  <dataset_dir>/<tower>/metadata.parquet  operating point, split, regime
                                    labels and lifetime weight
  <dataset_dir>/<tower>/sections.parquet  the 11 scored FLOATBench sections
  <dataset_dir>/<tower>/damage.parquet  reference damage (sim_id, section_id,
                                    damage; radius at the gauge)
  <dataset_dir>/parked.parquet      the 22 parked runs of every tower

Split names: `train` and `test` (the regime-aware partition, from
metadata.parquet), `val/train` and `val/val` (model selection) and
`fewshot/train_<n>_draw<k>` (adaptation sets); the last two are lists of
sim_id shipped with the code in `splits/`, the same on the three towers.
"""

import glob
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .constants import LOWPASS_HZ
from .constants import LOWPASS_ORDER
from .constants import MAX_TIME
from .constants import MIN_TIME
from .constants import SAMPLING_FREQUENCY
from .fatigue import compute_base_damage
from .fatigue import damage_filter

SPLITS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                          "splits")


class TowerGauges:
    """Radius and wall thickness used for the damage at the 11 gauges.

    Arrays are indexed by gauge, 0 (base) to 10 (top), in the order of
    sections.parquet. The radius is the outer radius at the gauge height and
    the thickness that of the section containing the gauge, so the stress is
    taken where the moment is recorded.

    Attributes:
        channels (List[str]): Gauge names ('tower_bottom', ..., 'tower_top').
        section_ids (List[int]): FLOATBench section containing each gauge.
        heights_gauges (np.ndarray): Gauge height above the base [m].
        z_over_h (np.ndarray): Gauge height over the tower height.
        radius_gauges (np.ndarray): Outer radius at the gauge height [m].
        thickness_gauges (np.ndarray): Thickness of the section containing
          the gauge [m].
        height (float): Tower height [m] (height of the top gauge).
    """

    def __init__(self, sections: pd.DataFrame):
        """Initializes the geometry from a sections.parquet table.

        Args:
            sections (pd.DataFrame): One row per gauge, base to top, with
              channel, section_id, gauge_height_m, z_over_h, gauge_radius_m
              and gauge_thickness_m.
        """
        if "gauge_radius_m" not in sections:
            raise KeyError("gauge_radius_m is not in sections.parquet; "
                           "download the current dataset.")
        gauges = sections.sort_values("gauge_height_m")
        self.channels = gauges["channel"].tolist()
        self.section_ids = gauges["section_id"].astype(int).tolist()
        self.heights_gauges = gauges["gauge_height_m"].to_numpy(float)
        self.z_over_h = gauges["z_over_h"].to_numpy(float)
        self.radius_gauges = gauges["gauge_radius_m"].to_numpy(float)
        self.thickness_gauges = gauges["gauge_thickness_m"].to_numpy(float)
        self.height = float(self.heights_gauges[-1])

    def damage(self,
               moment_series: np.ndarray,
               gauge: int,
               sn_intercepts_log10: Optional[List[float]] = None,
               sn_slopes: Optional[List[float]] = None) -> float:
        """Fatigue damage of a moment series (true or predicted) at a gauge.

        Args:
            moment_series (np.ndarray): Bending moment [kN.m] at the gauge.
            gauge (int): Gauge index, 0 (base) to 10 (top).
            sn_intercepts_log10 (List[float], optional): SN log10 intercepts.
            sn_slopes (List[float], optional): SN curve slopes.

        Returns:
            float: Miner damage (unitless).
        """
        return compute_base_damage(moment_series,
                                   self,
                                   sn_intercepts_log10,
                                   sn_slopes,
                                   gauge=gauge)


class ReleasedTower:
    """One tower of the released dataset.

    Attributes:
        tower_dir (str): Directory of the tower.
        name (str): Tower name (ref, opt1 or opt2).
        channels (List[str]): The 37 channel names, in file order.
        sampling_frequency (float): Sampling rate [Hz].
        metadata (pd.DataFrame): metadata.parquet indexed by sim_id.
        sections (pd.DataFrame): sections.parquet, base to top.
        geometry (TowerGauges): Gauge properties for the damage.
    """

    def __init__(self, tower_dir: str):
        """Reads the metadata and indexes the series shards.

        Args:
            tower_dir (str): <dataset_dir>/<tower>.
        """
        self.tower_dir = tower_dir
        self.name = os.path.basename(os.path.normpath(tower_dir))
        self.sampling_frequency = SAMPLING_FREQUENCY
        self.metadata = pd.read_parquet(
            os.path.join(tower_dir, "metadata.parquet")).set_index("sim_id")
        self.sections = pd.read_parquet(
            os.path.join(tower_dir, "sections.parquet"))
        self.geometry = TowerGauges(self.sections)
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

    def scored_moment(self,
                      sim_id: int,
                      gauge: str,
                      min_time: float = MIN_TIME,
                      max_time: float = MAX_TIME) -> np.ndarray:
        """True fore-aft moment of one gauge over the scored window.

        The window is the one of the evaluation and of FLOATBench: from
        `min_time` to `max_time` inclusive (6,001 samples), mean removed.

        Args:
            sim_id (int): Simulation ID.
            gauge (str): Gauge name ('tower_bottom', 'tower_1', ...,
              'tower_top').
            min_time (float): Start of the scored window [s].
            max_time (float): End of the scored window [s], inclusive.

        Returns:
            np.ndarray: Moment [kN.m].
        """
        start = int(round(min_time * self.sampling_frequency))
        stop = int(round(max_time * self.sampling_frequency)) + 1
        series = self.load(sim_id)[start:stop,
                                   self.channels.index(f"{gauge}_mfa")]
        series = series.astype(float)
        return series - series.mean()

    def gauge_damage(
            self,
            moment_series: np.ndarray,
            gauge: str,
            apply_lowpass: bool = True,
            lowpass_hz: float = LOWPASS_HZ,
            lowpass_order: int = LOWPASS_ORDER,
            sn_intercepts_log10: Optional[List[float]] = None,
            sn_slopes: Optional[List[float]] = None) -> float:
        """Fatigue damage of any moment series (true or predicted) at a gauge.

        The series goes through the low-pass of the benchmark metric (zero-
        phase Butterworth, 3 Hz) unless `apply_lowpass` is False.

        Args:
            moment_series (np.ndarray): Bending moment [kN.m] at the gauge.
            gauge (str): Gauge name ('tower_bottom', ..., 'tower_top').
            apply_lowpass (bool): Low-pass before the damage (as the metric).
            lowpass_hz (float): Cutoff of the low-pass [Hz].
            lowpass_order (int): Butterworth order of one pass.
            sn_intercepts_log10 (List[float], optional): SN log10 intercepts.
            sn_slopes (List[float], optional): SN curve slopes.

        Returns:
            float: Miner damage (unitless), as in damage.parquet for the true
              moment of scored_moment.
        """
        if gauge not in self.geometry.channels:
            raise KeyError(f"Unknown gauge '{gauge}'.")
        series = damage_filter(moment_series, self.sampling_frequency,
                               apply_lowpass, lowpass_hz, lowpass_order)
        return self.geometry.damage(series,
                                    self.geometry.channels.index(gauge),
                                    sn_intercepts_log10, sn_slopes)

    def damage(self) -> pd.DataFrame:
        """damage.parquet as a (sim_id x section_id) table."""
        damage = pd.read_parquet(os.path.join(self.tower_dir, "damage.parquet"))
        return damage.pivot(index="sim_id", columns="section_id",
                            values="damage")


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


def load_gauges(dataset_dir: str, name: str) -> TowerGauges:
    """The 11 gauges of <dataset_dir>/<name> (sections.parquet only)."""
    return TowerGauges(
        pd.read_parquet(os.path.join(dataset_dir, name, "sections.parquet")))


def load_tower(dataset_dir: str, name: str) -> ReleasedTower:
    """Opens <dataset_dir>/<name>."""
    return ReleasedTower(os.path.join(dataset_dir, name))


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
