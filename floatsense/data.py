# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-instance-attributes
# pylint: disable=unsubscriptable-object
"""Torch dataset over the released FLOATSense time series.

Every simulation is one row group of a Parquet shard (`floatsense.release`).
The dataset yields (input channels, condition scalars, target channel)
tensors for sequence-to-sequence moment reconstruction.
"""

from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .constants import MAX_TIME
from .constants import MIN_TIME
from .constants import OPERATING_POWER_KW
from .physics import DIRECTION_CHANNELS
from .physics import Calibration
from .physics import band_masks
from .physics import harmonic_windows
from .release import ReleasedTower

DEFAULT_CONDITION_CHANNELS = ["rotor_speed", "blade_pitch", "wind_speed"]


class SequenceDataset(Dataset):
    """Sequence dataset of (acceleration + state) -> moment pairs.

    Each item is a dict with:
      - 'inputs': float32 (1 + num_condition_channels, length): the
        acceleration followed by the time-varying condition channels;
      - 'condition': float32 (num_condition_channels,): the mean of each
        condition channel over the item window;
      - 'target': float32 (1, length): the zero-mean target moment;
      - 'physics_gain' (only with a calibration): the physics band gain per
        rFFT bin of the window, in normalized units, for the hybrid model.

    Channels are standardized with `norm_stats` (computed on the training
    set by `compute_norm_stats`); the target is scaled by its std only, as
    it is already zero-mean per item.
    """

    def __init__(self,
                 release: ReleasedTower,
                 sim_ids: List[int],
                 direction: str = "fa",
                 condition_channels: Optional[List[str]] = None,
                 min_time: float = MIN_TIME,
                 max_time: float = MAX_TIME,
                 crop_length: Optional[int] = None,
                 norm_stats: Optional[Dict[str, List[float]]] = None,
                 calibration_path: Optional[str] = None,
                 target_channel: Optional[str] = None,
                 input_channels: Optional[List[str]] = None,
                 height_targets: bool = False,
                 height_factors: Optional[List[float]] = None,
                 window_length: Optional[int] = None):
        """Initializes the dataset.

        Args:
            release (ReleasedTower): Tower of the released dataset.
            sim_ids (List[int]): Simulation IDs to include.
            direction (str): Reconstruction direction ('fa' or 'ss').
            condition_channels (List[str], optional): Operating-state
              channels appended to the input and averaged into the
              condition vector.
            min_time (float): Start of the usable window [s].
            max_time (float): End of the usable window [s], inclusive.
            crop_length (int, optional): If set, a random crop of this many
              samples is returned (training); otherwise the full scored
              window (6,001 samples, 400.0-1,000.0 s, as FLOATBench).
            norm_stats (dict, optional): Mapping channel -> [mean, std].
            calibration_path (str, optional): Physics calibration JSON that
              adds the per-simulation 'physics_gain' item.
            target_channel (str, optional): Overrides the target channel of
              the direction (e.g. 'tower_5_mfa' for an intermediate
              section).
            input_channels (List[str], optional): Replaces the default
              [acceleration + condition channels] input stack (sensor
              ablations).
        """
        self.release = release
        self.sim_ids = list(sim_ids)
        self.direction = direction
        self.condition_channels = list(condition_channels or
                                       DEFAULT_CONDITION_CHANNELS)
        self.crop_length = crop_length
        self.norm_stats = norm_stats
        self.calibration = (Calibration.from_json(calibration_path)
                            if calibration_path else None)

        self.channels = release.channels
        self.sampling_frequency = release.sampling_frequency
        self.start_index = int(round(min_time * self.sampling_frequency))
        self.stop_index = int(round(max_time * self.sampling_frequency)) + 1
        self.operating_power_kw = OPERATING_POWER_KW

        self.accel_channel = DIRECTION_CHANNELS[direction]["accel"]
        self.moment_channel = (target_channel or
                               DIRECTION_CHANNELS[direction]["moment"])
        self.input_channels = (list(input_channels) if input_channels else
                               [self.accel_channel] + self.condition_channels)
        # Height-conditioned task: a constant input channel z/H, the target
        # is the moment at that height (random during training, `section`
        # at evaluation) and the condition vector gets z/H as well.
        self.height_targets = height_targets
        self.height_factors = height_factors
        self.section = None
        # Length-fixed models see `window_length` samples from
        # `window_offset` (their input size); None keeps the full window.
        self.window_length = window_length
        self.window_offset = 0
        if height_targets:
            self.input_channels = self.input_channels + ["height"]
            self.height_channels = [
                f"{stem}_m{direction}" for stem in release.geometry.channels
            ]
            self.height_z_over_h = release.geometry.z_over_h
        # Field-SCADA channels 'stat:<channel>:<mean|std|min|max>' read the
        # released per-window statistics (series_stats.parquet); the series
        # of that channel is never touched.
        self.series_stats = None
        if any(c.startswith("stat:") for c in self.input_channels):
            self.series_stats = release.series_stats

    def __len__(self) -> int:
        return len(self.sim_ids)

    def _channel_index(self, channel: str) -> int:
        return self.channels.index(channel)

    def load_window(self, sim_id: int) -> np.ndarray:
        """Loads the (num_samples, num_channels) window of one simulation."""
        return self.release.load(sim_id)[self.start_index:self.stop_index]

    def _normalize(self, channel: str, values: np.ndarray) -> np.ndarray:
        if not self.norm_stats or channel not in self.norm_stats:
            return values
        mean, std = self.norm_stats[channel]
        std = std or 1.0
        if channel == self.moment_channel:
            return values / std
        return (values - mean) / std

    def denormalize_target(self, values: np.ndarray) -> np.ndarray:
        """Rescales a normalized target or prediction back to kN.m."""
        if not self.norm_stats or self.moment_channel not in self.norm_stats:
            return values
        return values * self.norm_stats[self.moment_channel][1]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sim_id = self.sim_ids[index]
        window = self.load_window(sim_id)
        # Field-SCADA channels 'stat:<channel>:<mean|std|min|max>': the
        # statistic of the whole scored window (the ten-minute value a
        # turbine logs), broadcast as a constant channel; normalized with
        # the statistics of the underlying channel.
        stats = {}
        for channel in self.input_channels:
            if channel.startswith("stat:"):
                _, base, kind = channel.split(":")
                if self.series_stats is not None:
                    value = float(self.series_stats.loc[sim_id, f"{base}_{kind}"])
                else:
                    value = float(getattr(np, kind)(window[:, self._channel_index(base)]))
                if self.norm_stats and base in self.norm_stats:
                    mean, std = self.norm_stats[base]
                    std = std or 1.0
                    value = value / std if kind == "std" else (value - mean) / std
                stats[channel] = value

        if self.crop_length:
            max_start = window.shape[0] - self.crop_length
            offset = int(np.random.randint(0, max_start + 1))
            window = window[offset:offset + self.crop_length]
        elif self.window_length:
            window = window[self.window_offset:self.window_offset +
                            self.window_length]

        section, height = None, 0.0
        if self.height_targets:
            section = (self.section if self.section is not None else int(
                np.random.randint(len(self.height_channels))))
            self.moment_channel = self.height_channels[section]
            height = float(self.height_z_over_h[section])
        inputs = [
            np.full(window.shape[0], height) if channel == "height" else
            np.full(window.shape[0], stats[channel]) if channel in stats else
            self._normalize(channel, window[:, self._channel_index(channel)])
            for channel in self.input_channels
        ]
        condition = np.array([
            float(np.mean(window[:, self._channel_index(channel)]))
            for channel in self.condition_channels
        ])
        if self.norm_stats:
            condition = np.array([
                (value - self.norm_stats[channel][0]) /
                (self.norm_stats[channel][1] or 1.0)
                for channel, value in zip(self.condition_channels, condition)
            ])

        if self.height_targets:
            condition = np.append(condition, height)

        target = window[:, self._channel_index(self.moment_channel)]
        target = self._normalize(self.moment_channel, target - np.mean(target))

        item = {
            "sim_id": sim_id,
            "inputs": torch.from_numpy(np.stack(inputs)).float(),
            "condition": torch.from_numpy(condition).float(),
            "target": torch.from_numpy(target[None, :]).float(),
        }
        if self.calibration is not None:
            gain = self._physics_gain(window)
            if section is not None and self.height_factors:
                gain = gain * self.height_factors[section]
            item["physics_gain"] = torch.from_numpy(gain)
        return item

    def _physics_gain(self, window: np.ndarray) -> np.ndarray:
        """Complex physics gain per rFFT bin of the window, normalized."""
        freqs = np.fft.rfftfreq(window.shape[0],
                                d=1.0 / self.sampling_frequency)
        windows = harmonic_windows(
            window[:, self._channel_index("rotor_speed")],
            window[:, self._channel_index("electrical_power")]
            > self.operating_power_kw)
        low_mask, masks = band_masks(freqs,
                                     self.calibration.transition_frequency(),
                                     windows)
        harmonic_mask = np.zeros_like(freqs, dtype=bool)
        for mask in masks:
            harmonic_mask |= mask
        gains = self.calibration.gain(freqs, low_mask, harmonic_mask)
        if self.norm_stats:
            gains = gains * (self.norm_stats[self.accel_channel][1] /
                             self.norm_stats[self.moment_channel][1])
        return gains.astype(np.complex64)


def compute_norm_stats(release: ReleasedTower,
                       sim_ids: List[int],
                       channels: List[str],
                       min_time: float = MIN_TIME,
                       max_time: float = MAX_TIME,
                       max_sims: int = 200) -> Dict[str, List[float]]:
    """Per-channel [mean, std] over a sample of training simulations.

    Args:
        release (ReleasedTower): Tower of the released dataset.
        sim_ids (List[int]): Training simulation IDs.
        channels (List[str]): Channels to compute statistics for.
        min_time (float): Start of the usable window [s].
        max_time (float): End of the usable window [s], inclusive.
        max_sims (int): Maximum number of simulations sampled (evenly).

    Returns:
        dict: Mapping channel -> [mean, std].
    """
    start = int(round(min_time * release.sampling_frequency))
    stop = int(round(max_time * release.sampling_frequency)) + 1
    indices = [release.channels.index(c) for c in channels]
    step = max(1, len(sim_ids) // max_sims)
    samples = []
    for sim_id in sim_ids[::step]:
        samples.append(release.load(sim_id)[start:stop, indices])
    stacked = np.concatenate(samples, axis=0)
    return {
        channel: [float(stacked[:, i].mean()),
                  float(stacked[:,
                                i].std())] for i, channel in enumerate(channels)
    }
