# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-locals
# pylint: disable=no-member
"""Physics baseline: band-gain moment reconstruction of Pimenta et al. (2024).

Reconstructs the tower-base bending moment from the tower-top acceleration
with a frequency-dependent gain made of three bands (Pimenta et al. 2024,
Renewable Energy 223, 119981, Eqs. 21-25, in the form used for this
dataset). With omega = 2*pi*f and C1 the background gain:

  - low frequency, 0 < f <= f_lim:   G(f) = C1 * (C4 + C5 * omega^2)
  - rotor harmonics 3P, 6P, 9P:      G(f) = C1 / (C2 + C3 * omega^2 - i*C_d)
  - elsewhere:                       G(f) = C1

f_lim is where the low-frequency polynomial reaches one. The harmonic
denominator carries a constant imaginary part (the damping term C_d), so it
never reaches zero inside a window and the branch has a phase lag. Both
accelerations include the gravity projection of the platform tilt (Eq. 10).

Calibration, one estimate per training simulation and the median over them:
  1. C1 comes from parked setups: the least-squares slope, with a free
     intercept, of the standard deviation of the moment against that of the
     acceleration over the 22 parked runs of the tower (`parked_constants`);
     it is a property of the structure.
  2. C4, C5: least squares of (|M|/|a|)/C1 = C4 + C5*omega^2 over the bins
     between 0.01 and 0.05 Hz of the Welch PSD ratio.
  3. C2, C3, C_d: one point per harmonic window, the bin of maximum moment
     PSD, through C1^2 * S_a/S_M = (C2 + C3*omega^2)^2 + C_d^2, a quadratic in
     omega^2 that three windows determine exactly.

Scoring: the gain is applied to the series from 300 to 1000 s and the
last 50 s mirrored about the record end, and 50 s cut on each side, so the
scored window is 400-1,000 s (the window of the FLOATBench labels) with a
margin on both sides against the circular convolution; both the reconstruction and the simulated moment
are low-passed at 3 Hz before the damage is computed.
"""

import dataclasses
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.signal
from tqdm import tqdm

from .fatigue import compute_base_damage
from .release import ReleasedTower
from .release import load_parked

# `sign` restores the phase of the reconstruction: the gains are fitted on
# amplitude ratios, so the leading minus of Eq. (21a), lambda = -M*dH*a, is
# lost, and in OpenFAST's conventions the fore-aft moment is in antiphase
# with the gravity-corrected fore-aft acceleration. Damage does not depend
# on it; time-series plots do.
DIRECTION_CHANNELS = {
    "fa": {
        "accel": "tower_top_afa_mod",
        "moment": "tower_bottom_mfa",
        "sign": -1.0
    },
    "ss": {
        "accel": "tower_top_ass_mod",
        "moment": "tower_bottom_mss",
        "sign": 1.0
    },
}
HARMONIC_ORDERS = (3, 6, 9)
PSD_EPS = 1e-30


def _linear_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Fits y = a + b*x by ordinary least squares; returns (a, b)."""
    design = np.stack([np.ones_like(x), x], axis=1)
    coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(coeffs[0]), float(coeffs[1])


def lowpass(series: np.ndarray, sampling_frequency: float,
            cutoff_hz: float) -> np.ndarray:
    """Zeroes every rFFT bin above `cutoff_hz` (no-op when cutoff is 0)."""
    if not cutoff_hz:
        return series
    spectrum = np.fft.rfft(series)
    freqs = np.fft.rfftfreq(len(series), d=1.0 / sampling_frequency)
    spectrum[freqs > cutoff_hz] = 0.0
    return np.fft.irfft(spectrum, n=len(series))


@dataclasses.dataclass
class Calibration:
    """Band gains mapping tower-top acceleration to base moment.

    Attributes:
        c_theta (float): Background gain C1 [kN.s^2].
        c_m_l (float): Low-frequency constant C4.
        c_omega_l (float): Low-frequency omega^2 term C5.
        c_r_h (float): Harmonic constant C2.
        c_omega_h (float): Harmonic omega^2 term C3.
        c_d_h (float): Harmonic damping term C_d.
        lf_cutoff (float): Fallback low-frequency limit [Hz] when the
          polynomial never reaches one.
    """

    c_theta: float
    c_m_l: float = 1.0
    c_omega_l: float = 0.0
    c_r_h: float = 1.0
    c_omega_h: float = 0.0
    c_d_h: float = 0.0
    lf_cutoff: float = 0.05

    def transition_frequency(self) -> float:
        """Frequency [Hz] where C4 + C5*omega^2 = 1, else `lf_cutoff`."""
        if self.c_omega_l == 0.0:
            return self.lf_cutoff
        omega_sq = (1.0 - self.c_m_l) / self.c_omega_l
        if not np.isfinite(omega_sq) or omega_sq <= 0.0:
            return self.lf_cutoff
        return float(np.sqrt(omega_sq) / (2.0 * np.pi))

    def gain(self, freqs: np.ndarray, low_mask: np.ndarray,
             harmonic_mask: np.ndarray) -> np.ndarray:
        """Complex gain [kN.s^2] of every frequency bin."""
        omega_sq = (2.0 * np.pi * freqs)**2
        gains = np.full(freqs.shape, self.c_theta, dtype=complex)
        gains[low_mask] = self.c_theta * (self.c_m_l +
                                          self.c_omega_l * omega_sq[low_mask])
        poly = self.c_r_h + self.c_omega_h * omega_sq[harmonic_mask]
        gains[harmonic_mask] = self.c_theta / (poly - 1j * self.c_d_h)
        gains[freqs == 0.0] = 0.0
        return gains

    def to_json(self, path: str, extra: Optional[dict] = None) -> None:
        """Saves the calibration, plus optional metadata, to JSON."""
        data = dataclasses.asdict(self)
        if extra:
            data["meta"] = extra
        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)

    @classmethod
    def from_json(cls, path: str) -> "Calibration":
        """Loads a calibration from JSON, ignoring metadata keys."""
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        names = {field.name for field in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})


def harmonic_windows(
        rotor_speed_rpm: np.ndarray,
        operating: np.ndarray,
        orders: Tuple[int, ...] = HARMONIC_ORDERS) -> List[Tuple[float, float]]:
    """Rotor harmonic windows [Hz], order * [min, max] rotor frequency.

    Only the samples where the turbine generates count; a setup that never
    operates has no harmonic window at all.
    """
    if np.sum(operating) < 2:
        return []
    speeds = rotor_speed_rpm[operating] / 60.0
    f_low, f_high = float(np.min(speeds)), float(np.max(speeds))
    return [(order * f_low, order * f_high) for order in orders]


def band_masks(
        freqs: np.ndarray, lf_cutoff: float,
        windows: List[Tuple[float,
                            float]]) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Low-frequency mask and one mask per harmonic window, all disjoint.

    The low-frequency band takes precedence, and where two windows overlap
    the lower order wins.
    """
    low_mask = (freqs > 0.0) & (freqs <= lf_cutoff)
    taken = low_mask.copy()
    masks = []
    for f_min, f_max in windows:
        mask = (freqs >= f_min) & (freqs <= f_max) & ~taken
        masks.append(mask)
        taken |= mask
    return low_mask, masks


GAUGE_STEMS = (["tower_bottom"] + [f"tower_{k}" for k in range(1, 10)] +
               ["tower_top"])


def parked_constants(dataset_dir: str, tower: str,
                     min_time: float = 400.0) -> pd.DataFrame:
    """Parked C1 of one tower at the 11 gauges, per direction.

    C1 is the least-squares slope, with a free intercept, of the standard
    deviation of the moment against that of the gravity-corrected tower-top
    acceleration over the 22 parked runs, from `min_time` to the end of the
    record (Pimenta et al., 2024).

    Args:
        dataset_dir (str): Released dataset directory (parked.parquet).
        tower (str): Tower name.
        min_time (float): Start of the scored window [s].

    Returns:
        pd.DataFrame: One row per gauge, base to top, with columns
          'gauge' and C1 [kN.s^2] in 'fa' and 'ss'.
    """
    runs = load_parked(dataset_dir, tower)
    channels = runs.pop("_channels")
    start = int(min_time * 10.0)
    stds = pd.DataFrame([{name: float(data[start:, i].std())
                          for i, name in enumerate(channels)}
                         for data in runs.values()])
    rows = []
    for stem in GAUGE_STEMS:
        row = {"gauge": stem}
        for direction, config in DIRECTION_CHANNELS.items():
            row[direction] = float(np.polyfit(stds[config["accel"]],
                                              stds[f"{stem}_m{direction}"],
                                              1)[0])
        rows.append(row)
    return pd.DataFrame(rows)


def parked_c_theta(dataset_dir: str, tower: str) -> Dict[str, float]:
    """Base C1 per direction [kN.s^2], rounded to 0.1 as in the paper."""
    base = parked_constants(dataset_dir, tower).iloc[0]
    return {d: round(float(base[d]), 1) for d in DIRECTION_CHANNELS}


class PhysicsReconstruction:
    """Calibrates and evaluates the band-gain baseline on the release."""

    def __init__(self,
                 release: ReleasedTower,
                 output_dir: str,
                 parked_c_theta: Dict[str, float],
                 min_time: float = 400.0,
                 max_time: float = 1000.0,
                 pad_seconds: float = 50.0,
                 segment_length: int = 4096,
                 lf_fit_band: Tuple[float, float] = (0.01, 0.05),
                 operating_power_kw: float = 100.0,
                 lowpass_hz: float = 3.0,
                 sn_intercepts_log10: Optional[List[float]] = None,
                 sn_slopes: Optional[List[float]] = None):
        """Initializes the baseline.

        Args:
            release (ReleasedTower): Tower of the released dataset; its
              sections give the damage metric.
            output_dir (str): Where calibrations, CSVs and summaries go.
            parked_c_theta (Dict[str, float]): C1 per direction [kN.s^2],
              measured on parked setups of this tower.
            min_time (float): Start of the scored window [s].
            max_time (float): End of the scored window [s].
            pad_seconds (float): Margin on each side of the scored window
              over which the gain is applied and then cut away. Before the
              window it comes from the record; after it, where the record
              ends, the last `pad_seconds` are mirrored about the end.
            segment_length (int): Welch segment length of the calibration.
            lf_fit_band (Tuple[float, float]): Bins [Hz] fitting C4 and C5.
            operating_power_kw (float): Generated power above which a sample
              counts as operating, for the rotor-speed windows.
            lowpass_hz (float): Low-pass cutoff applied to both the
              reconstruction and the simulated moment before the damage.
            sn_intercepts_log10 (List[float], optional): SN log10 intercepts.
            sn_slopes (List[float], optional): SN slopes.
        """
        self.release = release
        self.tower = release.geometry
        self.output_dir = output_dir
        self.parked_c_theta = dict(parked_c_theta)
        self.min_time = min_time
        self.max_time = max_time
        self.pad_seconds = pad_seconds
        self.segment_length = segment_length
        self.lf_fit_band = tuple(lf_fit_band)
        self.operating_power_kw = operating_power_kw
        self.lowpass_hz = lowpass_hz
        self.sn_intercepts_log10 = sn_intercepts_log10
        self.sn_slopes = sn_slopes
        self.channels = release.channels
        self.sampling_frequency = release.sampling_frequency

    def _index(self, seconds: float) -> int:
        return int(round(seconds * self.sampling_frequency))

    @property
    def pad_samples(self) -> int:
        """Samples of margin on each side of the scored window."""
        return self._index(self.pad_seconds)

    def load(self, sim_id: int) -> np.ndarray:
        """Loads the full (num_samples, num_channels) array of one sim."""
        return self.release.load(sim_id).astype(float)

    def series(self,
               data: np.ndarray,
               channel: str,
               start: float,
               stop: Optional[float] = None) -> np.ndarray:
        """One channel between `start` and `stop` seconds (inclusive)."""
        column = data[:, self.channels.index(channel)]
        end = None if stop is None else self._index(stop) + 1
        return column[self._index(start):end]

    def scored(self, data: np.ndarray, channel: str) -> np.ndarray:
        """One channel over the scored window."""
        return self.series(data, channel, self.min_time, self.max_time)

    def windows(self, data: np.ndarray) -> List[Tuple[float, float]]:
        """Harmonic windows of one simulation from its operating samples."""
        rotor = self.series(data, "rotor_speed", self.min_time)
        power = self.series(data, "electrical_power", self.min_time)
        return harmonic_windows(rotor, power > self.operating_power_kw)

    def psd(self, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Welch PSD with the configured segment length."""
        return scipy.signal.welch(values,
                                  fs=self.sampling_frequency,
                                  nperseg=self.segment_length)

    def estimate(self, data: np.ndarray, direction: str) -> Dict[str, float]:
        """Per-simulation estimates of C4, C5 and (C2, C3, C_d)."""
        config = DIRECTION_CHANNELS[direction]
        c_theta = self.parked_c_theta[direction]
        freqs, psd_accel = self.psd(
            self.series(data, config["accel"], self.min_time))
        _, psd_moment = self.psd(
            self.series(data, config["moment"], self.min_time))
        ratio = np.sqrt(psd_moment / np.maximum(psd_accel, PSD_EPS))
        omega_sq = (2.0 * np.pi * freqs)**2
        estimates = {
            "c_m_l": np.nan,
            "c_omega_l": np.nan,
            "c_r_h": np.nan,
            "c_omega_h": np.nan,
            "c_d_h": np.nan
        }
        low_mask, masks = band_masks(freqs, self.lf_fit_band[1],
                                     self.windows(data))
        low_fit = low_mask & (freqs >= self.lf_fit_band[0])
        if np.sum(low_fit) >= 3:
            estimates["c_m_l"], estimates["c_omega_l"] = _linear_fit(
                omega_sq[low_fit], ratio[low_fit] / c_theta)
        # One point per window, the bin where the moment PSD peaks, through
        # C1^2 * S_a / S_M = (C2 + C3*w^2)^2 + C_d^2, quadratic in w^2.
        points_x, points_y = [], []
        for mask in masks:
            if not np.any(mask):
                continue
            index = np.flatnonzero(mask)[np.argmax(psd_moment[mask])]
            points_x.append(omega_sq[index])
            points_y.append((c_theta / max(ratio[index], PSD_EPS))**2)
        if len(points_x) >= 3:
            quad, linear, constant = np.polyfit(np.array(points_x),
                                                np.array(points_y), 2)
            if quad > 0.0:
                c_omega_h = float(np.sqrt(quad))
                c_r_h = float(linear / (2.0 * c_omega_h))
                estimates["c_r_h"], estimates["c_omega_h"] = c_r_h, c_omega_h
                damping_sq = constant - c_r_h**2
                if damping_sq > 0.0:
                    estimates["c_d_h"] = float(np.sqrt(damping_sq))
        return estimates

    def calibrate(self,
                  direction: str,
                  sim_ids: List[int],
                  save: bool = True) -> Calibration:
        """Calibrates one direction as the median of per-sim estimates."""
        rows = [{
            "sim_id": sim_id,
            **self.estimate(self.load(sim_id), direction)
        } for sim_id in tqdm(sim_ids, desc=f"Calibrating {direction}")]
        estimates_df = pd.DataFrame(rows)
        median = estimates_df.drop(columns="sim_id").median()
        calibration = Calibration(c_theta=self.parked_c_theta[direction],
                                  c_m_l=float(median["c_m_l"]),
                                  c_omega_l=float(median["c_omega_l"]),
                                  c_r_h=float(median["c_r_h"]),
                                  c_omega_h=float(median["c_omega_h"]),
                                  c_d_h=float(median["c_d_h"]),
                                  lf_cutoff=self.lf_fit_band[1])
        if save:
            os.makedirs(self.output_dir, exist_ok=True)
            estimates_df.to_csv(os.path.join(self.output_dir,
                                             f"estimates_{direction}.csv"),
                                index=False)
            calibration.to_json(os.path.join(self.output_dir,
                                             f"calibration_{direction}.json"),
                                extra={
                                    "num_sims":
                                        len(sim_ids),
                                    "transition_frequency_hz":
                                        calibration.transition_frequency(),
                                    "segment_length":
                                        self.segment_length,
                                    "estimator":
                                        "per_simulation_median",
                                    "c_theta_source":
                                        "parked",
                                })
        return calibration

    def reconstruct(self, data: np.ndarray, direction: str,
                    calibration: Calibration) -> np.ndarray:
        """Zero-mean reconstructed moment [kN.m] over the scored window."""
        pad = self.pad_samples
        accel = self.series(data, DIRECTION_CHANNELS[direction]["accel"],
                            self.min_time - self.pad_seconds, self.max_time)
        # Mirror the tail about the record end so the transform has a
        # margin on both sides of the scored window.
        accel = np.concatenate([accel, accel[-2:-pad - 2:-1]])
        accel = accel - np.mean(accel)
        num_samples = len(accel)
        freqs = np.fft.rfftfreq(num_samples, d=1.0 / self.sampling_frequency)
        low_mask, masks = band_masks(freqs, calibration.transition_frequency(),
                                     self.windows(data))
        harmonic_mask = np.zeros_like(freqs, dtype=bool)
        for mask in masks:
            harmonic_mask |= mask
        gains = calibration.gain(freqs, low_mask, harmonic_mask)
        if self.lowpass_hz:
            gains[freqs > self.lowpass_hz] = 0.0
        moment = np.fft.irfft(np.fft.rfft(accel) * gains, n=num_samples)
        return DIRECTION_CHANNELS[direction]["sign"] * moment[pad:num_samples - pad]

    def damage(self, moment: np.ndarray) -> float:
        """Base-section damage of a moment series."""
        return compute_base_damage(moment, self.tower, self.sn_intercepts_log10,
                                   self.sn_slopes)

    def evaluate(self,
                 sim_ids: List[int],
                 calibrations: Dict[str, Calibration],
                 csv_name: str = "damage_comparison.csv") -> pd.DataFrame:
        """Damage of true and reconstructed moments for every simulation.

        Returns:
            pd.DataFrame with sim_id and, per direction, damage_true_<d>,
              damage_rec_<d> and var_ratio_<d>; also written to `csv_name`.
        """
        rows = []
        for sim_id in tqdm(sim_ids, desc="Evaluating physics"):
            data = self.load(sim_id)
            row = {"sim_id": sim_id}
            for direction, calibration in calibrations.items():
                # The reconstruction is already low-passed through its gain;
                # the simulated moment gets the same filter here.
                moment_true = lowpass(
                    self.scored(data, DIRECTION_CHANNELS[direction]["moment"]),
                    self.sampling_frequency, self.lowpass_hz)
                moment_rec = self.reconstruct(data, direction, calibration)
                row[f"damage_true_{direction}"] = self.damage(moment_true)
                row[f"damage_rec_{direction}"] = self.damage(moment_rec)
                row[f"var_ratio_{direction}"] = float(
                    np.var(moment_rec) / np.var(moment_true))
            rows.append(row)
        df = pd.DataFrame(rows)
        os.makedirs(self.output_dir, exist_ok=True)
        df.to_csv(os.path.join(self.output_dir, csv_name), index=False)
        return df
