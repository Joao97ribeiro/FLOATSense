# pylint: disable=duplicate-code
# pylint: disable=import-error
# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-locals
"""Standalone base-section fatigue damage for sequence-model evaluation.

Rainflow counting, bilinear SN curve with thickness correction and Miner's
rule on a bending-moment series at one gauge. The same function
scores the physics baseline and every learned model.
"""

from typing import TYPE_CHECKING, List, Optional

import numpy as np
import rainflow
import scipy.signal

from .constants import FATIGUE_LIFE_THRESHOLD
from .constants import LOWPASS_ORDER
from .constants import SN_INTERCEPTS_LOG10
from .constants import SN_SLOPES
from .constants import THICKNESS_EXPONENT
from .constants import THICKNESS_REFERENCE_MM

if TYPE_CHECKING:
    from .release import TowerGauges


def lowpass(series: np.ndarray,
            sampling_frequency: float,
            cutoff_hz: float,
            order: int = LOWPASS_ORDER) -> np.ndarray:
    """Zero-phase Butterworth low-pass (forward and backward, sosfiltfilt).

    Run in both directions, an order-4 filter has the attenuation of an
    order-8 one and no phase shift. Unlike a hard cut of the FFT bins, it
    does not ring at the window edges (which added spurious rainflow
    cycles).

    Args:
        series (np.ndarray): Time series.
        sampling_frequency (float): Sampling frequency [Hz].
        cutoff_hz (float): Cutoff frequency [Hz].
        order (int): Butterworth order of one pass.

    Returns:
        np.ndarray: Filtered series, same length.

    Raises:
        ValueError: If the cutoff is not between 0 and the Nyquist frequency.
    """
    if not 0.0 < cutoff_hz < sampling_frequency / 2:
        raise ValueError(f"Low-pass cutoff {cutoff_hz} Hz must be between 0 "
                         f"and {sampling_frequency / 2} Hz.")
    sos = scipy.signal.butter(order,
                              cutoff_hz,
                              fs=sampling_frequency,
                              output="sos")
    return scipy.signal.sosfiltfilt(sos, np.asarray(series, dtype=float))


def damage_filter(series: np.ndarray,
                  sampling_frequency: float,
                  apply_lowpass: bool,
                  cutoff_hz: float,
                  order: int = LOWPASS_ORDER) -> np.ndarray:
    """Moment fed to the damage metric: low-passed only if `apply_lowpass`.

    Args:
        series (np.ndarray): Moment series.
        sampling_frequency (float): Sampling frequency [Hz].
        apply_lowpass (bool): Whether to low-pass (on in the benchmark: the
          fatigue damage of these towers lies below 3 Hz).
        cutoff_hz (float): Cutoff [Hz] when `apply_lowpass` is True.
        order (int): Butterworth order of one pass.

    Returns:
        np.ndarray: The series, low-passed or unchanged.
    """
    if not apply_lowpass:
        return series
    return lowpass(series, sampling_frequency, cutoff_hz, order)


def compute_base_damage(moment_series: np.ndarray,
                        tower: "TowerGauges",
                        sn_intercepts_log10: Optional[List[float]] = None,
                        sn_slopes: Optional[List[float]] = None,
                        thickness_reference: float = THICKNESS_REFERENCE_MM,
                        thickness_exponent: float = THICKNESS_EXPONENT,
                        fatigue_life_threshold: float = FATIGUE_LIFE_THRESHOLD,
                        gauge: int = 0) -> float:
    """Computes fatigue damage of a moment series at one tower section.

    Args:
        moment_series (np.ndarray): Bending moment time series [kN.m].
        tower (TowerGauges): Outer radius and wall thickness at the
          gauges (radius_gauges, thickness_gauges).
        sn_intercepts_log10 (List[float], optional): SN log10 intercepts.
        sn_slopes (List[float], optional): SN curve slopes.
        thickness_reference (float): Reference thickness for SN [mm].
        thickness_exponent (float): Exponent for SN thickness correction.
        fatigue_life_threshold (float): Cycle threshold for slope switch.
        gauge (int): Gauge index, 0 (base) to 10 (top).

    Returns:
        float: Total fatigue damage (unitless).
    """
    sn_intercepts_log10 = sn_intercepts_log10 or list(SN_INTERCEPTS_LOG10)
    sn_slopes = sn_slopes or list(SN_SLOPES)

    if not 0 <= gauge < len(tower.radius_gauges):
        raise ValueError(f"Gauge index {gauge} is not one of the "
                         f"{len(tower.radius_gauges)} gauges.")
    radius = tower.radius_gauges[gauge]
    thickness = tower.thickness_gauges[gauge]
    inner_radius = radius - thickness
    modulus = (np.pi / 4) * (radius**4 - inner_radius**4) / radius

    # Zero-range cycles add no damage; dropping them avoids log10(0).
    cycles = [cycle for cycle in rainflow.extract_cycles(
        np.asarray(moment_series, dtype=float)) if cycle[0] > 0]
    if not cycles:
        return 0.0
    cycle_counts, moment_ranges = zip(
        *[(cycle[2], cycle[0]) for cycle in cycles])
    cycle_counts = np.array(cycle_counts)
    stress_ranges = 1e-3 * np.array(moment_ranges) / modulus

    thickness_mm = thickness * 1000
    correction = (max(thickness_mm, thickness_reference) /
                  thickness_reference)**thickness_exponent
    corrected_ranges = stress_ranges * correction

    fatigue_life = 10**(sn_intercepts_log10[0] -
                        sn_slopes[0] * np.log10(corrected_ranges))
    mask = fatigue_life > fatigue_life_threshold
    fatigue_life[mask] = 10**(sn_intercepts_log10[1] -
                              sn_slopes[1] * np.log10(corrected_ranges[mask]))

    return float(np.sum(cycle_counts / fatigue_life))
