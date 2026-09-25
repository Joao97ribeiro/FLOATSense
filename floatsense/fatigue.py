# pylint: disable=duplicate-code
# pylint: disable=import-error
# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-locals
"""Standalone base-section fatigue damage for sequence-model evaluation.

Rainflow counting, bilinear SN curve with thickness correction and Miner's
rule on a bending-moment series at one tower section. The same function
scores the physics baseline and every learned model.
"""

from typing import List, Optional

import numpy as np
import rainflow

from .release import TowerSections


def compute_base_damage(moment_series: np.ndarray,
                        tower: TowerSections,
                        sn_intercepts_log10: Optional[List[float]] = None,
                        sn_slopes: Optional[List[float]] = None,
                        thickness_reference: float = 25.0,
                        thickness_exponent: float = 0.2,
                        fatigue_life_threshold: float = 1e7,
                        section: int = 0) -> float:
    """Computes fatigue damage of a moment series at one tower section.

    Args:
        moment_series (np.ndarray): Bending moment time series [kN.m].
        tower (TowerSections): Mean outer radius and wall thickness of the
          FLOATBench sections.
        sn_intercepts_log10 (List[float], optional): SN log10 intercepts.
        sn_slopes (List[float], optional): SN curve slopes.
        thickness_reference (float): Reference thickness for SN [mm].
        thickness_exponent (float): Exponent for SN thickness correction.
        fatigue_life_threshold (float): Cycle threshold for slope switch.
        section (int): Zero-based FLOATBench section (section_id - 1).

    Returns:
        float: Total fatigue damage (unitless).
    """
    sn_intercepts_log10 = sn_intercepts_log10 or [12.010, 15.350]
    sn_slopes = sn_slopes or [3, 5]

    radius = tower.mean_radius_sections[section]
    thickness = tower.thickness_sections[section]
    inner_radius = radius - thickness
    modulus = (np.pi / 4) * (radius**4 - inner_radius**4) / radius

    cycles = rainflow.extract_cycles(np.asarray(moment_series, dtype=float))
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
