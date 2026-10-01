# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-locals
"""Physics reconstruction along the tower (full-tower extension).

The band-gain baseline reconstructs the base moment. Higher up the tower
the same acceleration is multiplied by a height factor f(z) = M(z)/M(0)
derived from the inertial model of Pimenta et al. (2024): the moment at
height z is the first mass moment of everything above z (rotor-nacelle
assembly plus the tower mass distribution) times the tower-top
acceleration. The one free parameter, the effective lever arm of the RNA
above the tower top, is calibrated so that the profile passes through the
parked C_theta measured at the highest instrumented section (two-point
calibration, base and top gauge). The harmonic and low-frequency constants
stay those of the base.

The tower mass per unit length is the ElastoDyn tower mass density
(TMassDen) of each tower, shipped in `towers/<tower>_mass.csv`.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .physics import PhysicsReconstruction
from .physics import damage_filter
from .physics import parked_constants

BASE_Z, TOP_Z = 15.0, 164.386
HEIGHT = TOP_Z - BASE_Z
# Rotor-nacelle assembly of the IEA 22 MW (ElastoDyn inputs).
NAC_MASS, NAC_CM_Z = 821239.8004933242, 4.2647901842947595
HUB_MASS, YAW_MASS, BLADE_MASS = 120447.70224890654, 28740.99049474962, 82427.5
TWR2SHFT, OVERHANG, SHFT_TILT = 4.142540706280534, -14.07711591388923, -6.0
# Target channels along the tower and the tower section each one scores.
# Gauge k sits at ElastoDyn node 3k, i.e. section 3k-1 of the 30 sections.
HEIGHT_CHANNELS = ([("tower_bottom", 0)] +
                   [(f"tower_{k}", 3 * k - 1) for k in range(1, 10)] +
                   [("tower_top", 29)])
GAUGE_Z = np.array([
    0.0, 12.4488, 27.3874, 42.3260, 57.2646, 72.2032, 87.1418, 102.0804,
    117.0190, 131.9576, HEIGHT
])


def rna_properties() -> Tuple[float, float]:
    """(M_rna, h): RNA mass and its centre of mass above the tower top."""
    apex = TWR2SHFT + abs(OVERHANG) * np.sin(np.radians(abs(SHFT_TILT)))
    rotor = HUB_MASS + 3 * BLADE_MASS
    mass = rotor + NAC_MASS + YAW_MASS
    return mass, (rotor * apex + NAC_MASS * NAC_CM_Z) / mass


def height_factor(z: np.ndarray, s: np.ndarray, m: np.ndarray, h_rna: float,
                  mass_rna: float) -> np.ndarray:
    """f(z) = M(z)/M(0) of the inertial model, tower mass included."""

    def upper(power):
        weight = m * s**power
        seg = 0.5 * (weight[1:] + weight[:-1]) * np.diff(s)
        return np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])

    first, second = upper(1), upper(2)

    def moment(zz):
        return (
            mass_rna * (1 + h_rna / HEIGHT) * (HEIGHT + h_rna - zz) +
            (np.interp(zz, s, second) - zz * np.interp(zz, s, first)) / HEIGHT)

    return moment(np.asarray(z, float)) / moment(0.0)


TOWERS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                          "towers")


def calibrate_profile(dataset_dir: str,
                      tower: str,
                      direction: str = "fa",
                      mass_csv: Optional[str] = None) -> Dict:
    """Height factors of a tower from its mass and parked profile.

    Args:
        dataset_dir (str): Released dataset directory (parked runs).
        tower (str): Tower name.
        direction (str): 'fa' or 'ss'.
        mass_csv (str, optional): Two-column CSV (height above the base [m],
          mass per length [kg/m]); defaults to towers/<tower>_mass.csv.

    Returns:
        dict: The effective RNA lever arm, the factor at the 11 target
          heights and the parked profile it was fitted to.
    """
    mass = pd.read_csv(mass_csv or
                       os.path.join(TOWERS_DIR, f"{tower}_mass.csv"))
    s, m = mass.iloc[:, 0].values, mass.iloc[:, 1].values
    parked = parked_constants(dataset_dir, tower)
    parked_f = (parked[direction] / parked[direction].iloc[0]).values
    anchor = len(parked_f) - 1
    mass_rna, h_nominal = rna_properties()
    h_eff = brentq(
        lambda hh: height_factor(GAUGE_Z[anchor], s, m, hh, mass_rna) -
        parked_f[anchor], -100.0, 5000.0)
    return {
        "h_rna_nominal": float(h_nominal),
        "h_rna_effective": float(h_eff),
        "factors": height_factor(GAUGE_Z, s, m, h_eff, mass_rna).tolist(),
        "parked_factors": parked_f.tolist(),
        "heights": GAUGE_Z.tolist(),
    }


def evaluate_heights(physics: PhysicsReconstruction, sim_ids: List[int],
                     calibration, factors: List[float], direction: str,
                     csv_path: str) -> pd.DataFrame:
    """Damage of true and reconstructed moments at the 11 heights."""
    rows = []
    fs = physics.sampling_frequency
    for sim_id in sim_ids:
        data = physics.load(sim_id)
        base = physics.reconstruct(data, direction, calibration)
        row = {"sim_id": sim_id}
        for (channel, section), factor in zip(HEIGHT_CHANNELS, factors):
            name = f"{channel}_m{direction}"
            if name not in physics.channels:
                continue
            true = damage_filter(physics.scored(data, name), fs,
                                 physics.apply_lowpass, physics.lowpass_hz)
            rec = damage_filter(base * factor, fs, physics.apply_lowpass,
                                physics.lowpass_hz)
            row[f"damage_true_{channel}"] = physics.tower.damage(
                true, section, physics.sn_intercepts_log10, physics.sn_slopes)
            row[f"damage_rec_{channel}"] = physics.tower.damage(
                rec, section, physics.sn_intercepts_log10, physics.sn_slopes)
        rows.append(row)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df
