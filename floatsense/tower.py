# pylint: disable=too-many-instance-attributes
# pylint: disable=too-few-public-methods
"""Tower geometry needed by the damage metric.

A tower is a stack of conical segments given by outer diameters at the
transition heights and one wall thickness per segment (the tower geometry
JSON of the OpenFAST campaign). The damage at a gauge uses the outer radius
at the gauge height and the thickness of the segment that contains it.
"""

import json
from typing import List, Optional, Tuple

import numpy as np


class Tower:
    """Tower geometry from direct arrays or from a structured JSON file.

    Attributes:
        diameter_transitions (np.ndarray): Outer diameters at the transition
          heights [m].
        z_transitions (np.ndarray): Vertical coordinates of the transitions
          [m].
        thickness_sections (np.ndarray): Wall thickness per segment [m].
        radius_transitions (np.ndarray): Outer radii at the transitions [m].
        mean_radius_sections (np.ndarray): Mean outer radius per segment [m].
        mean_z_sections (np.ndarray): Mean height per segment [m].
        num_sections (int): Number of segments.
        height (float): Tower height [m].
    """

    def __init__(self,
                 diameter_transitions: Optional[List[float]] = None,
                 z_transitions: Optional[List[float]] = None,
                 thickness_sections: Optional[List[float]] = None,
                 json_path: Optional[str] = None):
        """Initializes the tower from arrays or from a JSON file.

        Args:
            diameter_transitions (List[float], optional): Outer diameters [m].
            z_transitions (List[float], optional): Transition heights [m].
            thickness_sections (List[float], optional): Thicknesses [m].
            json_path (str, optional): Structured JSON with the three fields
              `diameter_transitions (m)`, `z_transitions (m)` and
              `thickness_sections (m)`, dicts keyed d0.., h0.. and t1..
        """
        if json_path:
            self._load_json(json_path)
        elif (diameter_transitions is not None and z_transitions is not None and
              thickness_sections is not None):
            self.diameter_transitions = np.array(diameter_transitions,
                                                 dtype=float)
            self.z_transitions = np.array(z_transitions, dtype=float)
            self.thickness_sections = np.array(thickness_sections, dtype=float)
        else:
            raise ValueError("Provide either a JSON path or all three arrays.")
        if len(self.diameter_transitions) != len(self.z_transitions):
            raise ValueError("diameter_transitions and z_transitions must "
                             "have the same length.")
        if len(self.thickness_sections) != len(self.z_transitions) - 1:
            raise ValueError("thickness_sections must have length "
                             "len(z_transitions) - 1.")
        self.radius_transitions = self.diameter_transitions / 2
        self.num_sections = len(self.thickness_sections)
        self.height = float(self.z_transitions[-1] - self.z_transitions[0])
        self.mean_radius_sections = (self.radius_transitions[:-1] +
                                     self.radius_transitions[1:]) / 2
        self.mean_z_sections = (self.z_transitions[:-1] +
                                self.z_transitions[1:]) / 2

    def _load_json(self, json_path: str) -> None:
        """Loads the three geometry fields from a structured JSON file."""
        with open(json_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        def ordered(field: str) -> List[float]:
            values = data[field]
            return [values[k] for k in sorted(values, key=lambda x: int(x[1:]))]

        self.diameter_transitions = np.array(
            ordered("diameter_transitions (m)"), dtype=float)
        self.z_transitions = np.array(ordered("z_transitions (m)"), dtype=float)
        self.thickness_sections = np.array(ordered("thickness_sections (m)"),
                                           dtype=float)

    def section_of(self, z: float) -> int:
        """Index of the segment that contains height z [m, absolute]."""
        return int(
            np.clip(
                np.searchsorted(self.z_transitions, z, side="right") - 1, 0,
                self.num_sections - 1))

    def radius_at(self, z: float) -> float:
        """Outer radius at height z [m, absolute], linear within a segment."""
        return float(np.interp(z, self.z_transitions, self.radius_transitions))

    def gauge_properties(
            self, gauge_heights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Outer radius and wall thickness at the gauge heights.

        Args:
            gauge_heights (np.ndarray): Gauge heights above the tower base [m].

        Returns:
            Tuple[np.ndarray, np.ndarray]: Radius at the gauge height [m] and
              thickness of the segment that contains the gauge [m].
        """
        heights = np.asarray(gauge_heights, dtype=float) + self.z_transitions[0]
        radius = np.array([self.radius_at(z) for z in heights])
        thickness = np.array(
            [self.thickness_sections[self.section_of(z)] for z in heights])
        return radius, thickness

    def to_json(self, path: str) -> None:
        """Writes the geometry in the structured JSON layout."""
        data = {
            "diameter_transitions (m)": {
                f"d{i}": float(v)
                for i, v in enumerate(self.diameter_transitions)
            },
            "z_transitions (m)": {
                f"h{i}": float(v) for i, v in enumerate(self.z_transitions)
            },
            "thickness_sections (m)": {
                f"t{i + 1}": float(v)
                for i, v in enumerate(self.thickness_sections)
            },
        }
        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)
