# pylint: disable=wrong-import-position
"""Tests of the damage metric.

Run from the repository root with `python -m unittest discover tests`.
The dataset test runs when FLOATSENSE_DATA points to a released dataset
(e.g. data/FLOATSense) and is skipped otherwise.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from floatsense.fatigue import damage_filter
from floatsense.fatigue import lowpass

FS = 10.0


class LowpassTest(unittest.TestCase):
    """The metric low-pass."""

    def test_rejects_invalid_cutoff(self):
        """A cutoff at 0, below 0 or at Nyquist raises."""
        series = np.random.default_rng(0).standard_normal(6001)
        for cutoff in (0.0, -1.0, FS / 2):
            with self.assertRaises(ValueError):
                lowpass(series, FS, cutoff)

    def test_off_returns_the_series(self):
        """With the filter off the series is returned unchanged."""
        series = np.random.default_rng(0).standard_normal(6001)
        self.assertIs(damage_filter(series, FS, False, 3.0), series)

    def test_keeps_low_and_removes_high_frequencies(self):
        """A 0.5 Hz sine passes, a 4 Hz sine is removed; same length."""
        time = np.arange(6001) / FS
        low = np.sin(2 * np.pi * 0.5 * time)
        high = np.sin(2 * np.pi * 4.0 * time)
        filtered = damage_filter(low + high, FS, True, 3.0)
        self.assertEqual(len(filtered), len(low))
        self.assertLess(np.max(np.abs(filtered - low)[100:-100]), 0.05)


@unittest.skipUnless(os.environ.get("FLOATSENSE_DATA"),
                     "set FLOATSENSE_DATA to a released dataset")
class ReleasedDamageTest(unittest.TestCase):
    """The released labels against the damage of the released series."""

    def test_gauge_damage_matches_damage_parquet(self):
        """gauge_damage(scored_moment) equals damage.parquet."""
        from floatsense import load_tower  # pylint: disable=import-outside-toplevel
        tower = load_tower(os.environ["FLOATSENSE_DATA"], "opt2")
        damage = tower.damage()
        for sim_id in tower.sim_ids[:5]:
            for gauge, section_id in zip(tower.sections["channel"],
                                         tower.sections["section_id"]):
                value = tower.gauge_damage(tower.scored_moment(sim_id, gauge),
                                           gauge)
                self.assertAlmostEqual(value / damage.loc[sim_id, section_id],
                                       1.0,
                                       places=9)


if __name__ == "__main__":
    unittest.main()
