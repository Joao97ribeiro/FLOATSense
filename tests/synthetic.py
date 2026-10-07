# pylint: disable=wrong-import-position
# pylint: disable=too-many-locals
"""A tiny synthetic tower in the released layout, for CPU tests.

The series are sums of sines with random phases and amplitudes per
simulation; the moment at each gauge is a scaled, slightly delayed copy of
the acceleration plus noise, so a model can learn it. damage.parquet is
computed with the metric of the benchmark (ReleasedTower.gauge_damage).
"""

import os
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from floatsense import load_tower

TOWER = "syn"
STEMS = ["tower_bottom"] + [f"tower_{i}" for i in range(1, 10)] + ["tower_top"]
NUM_SAMPLES = 10001  # 0 to 1,000 s at 10 Hz


def write_tower(dataset_dir: str,
                num_train: int = 12,
                num_test: int = 6,
                seed: int = 0) -> str:
    """Writes <dataset_dir>/syn and returns the dataset directory.

    Simulations 1..num_train are 'train', the next num_test are 'test'.
    """
    rng = np.random.default_rng(seed)
    tower_dir = os.path.join(dataset_dir, TOWER)
    os.makedirs(tower_dir, exist_ok=True)
    time_s = np.arange(NUM_SAMPLES) / 10.0
    sim_ids = list(range(1, num_train + num_test + 1))
    writer = None
    path = os.path.join(tower_dir, "series-00000-of-00001.parquet")
    for sim_id in sim_ids:
        freqs = np.array([0.05, 0.3, 0.9, 1.6])
        amps = rng.uniform(0.5, 2.0, size=4)
        phases = rng.uniform(0, 2 * np.pi, size=4)
        accel = np.sum(
            amps[:, None] *
            np.sin(2 * np.pi * freqs[:, None] * time_s + phases[:, None]),
            axis=0) + 0.05 * rng.standard_normal(NUM_SAMPLES)
        columns = {
            "sim_id": np.full(NUM_SAMPLES, sim_id, dtype=np.int64),
            "time_s": time_s,
            "tower_top_afa_mod": accel,
            "rotor_speed": 5 + rng.uniform(0, 3) + 0 * time_s,
            "blade_pitch": rng.uniform(0, 10) + 0.1 * np.sin(time_s),
            "wind_speed": rng.uniform(4, 20) + 0.5 * np.sin(0.1 * time_s),
            "electrical_power": np.full(NUM_SAMPLES, 1000.0),
        }
        for gauge, stem in enumerate(STEMS):
            scale = 1e4 * (11 - gauge)
            columns[f"{stem}_mfa"] = scale * (
                np.roll(accel, gauge) + 0.1 * rng.standard_normal(NUM_SAMPLES))
        table = pa.table({
            k: np.asarray(v, dtype=np.float32 if k != "sim_id" else np.int64)
            for k, v in columns.items()
        })
        if writer is None:
            writer = pq.ParquetWriter(path, table.schema)
        writer.write_table(table)
    writer.close()
    pd.DataFrame({
        "sim_id": sim_ids,
        "split": ["train"] * num_train + ["test"] * num_test,
    }).to_parquet(os.path.join(tower_dir, "metadata.parquet"))
    heights = np.linspace(1.0, 150.0, len(STEMS))
    pd.DataFrame({
        "section_id": np.arange(1, 3 * len(STEMS), 3),
        "channel": STEMS,
        "gauge_height_m": heights,
        "z_over_h": heights / heights[-1],
        "gauge_radius_m": np.linspace(5.0, 3.0, len(STEMS)),
        "gauge_thickness_m": np.linspace(0.066, 0.038, len(STEMS)),
    }).to_parquet(os.path.join(tower_dir, "sections.parquet"))
    tower = load_tower(dataset_dir, TOWER)
    rows = []
    for sim_id in sim_ids:
        for gauge, stem in enumerate(STEMS):
            rows.append({
                "sim_id":
                    sim_id,
                "section_id":
                    tower.geometry.section_ids[gauge],
                "damage":
                    tower.gauge_damage(tower.scored_moment(sim_id, stem), stem),
            })
    pd.DataFrame(rows).to_parquet(os.path.join(tower_dir, "damage.parquet"))
    return dataset_dir
