# pylint: disable=too-many-locals
"""Damage metrics, regime breakdown and cluster bootstrap.

Five numbers score a set of simulations: the R^2 of log10 damage, the
median of the ratio reconstructed/true, the fraction of simulations whose
ratio lies within a factor of two (0.5 <= ratio <= 2), the mean relative
error and the within-condition correlation. The first three carry
confidence intervals; they come from a bootstrap that resamples operating
conditions, not simulations: the six turbulence seeds of one condition are
not independent draws, and resampling them one by one understates the
uncertainty.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

METRIC_NAMES = ("r2_log_damage", "median_damage_ratio",
                "fraction_within_factor2")
CONDITION_COLUMNS = ["wind_speed", "wave_hs", "wave_tp"]


def summarize_damage(damage_true: np.ndarray,
                     damage_rec: np.ndarray) -> Dict[str, float]:
    """The three damage metrics of one set of simulations.

    Simulations where either damage is not positive are dropped.
    """
    true = np.asarray(damage_true, dtype=float)
    rec = np.asarray(damage_rec, dtype=float)
    valid = (true > 0) & (rec > 0)
    log_true, log_rec = np.log10(true[valid]), np.log10(rec[valid])
    residual = np.sum((log_true - log_rec)**2)
    total = np.sum((log_true - np.mean(log_true))**2)
    ratio = rec[valid] / true[valid]
    return {
        "num_sims":
            int(np.sum(valid)),
        "r2_log_damage":
            float(1.0 - residual / total) if total > 0 else float("nan"),
        "median_damage_ratio":
            float(np.median(ratio)),
        "fraction_within_factor2":
            float(np.mean((ratio >= 0.5) & (ratio <= 2.0))),
    }


def mean_relative_error(damage_true: np.ndarray,
                        damage_rec: np.ndarray) -> float:
    """Mean of |D_rec - D_true| / D_true over simulations with D > 0."""
    true = np.asarray(damage_true, dtype=float)
    rec = np.asarray(damage_rec, dtype=float)
    valid = (true > 0) & (rec > 0)
    return float(np.mean(np.abs(rec[valid] - true[valid]) / true[valid]))


def within_condition_correlation(damage_true: np.ndarray,
                                 damage_rec: np.ndarray,
                                 conditions: np.ndarray) -> float:
    """Pearson correlation of log damage deviations from condition means.

    Args:
        damage_true (np.ndarray): True damage per simulation.
        damage_rec (np.ndarray): Reconstructed damage per simulation.
        conditions (np.ndarray): Operating-condition label per simulation.

    Returns:
        float: corr(delta, delta_hat), where delta is log10 D minus its mean
          over the realizations of the same condition. A predictor that
          maps the condition to a damage scores zero.
    """
    true = np.asarray(damage_true, dtype=float)
    rec = np.asarray(damage_rec, dtype=float)
    valid = (true > 0) & (rec > 0)
    frame = pd.DataFrame({"true": np.log10(true[valid]),
                          "rec": np.log10(rec[valid]),
                          "condition": np.asarray(conditions)[valid]})
    means = frame.groupby("condition")[["true", "rec"]].transform("mean")
    deviations = frame[["true", "rec"]] - means
    return float(np.corrcoef(deviations["true"], deviations["rec"])[0, 1])


def condition_key(index_df: pd.DataFrame) -> pd.Series:
    """One label per operating condition (wind, Hs, Tp), shared by seeds."""
    return index_df[CONDITION_COLUMNS].round(6).astype(str).agg("|".join,
                                                                axis=1)


def cluster_bootstrap(damage_true: np.ndarray,
                      damage_rec: np.ndarray,
                      clusters: np.ndarray,
                      num_resamples: int = 1000,
                      seed: int = 0) -> Dict[str, List[float]]:
    """Percentile 95% CI of each metric, resampling clusters with replacement.

    Args:
        damage_true (np.ndarray): True damage per simulation.
        damage_rec (np.ndarray): Reconstructed damage per simulation.
        clusters (np.ndarray): Cluster label per simulation (the operating
          condition); every simulation of a resampled cluster is kept.
        num_resamples (int): Bootstrap resamples.
        seed (int): RNG seed.

    Returns:
        dict: metric -> [low, high].
    """
    rng = np.random.default_rng(seed)
    labels, inverse = np.unique(np.asarray(clusters), return_inverse=True)
    members = [np.flatnonzero(inverse == k) for k in range(len(labels))]
    samples = {name: [] for name in METRIC_NAMES}
    for _ in range(num_resamples):
        picked = rng.integers(0, len(labels), size=len(labels))
        rows = np.concatenate([members[k] for k in picked])
        summary = summarize_damage(damage_true[rows], damage_rec[rows])
        for name in METRIC_NAMES:
            samples[name].append(summary[name])
    return {
        name: [
            float(np.nanpercentile(values, 2.5)),
            float(np.nanpercentile(values, 97.5))
        ] for name, values in samples.items()
    }


def summarize_by_group(df: pd.DataFrame,
                       direction: str,
                       groups: Optional[pd.Series] = None,
                       clusters: Optional[pd.Series] = None,
                       num_resamples: int = 0) -> pd.DataFrame:
    """Metrics of a damage CSV, overall and per regime group.

    Args:
        df (pd.DataFrame): Rows with sim_id, damage_true_<d>, damage_rec_<d>.
        direction (str): 'fa' or 'ss'.
        groups (pd.Series, optional): Group label per sim_id (index) such as
          the wind/wave regime cell; adds one row per group.
        clusters (pd.Series, optional): Condition label per sim_id (index)
          for the bootstrap CIs.
        num_resamples (int): Bootstrap resamples; 0 disables the CIs.

    Returns:
        pd.DataFrame: One row per group ('all' first) with the metrics and,
          with CIs, <metric>_low / <metric>_high columns.
    """
    df = df.set_index("sim_id") if "sim_id" in df.columns else df
    true_col, rec_col = f"damage_true_{direction}", f"damage_rec_{direction}"
    rows = []
    labels = {"all": df.index}
    if groups is not None:
        for label, ids in groups.groupby(groups).groups.items():
            labels[str(label)] = df.index.intersection(ids)
    for label, ids in labels.items():
        subset = df.loc[ids]
        row = {
            "group": label,
            **summarize_damage(subset[true_col].values, subset[rec_col].values)
        }
        if num_resamples and clusters is not None:
            interval = cluster_bootstrap(subset[true_col].values,
                                         subset[rec_col].values,
                                         clusters.loc[subset.index].values,
                                         num_resamples)
            for name, (low, high) in interval.items():
                row[f"{name}_low"], row[f"{name}_high"] = low, high
        rows.append(row)
    return pd.DataFrame(rows)
