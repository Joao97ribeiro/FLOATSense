# pylint: disable=wrong-import-position
# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-locals
"""Decision rule of the validation-tuned track: does the ranking hold?

Compares two rankings of the 20 learned entries on the test split: the
fixed-budget one of the paper runs and the tuned one (hpo/final.py,
sealed). Each entry scores the R^2 of log10 damage at the top gauge and as
the mean over the 11 gauges; per tower the median over the seeds, then the
mean over the three towers. The statistic is Kendall's tau-b between the two
rankings, with a 95% percentile bootstrap that resamples the operating
points (clusters common to the towers and to both rankings) and, within
each entry and tower, the seeds:

    lower bound > TAU_AGREE      the ranking holds
    upper bound < TAU_ARTIFACT   the ranking is an artifact
    otherwise                    inconclusive

Also reported: the top-3 and top-5 overlaps (consistent from TOP_K), the
entries above the floor (naive gain) at the top gauge in each ranking, and
the guard: an entry whose tuned winner validates below its fixed-budget
validation score (phase 2 against --fixed_val_root) is flagged, and the
verdict is also given without the flagged entries.

    python hpo/analyze.py --paper_root=outputs/within \
        --tuned_root=outputs/hpo/sealed/last --dataset_dir=data/FLOATSense \
        --hpo_root=outputs/hpo --fixed_val_root=outputs/val_select
    python hpo/analyze.py --dry_run --out=/tmp/analyze   # synthetic rankings

Both roots use the layout of the within-tower runs:
<root>/<tower>/seed<k>/damage_comparison_<model>_fa.csv.
"""

import argparse
import glob
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from floatsense.metrics import condition_key
from hpo import common
from hpo import confirm
from hpo import search_space as S

GAUGES = ["tower_bottom"] + [f"tower_{i}" for i in range(1, 10)] + ["tower_top"]
METRICS = ("top", "mean11")
FLOOR = "naive"


def kendall_b(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Kendall tau-b along axis 1 of two (rows, entries) arrays."""
    i, j = np.triu_indices(x.shape[1], 1)
    sx, sy = np.sign(x[:, i] - x[:, j]), np.sign(y[:, i] - y[:, j])
    den = np.sqrt((sx != 0).sum(1) * (sy != 0).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (sx * sy).sum(1) / den


def topk_overlap(x: np.ndarray, y: np.ndarray, k: int) -> int:
    """Entries shared by the top-k of two score vectors."""
    return len(set(np.argsort(-x)[:k]) & set(np.argsort(-y)[:k]))


def verdict(low: float, high: float) -> str:
    """The pre-registered reading of a tau-b interval."""
    if low > S.TAU_AGREE:
        return "holds"
    if high < S.TAU_ARTIFACT:
        return "artifact"
    return "inconclusive"


def clusters(dataset_dir: str, towers: List[str],
             sim_ids: np.ndarray) -> Tuple[np.ndarray, int]:
    """Operating point of each simulation, checked equal on the towers."""
    keys = None
    for tower in towers:
        meta = pd.read_parquet(
            os.path.join(dataset_dir, tower,
                         "metadata.parquet")).set_index("sim_id").loc[sim_ids]
        key = condition_key(meta).to_numpy()
        if keys is not None and not (key == keys).all():
            raise ValueError("operating points differ between towers")
        keys = key
    labels, inverse = np.unique(keys, return_inverse=True)
    return inverse, len(labels)


def cluster_stats(path: str, sim_ids: np.ndarray, inverse: np.ndarray,
                  n_clusters: int) -> np.ndarray:
    """(4, gauge, cluster) sums of one damage CSV: n, sum y, sum y^2 and sum
    of squared residuals (y = log10 D centred per gauge; simulations where
    either damage is not positive are dropped, as in summarize_damage)."""
    df = pd.read_csv(path).set_index("sim_id").loc[sim_ids]
    out = np.zeros((4, len(GAUGES), n_clusters))
    for g, gauge in enumerate(GAUGES):
        true = df[f"damage_true_{gauge}"].to_numpy(float)
        rec = df[f"damage_rec_{gauge}"].to_numpy(float)
        ok = (true > 0) & (rec > 0) & np.isfinite(rec)
        log_true, log_rec = np.log10(true[ok]), np.log10(rec[ok])
        centre = log_true.mean()
        cluster = inverse[ok]
        out[0, g] = np.bincount(cluster, minlength=n_clusters)
        out[1, g] = np.bincount(cluster, log_true - centre, n_clusters)
        out[2, g] = np.bincount(cluster, (log_true - centre)**2, n_clusters)
        out[3, g] = np.bincount(cluster, (log_true - log_rec)**2, n_clusters)
    return out


def r2_rows(weights: np.ndarray, stats: np.ndarray) -> np.ndarray:
    """R^2 of log10 D per (bootstrap row, gauge) for cluster weights."""
    n, sy, syy, sres = (weights @ stats[i].T for i in range(4))
    with np.errstate(invalid="ignore", divide="ignore"):
        return 1.0 - sres / (syy - sy**2 / n)


def csv_path(root: str, tower: str, seed: int, model: str) -> str:
    """Damage CSV of one run."""
    return os.path.join(root, tower, f"seed{seed}",
                        f"damage_comparison_{model}_fa.csv")


def arm_scores(root: str, models: List[str], towers: List[str],
               seeds: List[int], weights: np.ndarray, sim_ids: np.ndarray,
               inverse: np.ndarray, n_clusters: int) -> Dict[str, np.ndarray]:
    """{metric: (rows, model, tower, seed)} R^2, NaN where a run is missing;
    row 0 is the point estimate."""
    shape = (weights.shape[0], len(models), len(towers), len(seeds))
    scores = {m: np.full(shape, np.nan) for m in METRICS}
    for i, model in enumerate(models):
        for j, tower in enumerate(towers):
            for k, seed in enumerate(seeds):
                path = csv_path(root, tower, seed, model)
                if not os.path.exists(path):
                    continue
                r2 = r2_rows(weights,
                             cluster_stats(path, sim_ids, inverse, n_clusters))
                scores["top"][:, i, j, k] = r2[:, -1]
                scores["mean11"][:, i, j, k] = r2.mean(1)
    return scores


def aggregate(scores: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """(rows, model): median over the seeds per tower, mean over towers.
    Bootstrap rows (1..) draw the seeds with replacement."""
    rows, models, towers, seeds = scores.shape
    pick = rng.integers(0, seeds, size=(rows, models, towers, seeds))
    pick[0] = np.arange(seeds)
    drawn = np.take_along_axis(scores, pick, axis=3)
    return np.nanmean(np.nanmedian(drawn, axis=3), axis=2)


def interval(values: np.ndarray) -> Tuple[float, float, float]:
    """Point (row 0) and percentile 95% interval over the other rows."""
    boot = values[1:][np.isfinite(values[1:])]
    return (float(values[0]), float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5)))


def compare(paper: np.ndarray, tuned: np.ndarray, names: List[str]) -> Dict:
    """tau-b with its interval and verdict, and the top-k overlaps."""
    tau, low, high = interval(kendall_b(paper, tuned))
    out = {
        "n": len(names),
        "tau_b": tau,
        "low": low,
        "high": high,
        "verdict": verdict(low, high)
    }
    for k, need in S.TOP_K.items():
        overlap = topk_overlap(paper[0], tuned[0], k)
        out[f"top{k}"] = overlap
        out[f"top{k}_consistent"] = overlap >= need
    out["paper_order"] = [names[i] for i in np.argsort(-paper[0])]
    out["tuned_order"] = [names[i] for i in np.argsort(-tuned[0])]
    return out


def fixed_val_score(root: str, model: str, tower: str,
                    dataset_dir: str) -> float:
    """Validation R^2 (mean of 11 gauges) of the fixed-budget runs, median
    over the seeds found."""
    values = []
    for path in sorted(
            glob.glob(
                os.path.join(root, tower, "seed*",
                             f"damage_comparison_{model}_fa.csv"))):
        sim_ids = pd.read_csv(path, usecols=["sim_id"])["sim_id"].to_numpy()
        inverse, n_clusters = clusters(dataset_dir, [tower], sim_ids)
        r2 = r2_rows(np.ones((1, n_clusters)),
                     cluster_stats(path, sim_ids, inverse, n_clusters))
        values.append(float(r2.mean()))
    return float(np.median(values)) if values else np.nan


def guard(hpo_root: str, fixed_root: str, models: List[str], towers: List[str],
          dataset_dir: str) -> pd.DataFrame:
    """Tuned winner (phase-2 median) vs fixed-budget validation score."""
    rows = []
    for model in models:
        for tower in towers:
            record = common.read_json(
                confirm.winner_path(hpo_root, model, tower)) or {}
            tuned = record.get("winner_median", np.nan)
            fixed = fixed_val_score(fixed_root, model, tower, dataset_dir)
            rows.append({
                "model": model,
                "tower": tower,
                "tuned_val": tuned,
                "fixed_val": fixed,
                "flag": bool(tuned < fixed)
            })
    return pd.DataFrame(rows)


def analyze(args: argparse.Namespace) -> Dict:
    """Runs the comparison and writes <out>/analysis.json and the scores."""
    rng = np.random.default_rng(args.seed)
    models = list(S.LEARNED)
    towers = list(S.TOWERS_SEARCHED)
    seeds = list(range(S.N_SEEDS))
    probe = sorted(
        glob.glob(
            os.path.join(args.paper_root, towers[0], "seed*",
                         "damage_comparison_*_fa.csv")))
    if not probe:
        sys.exit(f"no damage CSV under {args.paper_root}/{towers[0]}")
    sim_ids = pd.read_csv(probe[0], usecols=["sim_id"])["sim_id"].to_numpy()
    inverse, n_clusters = clusters(args.dataset_dir, towers, sim_ids)
    weights = np.vstack([
        np.ones(n_clusters),
        rng.multinomial(n_clusters,
                        np.full(n_clusters, 1.0 / n_clusters),
                        size=args.n_boot)
    ])
    with_floor = models + [FLOOR]
    arms = {}
    for arm, root in (("paper", args.paper_root), ("tuned", args.tuned_root)):
        raw = arm_scores(root, with_floor, towers, seeds, weights, sim_ids,
                         inverse, n_clusters)
        missing = int(np.isnan(raw["top"][0, :len(models)]).sum())
        if missing:
            print(
                f"{arm}: {missing} of {len(models) * len(towers) * len(seeds)}"
                " runs missing (ignored in the medians)")
        arms[arm] = {m: aggregate(raw[m], rng) for m in METRICS}
    result = {"n_clusters": n_clusters, "n_boot": args.n_boot, "metrics": {}}
    flags = []
    if args.hpo_root and args.fixed_val_root:
        table = guard(args.hpo_root, args.fixed_val_root, models, towers,
                      args.dataset_dir)
        table.to_csv(os.path.join(args.out, "guard.csv"), index=False)
        flags = sorted(set(table.loc[table["flag"], "model"]))
        result["guard_flagged"] = flags
    keep = [i for i, m in enumerate(models) if m not in flags]
    for metric in METRICS:
        paper = arms["paper"][metric][:, :len(models)]
        tuned = arms["tuned"][metric][:, :len(models)]
        entry = {"all": compare(paper, tuned, models)}
        if flags:
            entry["without_flagged"] = compare(paper[:, keep], tuned[:, keep],
                                               [models[i] for i in keep])
        if metric == "top":
            for arm in ("paper", "tuned"):
                floor = arms[arm][metric][0, -1]
                if np.isfinite(floor):
                    entry[f"{arm}_above_floor"] = int(
                        (arms[arm][metric][0, :len(models)] > floor).sum())
        result["metrics"][metric] = entry
    columns = {
        f"{arm}_{m}": scores[m][0, :len(models)] for arm, scores in arms.items()
        for m in METRICS
    }
    pd.DataFrame({
        "model": models,
        **columns
    }).to_csv(os.path.join(args.out, "scores.csv"), index=False)
    common.write_json(os.path.join(args.out, "analysis.json"), result)
    for metric, entry in result["metrics"].items():
        for name, comp in entry.items():
            if isinstance(comp, dict):
                print(f"{metric:6s} {name:16s} tau_b {comp['tau_b']:.3f} "
                      f"[{comp['low']:.3f}, {comp['high']:.3f}] "
                      f"{comp['verdict']}; top-3 {comp['top3']}, "
                      f"top-5 {comp['top5']}")
    return result


def write_synthetic(out: str,
                    shuffle: bool,
                    rng: np.random.Generator,
                    n_clusters: int = 30) -> Tuple[str, str, str]:
    """Synthetic paper and tuned runs (and metadata) for a dry run.

    Entry m has a reconstruction noise growing with m in the paper arm; the
    tuned arm keeps the order (small changes) or, with `shuffle`, permutes
    the entries. Returns (paper_root, tuned_root, dataset_dir).
    """
    sims = np.arange(1, 6 * n_clusters + 1)
    point = (sims - 1) // 6
    dataset_dir = os.path.join(out, "data")
    models = list(S.LEARNED) + [FLOOR]
    quality = np.linspace(0.05, 0.6, len(S.LEARNED))
    tuned_quality = (rng.permutation(quality) if shuffle else quality *
                     rng.uniform(0.9, 1.1, len(quality)))
    roots = (os.path.join(out, "paper"), os.path.join(out, "tuned"))
    for tower in S.TOWERS_SEARCHED:
        os.makedirs(os.path.join(dataset_dir, tower), exist_ok=True)
        pd.DataFrame({
            "sim_id": sims,
            "wind_speed": 4.0 + point % 10,
            "wave_hs": 1.0 + point // 10,
            "wave_tp": 8.0
        }).to_parquet(os.path.join(dataset_dir, tower, "metadata.parquet"))
        level = rng.normal(0, 1, n_clusters)[point]
        true = {
            g: 10**(level + 0.3 * rng.normal(0, 1, len(sims)) - 6 + 0.1 * i)
            for i, g in enumerate(GAUGES)
        }
        for root, qualities in zip(roots, (quality, tuned_quality)):
            for seed in range(S.N_SEEDS):
                folder = os.path.join(root, tower, f"seed{seed}")
                os.makedirs(folder, exist_ok=True)
                for model, noise in zip(models, list(qualities) + [0.8]):
                    frame = {"sim_id": sims}
                    for gauge in GAUGES:
                        frame[f"damage_true_{gauge}"] = true[gauge]
                        frame[f"damage_rec_{gauge}"] = true[gauge] * 10**(
                            rng.normal(0, noise + 0.01 * seed, len(sims)))
                    pd.DataFrame(frame).to_csv(csv_path(root, tower, seed,
                                                        model),
                                               index=False)
    return roots[0], roots[1], dataset_dir


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--paper_root", default="outputs/within")
    parser.add_argument("--tuned_root", default="outputs/hpo/sealed/last")
    parser.add_argument("--dataset_dir", default="data/FLOATSense")
    parser.add_argument("--hpo_root",
                        default="",
                        help="Track root with the phase-2 winners (guard).")
    parser.add_argument("--fixed_val_root",
                        default="",
                        help="Fixed-budget runs scored on val/val (guard).")
    parser.add_argument("--out", default="outputs/hpo/analysis")
    parser.add_argument("--n_boot", type=int, default=S.N_BOOT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry_run",
                        action="store_true",
                        help="Synthetic rankings (agreeing and permuted).")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> Dict:
    """Analyzes the two rankings (or two synthetic cases with --dry_run)."""
    args = parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    if not args.dry_run:
        return analyze(args)
    results = {}
    base = args.out
    for case in ("agree", "permuted"):
        args.out = os.path.join(base, case)
        os.makedirs(args.out, exist_ok=True)
        args.paper_root, args.tuned_root, args.dataset_dir = write_synthetic(
            args.out, case == "permuted", np.random.default_rng(args.seed))
        print(f"== synthetic case: {case}")
        results[case] = analyze(args)
    return results


if __name__ == "__main__":
    main()
