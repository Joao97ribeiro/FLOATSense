# pylint: disable=wrong-import-position
"""Phase 2 of the validation-tuned track: confirm the best configurations.

The search ranks configurations on one seed and EPOCHS_TRIAL epochs. Phase 2
takes the N_TOP best eligible trials of each (model, tower) study, trains
each with N_SEEDS seeds for EPOCHS_FINAL epochs on val/train (validated on
val/val), and picks the winner by the highest MEDIAN over the seeds of the
last-epoch validation score; never by the best seed.

    python hpo/confirm.py --model=tcn --tower=opt2 --freeze     # plan, once
    python hpo/confirm.py --model=tcn --tower=opt2              # the units
    python hpo/confirm.py --model=tcn --tower=opt2 --only=3     # one unit
    python hpo/confirm.py --model=tcn --tower=opt2 --summarize  # winner

The plan (the N_TOP configurations) is written once (--freeze), when the
study has its full budget and nothing running, so every worker confirms
the same configurations; they are taken among the first `target` counted
trials (by number), the trials that decided the extension rule. If the
winner's median is not finite (most seeds diverged), the (model, tower)
is parked with an alert instead of naming a winner; a parked (model,
tower) has no winner and is recorded as missing in the test and the
leaderboard. A unit is one (configuration, seed); workers claim
units, so several can share a study. The winner record also holds the
median best validation epoch (rounded to BEST_EPOCH_ROUND), the secondary
test epoch of phase 3. --dry_run trains nothing (common.stub_unit).
"""

import argparse
import os
import sys
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hpo import common
from hpo import constants as C
from hpo import search
from hpo import search_space as S


def study_dir(root: str, model: str, tower: str) -> str:
    """Phase-2 directory of a (model, tower)."""
    return os.path.join(root, "phase2", f"{model}_{tower}")


def plan_path(root: str, model: str, tower: str) -> str:
    """The frozen plan of a (model, tower)."""
    return os.path.join(study_dir(root, model, tower), "plan.json")


def winner_path(root: str, model: str, tower: str) -> str:
    """The winner record of a (model, tower)."""
    return os.path.join(study_dir(root, model, tower), "winner.json")


def unit_dir(root: str, model: str, tower: str, rank: int, seed: int) -> str:
    """Run directory of one (configuration, seed)."""
    return os.path.join(study_dir(root, model, tower), f"c{rank}_s{seed}")


def parked_path(root: str, model: str, tower: str) -> str:
    """Marker of a parked (model, tower) study."""
    return os.path.join(root, "parked", f"{model}_{tower}.json")


def park_study(root: str,
               model: str,
               tower: str,
               reason: str,
               hosts: Optional[List[str]] = None) -> bool:
    """Parks a study once, with an alert; True if this call parked it.
    `hosts`: hosts of the failures that parked it (hpo/pick.py un-parks a
    study once if they are a single host)."""
    if not common.write_once(parked_path(root, model, tower), {
            "reason": reason,
            "hosts": sorted(hosts or []),
            "time": common.now()
    }):
        return False
    common.alert(
        root, f"study/{model}_{tower}", f"study parked: {reason}", {
            "note": f"the other towers of {model} go on without it; "
                    f"{model}/{tower} has no winner and is missing in the "
                    "leaderboard",
            "hosts": sorted(hosts or [])
        })
    return True


def units(plan: Dict) -> List[tuple]:
    """(rank, seed) of every unit of a plan."""
    return [(c["rank"], seed)
            for c in plan["configs"]
            for seed in range(plan["n_seeds"])]


def freeze(args: argparse.Namespace) -> Dict:
    """Writes the plan once, from a finished search study."""
    path = plan_path(args.root, args.model, args.tower)
    existing = common.read_json(path)
    if existing is not None:
        print(f"plan already frozen in {path}")
        return existing
    study = search.open_study(args.root, args.model, args.tower, create=False)
    if study is None:
        sys.exit("no study " +
                 search.journal_path(args.root, args.model, args.tower))
    done, running, waiting = search.budget(study)
    target = search.target_trials(args.root, args.model, args.n_trials)
    if done < target or running or waiting:
        sys.exit(f"search not finished: {done}/{target} trials counted, "
                 f"{running} running, {waiting} waiting")
    first = [
        t for t in sorted(study.trials, key=lambda t: t.number)
        if search.counted(t)
    ][:target]
    trials = sorted([t for t in first if search.eligible(t)],
                    key=lambda t: (-t.value, t.number))
    if len(trials) < C.N_TOP:
        sys.exit(f"{len(trials)} eligible trials, fewer than {C.N_TOP}")
    top = trials[:C.N_TOP]
    margin = (top[-1].value -
              trials[C.N_TOP].value if len(trials) > C.N_TOP else None)
    plan = {
        "study": study.study_name,
        "model": args.model,
        "tower": args.tower,
        "n_trials_counted": done,
        "n_seeds": C.N_SEEDS,
        "epochs": args.epochs,
        "margin_top_to_next": margin,
        "configs": [{
            "rank": rank,
            "trial": t.number,
            "value": t.value,
            "config": t.user_attrs["config"]
        } for rank, t in enumerate(top)],
        "time": common.now(),
    }
    if not common.write_once(path, plan):
        return common.read_json(path)
    print(f"plan frozen: {path}")
    return plan


def run_one(args: argparse.Namespace, plan: Dict, rank: int,
            seed: int) -> Optional[Dict]:
    """Trains one unit (if not done and not claimed) and records it."""
    run_dir = unit_dir(args.root, args.model, args.tower, rank, seed)
    result_path = os.path.join(run_dir, "result.json")
    if os.path.exists(result_path):
        return common.read_json(result_path)
    if os.path.exists(os.path.join(run_dir, "PARKED")):
        print(f"c{rank}_s{seed} parked, see {args.root}/alerts", flush=True)
        return None
    if not common.claim(run_dir):
        return None
    try:
        cfg = plan["configs"][rank]["config"]
        if args.dry_run:
            result = common.stub_unit(cfg,
                                      100 + 10 * rank + seed,
                                      plan["epochs"],
                                      run_dir=run_dir)
        else:
            cmd = common.train_command(args, args.model, args.tower, run_dir,
                                       cfg, C.SEARCH_TRAIN_SPLIT,
                                       plan["epochs"], seed)
            result = common.run_unit(
                f"phase2/{args.model}_{args.tower}/c{rank}_s{seed}",
                cmd,
                run_dir,
                args.root,
                args.model,
                config=cfg)
        if result["status"] in ("parked", "stopped"):
            return None
        record = {"rank": rank, "seed": seed, "status": result["status"]}
        curve = (result.get("history") or {}).get("val_r2", [])
        last = curve[-1] if curve else None
        if (result["status"] == "ok" and last and
                last["epoch"] == plan["epochs"] and
                np.isfinite(last["r2_mean"])):
            best = max(curve, key=lambda e: e["r2_mean"])
            record.update(score=last["r2_mean"],
                          r2_top=last["r2_top"],
                          r2_base=last["r2_base"],
                          r2_gauges=last["r2_gauges"],
                          best_epoch=best["epoch"],
                          curve=[[e["epoch"], e["r2_mean"]] for e in curve])
        else:
            record.update(status="diverged", score=None, best_epoch=None)
        common.write_once(result_path, record)
        return common.read_json(result_path)
    finally:
        common.release(run_dir)


def summarize(args: argparse.Namespace, plan: Dict) -> Optional[Dict]:
    """The winner: highest median over seeds of the last-epoch score (a
    diverged seed counts as -inf); written once. If that median is not
    finite, the study is parked (no winner) and None is returned."""
    existing = common.read_json(winner_path(args.root, args.model, args.tower))
    if existing is not None:
        return existing
    rows = []
    for config in plan["configs"]:
        records = [
            common.read_json(
                os.path.join(
                    unit_dir(args.root, args.model, args.tower, config["rank"],
                             seed), "result.json"))
            for seed in range(plan["n_seeds"])
        ]
        if any(r is None for r in records):
            print(f"config {config['rank']}: "
                  f"{sum(r is not None for r in records)}/{plan['n_seeds']} "
                  "seeds done; the winner waits for all")
            return None
        scores = [
            -np.inf if r["score"] is None else r["score"] for r in records
        ]
        epochs = [r["best_epoch"] for r in records if r["best_epoch"]]
        rows.append({
            **config, "scores": [None if np.isinf(v) else v for v in scores],
            "median":
                float(np.median(scores)),
            "best_epoch_median": (int(
                round(np.median(epochs) / C.BEST_EPOCH_ROUND) *
                C.BEST_EPOCH_ROUND) if epochs else None)
        })
    rows.sort(key=lambda r: (-r["median"], r["rank"]))
    winner = rows[0]
    if not np.isfinite(winner["median"]):
        park_study(args.root, args.model, args.tower,
                   "no finite median over the seeds of phase 2")
        return None
    record = {
        "model": args.model,
        "tower": args.tower,
        "selection_rule": "highest median over seeds of the last-epoch "
                          "validation R^2 (mean of 11 gauges)",
        "winner_rank": winner["rank"],
        "winner_config": winner["config"],
        "winner_median": winner["median"],
        "best_epoch": winner["best_epoch_median"],
        "margin_to_second":
            (winner["median"] - rows[1]["median"] if len(rows) > 1 else None),
        "configs": rows,
        "time": common.now(),
    }
    common.write_once(winner_path(args.root, args.model, args.tower), record)
    return common.read_json(winner_path(args.root, args.model, args.tower))


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, choices=sorted(S.SPACE))
    parser.add_argument("--tower", required=True, choices=C.TOWERS_SEARCHED)
    parser.add_argument("--root", default="outputs/hpo")
    parser.add_argument("--dataset_dir", default="data/FLOATSense")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--only",
                        type=int,
                        default=-1,
                        help="Run only the n-th unit (rank-major).")
    parser.add_argument("--extra", default="")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--n_trials", type=int, default=C.N_TRIALS)
    parser.add_argument("--epochs", type=int, default=C.EPOCHS_FINAL)
    parser.add_argument("--checkpoint_seconds",
                        type=float,
                        default=C.CHECKPOINT_SECONDS,
                        help="Wall time between two resume saves of a run.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> Optional[Dict]:
    """Freezes the plan, runs the units or names the winner."""
    args = parse_args(argv)
    if args.freeze:
        return freeze(args)
    plan = common.read_json(plan_path(args.root, args.model, args.tower))
    if plan is None:
        sys.exit("no frozen plan; run with --freeze first")
    if args.summarize:
        record = summarize(args, plan)
        if record:
            print(f"winner: config {record['winner_rank']} median "
                  f"{record['winner_median']:.4f}, margin "
                  f"{record['margin_to_second']}, best epoch "
                  f"{record['best_epoch']}: {record['winner_config']}")
        return record
    todo = units(plan)
    if args.only >= 0:
        todo = todo[args.only:args.only + 1]
    for rank, seed in todo:
        record = run_one(args, plan, rank, seed)
        if record:
            print(
                f"c{rank}_s{seed}: {record['status']} "
                f"score {record.get('score')}",
                flush=True)
    return plan


if __name__ == "__main__":
    main()
