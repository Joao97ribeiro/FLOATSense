# pylint: disable=wrong-import-position
"""Phase 3 of the validation-tuned track: retrain the winners, test once.

The winner of every (model, tower) (hpo/confirm.py) is retrained on the
full training split (FINAL_TRAIN_SPLIT, 1,728 simulations) with N_SEEDS
seeds for EPOCHS_FINAL epochs, keeping the weights of the phase-2 median
best epoch too. The test split is opened once, at the end, for the
requested models and towers together, and only if every one of their
units is trained:

    python hpo/final.py --model=tcn --tower=opt2              # train
    python hpo/final.py --open_test --models=all --towers=all  # test, once

Each test score is written into a sealed directory,
<root>/sealed/<last|best>/<tower>/seed<k>/, in the layout of the
within-tower runs (damage_comparison_<model>_fa.csv), with a
SEALED_<model>.json marker; a sealed score is never computed again.
'last' (the headline) is the last epoch; 'best' (secondary) is the median
best validation epoch of phase 2, rounded to BEST_EPOCH_ROUND.
--dry_run trains and scores nothing.
"""

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hpo import common
from hpo import confirm
from hpo import constants as C
from hpo import search_space as S

VARIANTS = ("last", "best")


def unit_dir(root: str, model: str, tower: str, seed: int) -> str:
    """Run directory of one retraining."""
    return os.path.join(root, "final", f"{model}_{tower}", f"s{seed}")


def sealed_dir(root: str, variant: str, tower: str, seed: int) -> str:
    """Directory of the sealed test scores of one variant, tower and seed."""
    return os.path.join(root, "sealed", variant, tower, f"seed{seed}")


def winner(root: str, model: str, tower: str) -> Dict:
    """The phase-2 winner record (exits if it is missing)."""
    record = common.read_json(confirm.winner_path(root, model, tower))
    if record is None:
        sys.exit(f"no phase-2 winner for {model}/{tower}: run "
                 "hpo/confirm.py --summarize first")
    return record


def train(args: argparse.Namespace,
          model: str,
          tower: str,
          seeds: Optional[List[int]] = None) -> None:
    """Trains the N_SEEDS units of one (model, tower) (or `seeds`)."""
    record = winner(args.root, model, tower)
    best_epoch = record["best_epoch"]
    for seed in range(C.N_SEEDS) if seeds is None else seeds:
        if common.STOP.is_set():
            return
        run_dir = unit_dir(args.root, model, tower, seed)
        result_path = os.path.join(run_dir, "result.json")
        if os.path.exists(result_path) or os.path.exists(
                os.path.join(run_dir, "PARKED")) or not common.claim(run_dir):
            continue
        try:
            if args.dry_run:
                result = common.stub_unit(record["winner_config"], 200 + seed,
                                          args.epochs)
            else:
                keep = ([f"--save_epochs={best_epoch}"]
                        if best_epoch and best_epoch < args.epochs else [])
                cmd = common.train_command(args,
                                           model,
                                           tower,
                                           run_dir,
                                           record["winner_config"],
                                           C.FINAL_TRAIN_SPLIT,
                                           args.epochs,
                                           seed,
                                           validate=False,
                                           extra=keep)
                result = common.run_unit(f"final/{model}_{tower}/s{seed}", cmd,
                                         run_dir, args.root, model)
            if result["status"] not in ("parked", "stopped"):
                common.write_once(
                    result_path, {
                        "status": result["status"],
                        "config": record["winner_config"],
                        "best_epoch": best_epoch,
                        "epochs": args.epochs,
                        "time": common.now()
                    })
            print(f"final {model}/{tower} s{seed}: {result['status']}",
                  flush=True)
        finally:
            common.release(run_dir)


def _link(source: str, target: str) -> None:
    """Hard link (copy across file systems) of a checkpoint."""
    if os.path.exists(target):
        os.remove(target)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def score(args: argparse.Namespace, model: str, tower: str, seed: int,
          variant: str) -> None:
    """Scores one checkpoint on the test split into its sealed directory."""
    out = sealed_dir(args.root, variant, tower, seed)
    marker = os.path.join(out, f"SEALED_{model}.json")
    if os.path.exists(marker):
        return
    run_dir = unit_dir(args.root, model, tower, seed)
    result = common.read_json(os.path.join(run_dir, "result.json"))
    if result["status"] != "ok":
        common.write_once(marker, {"skipped": result["status"]})
        return
    epoch = result["epochs"]
    if variant == "best" and result["best_epoch"]:
        epoch = min(result["best_epoch"], epoch)
    os.makedirs(out, exist_ok=True)
    if not args.dry_run:
        suffix = f"_epoch{epoch}" if epoch < result["epochs"] else ""
        _link(os.path.join(run_dir, f"{model}_fa{suffix}.pt"),
              os.path.join(out, f"{model}_fa.pt"))
        cmd = [
            args.python, common.RUN_PY, f"--flagfile={common.CONFIG}",
            f"--dataset_dir={args.dataset_dir}", f"--tower={tower}",
            f"--models={model}", f"--train_split={C.FINAL_TRAIN_SPLIT}",
            f"--test_split={C.TEST_SPLIT}", f"--seed={seed}",
            f"--output_dir={out}", "--run_training=False"
        ] + shlex.split(args.extra)
        with open(os.path.join(out, f"log_{model}.txt"), "w",
                  encoding="utf-8") as log:
            code = subprocess.call(cmd,
                                   stdout=log,
                                   stderr=subprocess.STDOUT,
                                   cwd=common.REPO)
        if code:
            common.alert(args.root, f"test/{variant}/{model}_{tower}/s{seed}",
                         "test scoring failed", {"command": cmd})
            return
    common.write_once(
        marker, {
            "model":
                model,
            "tower":
                tower,
            "seed":
                seed,
            "variant":
                variant,
            "epoch":
                epoch,
            "summary":
                common.read_json(os.path.join(out, f"summary_{model}_fa.json")),
            "time":
                common.now()
        })


def open_test(args: argparse.Namespace) -> None:
    """Opens the test split once for every requested unit."""
    if (set(args.models) != set(S.LEARNED) or
            set(args.towers) != set(C.TOWERS_SEARCHED)) and not args.partial:
        sys.exit("the protocol opens the test once for every model and tower;"
                 " pass --partial to open it for a subset")
    pairs = [(m, t) for m in args.models for t in args.towers]
    missing = [
        f"{m}/{t}/s{seed}" for m, t in pairs for seed in range(C.N_SEEDS)
        if not os.path.exists(
            os.path.join(unit_dir(args.root, m, t, seed), "result.json"))
    ]
    if missing:
        sys.exit(f"the test opens when every unit is trained; missing: "
                 f"{', '.join(missing)}")
    for model, tower in pairs:
        for seed in range(C.N_SEEDS):
            for variant in VARIANTS:
                score(args, model, tower, seed, variant)
    print(f"test scored into {os.path.join(args.root, 'sealed')}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=sorted(S.SPACE))
    parser.add_argument("--tower", choices=C.TOWERS_SEARCHED)
    parser.add_argument("--open_test", action="store_true")
    parser.add_argument("--models",
                        default="all",
                        help="Models of --open_test (comma list or all).")
    parser.add_argument("--towers",
                        default="all",
                        help="Towers of --open_test (comma list or all).")
    parser.add_argument("--partial",
                        action="store_true",
                        help="Allow --open_test on a subset.")
    parser.add_argument("--root", default="outputs/hpo")
    parser.add_argument("--dataset_dir", default="data/FLOATSense")
    parser.add_argument("--extra", default="")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--epochs", type=int, default=C.EPOCHS_FINAL)
    args = parser.parse_args(argv)
    args.models = (list(S.LEARNED)
                   if args.models == "all" else args.models.split(","))
    args.towers = (list(C.TOWERS_SEARCHED)
                   if args.towers == "all" else args.towers.split(","))
    if not args.open_test and not (args.model and args.tower):
        parser.error("--model and --tower (training) or --open_test")
    return args


def main(argv: Optional[List[str]] = None) -> None:
    """Trains the winners of one (model, tower) or opens the test."""
    args = parse_args(argv)
    if args.open_test:
        open_test(args)
    else:
        train(args, args.model, args.tower)


if __name__ == "__main__":
    main()
