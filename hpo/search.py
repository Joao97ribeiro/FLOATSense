# pylint: disable=wrong-import-position
# pylint: disable=import-error
"""Phase 1 of the validation-tuned track: the search of one (model, tower).

One Optuna study per (model, tower), in a journal file that any number of
workers share (one worker per GPU; workers can join and leave at any time):

    python hpo/search.py --model=tcn --tower=opt2 --dataset_dir=data/FLOATSense
    python hpo/search.py --model=tcn --tower=opt2 --dry_run --root=/tmp/hpo

Each trial draws a configuration (search_space.suggest; N_STARTUP random
trials, then TPE, seeded per study and trial number) and trains it with
scripts/train/run.py for EPOCHS_TRIAL epochs on val/train, validated on
val/val every VAL_EVERY epochs. Its score is the validation R^2 of log10
damage, mean over the 11 gauges, at the last epoch. A MedianPruner on the
best-so-far value stops poor trials after PRUNE_WARMUP epochs (not for the
models in NO_PRUNING).

Budget: N_TRIALS completed or pruned trials per study (failed, parked and
over-cap trials do not count). If a study's best trial comes after the
EXTEND_AFTER-th, the three studies of that model get EXTEND_BY more trials.

Failures: a diverged trial completes with the lowest completed score of
the study; a model above the parameter cap is rejected before training
(pruned, not counted) and the next trial draws again; a crash is resumed
up to MAX_ATTEMPTS times (common.run_unit), then parked with an alert. A
trial left RUNNING by a dead worker (stale heartbeat) is requeued with the
same configuration and resumes from its checkpoint. A worker that holds
the lock of the study (hpo/pick.py, --exclusive) is its only writer, so
every RUNNING trial it finds is orphaned and requeued at once. A trial
stopped on request (common.request_stop) stays RUNNING for the next
worker to resume.

--dry_run replaces the training with a deterministic CPU stub
(common.stub_unit), to test the driver itself.
"""

import argparse
import datetime
import math
import os
import sys
import zlib
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.storages.journal import JournalFileOpenLock
from optuna.trial import TrialState

from hpo import common
from hpo import constants as C
from hpo import search_space as S


def study_name(model: str, tower: str) -> str:
    """Name (and journal file stem) of the study of a (model, tower)."""
    return f"{model}_{tower}"


def journal_path(root: str, model: str, tower: str) -> str:
    """Journal file of a study."""
    return os.path.join(root, "optuna", f"{study_name(model, tower)}.log")


def study_seed(model: str, tower: str) -> int:
    """Fixed sampler seed of a study."""
    return zlib.crc32(study_name(model, tower).encode())


def pruner(model: str) -> optuna.pruners.BasePruner:
    """MedianPruner of the protocol (none for the NO_PRUNING models)."""
    if model in C.NO_PRUNING:
        return optuna.pruners.NopPruner()
    return optuna.pruners.MedianPruner(n_startup_trials=C.N_STARTUP,
                                       n_warmup_steps=C.PRUNE_WARMUP,
                                       interval_steps=C.VAL_EVERY)


def sampler(seed: int) -> optuna.samplers.BaseSampler:
    """TPE after N_STARTUP random trials."""
    return optuna.samplers.TPESampler(n_startup_trials=C.N_STARTUP, seed=seed)


def open_study(root: str,
               model: str,
               tower: str,
               create: bool = True) -> Optional[optuna.Study]:
    """Opens (or creates) the study of a (model, tower).

    With create=False the journal is only read (optuna.load_study writes
    nothing; create_study appends a record each time), so readers such as
    the status page never write to a journal.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    path = journal_path(root, model, tower)
    if not create and not os.path.exists(path):
        return None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    storage = JournalStorage(
        JournalFileBackend(path, lock_obj=JournalFileOpenLock(path)))
    if not create:
        return optuna.load_study(study_name=study_name(model, tower),
                                 storage=storage,
                                 sampler=sampler(study_seed(model, tower)),
                                 pruner=pruner(model))
    return optuna.create_study(study_name=study_name(model, tower),
                               storage=storage,
                               direction="maximize",
                               load_if_exists=True,
                               sampler=sampler(study_seed(model, tower)),
                               pruner=pruner(model))


def counted(trial: optuna.trial.FrozenTrial) -> bool:
    """A trial that spends budget: completed, or pruned by the median rule
    (not rejected by the parameter cap)."""
    return (trial.state == TrialState.COMPLETE or
            (trial.state == TrialState.PRUNED and
             not trial.user_attrs.get("over_param_cap")))


def eligible(trial: optuna.trial.FrozenTrial) -> bool:
    """A trial that can be selected: completed, not diverged."""
    return (trial.state == TrialState.COMPLETE and
            not trial.user_attrs.get("diverged"))


def _running_alive(trial: optuna.trial.FrozenTrial) -> bool:
    """A RUNNING trial whose worker still beats."""
    run_dir = trial.user_attrs.get("run_dir")
    if run_dir:
        return not common.is_stale(run_dir)
    age = datetime.datetime.now() - trial.datetime_start
    return age.total_seconds() < C.STALE_MINUTES * 60


def budget(study: optuna.Study) -> Tuple[int, int, int]:
    """(counted, running, waiting) trials of a study."""
    trials = study.get_trials(deepcopy=False)
    return (sum(counted(t) for t in trials),
            sum(t.state == TrialState.RUNNING and _running_alive(t)
                for t in trials),
            sum(t.state == TrialState.WAITING for t in trials))


def requeue_stale(study: optuna.Study, exclusive: bool = False) -> List[int]:
    """Requeues the trials of dead workers with the same configuration;
    the new trial resumes from the run directory of the old one. With
    `exclusive` (the caller holds the lock of the study) every RUNNING
    trial belongs to a dead or stopped worker."""
    requeued = []
    for trial in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
        if not exclusive and _running_alive(trial):
            continue
        run_dir = trial.user_attrs.get("run_dir")
        if run_dir and not common.write_once(
                os.path.join(run_dir, f"requeued.{trial.number}.json"),
            {"time": common.now()}):
            continue  # another worker requeued it
        try:
            study.tell(trial.number, state=TrialState.FAIL)
        except (RuntimeError, ValueError):
            continue  # finished meanwhile
        if run_dir:
            study.enqueue_trial(trial.params,
                                user_attrs={
                                    "resume_from": run_dir,
                                    "requeued_from": trial.number
                                },
                                skip_if_exists=False)
            requeued.append(trial.number)
    return requeued


def extension_path(root: str, model: str) -> str:
    """Record of the extension of the three studies of a model."""
    return os.path.join(root, "extensions", f"{model}.json")


def target_trials(root: str, model: str, n_trials: int) -> int:
    """Budget of each study of `model` (extended or not)."""
    extended = os.path.exists(extension_path(root, model))
    return n_trials + (C.EXTEND_BY if extended else 0)


def best_position(trials: List[optuna.trial.FrozenTrial],
                  n_trials: int) -> Optional[int]:
    """1-based position, among the first `n_trials` counted trials, of the
    best eligible one (None if none is eligible)."""
    first = [t for t in sorted(trials, key=lambda t: t.number) if counted(t)
            ][:n_trials]
    scored = [(t.value, -i) for i, t in enumerate(first) if eligible(t)]
    if not scored:
        return None
    return -max(scored)[1] + 1


def maybe_extend(root: str, model: str, n_trials: int) -> bool:
    """Applies the extension rule once the first `n_trials` of a study of
    `model` are done; returns True if the model is (now) extended."""
    if os.path.exists(extension_path(root, model)):
        return True
    for tower in C.TOWERS_SEARCHED:
        study = open_study(root, model, tower, create=False)
        if study is None:
            continue
        trials = study.get_trials(deepcopy=False)
        if sum(counted(t) for t in trials) < n_trials:
            continue
        position = best_position(trials, n_trials)
        if position is not None and position > C.EXTEND_AFTER:
            common.write_once(
                extension_path(root, model), {
                    "model": model,
                    "tower": tower,
                    "best_position": position,
                    "extend_by": C.EXTEND_BY,
                    "time": common.now()
                })
            return True
    return False


def diverged_score(study: optuna.Study) -> float:
    """Score of a diverged trial: the lowest completed score of the study
    (else the lowest reported value, else DIVERGED_FALLBACK)."""
    values = [t.value for t in study.get_trials(deepcopy=False) if eligible(t)]
    if values:
        return min(values)
    reported = [
        v for t in study.get_trials(deepcopy=False)
        for v in t.intermediate_values.values() if math.isfinite(v)
    ]
    return min(reported + [C.DIVERGED_FALLBACK])


def _recorded(run_dir: str, number: int, status: str) -> None:
    """Marks the run directory of a trial told to the study (its files are
    no longer needed to resume it)."""
    common.write_json(os.path.join(run_dir, "recorded.json"), {
        "trial": number,
        "status": status,
        "time": common.now()
    })


def run_trial(study: optuna.Study, trial: optuna.Trial,
              args: argparse.Namespace) -> None:
    """Draws, trains and scores one trial, then tells the study."""
    cfg = S.suggest(trial, args.model)
    run_dir = trial.user_attrs.get("resume_from") or os.path.join(
        args.root, "trials", study.study_name, f"t{trial.number:03d}")
    trial.set_user_attr("run_dir", run_dir)
    trial.set_user_attr("config", cfg)
    pruning = args.model not in C.NO_PRUNING
    reported = set()

    def on_val(scores: Dict) -> bool:
        epoch, value = scores["epoch"], scores["r2_mean"]
        if epoch in reported or not math.isfinite(value):
            return False
        reported.add(epoch)
        trial.report(value, epoch)
        return pruning and trial.should_prune()

    if args.dry_run:
        result = common.stub_unit(cfg, trial.number, args.epochs, on_val)
    else:
        cmd = common.train_command(args, args.model, args.tower, run_dir, cfg,
                                   C.SEARCH_TRAIN_SPLIT, args.epochs,
                                   C.TRIAL_SEED)
        result = common.run_unit(f"search/{study.study_name}/t{trial.number}",
                                 cmd, run_dir, args.root, args.model, on_val)
    status = result["status"]
    if status == "stopped":
        print(f"trial {trial.number:3d} {args.model}/{args.tower} stopped",
              flush=True)
        return  # left RUNNING: the next worker resumes it
    if result.get("params") is not None:
        trial.set_user_attr("params_M", result["params"] / 1e6)
    if status == "ok":
        last = result["history"]["val_r2"][-1]
        trial.set_user_attr(
            "curve",
            [[e["epoch"], e["r2_mean"]] for e in result["history"]["val_r2"]])
        for key in ("r2_top", "r2_base", "r2_gauges", "epoch"):
            trial.set_user_attr(key, last[key])
        if last["epoch"] == args.epochs and math.isfinite(last["r2_mean"]):
            study.tell(trial, last["r2_mean"])
            _recorded(run_dir, trial.number, "ok")
            print(
                f"trial {trial.number:3d} {args.model}/{args.tower} "
                f"val {last['r2_mean']:.4f} {cfg}",
                flush=True)
            return
        status = "diverged"
    if status == "diverged":
        trial.set_user_attr("diverged", True)
        study.tell(trial, diverged_score(study))
    elif status == "pruned":
        trial.set_user_attr("pruned_at", max(reported) if reported else 0)
        study.tell(trial, state=TrialState.PRUNED)
    elif status == "too_large":
        trial.set_user_attr("over_param_cap", True)
        study.tell(trial, state=TrialState.PRUNED)
    else:  # parked
        trial.set_user_attr("parked", True)
        study.tell(trial, state=TrialState.FAIL)
    _recorded(run_dir, trial.number, status)
    print(f"trial {trial.number:3d} {args.model}/{args.tower} {status}",
          flush=True)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, choices=sorted(S.SPACE))
    parser.add_argument("--tower", required=True, choices=C.TOWERS_SEARCHED)
    parser.add_argument("--root",
                        default="outputs/hpo",
                        help="Root of the track (studies, runs, alerts).")
    parser.add_argument("--dataset_dir", default="data/FLOATSense")
    parser.add_argument("--max_new",
                        type=int,
                        default=0,
                        help="Run at most this many trials, then exit.")
    parser.add_argument("--extra",
                        default="",
                        help="Extra run.py flags of every trial.")
    parser.add_argument("--python",
                        default=sys.executable,
                        help="Python that runs scripts/train/run.py.")
    parser.add_argument("--dry_run",
                        action="store_true",
                        help="CPU stub instead of training.")
    parser.add_argument("--exclusive",
                        action="store_true",
                        help="The caller holds the lock of the study "
                        "(hpo/pick.py): requeue every RUNNING trial.")
    parser.add_argument("--n_trials",
                        type=int,
                        default=C.N_TRIALS,
                        help="Budget per study (the protocol: N_TRIALS).")
    parser.add_argument("--epochs",
                        type=int,
                        default=C.EPOCHS_TRIAL,
                        help="Epochs per trial (the protocol: EPOCHS_TRIAL).")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> optuna.Study:
    """Runs trials of one study until its budget is reached."""
    args = parse_args(argv)
    study = open_study(args.root, args.model, args.tower)
    seed = study_seed(args.model, args.tower)
    requeued = requeue_stale(study, args.exclusive)
    if requeued:
        print(f"requeued trials of dead workers: {requeued}", flush=True)
    started = 0
    while ((not args.max_new or started < args.max_new) and
           not common.STOP.is_set()):
        done, running, waiting = budget(study)
        target = target_trials(args.root, args.model, args.n_trials)
        if not waiting and done + running >= target:
            if (target == args.n_trials and not running and
                    maybe_extend(args.root, args.model, args.n_trials)):
                continue  # the budget grew
            break
        trial = study.ask()
        # Seeded per trial: reproducible, and two workers never share a draw.
        study.sampler = sampler(seed + trial.number)
        started += 1
        run_trial(study, trial, args)
    done, running, _ = budget(study)
    best = [t for t in study.get_trials(deepcopy=False) if eligible(t)]
    print(f"study {study.study_name}: {done} trials counted, {running} running",
          flush=True)
    if best:
        top = max(best, key=lambda t: t.value)
        print(f"best: trial {top.number}, val {top.value:.4f} {top.params}",
              flush=True)
    return study


if __name__ == "__main__":
    main()
