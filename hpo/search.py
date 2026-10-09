# pylint: disable=wrong-import-position
# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-locals
"""Phase 1 of the validation-tuned track: the search of one (model, tower).

One Optuna study per (model, tower), in a journal file (one worker at a
time when run by hpo/worker.py):

    python hpo/search.py --model=tcn --tower=opt2 --dataset_dir=data/FLOATSense
    python hpo/search.py --model=tcn --tower=opt2 --dry_run --root=/tmp/hpo

Each trial draws a configuration (search_space.suggest; N_STARTUP random
trials, then TPE, seeded per study and trial number) and trains it with
scripts/train/run.py for EPOCHS_TRIAL epochs on val/train, validated on
val/val every VAL_EVERY epochs. Its score is the validation R^2 of log10
damage, mean over the 11 gauges, at the last epoch. A MedianPruner on the
best-so-far value stops poor trials after PRUNE_WARMUP epochs (not for the
models in NO_PRUNING).

Budget: N_TRIALS completed or pruned trials per study (failed, shelved and
over-cap trials do not count). If a study's best trial comes after the
EXTEND_AFTER-th, the three studies of that model get EXTEND_BY more trials.

Failures: a diverged trial completes with the lowest completed score of
the study; a model above the parameter cap is rejected before training
and recorded as FAIL with the user attribute over_cap (not counted, and
not seen by TPE, whose startup counts completed and pruned trials only);
the next trial draws again at once (an over-cap draw does not count toward
--max_new). A trial out of memory is FAIL with the user attribute oom (not
counted; it is resumed once as a crash first, common.run_unit). A crash
is resumed up to MAX_ATTEMPTS times (common.run_unit), then the trial is
FAIL with the user attribute shelved. The study stops drawing (`capped`,
with the kind 'over_cap', 'oom' or 'shelved_trials') after MAX_OVER_CAP
over-cap draws, MAX_OOM trials out of memory or SHELVE_STUDY_AFTER shelved
trials (the last two since its retry); hpo/pick.py then shelves it, or
freezes its plan if it has its N_TRIALS counted trials. A trial whose
resume state belongs to another configuration stays RUNNING, held for an
operator (common.hold). A trial left RUNNING by a dead worker (stale
heartbeat) is enqueued again with the same configuration and resumes from
its checkpoint; a worker that holds the lock of the study (hpo/pick.py,
--exclusive) is its only writer, so every RUNNING trial it finds is
enqueued again at once. A journal line torn by a kill
or power loss is cut off when the study is opened for writing
(`repair_journal`). A trial stopped on request
(common.request_stop) stays RUNNING for the next worker to resume. A
worker whose lock was taken over (`owned` returns False) drops its result
instead of telling it.

The configuration stored in a trial ('config' user attribute, config.json
of its run) is the one passed to run.py: floats at 6 significant digits
(search_space.formatted); trial.params keep Optuna's unrounded draw.

--dry_run replaces the training with a deterministic CPU stub
(common.stub_unit), to test the driver itself.
"""

import argparse
import datetime
import math
import os
import sys
import zlib
from typing import Callable, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.storages.journal import JournalFileOpenLock
from optuna.trial import TrialState

from hpo import common
from hpo import constants as C
from hpo import search_space as S


def journal_path(root: str, model: str, tower: str) -> str:
    """Journal file of a study."""
    return os.path.join(root, "optuna", f"{common.study_id(model, tower)}.log")


def study_seed(model: str, tower: str) -> int:
    """Fixed sampler seed of a study."""
    return zlib.crc32(common.study_id(model, tower).encode())


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


def repair_journal(root: str, name: str, path: str) -> int:
    """Cuts a torn last line (an append cut by a kill or power loss) off a
    journal, which a later append would make unreadable; the cut bytes are
    kept in <path>.torn.<time>, with an alert of the unit `name`. The
    caller is the only writer (it holds the lock of the study). Returns
    the bytes cut."""
    try:
        with open(path, "rb") as file:
            data = file.read()
    except FileNotFoundError:
        return 0
    keep = data.rfind(b"\n") + 1
    if keep == len(data):
        return 0
    backup = f"{path}.torn.{datetime.datetime.now():%Y%m%d-%H%M%S}"
    with open(backup, "wb") as file:
        file.write(data[keep:])
    with open(path, "r+b") as file:
        file.truncate(keep)
    common.alert(root, name, "journal: torn last line cut", {
        "bytes": len(data) - keep,
        "backup": backup
    })
    return len(data) - keep


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
    if create:
        repair_journal(root, common.search_id(model, tower), path)
    storage = JournalStorage(
        JournalFileBackend(path, lock_obj=JournalFileOpenLock(path)))
    if not create:
        return optuna.load_study(study_name=common.study_id(model, tower),
                                 storage=storage,
                                 sampler=sampler(study_seed(model, tower)),
                                 pruner=pruner(model))
    return optuna.create_study(study_name=common.study_id(model, tower),
                               storage=storage,
                               direction="maximize",
                               load_if_exists=True,
                               sampler=sampler(study_seed(model, tower)),
                               pruner=pruner(model))


def over_cap(trial: optuna.trial.FrozenTrial) -> bool:
    """A draw rejected by the parameter cap (FAIL, not counted)."""
    return bool(trial.user_attrs.get("over_cap"))


def oom(trial: optuna.trial.FrozenTrial) -> bool:
    """A trial that ran out of memory (FAIL, not counted)."""
    return bool(trial.user_attrs.get("oom"))


def counted(trial: optuna.trial.FrozenTrial) -> bool:
    """A trial that spends budget: completed, or pruned by the median rule
    (never a draw rejected by the parameter cap)."""
    return (trial.state in (TrialState.COMPLETE, TrialState.PRUNED) and
            not over_cap(trial))


def eligible(trial: optuna.trial.FrozenTrial) -> bool:
    """A trial that can be selected: completed, not diverged."""
    return (trial.state == TrialState.COMPLETE and
            not trial.user_attrs.get("diverged"))


def trial_run_dir(trial: optuna.trial.FrozenTrial) -> Optional[str]:
    """Run directory of a trial: its own, or the one it resumes (a worker
    can die between ask() and setting 'run_dir')."""
    return trial.user_attrs.get("run_dir") or trial.user_attrs.get(
        "resume_from")


def _running_alive(trial: optuna.trial.FrozenTrial) -> bool:
    """A RUNNING trial whose worker still beats."""
    run_dir = trial_run_dir(trial)
    if run_dir:
        return not common.is_stale(run_dir)
    age = datetime.datetime.now() - trial.datetime_start
    return age.total_seconds() < C.STALE_SECONDS


def budget(study: optuna.Study) -> Tuple[int, int, int]:
    """(counted, running, waiting) trials of a study."""
    trials = study.get_trials(deepcopy=False)
    return (sum(counted(t) for t in trials),
            sum(t.state == TrialState.RUNNING and _running_alive(t)
                for t in trials),
            sum(t.state == TrialState.WAITING for t in trials))


def requeue_stale(study: optuna.Study, exclusive: bool = False) -> List[int]:
    """Enqueues the trials of dead workers again with the same
    configuration; the new trial resumes from the run directory of the old
    one. With `exclusive` (the caller holds the lock of the study) every
    RUNNING trial belongs to a dead or stopped worker.

    The old trial is told FAIL first (one worker only succeeds: a second
    tell raises), then its configuration is enqueued. A worker killed
    between the two loses at most the checkpoint (the trial is FAIL, not
    counted; the study draws again).
    """
    requeued = []
    for trial in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
        run_dir = trial_run_dir(trial)
        if not exclusive and _running_alive(trial):
            continue
        try:
            study.tell(trial.number, state=TrialState.FAIL)
        except (RuntimeError, ValueError):
            continue  # finished or requeued meanwhile
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
    """Write-once decision of the extension rule for the three studies of a
    model ('extended': true or false)."""
    return os.path.join(root, "extensions", f"{model}.json")


def failed_draws(root: str, model: str, tower: str, trials) -> Dict[str, int]:
    """Over-cap draws, trials out of memory and shelved trials of a study
    (the last two since its retry, if it was retried)."""
    retried = common.read_json(common.study_retried_path(root, model, tower))
    after = retried["after_trial"] if retried else -1
    return {
        "over_cap":
            sum(over_cap(t) for t in trials),
        "oom":
            sum(oom(t) and t.number > after for t in trials),
        "shelved_trials":
            sum(
                bool(t.user_attrs.get("shelved")) and t.number > after
                for t in trials)
    }


def capped(draws: Dict[str, int]) -> Optional[Dict[str, str]]:
    """Why a study stops drawing (see `failed_draws`): its 'kind'
    ('over_cap', 'oom' or 'shelved_trials') and a 'message'; or None."""
    if draws["over_cap"] >= C.MAX_OVER_CAP:
        return {
            "kind": "over_cap",
            "message": f"{draws['over_cap']} draws above the parameter cap"
        }
    if draws["oom"] >= C.MAX_OOM:
        return {
            "kind": "oom",
            "message": f"{draws['oom']} trials out of memory"
        }
    if draws["shelved_trials"] >= C.SHELVE_STUDY_AFTER:
        return {
            "kind": "shelved_trials",
            "message": f"{draws['shelved_trials']} trials shelved"
        }
    return None


def extension_decision(root: str, model: str) -> Optional[bool]:
    """The extension decision of a model (None if not taken yet)."""
    record = common.read_json(extension_path(root, model))
    return None if record is None else bool(record["extended"])


def target_trials(root: str, model: str, n_trials: int) -> int:
    """Budget of each study of `model` (extended or not)."""
    extended = bool(extension_decision(root, model))
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


def maybe_extend(root: str,
                 model: str,
                 n_trials: int,
                 towers: Optional[List[str]] = None,
                 decide: bool = False) -> bool:
    """Applies the extension rule once the first `n_trials` of a study of
    `model` are done; returns True if the model is (now) extended.
    `towers`: the studies that decide (default the non-shelved ones). The
    decision is one write-once file: an extension is written as soon as a
    study shows it; with `decide` (the gate of hpo/pick.py, every
    non-shelved study done) a 'not extended' decision is written too. A
    decision is never taken again."""
    decision = extension_decision(root, model)
    if decision is not None:
        return decision
    if towers is None:
        towers = [
            t for t in C.TOWERS_SEARCHED
            if not os.path.exists(common.study_shelved_path(root, model, t))
        ]
    for tower in towers:
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
                    "extended": True,
                    "tower": tower,
                    "towers": list(towers),
                    "best_position": position,
                    "extend_by": C.EXTEND_BY,
                    "time": common.now()
                })
            return bool(extension_decision(root, model))
    if decide:
        common.write_once(
            extension_path(root, model), {
                "model": model,
                "extended": False,
                "towers": list(towers),
                "time": common.now()
            })
        return bool(extension_decision(root, model))
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


def _tell(study: optuna.Study,
          trial: optuna.Trial,
          attrs: Dict,
          owned: Optional[Callable[[], bool]] = None,
          value: Optional[float] = None,
          state: Optional[TrialState] = None) -> bool:
    """Sets the user attributes of a trial and tells it, unless the lock of
    the study was lost (`owned`) or the trial was finished elsewhere (a
    takeover requeued it); then the result is dropped.

    Returns:
        bool: True if the trial was told.
    """
    if owned is not None and not owned():
        print(f"trial {trial.number}: lock lost, result dropped", flush=True)
        return False
    try:
        for key, item in attrs.items():
            trial.set_user_attr(key, item)
        study.tell(trial, value, state=state)
    except (RuntimeError, ValueError) as error:
        print(f"trial {trial.number}: not told ({error}), result dropped",
              flush=True)
        return False
    return True


def _result_of(study: optuna.Study, result: Dict, args: argparse.Namespace,
               reported: set) -> tuple:
    """What to tell the study for a finished run.

    Returns:
        (str, dict, float, TrialState): status ('diverged' for a run that
          ended early or non-finite), user attributes, value (completed
          trials) and state (other trials).
    """
    status = result["status"]
    attrs = {}
    if result.get("params") is not None:
        attrs["params_M"] = result["params"] / 1e6
    value, state = None, None
    if status == "ok":
        last = result["history"]["val_r2"][-1]
        attrs["curve"] = [
            [e["epoch"], e["r2_mean"]] for e in result["history"]["val_r2"]
        ]
        for key in ("r2_top", "r2_base", "r2_gauges", "epoch"):
            attrs[key] = last[key]
        if last["epoch"] == args.epochs and math.isfinite(last["r2_mean"]):
            value = last["r2_mean"]
        else:
            status = "diverged"
    if status == "diverged":
        attrs["diverged"] = True
        value = diverged_score(study)
    elif status == "pruned":
        attrs["pruned_at"] = max(reported) if reported else 0
        state = TrialState.PRUNED
    elif status == "too_large":
        attrs["over_cap"] = True
        state = TrialState.FAIL
    elif status == "oom":
        attrs["oom"] = True
        state = TrialState.FAIL
    elif status != "ok":  # shelved
        attrs["shelved"] = True
        state = TrialState.FAIL
    return status, attrs, value, state


def run_trial(study: optuna.Study,
              trial: optuna.Trial,
              args: argparse.Namespace,
              owned: Optional[Callable[[], bool]] = None) -> Dict:
    """Draws, trains and scores one trial, then tells the study.

    Args:
        study (optuna.Study): The study.
        trial (optuna.Trial): The trial (from study.ask()).
        args (argparse.Namespace): Options of the driver.
        owned (callable, optional): False once the lock of the study is
          lost (hpo/pick.py); the result is then dropped.

    Returns:
        dict: 'trial' (number), 'status' (that of common.run_unit, or
          'lost' when the result was dropped) and 'run_dir'.
    """
    cfg = S.formatted(S.suggest(trial, args.model))
    run_dir = trial.user_attrs.get("resume_from") or os.path.join(
        args.root, common.TRIALS_DIR, study.study_name, f"t{trial.number:03d}")
    trial.set_user_attr("run_dir", run_dir)
    trial.set_user_attr("config", cfg)
    outcome = {"trial": trial.number, "status": "lost", "run_dir": run_dir}
    # A requeued run that had finished (history written, never told)
    # replays its VAL lines: the replay is not pruned.
    replay = (not os.path.exists(os.path.join(run_dir, "recorded.json")) and
              os.path.exists(common.history_path(run_dir, args.model)))
    pruning = args.model not in C.NO_PRUNING and not replay
    reported = set()
    lost = []

    def on_val(scores: Dict) -> bool:
        epoch, value = scores["epoch"], scores["r2_mean"]
        if epoch in reported or not math.isfinite(value):
            return False
        reported.add(epoch)
        try:
            trial.report(value, epoch)
        except (RuntimeError, ValueError):
            lost.append(epoch)  # finished elsewhere: stop the run
            return True
        # Never pruned at its last epoch: the run is done.
        return pruning and epoch < args.epochs and trial.should_prune()

    if args.dry_run:
        result = common.stub_unit(cfg, trial.number, args.epochs, on_val,
                                  run_dir, args.stub_seconds)
    else:
        cmd = common.train_command(args, args.model, args.tower, run_dir, cfg,
                                   C.SEARCH_TRAIN_SPLIT, args.epochs,
                                   C.TRIAL_SEED)
        result = common.run_unit(common.search_id(args.model, args.tower),
                                 cmd,
                                 run_dir,
                                 args.root,
                                 args.model,
                                 on_val,
                                 config=cfg,
                                 owned=owned)
    status = result["status"]
    if lost:
        print(f"trial {trial.number}: finished elsewhere, result dropped",
              flush=True)
        return outcome
    if status in ("stopped", "config_mismatch"):
        print(f"trial {trial.number:3d} {args.model}/{args.tower} {status}",
              flush=True)
        outcome["status"] = status
        # Left RUNNING: the next worker resumes it (a held one, once an
        # operator removed its CONFIG_MISMATCH marker).
        return outcome
    status, attrs, value, state = _result_of(study, result, args, reported)
    if not _tell(study, trial, attrs, owned, value, state):
        return outcome
    _recorded(run_dir, trial.number, status)
    outcome["status"] = status
    tail = f" val {value:.4f} {cfg}" if status == "ok" else ""
    print(f"trial {trial.number:3d} {args.model}/{args.tower} {status}{tail}",
          flush=True)
    return outcome


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Command-line options."""
    parser = common.driver_parser(__doc__, C.EPOCHS_TRIAL)
    parser.add_argument("--max_new",
                        type=int,
                        default=0,
                        help="Run at most this many trials, then exit.")
    parser.add_argument("--exclusive",
                        action="store_true",
                        help="The caller holds the lock of the study "
                        "(hpo/pick.py): requeue every RUNNING trial.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None,
         outcomes: Optional[List[Dict]] = None,
         owned: Optional[Callable[[], bool]] = None) -> optuna.Study:
    """Runs trials of one study until its budget is reached.

    Args:
        argv (List[str], optional): Command-line options.
        outcomes (list, optional): Receives the outcome of every trial run
          (see `run_trial`).
        owned (callable, optional): False once the lock of the study is
          lost; passed to `run_trial`.

    Returns:
        optuna.Study: The study.
    """
    args = parse_args(argv)
    study = open_study(args.root, args.model, args.tower)
    seed = study_seed(args.model, args.tower)
    requeued = requeue_stale(study, args.exclusive)
    if requeued:
        print(f"trials of dead workers enqueued again: {requeued}", flush=True)
    started = 0
    while ((not args.max_new or started < args.max_new) and
           not common.STOP.is_set()):
        done, running, waiting = budget(study)
        target = target_trials(args.root, args.model, args.n_trials)
        reason = capped(
            failed_draws(args.root, args.model, args.tower,
                         study.get_trials(deepcopy=False)))
        if reason and not waiting:  # an enqueued trial still resumes
            print(f"study {study.study_name}: {reason['message']}, stopping",
                  flush=True)
            break
        if not waiting and done + running >= target:
            # Only the non-shelved towers decide (maybe_extend's default).
            if (target == args.n_trials and not running and
                    maybe_extend(args.root, args.model, args.n_trials)):
                continue  # the budget grew
            break
        trial = study.ask()
        # Seeded per trial: reproducible, and two workers never share a draw.
        study.sampler = sampler(seed + trial.number)
        outcome = run_trial(study, trial, args, owned)
        if outcomes is not None:
            outcomes.append(outcome)
        if outcome["status"] in ("lost", "config_mismatch"):
            break
        if outcome["status"] != "too_large":  # over-cap: draw again
            started += 1
    done, running, _ = budget(study)
    best = [t for t in study.get_trials(deepcopy=False) if eligible(t)]
    print(f"study {study.study_name}: {done} trials counted, {running} running",
          flush=True)
    if best:
        top = max(best, key=lambda t: t.value)
        print(
            f"best: trial {top.number}, val {top.value:.4f} "
            f"{top.user_attrs.get('config', top.params)}",
            flush=True)
    return study


if __name__ == "__main__":
    main()
