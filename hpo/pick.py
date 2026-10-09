# pylint: disable=wrong-import-position
# pylint: disable=use-dict-literal
# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-locals
# pylint: disable=too-many-instance-attributes
# pylint: disable=broad-exception-caught
"""Units, locks, phase advancement and pick order of the track's workers.

A worker is given a list of models and runs every phase of them, one unit
at a time:

  search    one trial of a (model, tower) study; the study has one worker
            at a time (lock 'study_<model>_<tower>'), so its trials are
            sequential and the lock holder is the only writer of its
            journal;
  confirm   one (configuration, seed) of a frozen plan (hpo/confirm.py);
  final     one seed of the retraining of a winner (hpo/final.py).

The next unit of a worker (`pick`) is, in this order: its own unit left
unfinished by its previous incarnation (same owner, whatever its restart
count); a search trial of the study with the most work left; a
confirmation unit of a frozen plan; a final unit of a winner. Within a
phase the order is the same (most work left first). Work is counted in
search trials (a 300-epoch unit is UNIT_FACTOR trials), times an optional
cost per model (`cost`, e.g. GPU-hours per trial, so the slowest models
start first). Units done, shelved, held or claimed by another live process
are skipped before any lock is taken.

Locks are files created with O_EXCL under <root>/locks/ holding an owner
token, beaten (mtime) every LOCK_BEAT_SECONDS. A lock is stale when its
mtime is STALE_SECONDS older than a probe file touched now
(common.server_time: no machine clock is compared), or at once when it
belongs to another incarnation of the same owner or to a dead process of
this host (so does a claim, common.claim). A stale lock is taken
over under a breaker lock (only one worker breaks it, and only it removes
the breaker); a worker that finds its token gone stops its unit.

Shelving: a study with fewer than N_TRIALS counted trials is shelved when
it stops drawing (search.capped); with N_TRIALS counted trials it is never
shelved for that, and its plan is frozen from the counted trials. A study
is also shelved with too few eligible trials for a plan, no finite median
in phase 2, or a unit shelved for good. A shelved study is left out of the
gate of its model, has no winner, and is missing in the test and the
leaderboard. Retry: `advance` retries a shelved unit, or a study shelved
for a kind in RETRIED_KINDS, once, RETRY_SHELVED_AFTER after its shelving,
with its failure counts reset (a study counts only the trials shelved or
out of memory after that); a second shelving is for good, and so is every
shelving once READY_FOR_TEST.json or sealed/ exists. A study capped by a
kind in RETRIED_KINDS during its extension is shelved and retried the same
way; capped again, its plan is frozen from its counted trials. An operator
retries a shelved unit or study at once, even after a shelving for good
(before READY_FOR_TEST.json), by the unit id of the alerts and the status:

    python hpo/pick.py --root=outputs/hpo --retry=final/tcn_opt2/s0
    python hpo/pick.py --root=outputs/hpo --retry=search/tcn_opt2  # study

(removing a marker by hand is not enough: the failure counts stay). A
unit held for an operator (configuration mismatch) is never picked.

`advance` moves the phases forward from the files alone, idempotently
(every decision is a write-once file): the shelvings and retries; the
extension decision (extended or not, taken once) once the studies of a
model not shelved for good have N_TRIALS counted trials (it waits while a
study is shelved with its retry pending); the plan of a study once
its search is over (with the extension decided); the winner once every
unit of a plan is done; READY_FOR_TEST once every final unit of every
learned model is trained (`mark_ready`; the test itself is opened by
hand, hpo/final.py --open_test).
"""

import argparse
import collections
import json
import os
import socket
import statistics
import sys
import threading
import time
import uuid
from typing import Callable, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from optuna.trial import TrialState

from hpo import common
from hpo import confirm
from hpo import final
from hpo import search
from hpo import constants as C
from hpo import search_space as S

PHASES = ("search", "confirm", "final")
# GPU cost of a 300-epoch unit relative to a search trial (approximate:
# the final units train on 1,728 instead of 1,380 simulations).
UNIT_FACTOR = C.EPOCHS_FINAL / C.EPOCHS_TRIAL

Unit = collections.namedtuple("Unit",
                              "phase model tower rank seed",
                              defaults=(None, None))


def unit_id(unit: Unit) -> str:
    """Name of a unit in the records, events, alerts and status."""
    if unit.phase == "search":
        return common.search_id(unit.model, unit.tower)
    if unit.phase == "confirm":
        return common.confirm_id(unit.model, unit.tower, unit.rank, unit.seed)
    return common.final_id(unit.model, unit.tower, unit.seed)


def _unit_of(phase: str, study: str, rest: List[str]) -> Unit:
    """The unit of a parsed id (ValueError if malformed)."""
    model, tower = study.rsplit("_", 1)
    if phase == "search" and not rest:
        return Unit("search", model, tower)
    if phase == "confirm" and len(rest) == 1:
        rank, seed = rest[0].removeprefix("c").split("_s")
        return Unit("confirm", model, tower, int(rank), int(seed))
    if phase == "final" and len(rest) == 1:
        return Unit("final", model, tower, None, int(rest[0][1:]))
    raise ValueError(phase)


def parse_unit(text: str) -> Unit:
    """Inverse of `unit_id`; ValueError for any other text."""
    phase, _, rest = text.partition("/")
    try:
        study, *more = rest.split("/")
        unit = _unit_of(phase, study, more)
    except ValueError:
        unit = None
    if (unit is None or unit_id(unit) != text or unit.model not in S.SPACE or
            unit.tower not in C.TOWERS_SEARCHED):
        raise ValueError(f"not a unit id: {text!r} (search/<model>_<tower>, "
                         "confirm/<model>_<tower>/c<rank>_s<seed> or "
                         "final/<model>_<tower>/s<seed>)")
    return unit


def lock_name(unit: Unit) -> str:
    """Lock of a unit: the study for a search trial, else the unit."""
    if unit.phase == "search":
        return f"study_{unit.model}_{unit.tower}"
    return unit_id(unit).replace("/", "_")


def run_dir(root: str, unit: Unit) -> Optional[str]:
    """Run directory of a confirmation or final unit (None for search)."""
    if unit.phase == "confirm":
        return confirm.unit_dir(root, unit.model, unit.tower, unit.rank,
                                unit.seed)
    if unit.phase == "final":
        return final.unit_dir(root, unit.model, unit.tower, unit.seed)
    return None


def locks_dir(root: str) -> str:
    """Directory of the lock files."""
    return os.path.join(root, "locks")


def read_lock(path: str) -> Optional[Dict]:
    """Content of a lock file: None if missing, {} if not readable yet."""
    try:
        with open(path, encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, OSError):
        return {}


class Lock:
    """An O_EXCL lock file with an owner token and a heartbeat (mtime).

    Args:
        root (str): Root of the track.
        name (str): Lock name (file <root>/locks/<name>.lock).
        owner (str): Stable identity of the worker (kept across its
          restarts).
        restart (int): Incarnation of the owner.
        stale_seconds (float, optional): Age of a stale lock (default
          STALE_SECONDS).
    """

    def __init__(self,
                 root: str,
                 name: str,
                 owner: str,
                 restart: int,
                 stale_seconds: Optional[float] = None):
        self.dir = locks_dir(root)
        self.path = os.path.join(self.dir, f"{name}.lock")
        self.owner = owner
        self.restart = int(restart)
        self.stale_seconds = (C.STALE_SECONDS
                              if stale_seconds is None else stale_seconds)
        self.token = (f"{owner}:{self.restart}:{socket.gethostname()}:"
                      f"{os.getpid()}:{uuid.uuid4().hex[:8]}")

    def acquire(self) -> bool:
        """Takes the lock (a stale one is taken over); False if held."""
        os.makedirs(self.dir, exist_ok=True)
        for _ in range(C.LOCK_TRIES):
            if self._create():
                return True
            holder = read_lock(self.path)
            if holder is None:
                continue  # released meanwhile
            if not self.stale(holder) or not self._break(holder):
                return False
        return False

    def _create(self) -> bool:
        try:
            handle = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        with os.fdopen(handle, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "token": self.token,
                    "owner": self.owner,
                    "restart": self.restart,
                    "host": socket.gethostname(),
                    "pid": os.getpid(),
                    "time": common.now()
                }, file)
        return True

    def age(self) -> float:
        """Seconds since the last beat (server clock; inf if missing)."""
        try:
            mtime = os.path.getmtime(self.path)
        except FileNotFoundError:
            return float("inf")
        return common.server_time(self.dir) - mtime

    def stale(self, holder: Dict) -> bool:
        """A lock of a dead worker (not beaten, or its process of this host
        gone), or of another incarnation of ours (same owner, different
        restart count: the worker was restarted)."""
        if (holder.get("owner") == self.owner and
                int(holder.get("restart", self.restart)) != self.restart):
            return True
        if common.dead_process(f"{holder.get('host')}:{holder.get('pid')}"):
            return True
        return self.age() > self.stale_seconds

    def _break(self, holder: Dict) -> bool:
        """Removes a stale lock under a breaker lock (one breaker wins)."""
        breaker = f"{self.path}.break"
        try:
            handle = os.open(breaker, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                age = common.server_time(self.dir) - os.path.getmtime(breaker)
            except FileNotFoundError:
                return False
            if age > C.BREAK_LOCK_SECONDS:  # its worker died while breaking
                try:
                    os.rename(breaker, f"{breaker}.{uuid.uuid4().hex}")
                except FileNotFoundError:
                    pass
            return False
        with os.fdopen(handle, "w", encoding="utf-8") as file:
            file.write(self.token)
        try:
            current = read_lock(self.path)
            if current is None:
                return True
            if (current.get("token") != holder.get("token") or
                    not self.stale(current)):
                return False
            moved = f"{self.path}.stale.{uuid.uuid4().hex}"
            try:
                os.rename(self.path, moved)
            except FileNotFoundError:
                return True  # released meanwhile
            with open(os.path.join(self.dir, "takeovers.log"),
                      "a",
                      encoding="utf-8") as log:
                log.write(f"{common.now()} {self.token} took over "
                          f"{os.path.basename(self.path)} from "
                          f"{current.get('token')}\n")
            os.remove(moved)  # a few bytes of lock record, logged above
            return True
        finally:
            self._remove_breaker(breaker)

    def _remove_breaker(self, breaker: str) -> None:
        """Removes the breaker if it is still ours (a slow breaker may have
        had it renamed away as stale, and another worker made a new one)."""
        try:
            with open(breaker, encoding="utf-8") as file:
                mine = file.read() == self.token
            if mine:
                os.remove(breaker)
        except FileNotFoundError:
            pass

    def state(self) -> str:
        """'held' (ours), 'lost' (readable, with another token), 'missing'
        or 'unknown' (not readable now: a transient error)."""
        holder = read_lock(self.path)
        if holder is None:
            return "missing"
        if not holder:
            return "unknown"
        return "held" if holder.get("token") == self.token else "lost"

    def owned(self,
              tries: int = C.OWNED_TRIES,
              wait: float = C.OWNED_WAIT_SECONDS) -> bool:
        """Is the lock still ours? A transient read error is retried; after
        `tries` unclear reads the answer is False (the safe side: a result
        is dropped rather than told twice)."""
        for attempt in range(tries):
            state = self.state()
            if state in ("held", "lost"):
                return state == "held"
            if attempt + 1 < tries:
                time.sleep(wait)
        return False

    def beat(self) -> str:
        """Beats the lock if it is still ours; returns its `state` (an
        error of utime raises OSError)."""
        state = self.state()
        if state == "held":
            os.utime(self.path, None)
        return state

    def release(self) -> None:
        """Removes the lock if it is still ours."""
        if self.owned():
            try:
                os.remove(self.path)
            except FileNotFoundError:
                pass


class Beater:
    """Beats a lock every `interval` seconds (and calls `on_beat`); calls
    `on_lost` (once) if the lock was taken over (readable, with another
    token) or after LOCK_BEAT_FAILURES unclear beats in a row (lock
    missing or unreadable, utime error): the unit is then stopped, on the
    safe side. A transient error is retried at the next beat."""

    def __init__(self,
                 lock: Lock,
                 on_lost: Callable[[], None],
                 interval: float = C.LOCK_BEAT_SECONDS,
                 on_beat: Optional[Callable[[], None]] = None):
        self.lock = lock
        self.on_lost = on_lost
        self.on_beat = on_beat
        self.interval = interval
        self.stop = threading.Event()
        self.lost = False
        self.failures = 0
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop.wait(self.interval):
            try:
                state = self.lock.beat()
            except OSError as error:
                state = f"unknown ({error})"
            if state == "held":
                self.failures = 0
                if self.on_beat is not None:
                    try:
                        self.on_beat()
                    except OSError:
                        pass
                continue
            if state != "lost":
                self.failures += 1
                print(
                    f"lock {os.path.basename(self.lock.path)}: {state} "
                    f"({self.failures}/{C.LOCK_BEAT_FAILURES})",
                    flush=True)
                if self.failures < C.LOCK_BEAT_FAILURES:
                    continue
            self.lost = True
            self.on_lost()
            return

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()


# --- State of a study, from its journal and files -----------------------------


def _hours(trial) -> Optional[float]:
    if trial.datetime_start is None or trial.datetime_complete is None:
        return None
    return (trial.datetime_complete -
            trial.datetime_start).total_seconds() / 3600


def study_shelved_for_good(root: str,
                           model: str,
                           tower: str,
                           frozen: Optional[bool] = None) -> bool:
    """A shelved study that will not be retried: a shelving that is not
    retried (its kind), a second shelving after its retry, or any shelving
    once the test is ready (`frozen`)."""
    marker = common.read_json(common.study_shelved_path(root, model, tower))
    if marker is None:
        return False
    if (common.test_frozen(root) if frozen is None else frozen):
        return True
    if not marker.get("retry"):
        return True
    record = common.read_json(common.study_retried_path(root, model, tower))
    return record is not None and record.get("shelved") != marker


def unit_done(root: str, unit: Unit) -> bool:
    """A confirmation or final unit with its result recorded."""
    return os.path.exists(common.result_path(run_dir(root, unit)))


def unit_shelved(root: str, unit: Unit) -> bool:
    """A confirmation or final unit shelved after its failures."""
    return common.is_shelved(run_dir(root, unit))


def _search_state(root: str, model: str, tower: str, n_trials: int) -> Dict:
    """The search of a study, from its journal: counted, pruned, failed,
    running and waiting trials, the failed draws and whether it stopped
    drawing (`capped`), the best eligible trial, the median hours of a
    completed trial, a trial held for an operator."""
    study = search.open_study(root, model, tower, create=False)
    trials = study.get_trials(deepcopy=False) if study else []
    states = collections.Counter(t.state for t in trials)
    eligible = [t for t in trials if search.eligible(t)]
    best = max(eligible, key=lambda t: t.value) if eligible else None
    durations = [
        h for h in (_hours(t) for t in trials if t.state == TrialState.COMPLETE)
        if h
    ]
    draws = search.failed_draws(root, model, tower, trials)
    return dict(
        target=search.target_trials(root, model, n_trials),
        counted=sum(search.counted(t) for t in trials),
        complete=states[TrialState.COMPLETE],
        pruned=sum(t.state == TrialState.PRUNED and not search.over_cap(t)
                   for t in trials),
        failed=states[TrialState.FAIL],
        capped=search.capped(draws),
        last_trial=max((t.number for t in trials), default=-1),
        running=states[TrialState.RUNNING],
        waiting=states[TrialState.WAITING],
        best=best.value if best else None,
        best_trial=best.number if best else None,
        hours_per_trial=(statistics.median(durations)
                         if len(durations) >= C.MIN_DURATIONS else None),
        # A RUNNING or WAITING trial whose run is held for an operator
        # (configuration mismatch): the study is not picked.
        held=any(
            common.held(search.trial_run_dir(t))
            for t in trials
            if t.state in (TrialState.RUNNING, TrialState.WAITING)),
        extension_decided=os.path.exists(search.extension_path(root, model)),
        **draws)


def _shelving_state(root: str, model: str, tower: str, frozen: bool) -> Dict:
    """The shelving of a study: shelved, for good, retry due, retried."""
    path = common.study_shelved_path(root, model, tower)
    shelved = os.path.exists(path)
    for_good = study_shelved_for_good(root, model, tower, frozen)
    return dict(shelved=shelved,
                shelved_for_good=for_good,
                retry_due=(shelved and not for_good and
                           common.marker_age(path) >= C.RETRY_SHELVED_AFTER),
                retried=os.path.exists(
                    common.study_retried_path(root, model, tower)))


def _units_state(root: str, model: str, tower: str, frozen: bool) -> Dict:
    """The confirmation and final units of a study: plan, winner, units
    done, shelved (for good, or with their retry due)."""
    plan = common.read_json(confirm.plan_path(root, model, tower))
    winner = common.read_json(confirm.winner_path(root, model, tower))
    confirm_units = [
        Unit("confirm", model, tower, rank, seed)
        for rank, seed in confirm.units(plan)
    ] if plan else []
    final_units = [
        Unit("final", model, tower, None, seed) for seed in range(C.N_SEEDS)
    ] if winner else []
    units = confirm_units + final_units
    return dict(
        plan=plan is not None,
        stopped_early=(plan or {}).get("stopped_early"),
        winner=winner["winner_median"] if winner else None,
        confirm_units=confirm_units,
        final_units=final_units,
        confirm_done=sum(unit_done(root, u) for u in confirm_units),
        final_done=sum(unit_done(root, u) for u in final_units),
        units_shelved=sum(unit_shelved(root, u) for u in units),
        units_shelved_for_good=[
            unit_id(u)
            for u in units
            if common.shelved_for_good(run_dir(root, u), frozen)
        ],
        units_retry_due=[
            u for u in units if common.retry_due(run_dir(root, u), frozen)
        ])


def study_state(root: str,
                model: str,
                tower: str,
                n_trials: int,
                cost: Optional[Dict[str, float]] = None) -> Dict:
    """Everything the picker, `advance` and the status page need to know
    about one (model, tower), read from the files only (`cost`: cost of a
    search trial per model, 1 if missing)."""
    frozen = common.test_frozen(root)
    state = dict(model=model,
                 tower=tower,
                 n_trials=n_trials,
                 cost=(cost or {}).get(model, 1.0))
    state.update(_search_state(root, model, tower, n_trials))
    state.update(_shelving_state(root, model, tower, frozen))
    state.update(_units_state(root, model, tower, frozen))
    state["search_pending"] = (not state["shelved"] and not state["held"] and (
        (not state["capped"] and state["counted"] < state["target"]) or
        state["running"] > 0 or state["waiting"] > 0))
    state["phase"] = phase_of(state)
    state["work_left"] = work_left(state)
    return state


def phase_of(state: Dict) -> str:
    """Phase of a study: search, waiting (for the other towers of its
    model), held (a trial held for an operator), confirm, final, done or
    shelved."""
    if state["shelved"]:
        return "shelved"
    if state["held"] and not state["plan"]:
        return "held"
    if state["final_units"]:
        return ("done" if state["final_done"] == len(state["final_units"]) else
                "final")
    if state["plan"]:
        return "confirm"
    return "search" if state["search_pending"] else "waiting"


def work_left(state: Dict) -> float:
    """Work left in a study: search trials (a 300-epoch unit counts
    UNIT_FACTOR) times the cost of a trial of its model."""
    if state["shelved"]:
        return 0.0
    trials = max(0, state["target"] - state["counted"])
    confirm_left = (len(state["confirm_units"]) -
                    state["confirm_done"] if state["plan"] else C.N_TOP *
                    C.N_SEEDS)
    final_left = (len(state["final_units"]) -
                  state["final_done"] if state["final_units"] else C.N_SEEDS)
    return state["cost"] * (trials + UNIT_FACTOR * (confirm_left + final_left))


_BROKEN: set = set()  # studies alerted as unreadable by this process


def broken(root: str, name: str, reason: str = "study unreadable") -> None:
    """An unreadable study (e.g. a journal line damaged mid-file) or model:
    one alert per process; the others go on."""
    if name not in _BROKEN:
        _BROKEN.add(name)
        common.alert(root, name, f"{reason}, skipped",
                     {"error": common.traceback_tail()})


def all_states(root: str,
               models,
               n_trials: int,
               cost: Optional[Dict[str, float]] = None) -> List[Dict]:
    """`study_state` of every readable (model, tower) of `models`."""
    states = []
    for model in models:
        for tower in C.TOWERS_SEARCHED:
            try:
                states.append(study_state(root, model, tower, n_trials, cost))
            except Exception:
                broken(root, common.search_id(model, tower))
    return states


# --- Phase advancement -------------------------------------------------------


def shelve_reason(state: Dict) -> Optional[Dict]:
    """Why a study must be shelved now, or None: the arguments of
    common.shelve_study ('kind', 'message', 'retry': retried once for a
    kind in RETRIED_KINDS, 'units'). A study with its N_TRIALS counted
    trials is shelved only once, for a retry, when such a kind stops its
    extension; else its plan is frozen."""
    if state["shelved"]:
        return None
    units = state["units_shelved_for_good"]
    if units:
        return dict(kind="unit_shelved",
                    message=f"unit shelved for good: {', '.join(units)}",
                    retry=False,
                    units=units)
    if state["plan"] or not state["capped"]:
        return None
    retry = state["capped"]["kind"] in C.RETRIED_KINDS
    if state["counted"] >= state["n_trials"]:  # in an extension
        if not (retry and not state["retried"] and
                state["counted"] < state["target"]):
            return None
    return dict(**state["capped"], retry=retry, units=[])


def retry_study(root: str, state: Dict, force: bool = False) -> bool:
    """Retries a shelved study once (the caller checked `retry_due`;
    `force`: an operator, even after a shelving for good): only the trials
    shelved or out of memory after this count toward a new shelving."""
    model, tower = state["model"], state["tower"]
    marker = common.read_json(common.study_shelved_path(root, model, tower))
    if marker is None:
        return False
    path = common.study_retried_path(root, model, tower)
    (common.write_json if force else common.write_once)(path, {
        "time": common.now(),
        "shelved": marker,
        "after_trial": state["last_trial"]
    })
    if (common.read_json(path) or {}).get("shelved") != marker:
        return False  # shelved again after its retry: for good
    try:
        os.remove(common.study_shelved_path(root, model, tower))
    except FileNotFoundError:
        return False
    common.alert(root, common.search_id(model, tower),
                 "shelved study retried (once)", {"shelved": marker})
    return True


def _driver_args(root: str, model: str, tower: str, n_trials: int):
    return confirm.parse_args([
        f"--model={model}", f"--tower={tower}", f"--root={root}",
        f"--n_trials={n_trials}"
    ])


def _advance_model(root: str, model: str, n_trials: int,
                   states: Dict[str, Dict]) -> List[str]:
    """The gate of a model: its non-shelved studies all have `n_trials`
    counted trials and no tower waits for its retry (a tower shelved for
    good never holds the others back); then the extension rule (on the
    non-shelved studies, decided once: a tower retried later does not
    reopen it), the plans and the winners."""
    events = []
    active = [t for t, s in states.items() if not s["shelved"]]
    if not active or not all(states[t]["counted"] >= n_trials for t in active):
        return events
    if not states[active[0]]["extension_decided"]:
        if any(s["shelved"] and not s["shelved_for_good"]
               for s in states.values()):
            return events  # a tower's retry is pending: it may decide too
        if search.maybe_extend(root, model, n_trials, active, decide=True):
            events.append(f"{model}: extended by {C.EXTEND_BY} trials")
        states = {
            t: study_state(root, model, t, n_trials) for t in C.TOWERS_SEARCHED
        }
    for tower, state in states.items():
        if state["shelved"]:
            continue
        args = _driver_args(root, model, tower, n_trials)
        # A held trial waits for an operator: no plan (and no shelving).
        if (not state["plan"] and not state["search_pending"] and
                not state["held"] and
            (state["counted"] >= state["target"] or state["capped"])):
            try:
                confirm.freeze(args)
                events.append(f"{model}/{tower}: plan frozen")
            except SystemExit as error:
                if common.shelve_study(root, model, tower, "no_plan",
                                       f"no plan: {error}"):
                    events.append(f"{model}/{tower}: shelved ({error})")
                continue
        plan = common.read_json(confirm.plan_path(root, model, tower))
        if (plan and state["winner"] is None and all(
                unit_done(root, Unit("confirm", model, tower, r, s))
                for r, s in confirm.units(plan))):
            if confirm.summarize(args, plan):
                events.append(f"{model}/{tower}: winner")
            elif os.path.exists(common.study_shelved_path(root, model, tower)):
                events.append(f"{model}/{tower}: shelved (no finite median)")
    return events


def mark_ready(root: str, n_trials: int = C.N_TRIALS) -> bool:
    """Writes READY_FOR_TEST.json (once, with an alert) if every final unit
    of every learned model is trained, whatever the caller's models.
    Returns True if this call wrote it."""
    path = common.ready_path(root)
    if (os.path.exists(path) or not all_done(root, S.LEARNED, n_trials) or
            not common.write_once(path, {
                "models": list(S.LEARNED),
                "time": common.now()
            })):
        return False
    common.alert(
        root, "track", "every final unit is trained: "
        "ready to open the test (hpo/final.py --open_test)",
        {"models": list(S.LEARNED)})
    return True


def _advance_all(root: str, models, n_trials: int) -> List[str]:
    """The body of `advance` (its lock held)."""
    events = []
    for model in models:
        try:
            events += _advance_one(root, model, n_trials)
        except Exception:
            broken(root, f"advance/{model}", "advance failed")
    if all_done(root, models, n_trials) and mark_ready(root, n_trials):
        events.append("ready for test")
    return events


def _advance_one(root: str, model: str, n_trials: int) -> List[str]:
    """`advance` of one model: retries, shelvings, then its gate."""
    events = []
    states = {
        t: study_state(root, model, t, n_trials) for t in C.TOWERS_SEARCHED
    }
    for tower, state in states.items():
        for unit in state["units_retry_due"]:
            if common.retry_unit(root, unit_id(unit), run_dir(root, unit)):
                events.append(f"{unit_id(unit)}: retried")
        if state["retry_due"] and retry_study(root, state):
            events.append(f"{model}/{tower}: retried")
            states[tower] = state = study_state(root, model, tower, n_trials)
        reason = shelve_reason(state)
        if reason is not None:
            state.update(shelved=True, shelved_for_good=not reason["retry"])
            if common.shelve_study(root, model, tower, **reason):
                events.append(f"{model}/{tower}: shelved ({reason['message']})")
    return events + _advance_model(root, model, n_trials, states)


def advance(root: str, models, n_trials: int = C.N_TRIALS) -> List[str]:
    """Advances the phases of `models` (see the module docstring); one
    caller at a time (lock 'advance', beaten every ADVANCE_BEAT_SECONDS and
    stale after ADVANCE_STALE_SECONDS, so a killed holder blocks the others
    briefly), others return at once.

    Returns:
        List[str]: What changed.
    """
    lock = Lock(root,
                "advance",
                f"advance-{socket.gethostname()}-{os.getpid()}",
                0,
                stale_seconds=C.ADVANCE_STALE_SECONDS)
    if not lock.acquire():
        return []
    # Every decision is a write-once file: a second advancer after a lost
    # lock is harmless, so a lost lock is only logged.
    try:
        with Beater(lock,
                    lambda: print("advance: lock lost", flush=True),
                    interval=C.ADVANCE_BEAT_SECONDS):
            events = _advance_all(root, models, n_trials)
    finally:
        lock.release()
    for event in events:
        print(f"advance: {event}", flush=True)
    return events


def all_done(root: str, models, n_trials: int = C.N_TRIALS) -> bool:
    """Every final unit of every study of `models` is trained, or the study
    is shelved for good (`study_shelved_for_good`; a shelving still to be
    retried is not terminal); never while a study is unreadable."""
    states = all_states(root, models, n_trials)
    return len(states) == len(models) * len(C.TOWERS_SEARCHED) and all(
        state["phase"] == "done" or
        (state["phase"] == "shelved" and state["shelved_for_good"])
        for state in states)


# --- Pick -------------------------------------------------------------------


def worker_path(root: str, owner: str) -> str:
    """Record of a worker (its current unit)."""
    return os.path.join(root, "workers", f"{owner}.json")


def candidates(states: List[Dict]) -> List[Unit]:
    """Units that can run now, in pick order (phase, then most work left
    first)."""
    ranked: List[Tuple[tuple, Unit]] = []
    order = {phase: i for i, phase in enumerate(PHASES)}
    for state in states:
        key = (-state["work_left"], state["model"], state["tower"])
        if state["phase"] == "search":
            ranked.append(((order["search"],) + key,
                           Unit("search", state["model"], state["tower"])))
        units = (state["confirm_units"] if state["phase"] == "confirm" else
                 state["final_units"] if state["phase"] == "final" else [])
        for unit in units:
            ranked.append(
                ((order[unit.phase],) + key + (unit.rank or 0, unit.seed),
                 unit))
    return [unit for _, unit in sorted(ranked)]


def pending(root: str,
            unit: Unit,
            n_trials: int,
            owner: Optional[str] = None) -> bool:
    """Is there work in this unit for this worker (`owner`)? Not for a unit
    done, shelved, held for an operator, or claimed by another live process
    (a claim of `owner`, any incarnation, is its own)."""
    if unit.phase == "search":
        return study_state(root, unit.model, unit.tower,
                           n_trials)["search_pending"]
    directory = run_dir(root, unit)
    return not (unit_done(root, unit) or unit_shelved(root, unit) or
                common.held(directory) or common.busy(directory, owner))


def _try(root: str, unit: Unit, owner: str, restart: int,
         n_trials: int) -> Optional[Lock]:
    """The lock of a unit, if free and the unit still pending under it."""
    lock = Lock(root, lock_name(unit), owner, restart)
    if not lock.acquire():
        return None
    try:
        if pending(root, unit, n_trials, owner):
            return lock
    except BaseException:
        lock.release()
        raise
    lock.release()  # finished meanwhile
    return None


def pick(
    root: str,
    owner: str,
    restart: int,
    models,
    n_trials: int = C.N_TRIALS,
    cost: Optional[Dict[str, float]] = None,
    skip=()
) -> Tuple[Optional[Unit], Optional[Lock]]:
    """The next unit of a worker, with its lock held.

    Args:
        root (str): Root of the track.
        owner (str): Stable identity of the worker.
        restart (int): Its incarnation (any other count of the same owner
          is a dead incarnation).
        models: Models of the worker.
        n_trials (int): Budget per study (the protocol: N_TRIALS).
        cost (dict, optional): Cost of a search trial per model.
        skip: Unit ids not to pick (set aside by the worker).

    Returns:
        (Unit, Lock), or (None, None) when nothing can run now.
    """
    record = common.read_json(worker_path(root, owner)) or {}
    if record.get("unit"):
        unit = parse_unit(record["unit"])
        if unit.model in models and unit_id(unit) not in skip:
            lock = _try(root, unit, owner, restart, n_trials)
            if lock:
                return unit, lock
    for unit in candidates(all_states(root, models, n_trials, cost)):
        # Checked before the lock (no lock churn on finished units); a
        # search study is pending by construction of the candidates.
        if unit_id(unit) in skip or (unit.phase != "search" and
                                     not pending(root, unit, n_trials, owner)):
            continue
        lock = _try(root, unit, owner, restart, n_trials)
        if lock:
            return unit, lock
    return None, None


def operator_retry(root: str, text: str) -> bool:
    """Retries a shelved unit or study at once (an operator; even after a
    shelving for good): `text` is a unit id (search/<model>_<tower> for a
    study). A study shelved for that unit is retried too."""
    if common.test_frozen(root):
        sys.exit("nothing is retried once the test is ready")
    unit = parse_unit(text)
    state = study_state(root, unit.model, unit.tower, C.N_TRIALS)
    if unit.phase == "search":
        return retry_study(root, state, force=True)
    marker = common.read_json(
        common.study_shelved_path(root, unit.model, unit.tower)) or {}
    done = common.retry_unit(root, text, run_dir(root, unit), force=True)
    if (done and marker.get("kind") == "unit_shelved" and
            text in marker.get("units", [])):
        retry_study(root, state, force=True)
    return done


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Command-line options of an operator retry."""
    parser = argparse.ArgumentParser(
        description="Retries a shelved unit or study at once.")
    parser.add_argument("--root", default=C.DEFAULT_ROOT)
    parser.add_argument("--retry",
                        required=True,
                        help="Unit id, as in the alerts and the status "
                        "(search/<model>_<tower> for a study).")
    args = parser.parse_args(argv)
    try:
        parse_unit(args.retry)
    except ValueError as error:
        parser.error(str(error))
    return args


def main(argv: Optional[List[str]] = None) -> bool:
    """Operator retry; returns True if something was retried."""
    args = parse_args(argv)
    retried = operator_retry(args.root, args.retry)
    print("retried" if retried else "not shelved: nothing to retry")
    return retried


if __name__ == "__main__":
    main()
