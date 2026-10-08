# pylint: disable=wrong-import-position
# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-locals
"""Units, locks, phase advancement and pick order of the track's workers.

A worker is given a list of models (which models run where is a choice of
each site, outside this repository) and runs every phase of them, one
unit at a time:

  search    one trial of a (model, tower) study; the study has one worker
            at a time (lock 'study_<model>_<tower>'), so its trials are
            sequential and the lock holder is the only writer of its
            journal;
  confirm   one (configuration, seed) of a frozen plan (hpo/confirm.py);
  final     one seed of the retraining of a winner (hpo/final.py).

The next unit of a worker (`pick`) is, in this order: its own unit left
unfinished by its previous incarnation (same owner, e.g. the same Slurm
job after a requeue, whatever its restart count); a search trial of the
study with the most work left; a confirmation unit of a frozen plan; a
final unit of a winner.
Within a phase the order is the same (most work left first). Work is
counted in search trials (a 300-epoch unit is UNIT_FACTOR trials), times
an optional cost per model (`cost`: e.g. GPU-hours per trial measured on
the site, so the slowest models start first).

Locks are files created with O_EXCL under <root>/locks/ (safe on NFSv4),
holding an owner token, and beaten (mtime set by the file server) every
LOCK_BEAT_SECONDS. A lock is stale when its mtime is STALE_MINUTES older
than a probe file touched now (common.server_time: no node clock is
compared), or at once when it belongs to another incarnation of the same
owner (the owner is the identity of the job: an incarnation with a
different restart count is an older one, now dead). A stale lock is taken
over under a breaker lock holding the breaker's token (only one worker
breaks it, and only it removes the breaker); a worker that finds its token
gone stops its unit.

Parking: a study is parked (marker in <root>/parked/) after
PARK_STUDY_AFTER parked trials, MAX_OVER_CAP draws above the parameter
cap, too few eligible trials to freeze a plan, no finite median in phase
2, or a confirmation or final unit whose parking is final. A parked study
is terminal: it is left out of the gate of its model (the extension rule
and the plans of the other towers go on without it), it has no winner,
and the test and the leaderboard record it as missing. Un-parking (once):
a unit or a study parked after failures that all came from one host is
un-parked by the first worker of another host that picks (common.unpark;
a study counts only the trials parked after that).

`advance` moves the phases forward from the files alone, idempotently
(every decision is a write-once file): the extension rule once the
non-parked studies of a model have N_TRIALS counted trials; the plan of a
study once its budget is spent (with the extension decided); the winner
once every unit of a plan is done; READY_FOR_TEST once every final unit
of every learned model is trained (the test itself is opened by hand,
hpo/final.py --open_test). A study that cannot go on is parked with an
alert (see Parking).
"""

import collections
import json
import os
import socket
import statistics
import sys
import threading
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
    """Readable name of a unit (records, events, status)."""
    study = f"{unit.model}_{unit.tower}"
    if unit.phase == "search":
        return f"search/{study}"
    if unit.phase == "confirm":
        return f"confirm/{study}/c{unit.rank}_s{unit.seed}"
    return f"final/{study}/s{unit.seed}"


def parse_unit(text: str) -> Unit:
    """Inverse of `unit_id`."""
    parts = text.split("/")
    model, tower = parts[1].rsplit("_", 1)
    if parts[0] == "search":
        return Unit("search", model, tower)
    if parts[0] == "confirm":
        rank, seed = parts[2][1:].split("_s")
        return Unit("confirm", model, tower, int(rank), int(seed))
    return Unit("final", model, tower, None, int(parts[2][1:]))


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
        owner (str): Stable identity of the worker (survives a requeue).
        restart (int): Incarnation of the owner (larger is newer).
    """

    def __init__(self, root: str, name: str, owner: str, restart: int):
        self.dir = locks_dir(root)
        self.path = os.path.join(self.dir, f"{name}.lock")
        self.owner = owner
        self.restart = int(restart)
        self.token = (f"{owner}:{self.restart}:{socket.gethostname()}:"
                      f"{os.getpid()}:{uuid.uuid4().hex[:8]}")

    def acquire(self) -> bool:
        """Takes the lock (a stale one is taken over); False if held."""
        os.makedirs(self.dir, exist_ok=True)
        for _ in range(3):
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
        """A lock of a dead worker, or of another incarnation of ours (same
        owner, different restart count: the job was restarted)."""
        if (holder.get("owner") == self.owner and
                int(holder.get("restart", self.restart)) != self.restart):
            return True
        return self.age() > C.STALE_MINUTES * 60

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

    def held(self) -> bool:
        """Is the lock still ours?"""
        holder = read_lock(self.path)
        return bool(holder) and holder.get("token") == self.token

    def beat(self) -> bool:
        """Beats the lock if it is still ours; False if it was taken."""
        if not self.held():
            return False
        os.utime(self.path, None)
        return True

    def release(self) -> None:
        """Removes the lock if it is still ours."""
        if self.held():
            os.remove(self.path)


class Beater:
    """Beats a lock every LOCK_BEAT_SECONDS; calls `on_lost` (once) if the
    lock was taken over."""

    def __init__(self, lock: Lock, on_lost: Callable[[], None]):
        self.lock = lock
        self.on_lost = on_lost
        self.stop = threading.Event()
        self.lost = False
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop.wait(C.LOCK_BEAT_SECONDS):
            if not self.lock.beat():
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


parked_path = confirm.parked_path
park_study = confirm.park_study


def unparked_path(root: str, model: str, tower: str) -> str:
    """Record of the (single) un-parking of a study."""
    return os.path.join(root, "parked", "unparked", f"{model}_{tower}.json")


def park_final(directory: str) -> bool:
    """A parked unit (run directory) that will not be un-parked: failures
    on several hosts (or none, e.g. a configuration mismatch), or un-parked
    once before."""
    marker = common.read_json(os.path.join(directory, "PARKED"))
    if marker is None:
        return False
    return (len(marker.get("hosts") or []) != 1 or
            bool(common.read_attempts(directory).get("unparked")))


def unit_done(root: str, unit: Unit) -> bool:
    """A confirmation or final unit with its result recorded."""
    return os.path.exists(os.path.join(run_dir(root, unit), "result.json"))


def unit_parked(root: str, unit: Unit) -> bool:
    """A confirmation or final unit parked after its failures."""
    return os.path.exists(os.path.join(run_dir(root, unit), "PARKED"))


def study_state(root: str,
                model: str,
                tower: str,
                n_trials: int,
                cost: Optional[Dict[str, float]] = None) -> Dict:
    """Everything the picker, `advance` and the status page need to know
    about one (model, tower), read from the files only (`cost`: cost of a
    search trial per model, 1 if missing)."""
    study = search.open_study(root, model, tower, create=False)
    trials = study.get_trials(deepcopy=False) if study else []
    states = collections.Counter(t.state for t in trials)
    unparked = common.read_json(unparked_path(root, model, tower))
    after = unparked["after_trial"] if unparked else -1
    parked_trials = [
        t for t in trials if t.user_attrs.get("parked") and t.number > after
    ]
    marker = common.read_json(parked_path(root, model, tower))
    eligible = [t for t in trials if search.eligible(t)]
    durations = [
        h for h in (_hours(t) for t in trials if t.state == TrialState.COMPLETE)
        if h
    ]
    state = {
        "model":
            model,
        "tower":
            tower,
        "target":
            search.target_trials(root, model, n_trials),
        "counted":
            sum(search.counted(t) for t in trials),
        "complete":
            states[TrialState.COMPLETE],
        "pruned":
            sum(t.state == TrialState.PRUNED and not search.over_cap(t)
                for t in trials),
        "over_cap":
            sum(search.over_cap(t) for t in trials),
        "failed":
            states[TrialState.FAIL],
        "parked_trials":
            len(parked_trials),
        "parked_hosts":
            sorted({
                h for t in parked_trials
                for h in t.user_attrs.get("parked_hosts") or ["unknown"]
            }),
        "last_trial":
            max((t.number for t in trials), default=-1),
        "running":
            states[TrialState.RUNNING],
        "waiting":
            states[TrialState.WAITING],
        "best":
            max((t.value for t in eligible), default=None),
        "best_trial":
            (max(eligible, key=lambda t: t.value).number if eligible else None),
        "hours_per_trial":
            (statistics.median(durations) if len(durations) >= 3 else None),
        "parked":
            marker is not None,
        "park_hosts": (marker or {}).get("hosts") or [],
        "unparked":
            unparked is not None,
        "extended":
            os.path.exists(search.extension_path(root, model)),
        "cost": (cost or {}).get(model, 1.0),
    }
    plan = common.read_json(confirm.plan_path(root, model, tower))
    winner = common.read_json(confirm.winner_path(root, model, tower))
    state["plan"] = plan is not None
    state["winner"] = winner["winner_median"] if winner else None
    state["confirm_units"] = ([
        Unit("confirm", model, tower, rank, seed)
        for rank, seed in confirm.units(plan)
    ] if plan else [])
    state["final_units"] = ([
        Unit("final", model, tower, None, seed) for seed in range(C.N_SEEDS)
    ] if winner else [])
    state["confirm_done"] = sum(
        unit_done(root, u) for u in state["confirm_units"])
    state["final_done"] = sum(unit_done(root, u) for u in state["final_units"])
    state["units_parked"] = sum(
        unit_parked(root, u)
        for u in state["confirm_units"] + state["final_units"])
    state["units_park_final"] = [
        unit_id(u)
        for u in state["confirm_units"] + state["final_units"]
        if park_final(run_dir(root, u))
    ]
    state["search_pending"] = (state["over_cap"] < C.MAX_OVER_CAP and
                               (state["counted"] < state["target"] or
                                state["running"] > 0 or state["waiting"] > 0))
    state["phase"] = phase_of(state)
    state["work_left"] = work_left(state)
    return state


def phase_of(state: Dict) -> str:
    """Phase of a study: search, waiting (for the other towers of its
    model), confirm, final, done or parked."""
    if state["parked"]:
        return "parked"
    if state["final_units"]:
        return ("done" if state["final_done"] == len(state["final_units"]) else
                "final")
    if state["plan"]:
        return "confirm"
    return "search" if state["search_pending"] else "waiting"


def work_left(state: Dict) -> float:
    """Work left in a study: search trials (a 300-epoch unit counts
    UNIT_FACTOR) times the cost of a trial of its model."""
    if state["parked"]:
        return 0.0
    trials = max(0, state["target"] - state["counted"])
    confirm_left = (len(state["confirm_units"]) -
                    state["confirm_done"] if state["plan"] else C.N_TOP *
                    C.N_SEEDS)
    final_left = (len(state["final_units"]) -
                  state["final_done"] if state["final_units"] else C.N_SEEDS)
    return state["cost"] * (trials + UNIT_FACTOR * (confirm_left + final_left))


def all_states(root: str,
               models,
               n_trials: int,
               cost: Optional[Dict[str, float]] = None) -> List[Dict]:
    """`study_state` of every (model, tower) of `models`."""
    return [
        study_state(root, model, tower, n_trials, cost)
        for model in models
        for tower in C.TOWERS_SEARCHED
    ]


# --- Phase advancement -------------------------------------------------------


def park_reason(state: Dict) -> Optional[Tuple[str, List[str]]]:
    """Why a study must be parked now (reason, hosts of the failures), or
    None."""
    if state["parked"]:
        return None
    if state["units_park_final"]:
        return (f"unit parked for good: {', '.join(state['units_park_final'])}",
                [])
    if state["plan"]:
        return None
    if state["parked_trials"] >= C.PARK_STUDY_AFTER:
        return (f"{state['parked_trials']} trials parked",
                state["parked_hosts"])
    if state["over_cap"] >= C.MAX_OVER_CAP:
        return f"{state['over_cap']} draws above the parameter cap", []
    return None


def unpark_study(root: str, state: Dict, host: str) -> bool:
    """Un-parks a study once if its parking failures all came from one host
    other than `host` (the caller holds the lock of the study); only the
    trials parked after this count toward a new parking."""
    model, tower = state["model"], state["tower"]
    marker = common.read_json(parked_path(root, model, tower))
    if marker is None:
        return False
    hosts = marker.get("hosts") or []
    record = common.read_json(unparked_path(root, model, tower))
    if record is None:
        if len(hosts) != 1 or hosts[0] == host:
            return False
        common.write_once(
            unparked_path(root, model, tower), {
                "time": common.now(),
                "host": host,
                "parked": marker,
                "after_trial": state["last_trial"]
            })
        record = common.read_json(unparked_path(root, model, tower))
    if record.get("parked") != marker:
        return False  # parked again after its un-parking: for good
    try:
        os.remove(parked_path(root, model, tower))
    except FileNotFoundError:
        return False
    common.alert(root, f"study/{model}_{tower}",
                 f"study un-parked by {host} (failures on {hosts[0]} only)", {})
    return True


def _driver_args(root: str, model: str, tower: str, n_trials: int):
    return confirm.parse_args([
        f"--model={model}", f"--tower={tower}", f"--root={root}",
        f"--n_trials={n_trials}"
    ])


def _advance_model(root: str, model: str, n_trials: int,
                   states: Dict[str, Dict]) -> List[str]:
    """The gate of a model: its non-parked studies (a parked tower never
    holds the others back) all have `n_trials` counted trials; then the
    extension rule (on those studies), the plans and the winners."""
    events = []
    active = [t for t, s in states.items() if not s["parked"]]
    if not active or not all(states[t]["counted"] >= n_trials for t in active):
        return events
    if not states[active[0]]["extended"]:
        if search.maybe_extend(root, model, n_trials, active):
            events.append(f"{model}: extended by {C.EXTEND_BY} trials")
            states = {
                t: study_state(root, model, t, n_trials)
                for t in C.TOWERS_SEARCHED
            }
    for tower, state in states.items():
        if state["parked"]:
            continue
        args = _driver_args(root, model, tower, n_trials)
        if not state["plan"] and not state["search_pending"]:
            try:
                confirm.freeze(args)
                events.append(f"{model}/{tower}: plan frozen")
            except SystemExit as error:
                if park_study(root, model, tower, f"no plan: {error}"):
                    events.append(f"{model}/{tower}: parked ({error})")
                continue
        plan = common.read_json(confirm.plan_path(root, model, tower))
        if (plan and state["winner"] is None and all(
                unit_done(root, Unit("confirm", model, tower, r, s))
                for r, s in confirm.units(plan))):
            if confirm.summarize(args, plan):
                events.append(f"{model}/{tower}: winner")
            elif os.path.exists(parked_path(root, model, tower)):
                events.append(f"{model}/{tower}: parked (no finite median)")
    return events


def ready_path(root: str) -> str:
    """Marker: every final unit of every learned model is trained."""
    return os.path.join(root, "READY_FOR_TEST.json")


def advance(root: str, models, n_trials: int = C.N_TRIALS) -> List[str]:
    """Advances the phases of `models` (see the module docstring); one
    caller at a time (lock 'advance'), others return at once.

    Returns:
        List[str]: What changed.
    """
    lock = Lock(root, "advance",
                f"advance-{socket.gethostname()}-{os.getpid()}", 0)
    if not lock.acquire():
        return []
    events = []
    try:
        for model in models:
            states = {
                t: study_state(root, model, t, n_trials)
                for t in C.TOWERS_SEARCHED
            }
            for state in states.values():
                reason = park_reason(state)
                if reason is None:
                    continue
                state["parked"] = True
                if park_study(root, model, state["tower"], *reason):
                    events.append(
                        f"{model}/{state['tower']}: parked ({reason[0]})")
            events += _advance_model(root, model, n_trials, states)
        # The marker is for the whole track, whatever the caller's models.
        if (not os.path.exists(ready_path(root)) and
                all_done(root, models, n_trials) and
                all_done(root, S.LEARNED, n_trials) and
                common.write_once(ready_path(root), {
                    "models": list(S.LEARNED),
                    "time": common.now()
                })):
            common.alert(
                root, "track", "every final unit is trained: "
                "ready to open the test (hpo/final.py --open_test)",
                {"models": list(S.LEARNED)})
            events.append("ready for test")
    finally:
        lock.release()
    for event in events:
        print(f"advance: {event}", flush=True)
    return events


def all_done(root: str, models, n_trials: int = C.N_TRIALS) -> bool:
    """Every final unit of every study of `models` is trained; a parked
    study is terminal."""
    return all(state["phase"] in ("done", "parked")
               for state in all_states(root, models, n_trials))


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
        if state["parked"]:
            continue
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
            host: Optional[str] = None) -> bool:
    """Is there still work in this unit (for a worker on `host`: a parked
    unit it may un-park counts)?"""
    if unit.phase == "search":
        state = study_state(root, unit.model, unit.tower, n_trials)
        return not state["parked"] and state["search_pending"]
    if unit_done(root, unit):
        return False
    if unit_parked(root, unit):
        return host is not None and common.can_unpark(run_dir(root, unit), host)
    return True


def _try(root: str, unit: Unit, owner: str, restart: int,
         n_trials: int) -> Optional[Lock]:
    host = socket.gethostname()
    lock = Lock(root, lock_name(unit), owner, restart)
    if not lock.acquire():
        return None
    if not pending(root, unit, n_trials, host):  # finished meanwhile
        lock.release()
        return None
    if unit.phase != "search" and unit_parked(root, unit):
        if not common.unpark(run_dir(root, unit), host):
            lock.release()
            return None
        common.alert(root, unit_id(unit), f"unit un-parked by {host}", {})
    return lock


def _unpark_studies(root: str, states: List[Dict], owner: str,
                    restart: int) -> bool:
    """Un-parks (once) the parked studies whose failures all came from
    another host; True if one was."""
    host = socket.gethostname()
    done = False
    for state in states:
        hosts = state["park_hosts"]
        if not state["parked"] or len(hosts) != 1 or hosts[0] == host:
            continue
        lock = Lock(root,
                    lock_name(Unit("search", state["model"], state["tower"])),
                    owner, restart)
        if not lock.acquire():
            continue
        try:
            done = unpark_study(root, state, host) or done
        finally:
            lock.release()
    return done


def pick(
    root: str,
    owner: str,
    restart: int,
    models,
    n_trials: int = C.N_TRIALS,
    cost: Optional[Dict[str, float]] = None
) -> Tuple[Optional[Unit], Optional[Lock]]:
    """The next unit of a worker, with its lock held.

    Args:
        root (str): Root of the track.
        owner (str): Stable identity of the worker.
        restart (int): Its incarnation (Slurm restart count; any other
          count of the same owner is a dead incarnation).
        models: Models of the worker.
        n_trials (int): Budget per study (the protocol: N_TRIALS).
        cost (dict, optional): Cost of a search trial per model.

    Returns:
        (Unit, Lock), or (None, None) when nothing can run now.
    """
    record = common.read_json(worker_path(root, owner)) or {}
    if record.get("unit"):
        unit = parse_unit(record["unit"])
        if unit.model in models:
            lock = _try(root, unit, owner, restart, n_trials)
            if lock:
                return unit, lock
    states = all_states(root, models, n_trials, cost)
    if _unpark_studies(root, states, owner, restart):
        states = all_states(root, models, n_trials, cost)
    for unit in candidates(states):
        lock = _try(root, unit, owner, restart, n_trials)
        if lock:
            return unit, lock
    return None, None
