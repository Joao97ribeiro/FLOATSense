# pylint: disable=wrong-import-position
# pylint: disable=too-many-return-statements
"""Worker of the validation-tuned track: picks units and runs them.

One worker per GPU. It loops: advance the phases (pick.advance), pick the
next unit of its models (pick.pick), run it through the drivers (a search
trial with hpo/search.py, a confirmation unit with hpo/confirm.py, a final
unit with hpo/final.py; each resumes from its checkpoint), and record it:

    python hpo/worker.py --root=<root> --models=tcn,fits \\
        --dataset_dir=<dataset> --python=<trainer python>
    python hpo/worker.py --dry_run --root=/tmp/hpo --models=tcn   # CPU stub

Several workers (on one machine or on a cluster sharing the root) can run
at once; which models a worker takes and how it is launched (a job
scheduler, one process per GPU) are left to the site. A site can import
`main` and pass a cost per model (pick order, slowest first).

Identity: --owner is the identity of the job (e.g. slurm-<job id>; it
survives a requeue) and --restart its incarnation (default: the Slurm
restart count, 0 outside Slurm). A lock of the same owner with another
restart count belongs to a dead incarnation and is taken back at once.

Progress: every run gets --checkpoint_seconds, the wall time between two
resume saves of the trainer, so a run killed without notice loses at most
that much (plus the epoch in progress at a stop request). A save costs a
few seconds of GPU time, so the value depends on the class of the worker:
CHECKPOINT_SECONDS (90 s, the default) on preemptible GPUs that can be
killed at once; e.g. 300 s on local or non-preemptible GPUs, which stop on
a signal at the end of an epoch. A unit whose last NO_PROGRESS_ATTEMPTS
attempts all ended stopped, preempted or killed without a new resume save
gets one alert (common.check_progress).

Exit codes (the contract with the launcher):
  0   every unit of its models done, idle for --idle_minutes, --max_units
      reached, or <root>/STOP; do not restart (an idle exit while work
      remains writes an alert record).
  97  EXIT_CONFIG: a run refused its resume state (written with another
      configuration, run.py exit code 6). The unit is held for an operator
      (CONFIG_MISMATCH marker in its run directory, alert record), not
      shelved; no worker picks it until the marker is removed.
  98  EXIT_BROKEN: a broken host. The host is listed in
      <root>/bad_hosts/<host>.json (reason, time) and no worker starts on
      it (each exits 98 at once). The launcher must not requeue a 98 onto
      the same host; an operator removes the file to re-admit the host.
  99  EXIT_REQUEUE: stopped mid-unit (a signal, or its lock lost); restart
      it (the same --owner resumes its unit first).

Stopping:
  - SIGUSR1 or SIGTERM (Slurm time limit or preemption, Ctrl-C): the
    running trainer gets SIGUSR1, ends its epoch, saves its resume state
    and exits; the worker keeps the unit as its own (the next incarnation
    resumes it first) and exits with EXIT_REQUEUE. A second signal is
    passed on as SIGTERM.
  - its lock taken over by another worker (it was believed dead), or not
    beaten LOCK_BEAT_FAILURES times in a row (the lock file unreadable):
    stops its unit and exits (EXIT_REQUEUE); a result of the unit that
    arrives after that is dropped, not recorded.
  - a breaker (EXIT_BROKEN): WORKER_FAILURES units in a row shelved or
    crashed, from at least two different studies (a broken host: bad
    environment, full disk, missing mount), or MAX_LOOP_ERRORS exceptions
    in a row of the loop itself (pick, advance). Failures of one study are
    the study's: DRIVER_ERRORS_PARK exceptions of the drivers in a study
    shelve the study (pick.driver_error), and the worker backs off
    (POLL_SECONDS) and goes on. Out of memory and a configuration mismatch
    never count. The units shelved by failures on a listed host are
    un-shelved by a worker of another host (hpo/pick.py).

Records: <root>/workers/<owner>.json (current unit, host, GPU; refreshed
at every lock beat and idle poll, with the exit code at the end; the status
page marks a record older than STALE_MINUTES without one as dead) and one
event line per unit in <root>/events/<owner>.jsonl (start, end, status,
seconds); both are read by hpo/status.py.
"""

import argparse
import json
import os
import signal
import socket
import sys
import time
import traceback
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hpo import common
from hpo import confirm
from hpo import final
from hpo import pick
from hpo import search
from hpo import constants as C
from hpo import search_space as S

_SIGNALS = {"count": 0}


def _on_signal(signum, frame) -> None:
    """First signal: stop at the end of the epoch; second: terminate."""
    del frame
    _SIGNALS["count"] += 1
    print(f"worker: signal {signal.Signals(signum).name}", flush=True)
    common.request_stop(signal.SIGUSR1 if _SIGNALS["count"] ==
                        1 else signal.SIGTERM)


def driver_args(args: argparse.Namespace, unit: pick.Unit,
                driver: str) -> List[str]:
    """Command-line options of a driver for one unit."""
    out = [
        f"--model={unit.model}", f"--tower={unit.tower}", f"--root={args.root}",
        f"--dataset_dir={args.dataset_dir}", f"--python={args.python}",
        f"--extra={args.extra}",
        f"--checkpoint_seconds={args.checkpoint_seconds}"
    ]
    if driver != "final":
        out.append(f"--n_trials={args.n_trials}")
    if args.dry_run:
        out.append("--dry_run")
    return out


def run(args: argparse.Namespace,
        unit: pick.Unit,
        lock: Optional[pick.Lock] = None) -> str:
    """Runs one unit through its driver; returns its status: 'trial' (a
    search trial told), 'done', 'parked' (shelved after its failures),
    'oom' (out of memory: not the host's), 'config_mismatch' (held for an
    operator), 'stopped', 'lost' (the result dropped after a lock
    takeover), 'empty' (no trial to run: nothing was done), 'busy' (claimed
    by another process, e.g. a driver run by hand) or 'unfinished'."""
    if unit.phase == "search":
        outcomes: List[Dict] = []
        search.main(driver_args(args, unit, "search") +
                    ["--max_new=1", "--exclusive"],
                    outcomes=outcomes,
                    owned=lock.owned if lock is not None else None)
        if common.STOP.is_set():
            return "stopped"
        if not outcomes:
            return "empty"
        last = outcomes[-1]["status"]
        return last if last in ("parked", "stopped", "lost", "oom",
                                "config_mismatch") else "trial"
    run_dir = pick.run_dir(args.root, unit)
    # The lock of the unit is held: a claim left in the run directory by an
    # earlier incarnation of this worker is void; a claim of another
    # process is not.
    common.release(run_dir, owner=args.owner)
    if common.busy(run_dir):
        return "busy"
    if unit.phase == "confirm":
        options = confirm.parse_args(driver_args(args, unit, "confirm"))
        plan = common.read_json(
            confirm.plan_path(args.root, unit.model, unit.tower))
        confirm.run_one(options, plan, unit.rank, unit.seed)
    else:
        options = final.parse_args(driver_args(args, unit, "final"))
        final.train(options, unit.model, unit.tower, [unit.seed])
    if pick.unit_done(args.root, unit):
        return "done"
    if common.held(run_dir):
        return "config_mismatch"
    if pick.unit_parked(args.root, unit):
        marker = common.read_json(os.path.join(run_dir, "PARKED")) or {}
        return "oom" if marker.get("reason") == "oom" else "parked"
    return "stopped" if common.STOP.is_set() else "unfinished"


def record(args: argparse.Namespace,
           unit: Optional[pick.Unit],
           gpu: Optional[str],
           exit_code: Optional[int] = None) -> None:
    """Writes the worker record (its current unit; the exit code at the
    end)."""
    entry = {
        "owner": args.owner,
        "tag": args.tag,
        "restart": args.restart,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "gpu": gpu,
        "unit": pick.unit_id(unit) if unit else None,
        "time": common.now()
    }
    if exit_code is not None:
        entry["exit"] = exit_code
    common.write_json(pick.worker_path(args.root, args.owner), entry)


def mark_exit(args: argparse.Namespace, code: int) -> int:
    """Adds the exit code to the worker record (its unit kept, for the next
    incarnation); returns the code."""
    try:
        path = pick.worker_path(args.root, args.owner)
        entry = common.read_json(path) or {"owner": args.owner}
        common.write_json(path, {**entry, "exit": code, "time": common.now()})
    except (OSError, ValueError) as error:
        print(f"worker {args.owner}: record not written ({error})", flush=True)
    return code


def touch_record(args: argparse.Namespace) -> None:
    """Refreshes the worker record (liveness for the status page)."""
    os.utime(pick.worker_path(args.root, args.owner), None)


def event(args: argparse.Namespace, entry: dict) -> None:
    """Appends one event line (one writer per file: the owner)."""
    path = os.path.join(args.root, "events", f"{args.owner}.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps({"owner": args.owner, **entry}) + "\n")


def last_error(args: argparse.Namespace, unit: pick.Unit) -> str:
    """The end of the log of the last parked run of a unit (from its alert
    record), for the alert of a broken worker."""
    if unit.phase == "search":
        study = search.open_study(args.root,
                                  unit.model,
                                  unit.tower,
                                  create=False)
        parked = [
            t for t in (study.get_trials(deepcopy=False) if study else [])
            if t.user_attrs.get("parked")
        ]
        run_dir = (search.trial_run_dir(parked[-1]) if parked else None)
    else:
        run_dir = pick.run_dir(args.root, unit)
    marker = common.read_json(os.path.join(run_dir,
                                           "PARKED")) if run_dir else None
    details = common.read_json(marker["alert"]) if marker else None
    return (details or {}).get("tail", "")[-1500:]


def run_locked(args: argparse.Namespace, unit: pick.Unit, lock: pick.Lock,
               gpu: Optional[str]) -> Dict:
    """Runs a unit while beating its lock; records it.

    Returns:
        dict: 'status' (see `run`, or 'crash' when the driver raised) and
          'error' (the traceback of a crash, the log tail of a parked unit).
    """
    record(args, unit, gpu)
    start = time.time()
    event(args, {"unit": pick.unit_id(unit), "start": common.now(), "gpu": gpu})

    def lost():
        print(f"worker: lock of {pick.unit_id(unit)} lost; stopping",
              flush=True)
        common.request_stop(signal.SIGTERM)

    error = ""
    with pick.Beater(lock, lost, on_beat=lambda: touch_record(args)) as beater:
        try:
            status = run(args, unit, lock)
        except Exception:  # pylint: disable=broad-exception-caught
            # A driver error (full disk, missing mount, bad environment, a
            # bad study): counted by the failure breaker of the worker and
            # by the driver errors of the study.
            status, error = "crash", traceback.format_exc()[-1500:]
            print(error, flush=True)
    if status == "parked":
        try:
            error = last_error(args, unit)
        except Exception:  # pylint: disable=broad-exception-caught
            error = traceback.format_exc()[-1500:]
    if beater.lost:
        status = "lost"
    else:
        lock.release()
    if status not in ("stopped", "lost"):
        record(args, None, gpu)  # else the next incarnation resumes it
    event(
        args, {
            "unit": pick.unit_id(unit),
            "end": common.now(),
            "status": status,
            "seconds": round(time.time() - start, 1),
            "gpu": gpu
        })
    print(f"worker {args.owner}: {pick.unit_id(unit)} {status}", flush=True)
    return {"status": status, "error": error}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default="outputs/hpo")
    parser.add_argument("--owner",
                        default=f"{socket.gethostname()}-{os.getpid()}",
                        help="Stable identity (e.g. slurm-<job id>).")
    parser.add_argument("--restart",
                        type=int,
                        default=int(os.environ.get("SLURM_RESTART_COUNT") or 0),
                        help="Incarnation of the owner (Slurm restart count; "
                        "a lock of the same owner with another count is "
                        "taken back at once).")
    parser.add_argument("--models",
                        default="all",
                        help="Models of this worker (comma list or all).")
    parser.add_argument("--tag",
                        default="",
                        help="Free label of the worker record (e.g. the "
                        "site's worker class).")
    parser.add_argument("--dataset_dir", default="data/FLOATSense")
    parser.add_argument("--python",
                        default=sys.executable,
                        help="Python that runs scripts/train/run.py.")
    parser.add_argument("--extra", default="")
    parser.add_argument("--n_trials", type=int, default=C.N_TRIALS)
    parser.add_argument("--dry_run",
                        action="store_true",
                        help="CPU stub instead of training.")
    parser.add_argument("--stub_seconds",
                        type=float,
                        default=0.0,
                        help="Seconds per validation of the dry-run stub.")
    parser.add_argument("--checkpoint_seconds",
                        type=float,
                        default=C.CHECKPOINT_SECONDS,
                        help="Wall time between two resume saves of every "
                        "run (run.py --checkpoint_seconds).")
    parser.add_argument("--idle_minutes", type=float, default=C.IDLE_MINUTES)
    parser.add_argument("--poll_seconds", type=float, default=C.POLL_SECONDS)
    parser.add_argument("--max_units",
                        type=int,
                        default=0,
                        help="Exit after this many units (0: no limit).")
    args = parser.parse_args(argv)
    args.models = (list(S.LEARNED)
                   if args.models == "all" else args.models.split(","))
    unknown = set(args.models) - set(S.LEARNED)
    if unknown:
        parser.error(f"unknown models: {sorted(unknown)}")
    return args


def _nothing_left(args: argparse.Namespace, idle_since: float) -> bool:
    """Nothing to pick: True (exit) if every unit is done or the worker
    has been idle for --idle_minutes (with an alert record if work
    remains: units held, shelved on one host, or running elsewhere)."""
    if pick.all_done(args.root, args.models, args.n_trials):
        print(f"worker {args.owner}: every unit done", flush=True)
        return True
    if time.monotonic() - idle_since > args.idle_minutes * 60:
        print(f"worker {args.owner}: idle, exiting", flush=True)
        common.alert(
            args.root, f"worker/{args.owner}",
            "worker idle while work remains (units held, running "
            "elsewhere, or shelved on this host only)", {
                "owner": args.owner,
                "models": args.models
            })
        return True
    return False


def broken(args: argparse.Namespace,
           failures: List[Dict],
           gpu: Optional[str],
           reason: Optional[str] = None) -> bool:
    """The failure breaker: True once WORKER_FAILURES units in a row ended
    parked or crashed on this worker, from at least two different studies
    (or, with `reason`, at once); then the host is listed in
    <root>/bad_hosts/ and an alert record (host, GPU, the units and their
    last errors) is written."""
    studies = {f.get("study") for f in failures}
    if reason is None:
        if len(failures) < C.WORKER_FAILURES or len(studies) < 2:
            return False
        reason = (f"{len(failures)} failed units in a row, in "
                  f"{len(studies)} studies")
    details = {"owner": args.owner, "gpu": gpu, "failures": failures}
    host = common.add_bad_host(args.root, socket.gethostname(), reason, details)
    path = common.alert(args.root, f"worker/{args.owner}",
                        f"worker stopped: {reason}", {
                            **details, "bad_host": host
                        })
    print(f"worker {args.owner}: {reason}, exiting (alert {path})", flush=True)
    return True


def main(argv: Optional[List[str]] = None,
         cost: Optional[Dict[str, float]] = None) -> int:
    """Picks and runs units until done, idle, stopped or signalled.

    Args:
        argv (List[str], optional): Command-line options.
        cost (dict, optional): Cost of a search trial per model (pick
          order: most work left first).

    Returns:
        int: 0 (done, idle or STOP file), EXIT_CONFIG (a configuration
          mismatch), EXIT_BROKEN (a broken host) or EXIT_REQUEUE
          (signalled, or its lock lost); see the module docstring.
    """
    args = parse_args(argv)
    for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _on_signal)
    common.STUB_SECONDS = args.stub_seconds
    common.OWNER = args.owner
    host = socket.gethostname()
    if host in common.bad_hosts(args.root):
        print(
            f"worker {args.owner}: host {host} is listed in "
            f"{common.bad_hosts_dir(args.root)}; not starting",
            flush=True)
        return C.EXIT_BROKEN
    gpu = None if args.dry_run else common.gpu_name()
    print(
        f"worker {args.owner} ({args.tag or 'untagged'}, restart "
        f"{args.restart}): "
        f"{','.join(args.models)}",
        flush=True)
    return mark_exit(args, _loop(args, cost, gpu))


def _after(args: argparse.Namespace, unit: pick.Unit, outcome: Dict,
           failures: List[Dict], gpu: Optional[str]) -> Optional[int]:
    """Handles the outcome of a unit: the exit code of the worker, or None
    to go on (after a back-off of --poll_seconds if nothing was run or the
    unit failed). `failures` (the failed units in a row) is updated."""
    status = outcome["status"]
    if status in ("stopped", "lost"):
        return C.EXIT_REQUEUE
    if status == "config_mismatch":
        common.alert(
            args.root, f"worker/{args.owner}",
            "worker stopped: configuration mismatch of "
            f"{pick.unit_id(unit)} (held for an operator)", {
                "owner": args.owner,
                "unit": pick.unit_id(unit)
            })
        return C.EXIT_CONFIG
    if status in ("done", "trial"):
        failures.clear()
    if status not in ("parked", "crash"):
        if status in ("empty", "busy", "unfinished"):  # do not spin
            common.STOP.wait(args.poll_seconds)
        return None
    if status == "crash":
        try:
            reason = pick.driver_error(args.root, unit, args.owner,
                                       outcome["error"])
            if reason:
                print(
                    f"worker {args.owner}: study of {pick.unit_id(unit)} "
                    f"shelved ({reason})",
                    flush=True)
        except Exception:  # pylint: disable=broad-exception-caught
            print(traceback.format_exc()[-1500:], flush=True)
    failures.append({
        "unit": pick.unit_id(unit),
        "study": f"{unit.model}_{unit.tower}",
        **outcome
    })
    if broken(args, failures, gpu):
        return C.EXIT_BROKEN
    common.STOP.wait(args.poll_seconds)  # back off, then retry
    return None


def _loop(args: argparse.Namespace, cost: Optional[Dict[str, float]],
          gpu: Optional[str]) -> int:
    """The loop of `main`; returns the exit code."""
    idle_since = time.monotonic()
    done = 0
    failures: List[Dict] = []  # consecutive failed units of this worker
    errors: List[str] = []  # consecutive exceptions of the loop itself
    while True:
        if common.STOP.is_set():
            return C.EXIT_REQUEUE
        if os.path.exists(os.path.join(args.root, "STOP")):
            print(f"worker {args.owner}: STOP file, exiting", flush=True)
            return 0
        try:
            pick.advance(args.root, args.models, args.n_trials)
            unit, lock = pick.pick(args.root, args.owner, args.restart,
                                   args.models, args.n_trials, cost)
            if unit is None:
                record(args, None, gpu)  # liveness
                if _nothing_left(args, idle_since):
                    return 0
            errors = []
        except Exception:  # pylint: disable=broad-exception-caught
            # A transient error of the shared file system: back off.
            errors.append(traceback.format_exc()[-1500:])
            print(errors[-1], flush=True)
            if len(errors) >= C.MAX_LOOP_ERRORS and broken(
                    args, [{
                        "error": e
                    } for e in errors[-3:]], gpu,
                    f"{len(errors)} errors in a row of the worker loop"):
                return C.EXIT_BROKEN
            common.STOP.wait(args.poll_seconds)
            continue
        if unit is None:
            common.STOP.wait(args.poll_seconds)
            continue
        outcome = run_locked(args, unit, lock, gpu)
        code = _after(args, unit, outcome, failures, gpu)
        if code is not None:
            return code
        if outcome["status"] in ("empty", "busy", "unfinished"):
            continue  # nothing was run
        idle_since = time.monotonic()
        done += 1
        if args.max_units and done >= args.max_units:
            return 0


if __name__ == "__main__":
    sys.exit(main())
