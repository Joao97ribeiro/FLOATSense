# pylint: disable=wrong-import-position
# pylint: disable=too-many-return-statements
"""Worker of the validation-tuned track: picks units and runs them.

One worker per GPU. It loops: advance the phases (pick.advance), pick the
next unit of its models (pick.pick), run it through its driver
(hpo/search.py, hpo/confirm.py or hpo/final.py; each resumes from its
checkpoint), and record it:

    python hpo/worker.py --root=<root> --models=tcn,fits \\
        --dataset_dir=<dataset> --python=<trainer python>
    python hpo/worker.py --dry_run --root=/tmp/hpo --models=tcn   # CPU stub

The default is one machine, one worker per GPU (CUDA_VISIBLE_DEVICES) on a
shared --root; workers of several machines can share a root on a shared
file system. Launching and restarting workers is the launcher's job,
outside this repository; a launcher can import `main` and pass a cost per
model (pick order, slowest first). --owner is the identity of a worker,
stable across its restarts, and --restart its incarnation (default: the
SLURM_RESTART_COUNT variable, else 0): a lock of the same owner with
another restart count belongs to a dead incarnation and is taken back.

Every run saves its resume state every --checkpoint_seconds (90 s by
default, for GPUs that can be killed at once; e.g. 300 s for local GPUs).
A unit that keeps crashing is shelved, and retried once after
RETRY_SHELVED_AFTER (hpo/pick.py); a unit whose resume state belongs to
another configuration is held for an operator (CONFIG_MISMATCH marker)
and the worker goes on with other units. An exception of a driver is
logged and followed by a wait of --poll_seconds; it is a crash of its
confirmation or final unit (attempts.json, so MAX_ATTEMPTS of them shelve
it), and after MAX_ATTEMPTS of them in this worker the unit is set aside
for RETRY_SHELVED_AFTER (with an alert), so it never starves the others.
An exception of the loop itself (advance, pick) is followed by the same
wait, with an alert after LOOP_ERRORS_ALERT in a row. While work remains
but nothing can be picked (units held, shelved until their retry, set
aside, or running elsewhere) the worker keeps polling, with one alert
after --idle_minutes. Alerts are records in <root>/alerts/.

Exit codes (the contract with the launcher):
  0   every unit of its models done (or shelved for good); also
      --max_units reached or the file <root>/STOP. Do not restart.
  98  EXIT_BROKEN: a broken machine. Units of WORKER_FAILURES different
      studies shelved in a row after failing early (no attempt saved a
      resume state: bad environment, full disk, missing mount), or a GPU
      with less memory than --min_gpu_gb. Do not restart it on the same
      machine.
  99  EXIT_REQUEUE: stopped. A signal (SIGUSR1, SIGTERM, Ctrl-C: the
      trainer ends its epoch and saves; a second signal is passed on as
      SIGTERM), its lock taken over or not beaten LOCK_BEAT_FAILURES times
      in a row (the unit stopped, its result dropped), MAX_LOOP_ERRORS
      exceptions of the loop in a row, a GPU of unknown memory under
      --min_gpu_gb, or any other exception of the worker. Restart it; the
      same --owner resumes its unit first.

Records, read by hpo/status.py: <root>/workers/<owner>.json (current unit,
host, GPU; refreshed at every lock beat and idle poll, with the exit code
at the end) and one event line per unit in <root>/events/<owner>.jsonl.
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
GPU_QUERY_TRIES = 3  # nvidia-smi queries of the GPU memory (--min_gpu_gb)
GPU_QUERY_SECONDS = 10.0  # wait between two of them


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
    search trial told), 'done', 'parked' (shelved after its failures;
    'parked_early' if none of its last attempts saved a resume state),
    'oom' (a search trial out of memory), 'config_mismatch' (held for an
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
        if last == "parked":
            return _parked(outcomes[-1]["run_dir"])
        return last if last in ("stopped", "lost", "oom",
                                "config_mismatch") else "trial"
    run_dir = pick.run_dir(args.root, unit)
    # The lock of the unit is held: a claim left in the run directory by an
    # earlier incarnation of this worker is void; a claim of another
    # process is not.
    common.release(run_dir, owner=args.owner)
    if common.busy(run_dir, args.owner):
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
        return _parked(run_dir)
    return "stopped" if common.STOP.is_set() else "unfinished"


def _parked(run_dir: str) -> str:
    """'parked_early' for a unit shelved after attempts that all failed
    before saving a resume state (common.park), else 'parked'."""
    marker = common.read_json(os.path.join(run_dir, "PARKED")) or {}
    return "parked_early" if marker.get("early") else "parked"


def crash(args: argparse.Namespace, unit: pick.Unit, error: str) -> None:
    """A driver exception of a confirmation or final unit (its lock held)
    is a crash of the unit, recorded in its attempts.json: MAX_ATTEMPTS of
    them shelve it (then the single retry of hpo/pick.py). A search unit
    has no run directory of its own (`_unit_error` sets it aside)."""
    run_dir = pick.run_dir(args.root, unit)
    if run_dir is None:
        return
    try:
        attempts = common.read_attempts(run_dir)
        common.add_attempt(attempts, time.time(), "error", None,
                           common.gpu_name())
        attempts["crashes"] += 1
        common.save_attempts(run_dir, attempts)
        if (attempts["crashes"] >= C.MAX_ATTEMPTS and
                not pick.unit_parked(args.root, unit)):
            common.park(args.root, pick.unit_id(unit), run_dir, "driver errors",
                        attempts, {"error": error})
    except Exception:  # pylint: disable=broad-exception-caught
        print(traceback.format_exc()[-1500:], flush=True)


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
    except Exception as error:  # pylint: disable=broad-exception-caught
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


def run_locked(args: argparse.Namespace, unit: pick.Unit, lock: pick.Lock,
               gpu: Optional[str]) -> Dict:
    """Runs a unit while beating its lock, records it and releases the lock
    (if still ours) whatever happens.

    Returns:
        dict: 'status' (see `run`, or 'error' when the driver raised) and
          'error' (the traceback of an error).
    """
    start, status, error = time.time(), "error", ""
    try:
        record(args, unit, gpu)
        event(args, {
            "unit": pick.unit_id(unit),
            "start": common.now(),
            "gpu": gpu
        })

        def lost():
            print(f"worker: lock of {pick.unit_id(unit)} lost; stopping",
                  flush=True)
            common.request_stop(signal.SIGTERM)

        with pick.Beater(lock, lost,
                         on_beat=lambda: touch_record(args)) as beater:
            try:
                status = run(args, unit, lock)
            except Exception:  # pylint: disable=broad-exception-caught
                status, error = "error", traceback.format_exc()[-1500:]
                print(error, flush=True)
                crash(args, unit, error)
        if beater.lost:
            status = "lost"
    finally:
        lock.release()  # only if still ours (best effort after a loss)
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
                        help="Stable identity (e.g. a job id).")
    parser.add_argument("--restart",
                        type=int,
                        default=int(os.environ.get("SLURM_RESTART_COUNT") or 0),
                        help="Incarnation of the owner (a lock of the same "
                        "owner with another count is taken back at once).")
    parser.add_argument("--models",
                        default="all",
                        help="Models of this worker (comma list or all).")
    parser.add_argument("--tag",
                        default="",
                        help="Free label of the worker record.")
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
    parser.add_argument("--min_gpu_gb",
                        type=float,
                        default=0.0,
                        help="Refuse to start on a GPU with less memory "
                        "(exit 98), or of unknown memory after a few "
                        "queries (exit 99) (0: no check).")
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


def _alert(args: argparse.Namespace, reason: str, **details) -> None:
    """An alert record of this worker (best effort: never raises)."""
    print(f"worker {args.owner}: {reason}", flush=True)
    try:
        common.alert(args.root, f"worker/{args.owner}", reason, {
            "owner": args.owner,
            "models": args.models,
            **details
        })
    except Exception:  # pylint: disable=broad-exception-caught
        print(traceback.format_exc()[-1500:], flush=True)


def small_gpu(args: argparse.Namespace) -> Optional[int]:
    """None if --min_gpu_gb is unset or met; else the exit code, with an
    alert: EXIT_BROKEN for a GPU with less memory, EXIT_REQUEUE if its
    memory is still unknown after GPU_QUERY_TRIES queries."""
    if args.min_gpu_gb <= 0:
        return None
    memory = None
    for attempt in range(GPU_QUERY_TRIES):
        memory = common.gpu_memory_gb()
        if memory is not None:
            break
        if attempt + 1 < GPU_QUERY_TRIES:
            time.sleep(GPU_QUERY_SECONDS)
    if memory is not None and memory >= args.min_gpu_gb:
        return None
    _alert(args, f"worker not started: GPU memory {memory or 'unknown'} GB, "
           f"--min_gpu_gb {args.min_gpu_gb:g}",
           gpu=common.gpu_name())
    return C.EXIT_REQUEUE if memory is None else C.EXIT_BROKEN


def main(argv: Optional[List[str]] = None,
         cost: Optional[Dict[str, float]] = None) -> int:
    """Picks and runs units until done, stopped or signalled.

    Args:
        argv (List[str], optional): Command-line options.
        cost (dict, optional): Cost of a search trial per model (pick
          order: most work left first).

    Returns:
        int: 0 (done, --max_units or STOP file), EXIT_BROKEN or
          EXIT_REQUEUE; see the module docstring.
    """
    args = parse_args(argv)
    for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _on_signal)
    common.STUB_SECONDS = args.stub_seconds
    common.OWNER = args.owner
    refused = small_gpu(args)
    if refused:
        return mark_exit(args, refused)
    gpu = None if args.dry_run else common.gpu_name()
    print(
        f"worker {args.owner} ({args.tag or 'untagged'}, restart "
        f"{args.restart}): {','.join(args.models)}",
        flush=True)
    try:
        code = _loop(args, cost, gpu)
    except Exception:  # pylint: disable=broad-exception-caught
        # A failure of the worker itself (its records, alerts, locks): the
        # lock of its unit was released by run_locked.
        _alert(args,
               "worker stopped: exception",
               error=traceback.format_exc()[-1500:])
        code = C.EXIT_REQUEUE
    return mark_exit(args, code)


def _step(args: argparse.Namespace, cost: Optional[Dict[str, float]],
          gpu: Optional[str], loop: Dict) -> Dict:
    """One pass of the loop: advance, pick (not the units set aside), run.
    Returns the outcome of the unit, or status 'all_done' or 'idle' when
    nothing was picked."""
    pick.advance(args.root, args.models, args.n_trials)
    skip = {u for u, until in loop["skip"].items() if until > time.monotonic()}
    unit, lock = pick.pick(args.root, args.owner, args.restart, args.models,
                           args.n_trials, cost, skip)
    if unit is None:
        record(args, None, gpu)  # liveness
        if pick.all_done(args.root, args.models, args.n_trials):
            # Written here too: the advance lock may have been busy when
            # the last units ended.
            pick.mark_ready(args.root, args.n_trials)
            return {"status": "all_done"}
        return {"status": "idle"}
    return {**run_locked(args, unit, lock, gpu), "unit": unit}


def _breaker(args: argparse.Namespace, failures: List[Dict]) -> bool:
    """True (with an alert) once units of WORKER_FAILURES different studies
    were shelved in a row after failing early ('parked_early')."""
    studies = {f["study"] for f in failures}
    if len(studies) < C.WORKER_FAILURES:
        return False
    _alert(args, f"worker stopped: units of {len(studies)} studies in a row "
           "shelved after failing early (a broken machine?)",
           failures=failures)
    return True


def _unit_error(args: argparse.Namespace, outcome: Dict, loop: Dict) -> None:
    """Driver exceptions of one unit: after MAX_ATTEMPTS of them in this
    worker the unit is set aside for RETRY_SHELVED_AFTER, with an alert."""
    name = pick.unit_id(outcome["unit"])
    loop["unit_errors"][name] = loop["unit_errors"].get(name, 0) + 1
    if loop["unit_errors"][name] >= C.MAX_ATTEMPTS:
        del loop["unit_errors"][name]
        loop["skip"][name] = time.monotonic() + C.RETRY_SHELVED_AFTER
        _alert(args,
               f"{name}: {C.MAX_ATTEMPTS} driver errors, set aside",
               error=outcome["error"])


def _after(args: argparse.Namespace, outcome: Dict,
           loop: Dict) -> Optional[int]:
    """Handles the outcome of one pass of the loop (`loop` holds the counts
    of the loop); returns the exit code of the worker, or None to go on."""
    status = outcome["status"]
    if status == "all_done":
        print(f"worker {args.owner}: every unit done", flush=True)
        return 0
    if status in ("stopped", "lost"):
        return C.EXIT_REQUEUE
    if status == "error" and outcome.get("unit") is not None:
        _unit_error(args, outcome, loop)
        return None
    loop["errors"] = loop["errors"] + 1 if status == "error" else 0
    if status == "error":  # of the loop itself (advance, pick)
        stop = loop["errors"] >= C.MAX_LOOP_ERRORS
        if stop or loop["errors"] == C.LOOP_ERRORS_ALERT:
            _alert(args, f"{loop['errors']} errors in a row"
                   f"{', stopping' if stop else ''}",
                   error=outcome["error"])
        return C.EXIT_REQUEUE if stop else None
    if status in ("idle", "empty", "busy", "unfinished"):
        if (status == "idle" and not loop["idle_alerted"] and
                time.monotonic() - loop["idle_since"] > args.idle_minutes * 60):
            loop["idle_alerted"] = True
            _alert(
                args, "worker idle while work remains (units held, "
                "shelved until their retry, or running elsewhere)")
        return None
    unit = outcome["unit"]
    if status == "parked_early":
        loop["failures"].append({
            "unit": pick.unit_id(unit),
            "study": f"{unit.model}_{unit.tower}"
        })
        if _breaker(args, loop["failures"]):
            return C.EXIT_BROKEN
    elif status in ("done", "trial", "parked"):  # the unit ran
        loop["failures"].clear()
    loop.update(idle_since=time.monotonic(), idle_alerted=False)
    loop["done"] += 1
    return 0 if args.max_units and loop["done"] >= args.max_units else None


def _loop(args: argparse.Namespace, cost: Optional[Dict[str, float]],
          gpu: Optional[str]) -> int:
    """The loop of `main`; returns the exit code."""
    loop = {
        "idle_since": time.monotonic(),
        "idle_alerted": False,
        "done": 0,
        "errors": 0,  # exceptions of the loop in a row
        "unit_errors": {},  # driver exceptions per unit
        "skip": {},  # units set aside: unit id -> time.monotonic() until
        "failures": []  # units shelved early in a row
    }
    while True:
        if common.STOP.is_set():
            return C.EXIT_REQUEUE
        if os.path.exists(os.path.join(args.root, "STOP")):
            print(f"worker {args.owner}: STOP file, exiting", flush=True)
            return 0
        try:
            outcome = _step(args, cost, gpu, loop)
        except Exception:  # pylint: disable=broad-exception-caught
            outcome = {
                "status": "error",
                "error": traceback.format_exc()[-1500:]
            }
            print(outcome["error"], flush=True)
        code = _after(args, outcome, loop)
        if code is not None:
            return code
        if outcome["status"] in ("error", "idle", "empty", "busy",
                                 "unfinished"):
            common.STOP.wait(args.poll_seconds)  # back off, do not spin


if __name__ == "__main__":
    sys.exit(main())
