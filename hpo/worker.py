# pylint: disable=wrong-import-position
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

Stopping:
  - SIGUSR1 or SIGTERM (Slurm time limit or preemption, Ctrl-C): the
    running trainer gets SIGUSR1, ends its epoch, saves its resume state
    and exits; the worker keeps the unit as its own (the next incarnation
    resumes it first) and exits with EXIT_REQUEUE. A second signal is
    passed on as SIGTERM.
  - <root>/STOP: the worker exits before its next pick (0).
  - nothing to pick for --idle_minutes, or every unit of its models done:
    exits (0).
  - its lock taken over by another worker (it was believed dead): stops
    its unit at once and exits.

Records: <root>/workers/<owner>.json (current unit, host, GPU) and one
event line per unit in <root>/events/<owner>.jsonl (start, end, status,
seconds); both are read by hpo/supervisor.py.
"""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
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


def gpu_name() -> Optional[str]:
    """Name of the visible GPU (None without one)."""
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return out.strip().splitlines()[0] if out.strip() else None


def driver_args(args: argparse.Namespace, unit: pick.Unit,
                driver: str) -> List[str]:
    """Command-line options of a driver for one unit."""
    out = [
        f"--model={unit.model}", f"--tower={unit.tower}", f"--root={args.root}",
        f"--dataset_dir={args.dataset_dir}", f"--python={args.python}",
        f"--extra={args.extra}"
    ]
    if driver != "final":
        out.append(f"--n_trials={args.n_trials}")
    if args.dry_run:
        out.append("--dry_run")
    return out


def run(args: argparse.Namespace, unit: pick.Unit) -> str:
    """Runs one unit through its driver; returns its status."""
    if unit.phase == "search":
        search.main(
            driver_args(args, unit, "search") + ["--max_new=1", "--exclusive"])
        return "stopped" if common.STOP.is_set() else "trial"
    run_dir = pick.run_dir(args.root, unit)
    # The lock of the unit is held: a claim left in the run directory by a
    # dead worker is void.
    common.release(run_dir)
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
    if pick.unit_parked(args.root, unit):
        return "parked"
    return "stopped" if common.STOP.is_set() else "unfinished"


def record(args: argparse.Namespace, unit: Optional[pick.Unit],
           gpu: Optional[str]) -> None:
    """Writes the worker record (its current unit)."""
    common.write_json(
        pick.worker_path(args.root, args.owner), {
            "owner": args.owner,
            "tag": args.tag,
            "restart": args.restart,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "gpu": gpu,
            "unit": pick.unit_id(unit) if unit else None,
            "time": common.now()
        })


def event(args: argparse.Namespace, entry: dict) -> None:
    """Appends one event line (one writer per file: the owner)."""
    path = os.path.join(args.root, "events", f"{args.owner}.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps({"owner": args.owner, **entry}) + "\n")


def run_locked(args: argparse.Namespace, unit: pick.Unit, lock: pick.Lock,
               gpu: Optional[str]) -> str:
    """Runs a unit while beating its lock; records it."""
    record(args, unit, gpu)
    start = time.time()
    event(args, {"unit": pick.unit_id(unit), "start": common.now(), "gpu": gpu})

    def lost():
        print(f"worker: lock of {pick.unit_id(unit)} taken over; stopping",
              flush=True)
        common.request_stop(signal.SIGTERM)

    with pick.Beater(lock, lost) as beater:
        status = run(args, unit)
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
    return status


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default="outputs/hpo")
    parser.add_argument("--owner",
                        default=f"{socket.gethostname()}-{os.getpid()}",
                        help="Stable identity (e.g. slurm-<job id>).")
    parser.add_argument("--restart",
                        type=int,
                        default=int(
                            os.environ.get("SLURM_RESTART_COUNT",
                                           int(time.time()))),
                        help="Incarnation of the owner (larger is newer).")
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


def main(argv: Optional[List[str]] = None,
         cost: Optional[Dict[str, float]] = None) -> int:
    """Picks and runs units until done, idle, stopped or signalled.

    Args:
        argv (List[str], optional): Command-line options.
        cost (dict, optional): Cost of a search trial per model (pick
          order: most work left first).

    Returns:
        int: 0 (done, idle or STOP file) or EXIT_REQUEUE (signalled).
    """
    args = parse_args(argv)
    for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _on_signal)
    common.STUB_SECONDS = args.stub_seconds
    gpu = None if args.dry_run else gpu_name()
    print(
        f"worker {args.owner} ({args.tag or 'untagged'}, restart "
        f"{args.restart}): "
        f"{','.join(args.models)}",
        flush=True)
    idle_since = time.monotonic()
    done = 0
    while True:
        if common.STOP.is_set():
            return C.EXIT_REQUEUE
        if os.path.exists(os.path.join(args.root, "STOP")):
            print(f"worker {args.owner}: STOP file, exiting", flush=True)
            return 0
        pick.advance(args.root, args.models, args.n_trials)
        unit, lock = pick.pick(args.root, args.owner, args.restart, args.models,
                               args.n_trials, cost)
        if unit is None:
            if pick.all_done(args.root, args.models, args.n_trials):
                print(f"worker {args.owner}: every unit done", flush=True)
                return 0
            if time.monotonic() - idle_since > args.idle_minutes * 60:
                print(f"worker {args.owner}: idle, exiting", flush=True)
                return 0
            common.STOP.wait(args.poll_seconds)
            continue
        status = run_locked(args, unit, lock, gpu)
        if status in ("stopped", "lost"):
            return C.EXIT_REQUEUE
        idle_since = time.monotonic()
        done += 1
        if args.max_units and done >= args.max_units:
            return 0


if __name__ == "__main__":
    sys.exit(main())
