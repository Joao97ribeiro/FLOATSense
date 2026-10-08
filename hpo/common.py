# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-locals
# pylint: disable=too-many-return-statements
# pylint: disable=wrong-import-position
"""Pieces shared by the drivers of the validation-tuned track.

A *unit* is one training run of scripts/train/run.py in its own directory
(a search trial, a confirmation seed or a final retraining). `run_unit`
runs it with --resume, streams its VAL lines to a callback (pruning), and
handles failures:

  - NaN or divergence (exit code EXIT_DIVERGED) and a model above the
    parameter cap (EXIT_TOO_LARGE) are returned to the caller;
  - out of memory ('oom': "CUDA out of memory", OutOfMemoryError, "out of
    memory" in the last lines) is returned to the caller, not retried and
    not counted against the host (the configuration does not fit);
  - a resume state of another configuration (EXIT_CONFIG_MISMATCH) is
    returned as 'config_mismatch' with an alert and a CONFIG_MISMATCH
    marker in the run directory; the unit is not parked, and no worker
    picks it until an operator removes the marker (after removing the
    resume state, or the run directory);
  - a crash is resumed from the last checkpoint, up to MAX_ATTEMPTS
    attempts; then the unit is parked (PARKED marker, with the hosts of its
    failures) and an alert record is written to <root>/alerts/;
  - a preemption (the run killed by a signal) or a hardware fault (CUDA
    error, ECC, Xid) is resumed without counting as an attempt;
  - a stop request of the worker (`request_stop`: Slurm time limit,
    preemption notice, STOP file) is passed to the run as SIGUSR1 (the
    trainer saves at the end of the epoch and exits with EXIT_STOPPED);
    the unit returns 'stopped' and is resumed by the next worker.

Records of a unit, in its run directory: config.json (the hyperparameters
as passed to run.py and the exact command, written before the run starts)
and attempts.json (one entry per attempt: start, end, seconds, host, GPU
and status, pruned and preempted attempts included; analyze.py sums them
into GPU-hours). An attempt is recorded when it starts, with status
'running', and its end and seconds are updated by the heartbeat; an open
attempt older than STALE_MINUTES was hard-killed ('killed', `event_status`)
and its seconds still count.

A parked unit whose failures all came from a single host is un-parked once
(`unpark`) when a worker of another host picks it (hpo/pick.py): the
failures were probably the host's, not the unit's. A unit parked twice, or
after failures on two hosts, stays parked. Failures on a host listed in
<root>/bad_hosts/ (`bad_hosts`, written when a worker stops with
EXIT_BROKEN) do not count: a unit parked by them only is un-parked by any
other host. Nothing is un-parked once the test is ready (`test_frozen`).

Every file the drivers share is written atomically (temporary file and
rename), and the files that fix a decision (plans, sealed test results)
are written once.
"""

import datetime
import functools
import json
import math
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Callable, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from floatsense.constants import EXIT_CONFIG_MISMATCH
from floatsense.constants import EXIT_DIVERGED
from floatsense.constants import EXIT_STOPPED
from floatsense.constants import EXIT_TOO_LARGE
from hpo import constants as C
from hpo import search_space as S

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_PY = os.path.join(REPO, "scripts", "train", "run.py")
CONFIG = os.path.join(REPO, "scripts", "train", "config.cfg")
VAL_LINE = re.compile(r"^VAL epoch=(\d+) r2_mean=(\S+) r2_top=(\S+) "
                      r"r2_base=(\S+)(?: r2_gauges=(\S+))?")
PARAMS_LINE = re.compile(r"^PARAMS trainable=(\d+)")
# Signals of a preemption (Slurm sends TERM, then KILL after the grace time).
PREEMPT_SIGNALS = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGUSR1,
                   signal.SIGUSR2)
HARDWARE = re.compile(
    r"uncorrectable ECC|ECC error|Xid|illegal memory access|"
    r"unspecified launch failure|CUDA-capable device|device not ready|"
    r"NCCL error|cudaErrorLaunchFailure|GPU is lost|Bus error", re.IGNORECASE)
OOM = re.compile(r"CUDA out of memory|OutOfMemoryError|out of memory",
                 re.IGNORECASE)
HEARTBEAT_SECONDS = 30.0
RETRY_SECONDS = 5.0  # pause before resuming a failed run
TAIL_LINES = 60
# A stop request of the worker; the runs it started get SIGUSR1.
STOP = threading.Event()
_CHILDREN: set = set()
# Seconds per validation of the dry-run stub (0: instantaneous).
STUB_SECONDS = 0.0
# Outcomes of an attempt that are not failures of the host.
NOT_FAILURES = ("ok", "pruned", "stopped", "diverged", "too_large", "oom",
                "config_mismatch", "running")
READY = "READY_FOR_TEST.json"  # marker: every final unit is trained
# Marker of a unit whose resume state belongs to another configuration.
HOLD = "CONFIG_MISMATCH"
# Identity of the worker of this process (hpo/worker.py sets it): written
# into the claims, so a worker releases only its own (any incarnation).
OWNER = ""


def now() -> str:
    """Local time stamp of the records."""
    return datetime.datetime.now().isoformat(timespec="seconds")


def write_json(path: str, obj) -> None:
    """Writes JSON atomically (temporary file next to it, then rename)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as file:
        json.dump(obj, file, indent=1)
    os.replace(tmp, path)


def write_once(path: str, obj) -> bool:
    """Writes JSON only if `path` does not exist yet (hard link of a
    complete temporary file, so two writers never both succeed).

    Returns:
        bool: True if this call wrote the file.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as file:
        json.dump(obj, file, indent=1)
    try:
        os.link(tmp, path)
        return True
    except FileExistsError:
        return False
    finally:
        os.remove(tmp)


def read_json(path: str):
    """JSON content of `path`, or None if it does not exist."""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def parse_val(line: str) -> Optional[Dict]:
    """The scores of a VAL line of the trainer, or None for other lines."""
    match = VAL_LINE.match(line.strip())
    if not match:
        return None
    gauges = match.group(5)
    return {
        "epoch": int(match.group(1)),
        "r2_mean": float(match.group(2)),
        "r2_top": float(match.group(3)),
        "r2_base": float(match.group(4)),
        "r2_gauges": [float(v) for v in gauges.split(",")] if gauges else [],
    }


def alert(root: str, unit: str, reason: str, details: Dict) -> str:
    """Writes an alert record to <root>/alerts/ and returns its path."""
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", unit)
    path = os.path.join(root, "alerts", f"{stamp}_{name}.json")
    write_json(
        path, {
            "unit": unit,
            "reason": reason,
            "time": now(),
            "host": socket.gethostname(),
            **details
        })
    return path


def request_stop(sig: int = signal.SIGUSR1) -> None:
    """Asks the runs of this process to stop (SIGUSR1: save at the end of
    the epoch) and `run_unit` not to start or resume any."""
    STOP.set()
    for process in list(_CHILDREN):
        try:
            process.send_signal(sig)
        except (ProcessLookupError, OSError):
            pass


def server_time(directory: str) -> float:
    """Time of the file server of `directory`: the mtime of a probe file
    touched now (on NFS, utime without a time is set by the server), so
    that ages on a shared file system do not depend on the node clocks."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f".probe.{socket.gethostname()}")
    with open(path, "a", encoding="utf-8"):
        pass
    os.utime(path, None)
    return os.stat(path).st_mtime


@functools.lru_cache(maxsize=1)
def gpu_name() -> str:
    """Name of the first visible GPU, or 'unknown'.

    nvidia-smi is queried first (for the first device of
    CUDA_VISIBLE_DEVICES), so the driver does not create a CUDA context on
    the GPU of its runs; torch is asked only if it is already imported.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and not visible.strip():
        return "unknown"  # no GPU visible
    query = ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
    if visible:
        query.append(f"--id={visible.split(',')[0].strip()}")
    try:
        out = subprocess.run(query,
                             capture_output=True,
                             text=True,
                             timeout=30,
                             check=False).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        out = ""
    if out:
        return out.splitlines()[0].strip()
    torch = sys.modules.get("torch")
    try:
        if torch is not None and torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except (RuntimeError, AssertionError):
        pass
    return "unknown"


def heartbeat_age(run_dir: str) -> float:
    """Seconds since the heartbeat of a unit (inf if it never ran)."""
    path = os.path.join(run_dir, "heartbeat")
    if not os.path.exists(path):
        return float("inf")
    return server_time(run_dir) - os.path.getmtime(path)


def is_stale(run_dir: str) -> bool:
    """A unit whose worker stopped beating STALE_MINUTES ago."""
    return heartbeat_age(run_dir) > C.STALE_MINUTES * 60


def _process() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


def claim(run_dir: str) -> bool:
    """Claims a unit for this worker (another live worker keeps it).

    A claim whose heartbeat is stale is taken over: the old claim is
    renamed away first, and only one worker can rename it. The claim holds
    '<OWNER>|<host>:<pid>'.
    """
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "claim")
    owner = f"{OWNER}|{_process()}"
    for _ in range(2):
        try:
            handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if not is_stale(run_dir):
                return False
            try:
                os.rename(path, f"{path}.stale.{os.getpid()}.{time.time()}")
            except FileNotFoundError:
                return False
            continue
        with os.fdopen(handle, "w") as file:
            file.write(owner)
        touch(run_dir)
        return True
    return False


def claim_holder(run_dir: str) -> Optional[str]:
    """Content of the claim of a unit (None if unclaimed or unreadable)."""
    try:
        with open(os.path.join(run_dir, "claim"), encoding="utf-8") as file:
            return file.read()
    except OSError:
        return None


def release(run_dir: str, owner: Optional[str] = None) -> bool:
    """Gives a claimed unit back if the claim is ours: made by this process,
    or (with `owner`) by any incarnation of that worker identity. A claim of
    another process (e.g. a driver run by hand) is left alone.

    Returns:
        bool: True if no claim of ours is left (released, or none).
    """
    holder = claim_holder(run_dir)
    if holder is None:
        return True
    name, _, process = holder.rpartition("|")
    if process != _process() and not (owner and name == owner):
        return False
    try:
        os.remove(os.path.join(run_dir, "claim"))
    except FileNotFoundError:
        pass
    return True


def busy(run_dir: str) -> bool:
    """A unit claimed by another live process (not stale)."""
    holder = claim_holder(run_dir)
    if holder is None:
        return False
    return holder.rpartition("|")[2] != _process() and not is_stale(run_dir)


def bad_hosts_dir(root: str) -> str:
    """Records of the hosts a worker stopped on with EXIT_BROKEN."""
    return os.path.join(root, "bad_hosts")


def bad_hosts(root: str) -> frozenset:
    """The hosts listed in <root>/bad_hosts/ (an operator removes a file to
    re-admit a host)."""
    try:
        names = os.listdir(bad_hosts_dir(root))
    except FileNotFoundError:
        return frozenset()
    return frozenset(n[:-5] for n in names if n.endswith(".json"))


def add_bad_host(root: str, host: str, reason: str, details: Dict) -> str:
    """Lists `host` in <root>/bad_hosts/; returns the record path."""
    path = os.path.join(bad_hosts_dir(root), f"{host}.json")
    write_json(path, {"host": host, "reason": reason, "time": now(), **details})
    return path


def test_frozen(root: str) -> bool:
    """The test is ready (READY_FOR_TEST.json) or opened (sealed/): no
    parked unit or study is un-parked any more."""
    return (os.path.exists(os.path.join(root, READY)) or
            os.path.exists(os.path.join(root, "sealed")))


def touch(run_dir: str) -> None:
    """Beats the heartbeat of a unit."""
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "heartbeat"), "a", encoding="utf-8"):
        os.utime(os.path.join(run_dir, "heartbeat"))


def classify(returncode: int, tail: str) -> str:
    """Outcome of a finished run from its exit code and last lines.

    Returns:
        str: 'ok', 'diverged', 'too_large', 'stopped', 'config_mismatch',
          'preempted', 'oom' (out of memory), 'hardware' or 'crash' (every
          other error).
    """
    if returncode == 0:
        return "ok"
    if returncode == EXIT_STOPPED:
        return "stopped"
    if returncode == EXIT_CONFIG_MISMATCH:
        return "config_mismatch"
    if returncode == EXIT_DIVERGED:
        return "diverged"
    if returncode == EXIT_TOO_LARGE:
        return "too_large"
    if returncode < 0 and -returncode in PREEMPT_SIGNALS:
        return "preempted"
    if OOM.search(tail):
        return "oom"
    if HARDWARE.search(tail):
        return "hardware"
    return "crash"


def train_command(args,
                  model: str,
                  tower: str,
                  run_dir: str,
                  cfg: Dict,
                  train_split: str,
                  num_epochs: int,
                  seed: int,
                  validate: bool = True,
                  extra: Optional[List[str]] = None) -> List[str]:
    """The run.py command of a unit (no test evaluation; resumable, with a
    resume save every args.checkpoint_seconds of wall time).

    Args:
        args: Driver options (python, dataset_dir, checkpoint_seconds and
          extra run.py flags).
        model (str): Model name.
        tower (str): Tower name.
        run_dir (str): Output directory of the run.
        cfg (dict): Configuration drawn by search_space.suggest.
        train_split (str): Training split.
        num_epochs (int): Epochs of the run.
        seed (int): Training seed.
        validate (bool): Validate on SEARCH_VAL_SPLIT by the damage score.
        extra (List[str], optional): More run.py flags.

    Returns:
        List[str]: The command.
    """
    cmd = [
        args.python, RUN_PY, f"--flagfile={CONFIG}",
        f"--dataset_dir={args.dataset_dir}", f"--tower={tower}",
        f"--models={model}", f"--train_split={train_split}",
        f"--num_epochs={num_epochs}", f"--seed={seed}",
        f"--output_dir={run_dir}", "--run_evaluation=False", "--resume",
        "--checkpoint_seconds="
        f"{getattr(args, 'checkpoint_seconds', C.CHECKPOINT_SECONDS):g}"
    ] + S.as_args(cfg)
    if validate:
        cmd += [
            f"--val_split={C.SEARCH_VAL_SPLIT}", "--val_score=damage",
            f"--val_every={C.VAL_EVERY}"
        ]
    if model not in S.PRETRAINED:
        cmd.append(f"--max_params_m={S.MAX_PARAMS_M}")
    return cmd + shlex.split(args.extra) + list(extra or [])


def history_path(run_dir: str, model: str) -> str:
    """History JSON written by the trainer at the end of a run."""
    return os.path.join(run_dir, f"history_{model}_fa.json")


class _Heartbeat:
    """Touches the heartbeat of a unit while its run is alive (and calls
    `on_beat`, e.g. to update the open attempt record)."""

    def __init__(self,
                 run_dir: str,
                 on_beat: Optional[Callable[[], None]] = None):
        self.run_dir = run_dir
        self.on_beat = on_beat
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._beat, daemon=True)

    def _beat(self):
        while not self.stop.wait(HEARTBEAT_SECONDS):
            try:
                touch(self.run_dir)
                if self.on_beat is not None:
                    self.on_beat()
            except OSError as error:  # a transient file-system error
                print(f"heartbeat of {self.run_dir}: {error}", flush=True)

    def __enter__(self):
        touch(self.run_dir)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()


def attempts_path(run_dir: str) -> str:
    """Record of the attempts of a unit."""
    return os.path.join(run_dir, "attempts.json")


def resume_path(run_dir: str, model: str) -> str:
    """Resume state of the trainer in a run directory."""
    return os.path.join(run_dir, f"{model}_fa_resume.pt")


def _mtime(path: str) -> Optional[float]:
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def read_attempts(run_dir: str) -> Dict:
    """The attempts record of a unit (empty if it never ran)."""
    record = read_json(attempts_path(run_dir)) or {}
    record.setdefault("crashes", 0)
    record.setdefault("free", 0)
    record.setdefault("events", [])
    return record


def open_attempt(record: Dict,
                 start: float,
                 gpu: str,
                 resume_mtime: Optional[float] = None) -> Dict:
    """Appends an open attempt (status 'running') to an attempts record and
    returns it; `update_attempt` moves its end, `close_attempt` ends it."""
    event = {
        "start": _iso(start),
        "end": _iso(start),
        "seconds": 0.0,
        "host": socket.gethostname(),
        "gpu": gpu,
        "status": "running",
        "returncode": None,
        "resume_mtime": resume_mtime
    }
    record["events"].append(event)
    return event


def update_attempt(event: Dict, start: float) -> None:
    """Moves the end of an open attempt to now."""
    end = time.time()
    event["end"] = _iso(end)
    event["seconds"] = round(end - start, 1)


def close_attempt(event: Dict,
                  start: float,
                  status: str,
                  returncode: Optional[int],
                  resume_mtime: Optional[float] = None) -> None:
    """Ends an attempt: end, seconds, status, exit code and whether it
    saved a new resume state."""
    update_attempt(event, start)
    event["status"] = status
    event["returncode"] = returncode
    event["saved"] = (resume_mtime is not None and
                      resume_mtime != event.get("resume_mtime"))


def add_attempt(record: Dict, start: float, status: str,
                returncode: Optional[int], gpu: str) -> None:
    """Appends one finished attempt (wall-clock start and end, seconds,
    host, GPU, status) to an attempts record."""
    close_attempt(open_attempt(record, start, gpu), start, status, returncode)


def _iso(stamp: float) -> str:
    return datetime.datetime.fromtimestamp(stamp).isoformat(timespec="seconds")


def event_status(event: Dict) -> str:
    """Status of an attempt; an open one ('running') whose end was last
    updated STALE_MINUTES ago was hard-killed: 'killed'."""
    status = event.get("status") or "unknown"
    if status != "running":
        return status
    try:
        end = datetime.datetime.fromisoformat(event["end"])
    except (KeyError, TypeError, ValueError):
        return "killed"
    age = (datetime.datetime.now() - end).total_seconds()
    return "killed" if age > C.STALE_MINUTES * 60 else "running"


def close_killed(record: Dict, resume_mtime: Optional[float] = None) -> None:
    """Marks the open attempts of a unit 'killed' (called before a new
    attempt: the caller holds the unit, so an open attempt is dead)."""
    for event in record["events"]:
        if event.get("status") == "running":
            event["status"] = "killed"
            event["saved"] = (resume_mtime is not None and
                              resume_mtime != event.get("resume_mtime"))


def check_progress(root: str, name: str, run_dir: str, record: Dict) -> bool:
    """Alerts once if the last NO_PROGRESS_ATTEMPTS attempts all ended
    stopped, preempted or killed without a new resume save (the unit makes
    no progress: its checkpoint interval is longer than the time its
    workers get). Returns True if it alerted now."""
    last = record["events"][-C.NO_PROGRESS_ATTEMPTS:]
    if (record.get("no_progress_alert") or len(last) < C.NO_PROGRESS_ATTEMPTS or
            any(
                e.get("status") not in ("stopped", "preempted",
                                        "killed") or e.get("saved")
                for e in last)):
        return False
    record["no_progress_alert"] = alert(
        root, name, f"no progress: {len(last)} attempts in a row stopped "
        "without a new resume save", {
            "run_dir": run_dir,
            "events": last,
            "note": "lower --checkpoint_seconds or give the unit a longer "
                    "worker"
        })
    return True


def failure_hosts(record: Dict) -> List[str]:
    """Hosts of the failed attempts of a unit since its last un-parking."""
    since = record.get("since", (record.get("unparked") or
                                 {}).get("events_before", 0))
    return sorted({
        e.get("host") or "unknown"
        for e in record["events"][since:]
        if e.get("status") not in NOT_FAILURES
    })


def park(root: str, name: str, run_dir: str, reason: str, record: Dict,
         details: Dict) -> str:
    """Parks a unit: alert record and PARKED marker (with the hosts of its
    failures; none for a failure that is not the host's: out of memory).
    Returns the alert path."""
    hosts = [] if reason in ("config_mismatch",
                             "oom") else failure_hosts(record)
    path = alert(root, name, f"shelved after {reason}", {
        "run_dir": run_dir,
        "attempts": record,
        "hosts": hosts,
        **details
    })
    write_json(
        os.path.join(run_dir, "PARKED"), {
            "time": now(),
            "reason": reason,
            "host": socket.gethostname(),
            "hosts": hosts,
            "alert": path
        })
    return path


def live_hosts(hosts: List[str], bad=frozenset()) -> Optional[List[str]]:
    """The hosts of a parking that count: those not listed as bad; None if
    every host of a non-empty list is bad (the parking does not count)."""
    live = [h for h in hosts if h not in bad]
    return None if hosts and not live else live


def can_unpark(run_dir: str, host: str, bad=frozenset()) -> bool:
    """A parked unit whose failures all came from one host other than
    `host`, never un-parked before; or whose failures all came from hosts
    listed as bad (`bad`), whatever its history. A worker on a bad host
    un-parks nothing."""
    marker = read_json(os.path.join(run_dir, "PARKED"))
    if marker is None or host in bad:
        return False
    live = live_hosts(marker.get("hosts") or [], bad)
    if live is None:
        return True
    return (len(live) == 1 and live[0] != host and
            not read_attempts(run_dir).get("unparked"))


def unpark(run_dir: str, host: str, bad=frozenset()) -> bool:
    """Un-parks a unit (see `can_unpark`; the caller holds its lock): the
    PARKED marker is renamed away and the failure counts restart.

    Returns:
        bool: True if the unit was un-parked.
    """
    if not can_unpark(run_dir, host, bad):
        return False
    marker = read_json(os.path.join(run_dir, "PARKED"))
    record = read_attempts(run_dir)
    entry = {
        "time": now(),
        "host": host,
        "parked": marker,
        "events_before": len(record["events"])
    }
    if live_hosts(marker.get("hosts") or [], bad) is None:
        record.setdefault("unparked_bad_hosts", []).append(entry)
    else:
        record["unparked"] = entry
    record["since"] = len(record["events"])
    record["crashes"] = record["free"] = 0
    write_json(attempts_path(run_dir), record)
    try:
        os.replace(os.path.join(run_dir, "PARKED"),
                   os.path.join(run_dir, f"PARKED.unparked.{time.time():.0f}"))
    except FileNotFoundError:
        return False
    return True


def held(run_dir: Optional[str]) -> bool:
    """A unit held for an operator (CONFIG_MISMATCH marker)."""
    return bool(run_dir) and os.path.exists(os.path.join(run_dir, HOLD))


def hold(root: str, name: str, run_dir: str, details: Dict) -> str:
    """Holds a unit whose resume state belongs to another configuration:
    alert record and CONFIG_MISMATCH marker (no worker picks it until an
    operator removes the marker). Returns the alert path."""
    path = alert(
        root, name, "resume state of another configuration: held for an "
        "operator", {
            "run_dir": run_dir,
            "note": "remove the resume state (or the run directory), then "
                    f"the {HOLD} marker",
            **details
        })
    write_json(os.path.join(run_dir, HOLD), {"time": now(), "alert": path})
    return path


def _beat_attempt(record_path: str, record: Dict, event: Dict,
                  start: float) -> None:
    """Heartbeat of an open attempt: its end moved to now, the record
    written (atomically)."""
    update_attempt(event, start)
    write_json(record_path, record)


def _attempt(cmd: List[str], run_dir: str, env: Optional[Dict[str, str]],
             on_val: Optional[Callable[[Dict], bool]],
             on_beat: Callable[[], None]) -> tuple:
    """Runs one attempt of a unit, logging and streaming its output.

    Returns:
        (int, str, bool, Optional[int]): exit code, last lines of the log,
          stopped by `on_val` (pruned), trainable parameters (if printed).
    """
    tail: List[str] = []
    stopped, params = False, None
    with _Heartbeat(run_dir, on_beat), open(os.path.join(run_dir, "log.txt"),
                                            "a",
                                            encoding="utf-8") as log:
        log.write(f"=== {now()} {' '.join(cmd)}\n")
        with subprocess.Popen(cmd,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT,
                              text=True,
                              bufsize=1,
                              cwd=REPO,
                              env=env) as process:
            _CHILDREN.add(process)
            if STOP.is_set():
                process.send_signal(signal.SIGUSR1)
            for line in process.stdout:
                log.write(line)
                tail = (tail + [line])[-TAIL_LINES:]
                match = PARAMS_LINE.match(line)
                if match:
                    params = int(match.group(1))
                scores = parse_val(line)
                if scores and on_val is not None and on_val(scores):
                    stopped = True
                    process.terminate()
                    break
            returncode = process.wait()
            _CHILDREN.discard(process)
    return returncode, "".join(tail), stopped, params


def run_unit(name: str,
             cmd: List[str],
             run_dir: str,
             root: str,
             model: str,
             on_val: Optional[Callable[[Dict], bool]] = None,
             env: Optional[Dict[str, str]] = None,
             config: Optional[Dict] = None) -> Dict:
    """Runs one unit to completion, resuming it after failures.

    Args:
        name (str): Unit name of the records and alerts.
        cmd (List[str]): The run.py command (with --resume).
        run_dir (str): Output directory of the run (log, heartbeat,
          config.json, attempts.json).
        root (str): Root of the track (alerts go to <root>/alerts).
        model (str): Model name (history file of the run).
        on_val (callable, optional): Called with every parsed VAL line;
          returning True stops the run (pruned).
        env (dict, optional): Environment of the run.
        config (dict, optional): Hyperparameters of the run (written to
          config.json as passed to run.py, i.e. formatted).

    Returns:
        dict: 'status' ('ok', 'pruned', 'diverged', 'too_large', 'oom',
          'stopped', 'config_mismatch' or 'parked'), 'history' when ok,
          'params' (trainable count, when printed) and 'tail' (last lines
          of the log).
    """
    os.makedirs(run_dir, exist_ok=True)
    write_json(
        os.path.join(run_dir, "config.json"), {
            "config": S.formatted(config) if config is not None else None,
            "command": cmd,
            "command_line": shlex.join(cmd),
            "time": now()
        })
    record_path = attempts_path(run_dir)
    record = read_attempts(run_dir)
    params = None
    gpu = gpu_name()
    resume = resume_path(run_dir, model)
    while True:
        if STOP.is_set():
            return {"status": "stopped", "params": params, "tail": ""}
        close_killed(record, _mtime(resume))
        start = time.time()
        event = open_attempt(record, start, gpu, _mtime(resume))
        write_json(record_path, record)
        returncode, text, stopped, printed = _attempt(
            cmd, run_dir, env, on_val,
            functools.partial(_beat_attempt, record_path, record, event, start))
        params = printed if printed is not None else params
        status = "pruned" if stopped else classify(returncode, text)
        if (status not in ("ok", "pruned", "diverged", "too_large", "oom",
                           "config_mismatch") and STOP.is_set()):
            status = "stopped"  # the worker was asked to stop
        close_attempt(event, start, status, returncode, _mtime(resume))
        check_progress(root, name, run_dir, record)
        result = {"status": status, "params": params, "tail": text}
        if status in ("ok", "pruned", "diverged", "too_large", "stopped",
                      "oom"):
            write_json(record_path, record)
            if status == "ok":
                result["history"] = read_json(history_path(run_dir, model))
            return result
        if status == "config_mismatch":
            write_json(record_path, record)
            hold(root, name, run_dir, {"command": cmd, "tail": text[-4000:]})
            return result
        key = "crashes" if status == "crash" else "free"
        record[key] += 1
        write_json(record_path, record)
        if (record["crashes"] >= C.MAX_ATTEMPTS or
                record["free"] > C.MAX_FREE_RETRIES):
            park(root, name, run_dir, status, record, {
                "command": cmd,
                "tail": text[-4000:]
            })
            return {"status": "parked", "params": params, "tail": text}
        print(f"{name}: {status} (exit {returncode}), resuming", flush=True)
        time.sleep(RETRY_SECONDS)


def stub_curve(cfg: Dict, key: int, epochs: int) -> List[Dict]:
    """Validation curve of a dry run (no training): a saturating curve
    whose level depends on the learning rate and on `key`, deterministic."""
    level = 0.6 + 0.3 * ((key * 0.618034) % 1.0)
    level -= 0.05 * abs(math.log10(cfg["lr"] / 1e-3))
    curve = []
    for epoch in range(C.VAL_EVERY, epochs + 1, C.VAL_EVERY):
        value = level * (1.0 - 0.5**(epoch / 30.0))
        curve.append({
            "epoch": epoch,
            "r2_mean": value,
            "r2_top": value - 0.1,
            "r2_base": value + 0.05,
            "r2_gauges": [value] * 11
        })
    return curve


def stub_unit(cfg: Dict,
              key: int,
              epochs: int,
              on_val: Optional[Callable[[Dict], bool]] = None,
              run_dir: Optional[str] = None) -> Dict:
    """`run_unit` of a dry run: streams `stub_curve` to `on_val` (one
    validation every STUB_SECONDS; a stop request stops it). With
    `run_dir`, it writes config.json and records the attempt (GPU 'none')
    in attempts.json, as run_unit does."""
    start = time.time()

    def finish(status: str) -> None:
        if run_dir is None:
            return
        write_json(os.path.join(run_dir, "config.json"), {
            "config": S.formatted(cfg),
            "command": None,
            "time": now()
        })
        record = read_attempts(run_dir)
        add_attempt(record, start, status, None, "none")
        write_json(attempts_path(run_dir), record)

    curve = stub_curve(cfg, key, epochs)
    for scores in curve:
        if STOP.wait(STUB_SECONDS) if STUB_SECONDS else STOP.is_set():
            finish("stopped")
            return {"status": "stopped", "params": 1000, "tail": ""}
        if on_val is not None and on_val(scores):
            finish("pruned")
            return {"status": "pruned", "params": 1000, "tail": ""}
    finish("ok")
    return {
        "status": "ok",
        "history": {
            "train_loss": [0.0] * epochs,
            "val_r2": curve
        },
        "params": 1000,
        "tail": ""
    }
