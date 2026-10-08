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
  - a resume state of another configuration (EXIT_CONFIG_MISMATCH) parks
    the unit at once (retrying cannot help);
  - a crash (out of memory included) is resumed from the last checkpoint,
    up to MAX_ATTEMPTS attempts; then the unit is parked (PARKED marker,
    with the hosts of its failures) and an alert record is written to
    <root>/alerts/;
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
into GPU-hours).

A parked unit whose failures all came from a single host is un-parked once
(`unpark`) when a worker of another host picks it (hpo/pick.py): the
failures were probably the host's, not the unit's. A unit parked twice, or
after failures on two hosts, stays parked.

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
from floatsense import constants as FC
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
HEARTBEAT_SECONDS = 30.0
RETRY_SECONDS = 5.0  # pause before resuming a failed run
TAIL_LINES = 60
# A stop request of the worker; the runs it started get SIGUSR1.
STOP = threading.Event()
_CHILDREN: set = set()
# Seconds per validation of the dry-run stub (0: instantaneous).
STUB_SECONDS = 0.0
# Exit code of a resume refused because the stored run configuration differs
# (6 if the trainer does not define it).
EXIT_CONFIG_MISMATCH = getattr(FC, "EXIT_CONFIG_MISMATCH", 6)
# Outcomes of an attempt that are not failures of the host.
NOT_FAILURES = ("ok", "pruned", "stopped", "diverged", "too_large")


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


def claim(run_dir: str) -> bool:
    """Claims a unit for this worker (another live worker keeps it).

    A claim whose heartbeat is stale is taken over: the old claim is
    renamed away first, and only one worker can rename it.
    """
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "claim")
    owner = f"{socket.gethostname()}:{os.getpid()}"
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


def release(run_dir: str) -> None:
    """Gives a claimed unit back."""
    path = os.path.join(run_dir, "claim")
    if os.path.exists(path):
        os.remove(path)


def touch(run_dir: str) -> None:
    """Beats the heartbeat of a unit."""
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "heartbeat"), "a", encoding="utf-8"):
        os.utime(os.path.join(run_dir, "heartbeat"))


def classify(returncode: int, tail: str) -> str:
    """Outcome of a finished run from its exit code and last lines.

    Returns:
        str: 'ok', 'diverged', 'too_large', 'stopped', 'config_mismatch',
          'preempted', 'hardware' or 'crash' (out of memory and every
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
    """Touches the heartbeat of a unit while its run is alive."""

    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._beat, daemon=True)

    def _beat(self):
        while not self.stop.wait(HEARTBEAT_SECONDS):
            touch(self.run_dir)

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


def read_attempts(run_dir: str) -> Dict:
    """The attempts record of a unit (empty if it never ran)."""
    record = read_json(attempts_path(run_dir)) or {}
    record.setdefault("crashes", 0)
    record.setdefault("free", 0)
    record.setdefault("events", [])
    return record


def add_attempt(record: Dict, start: float, status: str,
                returncode: Optional[int], gpu: str) -> None:
    """Appends one attempt (wall-clock start and end, seconds, host, GPU,
    status) to an attempts record."""
    end = time.time()
    record["events"].append({
        "start": _iso(start),
        "end": _iso(end),
        "seconds": round(end - start, 1),
        "host": socket.gethostname(),
        "gpu": gpu,
        "status": status,
        "returncode": returncode
    })


def _iso(stamp: float) -> str:
    return datetime.datetime.fromtimestamp(stamp).isoformat(timespec="seconds")


def failure_hosts(record: Dict) -> List[str]:
    """Hosts of the failed attempts of a unit since its last un-parking."""
    since = (record.get("unparked") or {}).get("events_before", 0)
    return sorted({
        e.get("host") or "unknown"
        for e in record["events"][since:]
        if e.get("status") not in NOT_FAILURES
    })


def park(root: str, name: str, run_dir: str, reason: str, record: Dict,
         details: Dict) -> str:
    """Parks a unit: alert record and PARKED marker (with the hosts of its
    failures; none for a failure that is not the host's). Returns the alert
    path."""
    hosts = [] if reason == "config_mismatch" else failure_hosts(record)
    path = alert(root, name, f"parked after {reason}", {
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


def can_unpark(run_dir: str, host: str) -> bool:
    """A parked unit whose failures all came from one host other than
    `host`, never un-parked before."""
    marker = read_json(os.path.join(run_dir, "PARKED"))
    if marker is None:
        return False
    hosts = marker.get("hosts") or []
    return (len(hosts) == 1 and hosts[0] != host and
            not read_attempts(run_dir).get("unparked"))


def unpark(run_dir: str, host: str) -> bool:
    """Un-parks a unit once (see `can_unpark`; the caller holds its lock):
    the PARKED marker is renamed away and the failure counts restart.

    Returns:
        bool: True if the unit was un-parked.
    """
    if not can_unpark(run_dir, host):
        return False
    marker = read_json(os.path.join(run_dir, "PARKED"))
    record = read_attempts(run_dir)
    record["unparked"] = {
        "time": now(),
        "host": host,
        "parked": marker,
        "events_before": len(record["events"])
    }
    record["crashes"] = record["free"] = 0
    write_json(attempts_path(run_dir), record)
    try:
        os.replace(os.path.join(run_dir, "PARKED"),
                   os.path.join(run_dir, "PARKED.unparked"))
    except FileNotFoundError:
        return False
    return True


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
        dict: 'status' ('ok', 'pruned', 'diverged', 'too_large',
          'stopped' or 'parked'), 'history' when ok, 'params' (trainable
          count, when printed) and 'tail' (last lines of the log).
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
    while True:
        if STOP.is_set():
            return {"status": "stopped", "params": params, "tail": ""}
        tail: List[str] = []
        stopped = False
        start = time.time()
        with _Heartbeat(run_dir), open(os.path.join(run_dir, "log.txt"),
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
        text = "".join(tail)
        status = "pruned" if stopped else classify(returncode, text)
        if (status not in ("ok", "pruned", "diverged", "too_large") and
                STOP.is_set()):
            status = "stopped"  # the worker was asked to stop
        add_attempt(record, start, status, returncode, gpu)
        if status in ("ok", "pruned", "diverged", "too_large", "stopped"):
            write_json(record_path, record)
            result = {"status": status, "params": params, "tail": text}
            if status == "ok":
                result["history"] = read_json(history_path(run_dir, model))
            return result
        if status == "config_mismatch":
            write_json(record_path, record)
            park(root, name, run_dir, status, record, {
                "command": cmd,
                "tail": text[-4000:]
            })
            return {"status": "parked", "params": params, "tail": text}
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
