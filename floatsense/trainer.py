# pylint: disable=too-many-lines
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-instance-attributes
# pylint: disable=not-callable
# pylint: disable=wrong-import-position
# pylint: disable=too-many-branches
# pylint: disable=too-many-statements
"""Training and damage-based evaluation of moment reconstruction models.

Trains a sequence model on the released time series of a train split and
evaluates it on held-out simulations with the same fatigue-damage metrics
as the physics baseline (R^2 of log10 damage, median damage ratio, fraction
within a factor of 2), writing a per-simulation damage CSV with the same
columns as the physics baseline.

With the default arguments a run reproduces the published trainer bit for
bit, with one exception: a non-finite training loss raises DivergedError
(exit code 3 in scripts/train/run.py), where the published code finished
the run and saved a non-finite checkpoint. The two differ only for runs
that diverge.
"""

import glob
import hashlib
import json
import math
import multiprocessing
import os
import random
import shutil
import signal
import socket
import threading
import time
import uuid

# cuBLAS needs this before its first call to run deterministically.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .constants import CHECKPOINT_SECONDS
from .constants import INPUT_LENGTH
from .constants import LOWPASS_HZ
from .constants import LOWPASS_ORDER
from .constants import MAX_TIME
from .constants import MIN_TIME
from .constants import SN_INTERCEPTS_LOG10
from .constants import SN_SLOPES
from .data import SequenceDataset
from .data import compute_norm_stats
from .fatigue import damage_filter
from .metrics import summarize_damage
from .models import ACCEL_FIRST_MODELS
from .models import LENGTH_FIXED_MODELS
from .models import build_model
from .models import count_parameters
from .release import ReleasedTower
from .release import TowerGauges


def _damage_job(moment, gauge, tower: TowerGauges, intercepts, slopes) -> float:
    """Damage of one series at one gauge (picklable for the pool)."""
    return tower.damage(moment, gauge, intercepts, slopes)


class DivergedError(RuntimeError):
    """The training loss or a validation prediction is not finite."""


class ModelTooLargeError(RuntimeError):
    """The model has more trainable parameters than allowed."""


class StoppedError(RuntimeError):
    """Stopped on request (SIGUSR1) after saving the resume state."""


class ConfigMismatchError(ValueError):
    """The resume state in the output directory belongs to another run
    configuration."""


# Set by SIGUSR1 in a resumable run: finish the epoch, save, stop. Cleared
# when `train` returns or raises, so a request never outlives its run.
STOP_REQUESTED = threading.Event()


def request_stop(signum, frame) -> None:
    """SIGUSR1 handler of a resumable run."""
    del signum, frame
    STOP_REQUESTED.set()


def lr_factor(step: int, total_steps: int, warmup_steps: int,
              schedule: str) -> float:
    """Learning-rate multiplier of an optimizer step.

    A linear warm-up over `warmup_steps` (from 1/warmup_steps to 1), then
    constant or a cosine decay to zero at `total_steps`.

    Args:
        step (int): Optimizer step, from 0.
        total_steps (int): Steps of the whole run.
        warmup_steps (int): Steps of the linear warm-up (0 = none).
        schedule (str): 'constant' or 'cosine'.

    Returns:
        float: The multiplier of the base learning rate.
    """
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    if schedule == "constant":
        return 1.0
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def _rng_state() -> Dict:
    """Every random stream a run draws from."""
    return {
        "torch": torch.get_rng_state(),
        "cuda": (torch.cuda.get_rng_state_all()
                 if torch.cuda.is_available() else None),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def _set_rng_state(state: Dict) -> None:
    """Restores the streams saved by `_rng_state` (the generator states
    must be CPU ByteTensors, wherever they were loaded)."""
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def _atomic_save(obj, path: str) -> None:
    """torch.save through a temporary directory and a rename (never half
    written, even if the job is killed).

    The file is written as <path>.tmp.<host>.<pid>.<uuid8>/<name of path>,
    a directory unique across the hosts that share a file system (see
    `_temporary_in_flight`). torch names the root folder of the archive
    after the file name, so the archive holds the same names (and bytes) as
    a direct torch.save to `path`, without the host name or the pid.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = (f"{path}.tmp.{socket.gethostname()}.{os.getpid()}."
           f"{uuid.uuid4().hex[:8]}")
    os.mkdir(tmp)
    try:
        written = os.path.join(tmp, os.path.basename(path))
        torch.save(obj, written)
        os.replace(written, path)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _temporary_writer(path: str) -> Optional[tuple]:
    """(host, pid) of a temporary of `_atomic_save`, None if the name does
    not follow its pattern. The older <path>.tmp.<pid> pattern counts as
    written on this host."""
    suffix = path.rsplit(".tmp.", 1)[-1]
    if suffix.isdigit():
        return socket.gethostname(), int(suffix)
    parts = suffix.rsplit(".", 2)
    if (len(parts) == 3 and parts[0] and parts[1].isdigit() and
            len(parts[2]) == 8 and
            all(c in "0123456789abcdef" for c in parts[2])):
        return parts[0], int(parts[1])
    return None


def _temporary_mtime(path: str) -> float:
    """Last modification of a temporary of `_atomic_save`: of the file, or
    of the directory and the file being written in it."""
    mtime = os.path.getmtime(path)
    if os.path.isdir(path):
        for entry in os.scandir(path):
            mtime = max(mtime, entry.stat(follow_symlinks=False).st_mtime)
    return mtime


def _temporary_in_flight(path: str, max_age: float) -> bool:
    """Whether a temporary (directory, or file of the older pattern) of
    `_atomic_save` may be the save in flight of a concurrent writer: it was
    modified less than `max_age` seconds ago and, if written on this host,
    by another live process (a process of another user counts as alive). A
    pid seen from another host means nothing, so a temporary of another
    host is judged on its age alone; an old temporary of a live pid is a
    leak of a reused pid."""
    writer = _temporary_writer(path)
    if writer is None:
        return False
    host, pid = writer
    local = host == socket.gethostname()
    if local and pid in (0, os.getpid()):
        return False
    try:
        if time.time() - _temporary_mtime(path) >= max_age:
            return False
        if local:
            os.kill(pid, 0)
    except (ProcessLookupError, FileNotFoundError, OverflowError):
        return False
    except PermissionError:
        return True
    return True


def _file_sha256(path: Optional[str]) -> Optional[str]:
    """sha256 of the bytes of a file (None for no file)."""
    if path is None:
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _ids_hash(train_ids: List[int], val_ids: Optional[List[int]]) -> str:
    """Digest of the training and validation simulations, in order."""
    ids = [[int(i) for i in train_ids], [int(i) for i in val_ids or []]]
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


class SequenceModelTrainer:
    """Trains and evaluates a sequence model for one direction."""

    def __init__(self,
                 release: ReleasedTower,
                 output_dir: str,
                 direction: str = "fa",
                 model_name: str = "spectral",
                 condition_channels: Optional[List[str]] = None,
                 min_time: float = MIN_TIME,
                 max_time: float = MAX_TIME,
                 apply_lowpass: bool = True,
                 lowpass_hz: float = LOWPASS_HZ,
                 lowpass_order: int = LOWPASS_ORDER,
                 crop_length: int = 4096,
                 batch_size: int = 16,
                 learning_rate: float = 1e-3,
                 num_epochs: int = 50,
                 val_every: int = 0,
                 early_stopping_patience: int = 0,
                 num_workers: int = 4,
                 sn_intercepts_log10: Optional[List[float]] = None,
                 sn_slopes: Optional[List[float]] = None,
                 loss_name: str = "mse",
                 damage_loss_weight: float = 1.0,
                 damage_m: float = 4.0,
                 damage_freq_exponent: float = 1.0,
                 init_checkpoint: Optional[str] = None,
                 calibration_path: Optional[str] = None,
                 condition_bound: float = 0.5,
                 target_channel: Optional[str] = None,
                 damage_gauge: int = 0,
                 input_channels: Optional[List[str]] = None,
                 height_targets: bool = False,
                 height_factors: Optional[List[float]] = None,
                 seed: int = 0,
                 deterministic: bool = True,
                 device: Optional[str] = None,
                 weight_decay: Optional[float] = None,
                 schedule: str = "constant",
                 warmup_epochs: int = 0,
                 grad_clip: float = 0.0,
                 model_kwargs: Optional[Dict] = None,
                 val_score: str = "loss",
                 resume: bool = False,
                 max_params_m: float = 0.0,
                 save_epochs: Optional[List[int]] = None,
                 checkpoint_seconds: Optional[float] = None):
        """Initializes the trainer.

        Args:
            release (ReleasedTower): Training tower of the released dataset.
            output_dir (str): Directory for checkpoints, CSVs and summary.
            direction (str): Reconstruction direction ('fa' or 'ss').
            model_name (str): Model name understood by `build_model`.
            condition_channels (List[str], optional): Operating-state
              channels.
            min_time (float): Start time of the usable window [s].
            max_time (float): End time of the usable window [s].
            apply_lowpass (bool): Low-pass the true and the reconstructed
              moment before the damage (on by default: the damage of these
              towers lies below 3 Hz).
            lowpass_hz (float): Cutoff of that low-pass [Hz].
            lowpass_order (int): Butterworth order of one pass.
            crop_length (int): Training crop length; the length-fixed
              spectral models always train on the full window.
            batch_size (int): Training batch size.
            learning_rate (float): Adam learning rate.
            num_epochs (int): Number of training epochs.
            num_workers (int): DataLoader workers.
            sn_intercepts_log10 (List[float], optional): SN log10 intercepts.
            sn_slopes (List[float], optional): SN curve slopes.
            loss_name (str): 'mse', or 'damage' to add a damage-aware term
              penalizing the squared log-ratio of the differentiable spectral
              damage proxy sum(|A|^m * f^e) between prediction and target.
            damage_loss_weight (float): Weight of the damage-aware term.
            damage_m (float): SN-like amplitude exponent of the proxy.
            damage_freq_exponent (float): Frequency exponent of the proxy.
            init_checkpoint (str, optional): Checkpoint to fine-tune from.
              Training starts from its weights and keeps its normalization
              stats (so the model stays consistent with the source domain).
              It must not be the checkpoint this run writes.
            calibration_path (str, optional): Physics calibration JSON for
              models that consume the per-simulation physics gain (hybrid).
            condition_bound (float): Tanh bound of the hybrid model's
              condition-dependent correction (0 disables the bound).
            target_channel (str, optional): Overrides the target moment
              channel (e.g. an intermediate section gage 'tower_5_mfa').
            damage_gauge (int): Gauge index of the target channel in the
              single-height task, 0 (base) to 10 (top).
            input_channels (List[str], optional): Replaces the default
              [acceleration + condition channels] input stack (sensor
              ablations).
            seed (int): Seed of the weight initialization, the random crops
              and the batch order.
            deterministic (bool): Use deterministic cuDNN/CUDA kernels, so
              that a rerun of the same seed on the same GPU type is
              identical (the paper runs used False).
            device (str, optional): Torch device (default: cuda if available).
            weight_decay (float, optional): AdamW weight decay; None keeps
              Adam (the published recipe).
            schedule (str): 'constant' or 'cosine' (decay to zero at the
              last step of the run).
            warmup_epochs (int): Linear learning-rate warm-up (0 = none).
            grad_clip (float): Maximum gradient norm (0 = no clipping).
            model_kwargs (dict, optional): Architecture knobs passed to
              `build_model` (a 'condition_bound' entry sets the bound).
            val_score (str): Validation score, 'loss' (squared error on
              normalized crops) or 'damage' (R^2 of log10 damage over the
              11 gauges, the benchmark metric).
            resume (bool): Keep a resume state (at every validation and
              every `checkpoint_seconds`) and continue from it if present;
              SIGUSR1 saves it at the end of the epoch and raises
              StoppedError. A resume state written with another run
              configuration (see `_run_config`) raises ConfigMismatchError;
              `num_epochs` is part of it, so a run is extended in a new
              output directory, never in place.
            max_params_m (float): Refuse a model with more trainable
              parameters, in millions (0 = no limit).
            save_epochs (List[int], optional): Epochs whose weights are also
              kept as <model>_<direction>_epoch<e>.pt.
            checkpoint_seconds (float, optional): Wall time between two
              saves of the resume state, checked at the end of each epoch
              (default CHECKPOINT_SECONDS; 0 saves after every epoch).
        """
        self.release = release
        self.output_dir = output_dir
        self.direction = direction
        self.model_name = model_name
        self.condition_channels = condition_channels
        self.min_time = min_time
        self.max_time = max_time
        self.apply_lowpass = apply_lowpass
        self.lowpass_hz = lowpass_hz
        self.lowpass_order = lowpass_order
        self.crop_length = (crop_length
                            if model_name not in LENGTH_FIXED_MODELS else None)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.val_every = val_every
        self.early_stopping_patience = early_stopping_patience
        self.num_workers = num_workers
        self.sn_intercepts_log10 = sn_intercepts_log10
        self.sn_slopes = sn_slopes
        self.loss_name = loss_name
        self.damage_loss_weight = damage_loss_weight
        self.damage_m = damage_m
        self.damage_freq_exponent = damage_freq_exponent
        self.init_checkpoint = init_checkpoint
        self.calibration_path = calibration_path
        self.target_channel = target_channel
        self.damage_gauge = damage_gauge
        self.input_channels = input_channels
        self.height_targets = height_targets
        self.height_factors = height_factors
        self.seed = seed
        self.deterministic = deterministic
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.weight_decay = weight_decay
        self.schedule = schedule
        self.warmup_epochs = warmup_epochs
        self.grad_clip = grad_clip
        self.model_kwargs = dict(model_kwargs or {})
        # The bound is a constructor argument of its own (hybrids).
        self.condition_bound = self.model_kwargs.pop("condition_bound",
                                                     condition_bound)
        if val_score not in ("loss", "damage"):
            raise ValueError(f"Unknown val_score '{val_score}'.")
        if schedule not in ("constant", "cosine"):
            raise ValueError(f"Unknown schedule '{schedule}'.")
        self.val_score = val_score
        self.resume = resume
        self.max_params_m = max_params_m
        self.save_epochs = set(save_epochs or [])
        self.checkpoint_seconds = checkpoint_seconds
        self._previous_handler = None
        self.stop_requested = False
        self._config = None
        self.model = None
        self.norm_stats = None
        self._eval_length = None
        self._val_true = None

    def _make_dataset(self,
                      sim_ids: List[int],
                      crop_length: Optional[int],
                      release: Optional[ReleasedTower] = None,
                      window_length: Optional[int] = None) -> SequenceDataset:
        return SequenceDataset(release=release or self.release,
                               sim_ids=sim_ids,
                               direction=self.direction,
                               condition_channels=self.condition_channels,
                               min_time=self.min_time,
                               max_time=self.max_time,
                               crop_length=crop_length,
                               norm_stats=self.norm_stats,
                               calibration_path=self.calibration_path,
                               target_channel=self.target_channel,
                               input_channels=self.input_channels,
                               height_targets=self.height_targets,
                               height_factors=self.height_factors,
                               window_length=window_length)

    def _input_length(self) -> Optional[int]:
        """Window of a length-fixed model (None: crops or any length)."""
        return None if self.crop_length else self._eval_length

    def _predict_window(self, dataset: SequenceDataset, index: int,
                        item: Dict[str, torch.Tensor]) -> np.ndarray:
        """Normalized prediction over the full scored window.

        Every model trained on crops sees `INPUT_LENGTH` (6,000) samples per
        forward pass, the length the paper checkpoints were scored at (and a
        multiple of the patch and pooling sizes of PatchTST and U-Net); a
        length-fixed model sees its own input size (`_eval_length`). When
        that is shorter than the window, the first and the last `length`
        samples are predicted and the last samples of the second prediction,
        aligned on the overlap, complete the first. A variance head (Prob-TCN)
        is stitched before sampling, so one noise draw covers the window.

        Args:
            dataset (SequenceDataset): Evaluation dataset (full window).
            index (int): Item index.
            item (dict): dataset[index], already loaded.

        Returns:
            np.ndarray: Normalized prediction over the full window.
        """
        length = INPUT_LENGTH if self.crop_length else self._eval_length
        extra = item["target"].shape[-1] - length
        if extra <= 0:
            return self._predict(item)
        dataset.window_length = length
        try:
            dataset.window_offset = 0
            first = self._predict(dataset[index], sample=False)
            dataset.window_offset = extra
            last = self._predict(dataset[index], sample=False)
        finally:
            dataset.window_length, dataset.window_offset = None, 0
        # Channel 0 is the mean, shifted onto the first prediction over the
        # overlap; a log-variance channel (Prob-TCN) is appended as it is.
        first, last = np.atleast_2d(first), np.atleast_2d(last)
        shift = float(np.mean(first[0, extra:] - last[0, :-extra]))
        tail = last[:, -extra:].copy()
        tail[0] += shift
        stitched = np.concatenate([first, tail], axis=-1)
        if stitched.shape[0] == 1:
            return stitched[0]
        return self._sample(torch.from_numpy(stitched).to(self.device))

    def _damage_proxy_loss(self, prediction: torch.Tensor, target: torch.Tensor,
                           freq_weights: torch.Tensor) -> torch.Tensor:
        """Squared log-ratio of the spectral damage proxy sum(|A|^m * f^e)."""
        scale = 2.0 / prediction.shape[-1]
        proxies = []
        for series in (prediction, target):
            amplitudes = torch.abs(torch.fft.rfft(series, dim=-1)) * scale
            proxies.append(
                torch.sum(amplitudes**self.damage_m * freq_weights, dim=-1))
        log_ratio = torch.log(proxies[0] + 1e-12) - torch.log(proxies[1] +
                                                              1e-12)
        return torch.mean(log_ratio**2)

    def train(self,
              train_ids: List[int],
              val_ids: Optional[List[int]] = None) -> Dict[str, List[float]]:
        """Trains the model on the given simulations.

        Args:
            train_ids (List[int]): Training simulation IDs.
            val_ids (List[int], optional): Held-out simulations scored every
              few epochs (`val_score`); the score never influences training
              except through early stopping.

        Returns:
            dict: Per-epoch mean training loss under 'train_loss' and, with
            `val_ids`, the validation loss under 'val_loss' (epoch, value)
            or the damage scores under 'val_r2' (one dict per validation).
            A SIGUSR1 during the last epoch (or the epoch that stops it
            early) saves the state and lets the run complete;
            `stop_requested` is then True, and the caller should stop before
            its next run.

        Raises:
            DivergedError: A training loss or a damage-validation prediction
              is not finite. This guard is always on: the published code
              finished such a run with a non-finite checkpoint, so the
              default flags differ from it only for runs that diverge.
            ModelTooLargeError: More trainable parameters than
              `max_params_m`.
            StoppedError: SIGUSR1 in a resumable run (state saved). A
              request in the epoch where early stopping ends the run lets
              it complete, as in the last epoch.
            ValueError: `init_checkpoint` is the checkpoint this run
              writes (among other invalid settings).
            ConfigMismatchError: The resume state of the output directory
              was written with another run configuration (another
              `num_epochs` or `model_kwargs` included: a longer run needs a
              new directory).
        """
        self._previous_handler = None
        self.stop_requested = False
        try:
            return self._train(train_ids, val_ids)
        finally:
            # A request during the last epoch saves and lets the run
            # complete: the caller reads it here and stops.
            self.stop_requested = STOP_REQUESTED.is_set()
            STOP_REQUESTED.clear()
            if self._previous_handler is not None:
                signal.signal(signal.SIGUSR1, self._previous_handler[0])
                self._previous_handler = None

    def _train(self, train_ids: List[int],
               val_ids: Optional[List[int]]) -> Dict[str, List[float]]:
        """`train`, whose stop request is cleared by the caller."""
        if self.deterministic:
            # Same seed, same GPU type -> same weights: deterministic cuDNN
            # and CUDA kernels (ops without one only warn).
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        probe = SequenceDataset(release=self.release,
                                sim_ids=train_ids[:1],
                                direction=self.direction,
                                condition_channels=self.condition_channels,
                                min_time=self.min_time,
                                max_time=self.max_time,
                                target_channel=self.target_channel,
                                input_channels=self.input_channels,
                                height_targets=self.height_targets)
        if (self.model_name in ACCEL_FIRST_MODELS and
                probe.input_channels[0] != probe.accel_channel):
            raise ValueError(f"{self.model_name} reads the first input channel "
                             f"as the acceleration: put {probe.accel_channel} "
                             "first in --input_channels.")
        if (self.init_checkpoint and os.path.realpath(self.init_checkpoint)
                == os.path.realpath(self.checkpoint_path())):
            raise ValueError(
                f"init_checkpoint {self.init_checkpoint} is the checkpoint "
                "this run writes: fine-tune into another output directory.")
        self._remove_stale_temporaries()
        # Read only by a resume state (no work for a run without one).
        self._config = (self._run_config(train_ids, val_ids)
                        if self.resume else None)
        resume_state = self._load_resume()
        if resume_state is not None:
            self._check_run_config(resume_state)
        if resume_state is not None and resume_state.get("completed"):
            # Finished before: the final checkpoint is the result.
            self.load_checkpoint()
            self._print_val_history(resume_state["history"])
            return resume_state["history"]
        stat_channels = list(
            dict.fromkeys([
                c.split(":")[1] if c.startswith("stat:") else c
                for c in probe.input_channels
                if c != "height"
            ] + probe.condition_channels + [probe.moment_channel] + (
                probe.height_channels if self.height_targets else [])))
        init_state = None
        if self.init_checkpoint:
            init_state = torch.load(self.init_checkpoint,
                                    map_location=self.device,
                                    weights_only=False)
            self._check_setup(init_state, probe)
            # A resumed run keeps the stats it trained with.
            self.norm_stats = (init_state["norm_stats"] if resume_state is None
                               else resume_state["norm_stats"])
            self.model_kwargs = (self.model_kwargs or
                                 init_state.get("model_kwargs", {}))
        elif resume_state is not None:
            self.norm_stats = resume_state["norm_stats"]
        else:
            self.norm_stats = compute_norm_stats(self.release, train_ids,
                                                 stat_channels, self.min_time,
                                                 self.max_time)
        # Input length of the length-fixed models: the full scored window
        # (6,001 samples). Checkpoints trained before stored 6,000 and are
        # scored over the full window by _predict_window.
        self._eval_length = probe[0]["inputs"].shape[-1]

        dataset = self._make_dataset(train_ids,
                                     self.crop_length,
                                     window_length=self._input_length())
        # Full batches only, unless the split is smaller than one batch.
        loader = DataLoader(dataset,
                            batch_size=self.batch_size,
                            shuffle=True,
                            num_workers=self.num_workers,
                            drop_last=len(dataset) > self.batch_size)

        num_samples = self.crop_length or self._eval_length
        self.model = self._build_model(num_samples, len(dataset.input_channels),
                                       len(dataset.condition_channels))
        if init_state is not None:
            self.model.load_state_dict(init_state["state_dict"])
        optimizer = self._optimizer()
        scheduler = self._scheduler(optimizer, len(loader))

        freqs = torch.fft.rfftfreq(num_samples,
                                   d=1.0 / dataset.sampling_frequency).to(
                                       self.device)
        freq_weights = freqs**self.damage_freq_exponent

        if self.early_stopping_patience and not val_ids:
            raise ValueError("early_stopping_patience needs a validation "
                             "split (val_split).")
        history = {"train_loss": []}
        val_loader, val_dataset = None, None
        if val_ids and self.val_score == "damage":
            history["val_r2"] = []
            val_dataset = self._damage_dataset(val_ids)
        elif val_ids:
            history["val_loss"] = []
            val_loader = DataLoader(self._make_dataset(
                val_ids, self.crop_length, window_length=self._input_length()),
                                    batch_size=self.batch_size,
                                    shuffle=False,
                                    num_workers=0)
        maximize = self.val_score == "damage"
        val_interval = self.val_every or max(1, self.num_epochs // 10)
        best_val = -float("inf") if maximize else float("inf")
        best_epoch, best_state, start_epoch = 0, None, 0
        if resume_state is not None:
            if resume_state["model_kwargs"] != self.model_kwargs:
                raise ConfigMismatchError(
                    f"Resume state of {resume_state['model_kwargs']}"
                    f", run configured with {self.model_kwargs}.")
            self.model.load_state_dict(resume_state["state_dict"])
            optimizer.load_state_dict(resume_state["optimizer"])
            if scheduler is not None:
                scheduler.load_state_dict(resume_state["scheduler"])
            history = resume_state["history"]
            best_val, best_epoch = resume_state["best"]
            best_state = resume_state["best_state"]
            start_epoch = resume_state["epoch"]
            _set_rng_state(resume_state["rng"])
            print(
                f"[{self.model_name}/{self.direction}] resumed at epoch "
                f"{start_epoch}",
                flush=True)
            self._print_val_history(history)
        last_save = time.monotonic()
        checkpoint_seconds = self._checkpoint_interval()
        if (self.resume and
                threading.current_thread() is threading.main_thread()):
            # Restored by `train` when the run returns or raises.
            self._previous_handler = (signal.signal(signal.SIGUSR1,
                                                    request_stop),)
        for epoch in range(start_epoch, self.num_epochs):
            self.model.train()
            losses = []
            for batch in loader:
                inputs = batch["inputs"].to(self.device)
                condition = batch["condition"].to(self.device)
                target = batch["target"].to(self.device)
                if getattr(self.model, "needs_physics_gain", False):
                    prediction = self.model(
                        inputs, condition,
                        batch["physics_gain"].to(self.device))
                else:
                    prediction = self.model(inputs, condition)
                if getattr(self.model, "predicts_variance", False):
                    log_var = prediction[:, 1:2, :]
                    prediction = prediction[:, :1, :]
                    loss = torch.mean(
                        0.5 * (log_var +
                               (prediction - target)**2 / torch.exp(log_var)))
                else:
                    loss = torch.mean((prediction - target)**2)
                if self.loss_name == "damage":
                    loss = loss + self.damage_loss_weight * (
                        self._damage_proxy_loss(prediction[:, 0, :],
                                                target[:, 0, :], freq_weights))
                optimizer.zero_grad()
                loss.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.grad_clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                losses.append(float(loss.detach()))
                if not math.isfinite(losses[-1]):
                    raise DivergedError(f"Training loss {losses[-1]} at "
                                        f"epoch {epoch + 1}.")
            history["train_loss"].append(float(np.mean(losses)))
            log_now = (epoch + 1) % max(1, self.num_epochs // 10) == 0
            val_now = ((val_loader is not None or val_dataset is not None) and
                       (epoch + 1) % val_interval == 0)
            if val_now and val_dataset is not None:
                scores = self._damage_scores(val_dataset)
                history["val_r2"].append({"epoch": epoch + 1, **scores})
                self._print_val(history["val_r2"][-1])
                val_loss = scores["r2_mean"]
            elif val_now:
                val_loss = self._validation_loss(val_loader)
                history["val_loss"].append((epoch + 1, val_loss))
            if val_now and (val_loss > best_val
                            if maximize else val_loss < best_val):
                best_val, best_epoch = val_loss, epoch + 1
                if self.early_stopping_patience:
                    best_state = {
                        k: v.detach().clone().cpu()
                        for k, v in self.model.state_dict().items()
                    }
            if log_now or val_now:
                line = (f"[{self.model_name}/{self.direction}] "
                        f"epoch {epoch + 1}/{self.num_epochs} "
                        f"loss {history['train_loss'][-1]:.5f}")
                if val_now:
                    line += (f" val_r2 {val_loss:.5f}"
                             if maximize else f" val_loss {val_loss:.5f}")
                print(line)
            if epoch + 1 in self.save_epochs:
                self.save_checkpoint(self.checkpoint_path(epoch + 1))
            stop_now = self.resume and STOP_REQUESTED.is_set()
            if self.resume and (val_now or stop_now or time.monotonic() -
                                last_save >= checkpoint_seconds):
                self._save_resume(epoch + 1, optimizer, scheduler, history,
                                  (best_val, best_epoch), best_state)
                last_save = time.monotonic()
            early_stop = bool(self.early_stopping_patience and best_epoch and
                              epoch + 1 - best_epoch
                              >= self.early_stopping_patience * val_interval)
            # A run that ends here (last epoch or early stop) completes, and
            # the request is read by the caller in `stop_requested`.
            if stop_now and epoch + 1 < self.num_epochs and not early_stop:
                raise StoppedError(f"Stopped after epoch {epoch + 1}.")
            if early_stop:
                print(f"[{self.model_name}/{self.direction}] early stop at "
                      f"epoch {epoch + 1}, best {best_epoch} "
                      f"({'val_r2' if maximize else 'val_loss'} "
                      f"{best_val:.5f})")
                break

        restore = self.early_stopping_patience and best_state is not None
        if restore:
            self.model.load_state_dict(best_state)
        if restore or (maximize and best_epoch):
            history["best_epoch"] = best_epoch
            history["best_val_r2" if maximize else "best_val_loss"] = best_val
        self.save_checkpoint()
        os.makedirs(self.output_dir, exist_ok=True)
        name = f"history_{self.model_name}_{self.direction}.json"
        with open(os.path.join(self.output_dir, name), "w",
                  encoding="utf-8") as file:
            json.dump(history, file)
        if self.resume:
            # The weights are in the final checkpoint; keep only the record.
            _atomic_save(
                {
                    "completed": True,
                    "history": history,
                    "config": self._config
                }, self.resume_path())
        return history

    def _build_model(self, num_samples: int, num_inputs: int,
                     num_conditions: int) -> torch.nn.Module:
        """Builds the model, prints its size and enforces max_params_m."""
        model = build_model(self.model_name,
                            num_samples=num_samples,
                            input_channels=num_inputs,
                            condition_dim=num_conditions +
                            int(self.height_targets),
                            condition_bound=self.condition_bound,
                            **self.model_kwargs).to(self.device)
        trainable = count_parameters(model)
        print(
            f"PARAMS trainable={trainable} "
            f"total={count_parameters(model, trainable=False)}",
            flush=True)
        if self.max_params_m and trainable > self.max_params_m * 1e6:
            raise ModelTooLargeError(
                f"{self.model_name} {self.model_kwargs}: {trainable / 1e6:.2f}"
                f" M trainable parameters > {self.max_params_m} M.")
        return model

    def _optimizer(self) -> torch.optim.Optimizer:
        """Adam (published) or, with a weight decay, AdamW."""
        if self.weight_decay is None:
            return torch.optim.Adam(self.model.parameters(),
                                    lr=self.learning_rate)
        return torch.optim.AdamW(self.model.parameters(),
                                 lr=self.learning_rate,
                                 weight_decay=self.weight_decay)

    def _scheduler(self, optimizer: torch.optim.Optimizer,
                   steps_per_epoch: int):
        """Per-step learning-rate schedule (None: constant, no warm-up)."""
        if self.schedule == "constant" and not self.warmup_epochs:
            return None
        total = self.num_epochs * steps_per_epoch
        warmup = self.warmup_epochs * steps_per_epoch
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: lr_factor(step, total, warmup, self.schedule))

    def resume_path(self) -> str:
        """Path of the resume state of this model/direction."""
        return os.path.join(self.output_dir,
                            f"{self.model_name}_{self.direction}_resume.pt")

    def _load_resume(self) -> Optional[Dict]:
        """The resume state, if `resume` is on and one was saved.

        Loaded on the CPU: the random generator states must stay CPU
        ByteTensors, and load_state_dict copies the rest to the device.
        """
        if not self.resume or not os.path.exists(self.resume_path()):
            return None
        return torch.load(self.resume_path(),
                          map_location="cpu",
                          weights_only=False)

    def _run_config(self, train_ids: List[int],
                    val_ids: Optional[List[int]]) -> Dict:
        """Settings a resume state must have been written with.

        Every setting that changes the weights or the selected epoch: the
        tower, the task (inputs, target, scored window, damage metric), the
        recipe and the training simulations. The number of epochs is one of
        them, so a finished or interrupted run is never extended in place:
        a longer run goes to a new output directory. The calibration and
        the initial checkpoint are stored by real path, for information, and
        by the sha256 of their contents, which is compared (read once per
        run, only by a run with `resume`).
        """

        def as_list(values) -> Optional[List]:
            return None if values is None else list(values)

        def real_path(path: Optional[str]) -> Optional[str]:
            return None if path is None else os.path.realpath(path)

        return {
            "tower": getattr(self.release, "name", None),
            "model_name": self.model_name,
            "direction": self.direction,
            "input_channels": as_list(self.input_channels),
            "condition_channels": as_list(self.condition_channels),
            "target_channel": self.target_channel,
            "damage_gauge": self.damage_gauge,
            "height_targets": bool(self.height_targets),
            "height_factors": (None if self.height_factors is None else
                               [float(f) for f in self.height_factors]),
            "calibration_path": real_path(self.calibration_path),
            "calibration_sha256": _file_sha256(self.calibration_path),
            "init_checkpoint": real_path(self.init_checkpoint),
            "init_checkpoint_sha256": _file_sha256(self.init_checkpoint),
            "min_time": self.min_time,
            "max_time": self.max_time,
            "apply_lowpass": self.apply_lowpass,
            "lowpass_hz": self.lowpass_hz,
            "lowpass_order": self.lowpass_order,
            "sn_intercepts_log10": as_list(self.sn_intercepts_log10),
            "sn_slopes": as_list(self.sn_slopes),
            "model_kwargs": dict(self.model_kwargs),
            "condition_bound": self.condition_bound,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "schedule": self.schedule,
            "warmup_epochs": self.warmup_epochs,
            "num_epochs": self.num_epochs,
            "seed": self.seed,
            "grad_clip": self.grad_clip,
            "batch_size": self.batch_size,
            "crop_length": self.crop_length,
            "loss_name": self.loss_name,
            "damage_loss_weight": self.damage_loss_weight,
            "damage_m": self.damage_m,
            "damage_freq_exponent": self.damage_freq_exponent,
            "early_stopping_patience": self.early_stopping_patience,
            "val_every": self.val_every,
            "val_score": self.val_score,
            "ids_hash": _ids_hash(train_ids, val_ids),
        }

    def _check_run_config(self, resume_state: Dict) -> None:
        """Refuses a resume state (in progress or completed) of another run
        configuration (see `_run_config`). States written before the
        configuration was stored are not checked, and only the settings a
        state stored are compared (a state written before a setting was
        added keeps resuming). The calibration and the initial checkpoint
        are compared by contents (a moved file with the same contents
        resumes), or by real path in states written before the digest."""
        saved = resume_state.get("config")
        if saved is None:
            return
        current = self._config
        saved = dict(saved)
        for key, digest in (("calibration_path", "calibration_sha256"),
                            ("init_checkpoint", "init_checkpoint_sha256")):
            if digest in saved:
                # Same contents, same run: the file may have moved.
                saved.pop(key, None)
            elif saved.get(key) is not None:
                # States written before the digest (and the real path) were
                # stored.
                saved[key] = os.path.realpath(saved[key])
        changed = sorted(k for k in saved if saved[k] != current.get(k))
        if changed:
            details = ", ".join(
                f"{k}: {saved.get(k)!r} -> {current.get(k)!r}" for k in changed)
            raise ConfigMismatchError(
                f"{self.resume_path()} was written by another run "
                f"configuration ({details}); use another output directory "
                "or remove the resume state.")

    def _checkpoint_interval(self) -> float:
        """Wall time between two saves of the resume state [s]."""
        return (CHECKPOINT_SECONDS
                if self.checkpoint_seconds is None else self.checkpoint_seconds)

    def _remove_stale_temporaries(self) -> None:
        """Removes the temporaries (directories, and files of the older
        pattern) left by a killed `_atomic_save` of the files this run
        writes (only those, only in its directory). A temporary younger than
        max(3 checkpoint intervals, 1 h), of another live process on this
        host or of another host, is kept: it may be the save in flight of a
        concurrent writer (see `_temporary_in_flight`)."""
        max_age = max(3.0 * self._checkpoint_interval(), 3600.0)
        stems = [self.checkpoint_path(), self.resume_path()]
        patterns = [glob.escape(stem) + ".tmp.*" for stem in stems]
        patterns.append(
            glob.escape(
                os.path.join(self.output_dir,
                             f"{self.model_name}_{self.direction}_epoch")) +
            "[0-9]*.pt.tmp.*")
        for pattern in patterns:
            for path in glob.glob(pattern):
                if os.path.islink(path) or _temporary_in_flight(path, max_age):
                    continue
                if os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
                elif os.path.isfile(path):
                    os.remove(path)

    def _save_resume(self, epoch: int, optimizer, scheduler, history: Dict,
                     best: tuple, best_state: Optional[Dict]) -> None:
        """Everything needed to continue after `epoch` as if never stopped."""
        _atomic_save(
            {
                "epoch": epoch,
                "state_dict": self.model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler":
                    (scheduler.state_dict() if scheduler is not None else None),
                "history": history,
                "best": best,
                "best_state": best_state,
                "norm_stats": self.norm_stats,
                "model_kwargs": self.model_kwargs,
                "config": self._config,
                "rng": _rng_state(),
            }, self.resume_path())

    @staticmethod
    def _print_val(entry: Dict) -> None:
        """One parsable validation line (read by the hpo drivers)."""
        gauges = ",".join(f"{v:.6f}" for v in entry["r2_gauges"])
        print(
            f"VAL epoch={entry['epoch']} r2_mean={entry['r2_mean']:.6f} "
            f"r2_top={entry['r2_top']:.6f} r2_base={entry['r2_base']:.6f} "
            f"r2_gauges={gauges}",
            flush=True)

    def _print_val_history(self, history: Dict) -> None:
        """Repeats the VAL lines of a resumed run."""
        for entry in history.get("val_r2", []):
            self._print_val(entry)

    def _damage_dataset(self, sim_ids: List[int]) -> SequenceDataset:
        """Full-window dataset of the damage validation; each simulation is
        read once for its 11 gauges and two inputs."""
        if not self.height_targets:
            raise ValueError("val_score='damage' scores the 11 gauges of the "
                             "height task (height_targets).")
        dataset = self._make_dataset(sim_ids, None)
        load, cache = dataset.load_window, {}

        def load_once(sim_id: int) -> np.ndarray:
            if sim_id not in cache:
                cache.clear()
                cache[sim_id] = load(sim_id)
            return cache[sim_id]

        dataset.load_window = load_once
        return dataset

    def _metric_is_released(self) -> bool:
        """Whether damage.parquet holds the true damage of this metric."""
        return (self.apply_lowpass and self.lowpass_hz == LOWPASS_HZ and
                self.lowpass_order == LOWPASS_ORDER and
                self.min_time == MIN_TIME and self.max_time == MAX_TIME and
                list(self.sn_intercepts_log10 or
                     SN_INTERCEPTS_LOG10) == list(SN_INTERCEPTS_LOG10) and
                list(self.sn_slopes or SN_SLOPES) == list(SN_SLOPES))

    def _predict_batch(self, items: List[Dict[str, torch.Tensor]],
                       sample: bool) -> np.ndarray:
        """`_predict` of several items in one forward pass: (batch, length),
        or (batch, 2, length) for an unsampled variance head."""
        inputs = torch.stack([i["inputs"] for i in items]).to(self.device)
        condition = torch.stack([i["condition"] for i in items]).to(self.device)
        if getattr(self.model, "needs_physics_gain", False):
            gain = torch.stack([i["physics_gain"] for i in items])
            return self.model(inputs, condition,
                              gain.to(self.device))[:, 0].cpu().numpy()
        output = self.model(inputs, condition)
        if getattr(self.model, "predicts_variance", False):
            if sample:
                return np.stack([self._sample(o) for o in output])
            return output.cpu().numpy()
        return output[:, 0].cpu().numpy()

    def _predict_gauges(self, dataset: SequenceDataset, index: int,
                        gauges: List[int]) -> np.ndarray:
        """`_predict_window` of the gauges of one simulation, batched.

        Returns:
            np.ndarray: (gauges, window) normalized predictions.
        """

        def items(length=None, offset=0):
            dataset.window_length, dataset.window_offset = length, offset
            out = []
            for gauge in gauges:
                dataset.section = gauge
                out.append(dataset[index])
            return out

        length = INPUT_LENGTH if self.crop_length else self._eval_length
        window = dataset.stop_index - dataset.start_index
        extra = window - length
        try:
            if extra <= 0:
                return self._predict_batch(items(), sample=True)
            first = self._predict_batch(items(length, 0), sample=False)
            last = self._predict_batch(items(length, extra), sample=False)
        finally:
            dataset.window_length, dataset.window_offset = None, 0
        # As _predict_window: the mean channel of the second prediction is
        # shifted onto the first over the overlap; log-variance as it is.
        if first.ndim == 2:
            first, last = first[:, None], last[:, None]
        shift = np.mean(first[:, 0, extra:] - last[:, 0, :-extra], axis=-1)
        tail = last[:, :, -extra:].copy()
        tail[:, 0] += shift[:, None]
        stitched = np.concatenate([first, tail], axis=-1)
        if stitched.shape[1] == 1:
            return stitched[:, 0]
        return np.stack([
            self._sample(torch.from_numpy(s).to(self.device)) for s in stitched
        ])

    def _damage_scores(self, dataset: SequenceDataset) -> Dict:
        """R^2 of log10 damage at each gauge over the dataset simulations.

        The metric of `evaluate` (same filter, window, stitching and S-N
        curve), with the forward passes batched over the 11 gauges, the
        true damage read once from damage.parquet and no file written. The
        random streams and the train mode are restored, so validating does
        not change the training run. On a GPU the batched kernels can differ
        from the per-item passes of `evaluate`, so the scores can differ at
        float tolerance.

        Returns:
            dict: r2_gauges (base to top), r2_mean, r2_top and r2_base.
        """
        rng = _rng_state()
        torch.manual_seed(self.seed)  # fixed noise of a variance head
        self.model.eval()
        tower = dataset.release.geometry
        gauges = list(range(len(tower.channels)))
        sn_args = (tower, self.sn_intercepts_log10, self.sn_slopes)
        jobs = []
        try:
            with torch.no_grad():
                for index, _ in enumerate(dataset.sim_ids):
                    predictions = self._predict_gauges(dataset, index, gauges)
                    if not np.all(np.isfinite(predictions)):
                        raise DivergedError("Non-finite validation prediction.")
                    for gauge, prediction in zip(gauges, predictions):
                        dataset.moment_channel = dataset.height_channels[gauge]
                        jobs.append((damage_filter(
                            dataset.denormalize_target(prediction),
                            dataset.sampling_frequency, self.apply_lowpass,
                            self.lowpass_hz, self.lowpass_order), gauge))
                        if self._val_true is None and \
                                not self._metric_is_released():
                            dataset.section = gauge
                            true = dataset.denormalize_target(
                                dataset[index]["target"][0].numpy())
                            jobs.append(
                                (damage_filter(true, dataset.sampling_frequency,
                                               self.apply_lowpass,
                                               self.lowpass_hz,
                                               self.lowpass_order), gauge))
        finally:
            dataset.section = None
            _set_rng_state(rng)
            self.model.train()
        with multiprocessing.Pool(min(8, os.cpu_count() or 1)) as pool:
            damages = np.array(
                pool.starmap(_damage_job, [(m, g, *sn_args) for m, g in jobs],
                             chunksize=64))
        if self._val_true is None:
            if self._metric_is_released():
                table = dataset.release.damage()
                self._val_true = table.loc[dataset.sim_ids,
                                           tower.section_ids].to_numpy(float)
            else:
                pairs = np.reshape(damages, (len(dataset), len(gauges), 2))
                damages, self._val_true = pairs[..., 0], pairs[..., 1]
        rec = np.reshape(damages, (len(dataset), len(gauges)))
        r2 = [
            summarize_damage(self._val_true[:, g], rec[:, g])["r2_log_damage"]
            for g in gauges
        ]
        return {
            "r2_gauges": r2,
            "r2_mean": float(np.mean(r2)),
            "r2_top": r2[-1],
            "r2_base": r2[0],
        }

    def _validation_loss(self, loader: DataLoader) -> float:
        """Mean squared error on normalized crops of the validation sims.

        The crops (and the height, for the height task) are drawn with a
        fixed NumPy state so that the number is comparable across epochs;
        the training RNG streams are left untouched. The variance head, if
        any, is not sampled: the loss is on the mean prediction.
        """
        state = np.random.get_state()
        np.random.seed(12345)
        self.model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for batch in loader:
                inputs = batch["inputs"].to(self.device)
                condition = batch["condition"].to(self.device)
                target = batch["target"].to(self.device)
                if getattr(self.model, "needs_physics_gain", False):
                    prediction = self.model(
                        inputs, condition,
                        batch["physics_gain"].to(self.device))
                else:
                    prediction = self.model(inputs, condition)
                prediction = prediction[:, :1, :]
                total += float(torch.sum((prediction - target)**2))
                count += prediction.numel()
        np.random.set_state(state)
        self.model.train()
        return total / max(count, 1)

    def checkpoint_path(self, epoch: Optional[int] = None) -> str:
        """Returns the checkpoint path for this model/direction (with
        `epoch`, the weights kept at that epoch by `save_epochs`)."""
        suffix = f"_epoch{epoch}" if epoch else ""
        return os.path.join(self.output_dir,
                            f"{self.model_name}_{self.direction}{suffix}.pt")

    def save_checkpoint(self, path: Optional[str] = None) -> None:
        """Saves model weights, normalization stats and settings.

        The architecture knobs are stored only when set, so a checkpoint of
        the published recipe holds the bytes of the published trainer (a
        checkpoint without them loads with the published architecture).
        """
        checkpoint = {
            "state_dict": self.model.state_dict(),
            "norm_stats": self.norm_stats,
            "model_name": self.model_name,
            "direction": self.direction,
            "eval_length": self._eval_length,
            "crop_length": self.crop_length,
            "loss_name": self.loss_name,
            "condition_bound": self.condition_bound,
            "model_kwargs": self.model_kwargs,
            "seed": self.seed,
            "setup": self._setup(self._make_dataset([], None)),
        }
        if not self.model_kwargs:
            del checkpoint["model_kwargs"]
        _atomic_save(checkpoint, path or self.checkpoint_path())

    def _setup(self, probe: SequenceDataset) -> Dict:
        """Channel setup a checkpoint was trained with."""
        return {
            "direction": self.direction,
            "input_channels": list(probe.input_channels),
            "condition_channels": list(probe.condition_channels),
            "target_channel": probe.moment_channel,
            "height_targets": bool(self.height_targets)
        }

    def _check_setup(self, checkpoint: Dict, probe: SequenceDataset) -> None:
        """Refuses a checkpoint trained with other channels (checkpoints
        written before the setup was stored are not checked)."""
        saved = checkpoint.get("setup")
        if saved is None:
            return
        current = self._setup(probe)
        if current["height_targets"]:
            # The 11-height task samples the target channel per item.
            saved = {k: v for k, v in saved.items() if k != "target_channel"}
            current.pop("target_channel")
        if saved != current:
            raise ValueError(f"Checkpoint trained with {saved}, "
                             f"run configured with {current}.")

    def load_checkpoint(self, path: Optional[str] = None) -> None:
        """Loads a previously trained checkpoint (by default the final one
        of this model/direction); the architecture knobs it was trained with
        are rebuilt."""
        checkpoint = torch.load(path or self.checkpoint_path(),
                                map_location=self.device,
                                weights_only=False)
        self.norm_stats = checkpoint["norm_stats"]
        self._eval_length = checkpoint["eval_length"]
        probe = self._make_dataset([], None)
        self._check_setup(checkpoint, probe)
        num_samples = checkpoint["crop_length"] or self._eval_length
        self.model_kwargs = checkpoint.get("model_kwargs", {})
        self.model = build_model(self.model_name,
                                 num_samples=num_samples,
                                 input_channels=len(probe.input_channels),
                                 condition_dim=len(probe.condition_channels) +
                                 int(self.height_targets),
                                 condition_bound=checkpoint.get(
                                     "condition_bound", 0.5),
                                 **self.model_kwargs).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"])

    @staticmethod
    def _sample(output: torch.Tensor) -> np.ndarray:
        """One draw of a (mean, log-variance) prediction."""
        sigma = torch.exp(0.5 * output[1])
        return (output[0] + sigma * torch.randn_like(sigma)).cpu().numpy()

    def _predict(self,
                 item: Dict[str, torch.Tensor],
                 sample: bool = True) -> np.ndarray:
        """Normalized prediction of one item.

        Args:
            item (dict): Dataset item.
            sample (bool): Draw from a variance head; False returns its
              (mean, log-variance) channels instead.

        Returns:
            np.ndarray: (length,) prediction, or (2, length) when a variance
              head is not sampled.
        """
        inputs = item["inputs"][None].to(self.device)
        condition = item["condition"][None].to(self.device)
        if getattr(self.model, "needs_physics_gain", False):
            return self.model(inputs, condition, item["physics_gain"][None].to(
                self.device))[0, 0].cpu().numpy()
        if getattr(self.model, "predicts_variance", False):
            output = self.model(inputs, condition)[0]
            return self._sample(output) if sample else output.cpu().numpy()
        return self.model(inputs, condition)[0, 0].cpu().numpy()

    def evaluate(self,
                 eval_ids: List[int],
                 release: Optional[ReleasedTower] = None,
                 tag: str = "") -> Dict[str, float]:
        """Evaluates damage metrics on held-out simulations.

        Args:
            eval_ids (List[int]): Simulation IDs to evaluate.
            release (ReleasedTower, optional): Another tower, for zero-shot
              cross-tower evaluation (the model keeps the normalization
              statistics of its training tower); its sections give the
              damage metric.
            tag (str): Suffix of the output files (e.g. the target tower).

        Returns:
            dict: Summary metrics (R^2 of log damage, median damage ratio,
              fraction within factor 2).
        """
        release = release or self.release
        tower = release.geometry
        # Fixed noise for the variance head: the same checkpoint scores the
        # same, whatever ran before.
        torch.manual_seed(self.seed)
        dataset = self._make_dataset(eval_ids, None, release)
        sections = (list(range(len(tower.channels)))
                    if self.height_targets else [None])
        sn_args = (tower, self.sn_intercepts_log10, self.sn_slopes)
        rows, jobs = [], []
        self.model.eval()
        with torch.no_grad():
            for index in tqdm(range(len(dataset)),
                              desc=f"Evaluating {self.model_name}"):
                row = {"sim_id": dataset.sim_ids[index]}
                for section in sections:
                    dataset.section = section
                    item = dataset[index]
                    moment_rec = damage_filter(
                        dataset.denormalize_target(
                            self._predict_window(dataset, index, item)),
                        dataset.sampling_frequency, self.apply_lowpass,
                        self.lowpass_hz, self.lowpass_order)
                    moment_true = damage_filter(
                        dataset.denormalize_target(item["target"][0].numpy()),
                        dataset.sampling_frequency, self.apply_lowpass,
                        self.lowpass_hz, self.lowpass_order)
                    if section is None:
                        # Single height: columns named by the gauge.
                        stem = dataset.moment_channel.rsplit("_m", 1)[0]
                        damage_gauge = self.damage_gauge
                    else:
                        stem, damage_gauge = tower.channels[section], section
                    row[f"var_ratio_{stem}"] = float(
                        np.var(moment_rec) / np.var(moment_true))
                    # Rainflow is the bottleneck: it runs in a process pool.
                    jobs.append((index, f"damage_true_{stem}", moment_true,
                                 damage_gauge))
                    jobs.append(
                        (index, f"damage_rec_{stem}", moment_rec, damage_gauge))
                rows.append(row)
        with multiprocessing.Pool(min(8, os.cpu_count() or 1)) as pool:
            damages = pool.starmap(
                _damage_job, [(m, sec, *sn_args) for _, _, m, sec in jobs],
                chunksize=64)
        for (index, key, _, _), value in zip(jobs, damages):
            rows[index][key] = value

        os.makedirs(self.output_dir, exist_ok=True)
        df = pd.DataFrame(rows)
        stem = f"{self.model_name}_{self.direction}" + (f"_{tag}"
                                                        if tag else "")
        df.to_csv(os.path.join(self.output_dir,
                               f"damage_comparison_{stem}.csv"),
                  index=False)

        key = (tower.channels[0] if self.height_targets else
               dataset.moment_channel.rsplit("_m", 1)[0])
        scores = summarize_damage(df[f"damage_true_{key}"].values,
                                  df[f"damage_rec_{key}"].values)
        summary = {
            "model": self.model_name,
            "direction": self.direction,
            "seed": self.seed,
            "tag": tag,
            **scores,
        }
        summary_path = os.path.join(self.output_dir, f"summary_{stem}.json")
        with open(summary_path, "w", encoding="utf-8") as file:
            json.dump(summary, file, indent=4)
        return summary
