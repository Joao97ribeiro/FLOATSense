# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-instance-attributes
# pylint: disable=not-callable
"""Training and damage-based evaluation of moment reconstruction models.

Trains a sequence model on the released time series of a train split and
evaluates it on held-out simulations with the same fatigue-damage metrics
as the physics baseline (R^2 of log10 damage, median damage ratio, fraction
within a factor of 2), writing a per-simulation damage CSV with the same
columns as the physics baseline.
"""

import json
import multiprocessing
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import HEIGHT_TARGETS
from .data import SequenceDataset
from .data import compute_norm_stats
from .fatigue import compute_base_damage
from .metrics import summarize_damage
from .models import build_model
from .physics import lowpass
from .release import ReleasedTower
from .release import TowerSections


def _damage_job(moment, section, tower: TowerSections, intercepts,
                slopes) -> float:
    """Damage of one series at one section (picklable for the pool)."""
    return compute_base_damage(moment,
                               tower,
                               intercepts,
                               slopes,
                               section=section)


class SequenceModelTrainer:
    """Trains and evaluates a sequence model for one direction."""

    def __init__(self,
                 release: ReleasedTower,
                 output_dir: str,
                 direction: str = "fa",
                 model_name: str = "spectral",
                 condition_channels: Optional[List[str]] = None,
                 min_time: float = 400.0,
                 max_time: float = 1000.0,
                 lowpass_hz: float = 3.0,
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
                 damage_section: int = 0,
                 input_channels: Optional[List[str]] = None,
                 height_targets: bool = False,
                 height_factors: Optional[List[float]] = None,
                 seed: int = 0,
                 device: Optional[str] = None):
        """Initializes the trainer.

        Args:
            release (ReleasedTower): Training tower of the released dataset.
            output_dir (str): Directory for checkpoints, CSVs and summary.
            direction (str): Reconstruction direction ('fa' or 'ss').
            model_name (str): Model name understood by `build_model`.
            condition_channels (List[str], optional): Operating-state
              channels.
            min_time (float): Start time of the usable window [s].
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
            calibration_path (str, optional): Physics calibration JSON for
              models that consume the per-simulation physics gain (hybrid).
            condition_bound (float): Tanh bound of the hybrid model's
              condition-dependent correction (0 disables the bound).
            target_channel (str, optional): Overrides the target moment
              channel (e.g. an intermediate section gage 'tower_5_mfa').
            damage_section (int): Tower section index used for the damage
              evaluation of the target channel (0 = base).
            input_channels (List[str], optional): Replaces the default
              [acceleration + condition channels] input stack (sensor
              ablations).
            seed (int): Seed of the weight initialization, the random crops
              and the batch order.
            device (str, optional): Torch device (default: cuda if available).
        """
        self.release = release
        self.output_dir = output_dir
        self.direction = direction
        self.model_name = model_name
        self.condition_channels = condition_channels
        self.min_time = min_time
        self.max_time = max_time
        self.lowpass_hz = lowpass_hz
        self.crop_length = (crop_length
                            if model_name not in ("spectral", "hybrid",
                                                  "hybrid_tcn") else None)
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
        self.condition_bound = condition_bound
        self.target_channel = target_channel
        self.damage_section = damage_section
        self.input_channels = input_channels
        self.height_targets = height_targets
        self.height_factors = height_factors
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.norm_stats = None
        self._eval_length = None

    def _make_dataset(
            self,
            sim_ids: List[int],
            crop_length: Optional[int],
            release: Optional[ReleasedTower] = None) -> SequenceDataset:
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
                                     height_factors=self.height_factors)

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

    def train(self, train_ids: List[int],
              val_ids: Optional[List[int]] = None) -> Dict[str, List[float]]:
        """Trains the model on the given simulations.

        Args:
            train_ids (List[int]): Training simulation IDs.
            val_ids (List[int], optional): Held-out simulations on which the
              training loss (squared error on normalized crops) is also
              reported every few epochs; it never influences training.

        Returns:
            dict: Per-epoch mean training loss under 'train_loss' and, with
            `val_ids`, the validation loss under 'val_loss' (epoch, value).
        """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        probe = SequenceDataset(
            release=self.release,
            sim_ids=train_ids[:1],
            direction=self.direction,
            condition_channels=self.condition_channels,
            min_time=self.min_time,
            max_time=self.max_time,
            target_channel=self.target_channel,
            input_channels=self.input_channels,
            height_targets=self.height_targets)
        stat_channels = list(
            dict.fromkeys(
                [c.split(":")[1] if c.startswith("stat:") else c
                 for c in probe.input_channels if c != "height"] +
                probe.condition_channels + [probe.moment_channel] +
                (probe.height_channels if self.height_targets else [])))
        init_state = None
        if self.init_checkpoint:
            init_state = torch.load(self.init_checkpoint,
                                    map_location=self.device,
                                    weights_only=False)
            self.norm_stats = init_state["norm_stats"]
        else:
            self.norm_stats = compute_norm_stats(self.release, train_ids,
                                                 stat_channels, self.min_time,
                                                 self.max_time)
        self._eval_length = probe[0]["inputs"].shape[-1]

        dataset = self._make_dataset(train_ids, self.crop_length)
        loader = DataLoader(dataset,
                            batch_size=self.batch_size,
                            shuffle=True,
                            num_workers=self.num_workers,
                            drop_last=True)

        num_samples = self.crop_length or self._eval_length
        self.model = build_model(self.model_name,
                                 num_samples=num_samples,
                                 input_channels=len(dataset.input_channels),
                                 condition_dim=len(dataset.condition_channels) +
                                 int(self.height_targets),
                                 condition_bound=self.condition_bound).to(
                                     self.device)
        if init_state is not None:
            self.model.load_state_dict(init_state["state_dict"])
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.learning_rate)

        freqs = torch.fft.rfftfreq(num_samples,
                                   d=1.0 / dataset.sampling_frequency).to(
                                       self.device)
        freq_weights = freqs**self.damage_freq_exponent

        history = {"train_loss": []}
        val_loader = None
        if val_ids:
            history["val_loss"] = []
            val_loader = DataLoader(self._make_dataset(val_ids,
                                                       self.crop_length),
                                    batch_size=self.batch_size,
                                    shuffle=False, num_workers=0)
        val_interval = self.val_every or max(1, self.num_epochs // 10)
        best_val, best_epoch, best_state = float("inf"), 0, None
        for epoch in range(self.num_epochs):
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
                optimizer.step()
                losses.append(float(loss.detach()))
            history["train_loss"].append(float(np.mean(losses)))
            log_now = (epoch + 1) % max(1, self.num_epochs // 10) == 0
            val_now = (val_loader is not None and
                       (epoch + 1) % val_interval == 0)
            if val_now:
                val_loss = self._validation_loss(val_loader)
                history["val_loss"].append((epoch + 1, val_loss))
                if val_loss < best_val:
                    best_val, best_epoch = val_loss, epoch + 1
                    if self.early_stopping_patience:
                        best_state = {k: v.detach().clone().cpu() for k, v
                                      in self.model.state_dict().items()}
            if log_now or val_now:
                line = (f"[{self.model_name}/{self.direction}] "
                        f"epoch {epoch + 1}/{self.num_epochs} "
                        f"loss {history['train_loss'][-1]:.5f}")
                if val_now:
                    line += f" val_loss {val_loss:.5f}"
                print(line)
            if (self.early_stopping_patience and best_epoch and
                    epoch + 1 - best_epoch >=
                    self.early_stopping_patience * val_interval):
                print(f"[{self.model_name}/{self.direction}] early stop at "
                      f"epoch {epoch + 1}, best {best_epoch} "
                      f"(val_loss {best_val:.5f})")
                break

        if self.early_stopping_patience and best_state is not None:
            self.model.load_state_dict(best_state)
            history["best_epoch"] = best_epoch
            history["best_val_loss"] = best_val
        self.save_checkpoint()
        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir,
                               f"history_{self.model_name}_{self.direction}.json"),
                  "w", encoding="utf-8") as file:
            json.dump(history, file)
        return history

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

    def checkpoint_path(self) -> str:
        """Returns the checkpoint path for this model/direction."""
        return os.path.join(self.output_dir,
                            f"{self.model_name}_{self.direction}.pt")

    def save_checkpoint(self) -> None:
        """Saves model weights, normalization stats and settings."""
        os.makedirs(self.output_dir, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "norm_stats": self.norm_stats,
                "model_name": self.model_name,
                "direction": self.direction,
                "eval_length": self._eval_length,
                "crop_length": self.crop_length,
                "loss_name": self.loss_name,
                "condition_bound": self.condition_bound,
                "seed": self.seed,
            }, self.checkpoint_path())

    def load_checkpoint(self) -> None:
        """Loads a previously trained checkpoint."""
        checkpoint = torch.load(self.checkpoint_path(),
                                map_location=self.device,
                                weights_only=False)
        self.norm_stats = checkpoint["norm_stats"]
        self._eval_length = checkpoint["eval_length"]
        probe = self._make_dataset([], None)
        num_samples = checkpoint["crop_length"] or self._eval_length
        self.model = build_model(self.model_name,
                                 num_samples=num_samples,
                                 input_channels=len(probe.input_channels),
                                 condition_dim=len(probe.condition_channels) +
                                 int(self.height_targets),
                                 condition_bound=checkpoint.get(
                                     "condition_bound", 0.5)).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"])

    def _predict(self, item: Dict[str, torch.Tensor]) -> np.ndarray:
        """Normalized prediction of one item (sampling the variance head)."""
        inputs = item["inputs"][None].to(self.device)
        condition = item["condition"][None].to(self.device)
        if getattr(self.model, "needs_physics_gain", False):
            return self.model(inputs, condition, item["physics_gain"][None].to(
                self.device))[0, 0].cpu().numpy()
        if getattr(self.model, "predicts_variance", False):
            output = self.model(inputs, condition)[0]
            sigma = torch.exp(0.5 * output[1])
            return (output[0] + sigma * torch.randn_like(sigma)).cpu().numpy()
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
        dataset = self._make_dataset(eval_ids, None, release)
        sections = (list(range(len(HEIGHT_TARGETS)))
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
                    moment_rec = lowpass(
                        dataset.denormalize_target(self._predict(item)),
                        dataset.sampling_frequency, self.lowpass_hz)
                    moment_true = lowpass(
                        dataset.denormalize_target(item["target"][0].numpy()),
                        dataset.sampling_frequency, self.lowpass_hz)
                    if section is None:
                        stem, damage_section = self.direction, self.damage_section
                    else:
                        stem, damage_section = HEIGHT_TARGETS[section][:2]
                    row[f"var_ratio_{stem}"] = float(
                        np.var(moment_rec) / np.var(moment_true))
                    # Rainflow is the bottleneck: it runs in a process pool.
                    jobs.append((index, f"damage_true_{stem}", moment_true,
                                 damage_section))
                    jobs.append((index, f"damage_rec_{stem}", moment_rec,
                                 damage_section))
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

        key = "tower_bottom" if self.height_targets else self.direction
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
