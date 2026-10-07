# pylint: disable=wrong-import-position,protected-access
# pylint: disable=use-dict-literal
"""Tests of the training options of the validation-tuned track.

Run from the repository root with `python -m unittest discover tests`.
CPU only, on a tiny synthetic tower (tests/synthetic.py): the published
defaults are bit-identical to the published trainer (fingerprints of its
runs in tests/data/trainer_fingerprints.json), AdamW without weight decay
is Adam, the schedule, the damage validation (equal to `evaluate`), the
resume (equal to an uninterrupted run), the kept epochs, the parameter cap
and the NaN guard with their exit codes.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tests import synthetic
from floatsense import constants as C
from floatsense import compute_norm_stats
from floatsense import load_tower
from floatsense.metrics import summarize_damage
from floatsense.trainer import DivergedError
from floatsense.trainer import ModelTooLargeError
from floatsense.trainer import STOP_REQUESTED
from floatsense.trainer import StoppedError
from floatsense.trainer import SequenceModelTrainer
from floatsense.trainer import lr_factor

FINGERPRINTS = os.path.join(os.path.dirname(__file__), "data",
                            "trainer_fingerprints.json")
REPO = os.path.join(os.path.dirname(__file__), "..")
TINY = {"hidden_channels": 8, "num_levels": 3, "dropout": 0.1}
TMP = None
TOWER = None


def setUpModule():  # pylint: disable=invalid-name
    """Writes the synthetic tower once."""
    global TMP, TOWER  # pylint: disable=global-statement
    TMP = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
    synthetic.write_tower(TMP.name)
    TOWER = load_tower(TMP.name, synthetic.TOWER)


def tearDownModule():  # pylint: disable=invalid-name
    """Removes the synthetic tower."""
    TMP.cleanup()


def ids():
    """(train, validation) simulation IDs of the synthetic tower."""
    return TOWER.split_ids("train"), TOWER.split_ids("test")


def make(output_dir: str, **kwargs) -> SequenceModelTrainer:
    """A CPU trainer on the synthetic tower (small batches, short run)."""
    options = dict(release=TOWER,
                   output_dir=output_dir,
                   model_name="tcn",
                   crop_length=4096,
                   batch_size=4,
                   num_epochs=4,
                   num_workers=0,
                   seed=3,
                   device="cpu")
    options.update(kwargs)
    return SequenceModelTrainer(**options)


def tuned(output_dir: str, **kwargs) -> SequenceModelTrainer:
    """A trainer with every option of the tuned track switched on."""
    options = dict(height_targets=True,
                   model_kwargs=TINY,
                   weight_decay=1e-3,
                   schedule="cosine",
                   warmup_epochs=1,
                   grad_clip=1.0,
                   val_score="damage",
                   val_every=2,
                   num_workers=2,
                   resume=True,
                   save_epochs=[2])
    options.update(kwargs)
    return make(output_dir, **options)


class Interrupt(Exception):
    """Stands for a killed job."""


class InterruptedTrainer(SequenceModelTrainer):
    """Stops right after its first resume save."""

    def _save_resume(self, *args, **kwargs):
        super()._save_resume(*args, **kwargs)
        raise Interrupt


def state(trainer: SequenceModelTrainer):
    """Copy of the weights of a trainer."""
    return {k: v.clone() for k, v in trainer.model.state_dict().items()}


class DefaultsTest(unittest.TestCase):
    """The published recipe is unchanged."""

    def test_bit_identical_to_published(self):
        """Same losses and weights as the published trainer, run for run
        (Adam, no clipping, constant rate, validation by loss)."""
        with open(FINGERPRINTS, encoding="utf-8") as file:
            reference = json.load(file)
        train_ids, val_ids = ids()
        for model, expected in reference.items():
            with self.subTest(model=model), \
                    tempfile.TemporaryDirectory() as out:
                trainer = make(out, model_name=model, num_epochs=2, val_every=1)
                history = trainer.train(train_ids, val_ids)
                self.assertEqual(history["train_loss"],
                                 expected["history"]["train_loss"])
                self.assertEqual([list(v) for v in history["val_loss"]],
                                 expected["history"]["val_loss"])
                weights = [
                    v.double()
                    for v in trainer.model.state_dict().values()
                    if v.is_floating_point()
                ]
                self.assertAlmostEqual(float(sum(v.sum() for v in weights)),
                                       expected["param_sum"],
                                       places=9)
                self.assertFalse(os.path.exists(trainer.resume_path()))

    def test_adamw_without_decay_is_adam(self):
        """AdamW(weight_decay=0) takes exactly the steps of Adam."""
        models = []
        for decay in (None, 0.0):
            trainer = make("", weight_decay=decay)
            torch.manual_seed(0)
            trainer.model = torch.nn.Sequential(torch.nn.Linear(6, 16),
                                                torch.nn.GELU(),
                                                torch.nn.Linear(16, 1))
            optimizer = trainer._optimizer()
            self.assertIsInstance(
                optimizer,
                torch.optim.Adam if decay is None else torch.optim.AdamW)
            generator = torch.Generator().manual_seed(1)
            for _ in range(20):
                inputs = torch.randn(8, 6, generator=generator)
                loss = trainer.model(inputs).pow(2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            models.append(trainer.model)
        for first, second in zip(models[0].parameters(),
                                 models[1].parameters()):
            self.assertTrue(torch.equal(first, second))


class ScheduleTest(unittest.TestCase):
    """Learning-rate schedule."""

    def test_factor(self):
        """Linear warm-up, then constant or cosine to zero."""
        self.assertAlmostEqual(lr_factor(0, 100, 10, "cosine"), 0.1)
        self.assertAlmostEqual(lr_factor(9, 100, 10, "cosine"), 1.0)
        self.assertAlmostEqual(lr_factor(10, 100, 10, "cosine"), 1.0)
        self.assertAlmostEqual(lr_factor(55, 100, 10, "cosine"), 0.5)
        self.assertAlmostEqual(lr_factor(100, 100, 10, "cosine"), 0.0)
        self.assertAlmostEqual(lr_factor(4, 100, 10, "constant"), 0.5)
        self.assertAlmostEqual(lr_factor(50, 100, 10, "constant"), 1.0)
        self.assertAlmostEqual(lr_factor(0, 100, 0, "cosine"), 1.0)

    def test_scheduler(self):
        """Per-step rates over a run; none for the published recipe."""
        trainer = make("", num_epochs=4, learning_rate=1e-3)
        trainer.model = torch.nn.Linear(2, 1)
        self.assertIsNone(trainer._scheduler(trainer._optimizer(), 3))
        trainer = make("",
                       num_epochs=4,
                       learning_rate=1e-3,
                       schedule="cosine",
                       warmup_epochs=1)
        trainer.model = torch.nn.Linear(2, 1)
        optimizer = trainer._optimizer()
        scheduler = trainer._scheduler(optimizer, 3)
        rates = []
        for _ in range(12):
            rates.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
        expected = [1e-3 * lr_factor(s, 12, 3, "cosine") for s in range(12)]
        np.testing.assert_allclose(rates, expected, rtol=1e-12)
        self.assertAlmostEqual(rates[2], 1e-3)


class TunedRunTest(unittest.TestCase):
    """A run with every tuned-track option."""

    @classmethod
    def setUpClass(cls):
        cls.out = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        cls.trainer = tuned(cls.out.name)
        cls.history = cls.trainer.train(*ids())
        cls.weights = state(cls.trainer)

    @classmethod
    def tearDownClass(cls):
        cls.out.cleanup()

    def resumed(self, checkpoint_seconds: float) -> SequenceModelTrainer:
        """The same run, killed after its first resume save, resumed."""
        out = tempfile.mkdtemp(dir=self.out.name)
        with mock.patch("floatsense.trainer.CHECKPOINT_SECONDS",
                        checkpoint_seconds):
            first = tuned(out)
            first.__class__ = InterruptedTrainer
            with self.assertRaises(Interrupt):
                first.train(*ids())
        saved = torch.load(first.resume_path(), weights_only=False)
        trainer = tuned(out)
        history = trainer.train(*ids())
        return trainer, history, saved["epoch"]

    def assert_same_run(self, trainer, history):
        """Same weights and history as the uninterrupted run."""
        self.assertEqual(history, self.history)
        for key, value in state(trainer).items():
            self.assertTrue(torch.equal(value, self.weights[key]), key)

    def test_resume_at_validation(self):
        """Killed after the validation of epoch 2, resumed: identical."""
        trainer, history, epoch = self.resumed(1e9)
        self.assertEqual(epoch, 2)
        self.assert_same_run(trainer, history)

    def test_resume_on_wall_time(self):
        """Killed after a time-based save (epoch 1), resumed: identical."""
        trainer, history, epoch = self.resumed(0.0)
        self.assertEqual(epoch, 1)
        self.assert_same_run(trainer, history)

    def test_stop_request_saves_and_resumes(self):
        """SIGUSR1: the epoch ends, the state is saved, the run stops; the
        resumed run is identical."""
        out = tempfile.mkdtemp(dir=self.out.name)
        STOP_REQUESTED.set()
        try:
            with mock.patch("floatsense.trainer.CHECKPOINT_SECONDS", 1e9):
                with self.assertRaises(StoppedError):
                    tuned(out).train(*ids())
        finally:
            STOP_REQUESTED.clear()
        saved = torch.load(tuned(out).resume_path(), weights_only=False)
        self.assertEqual(saved["epoch"], 1)
        trainer = tuned(out)
        self.assert_same_run(trainer, trainer.train(*ids()))

    def test_completed_run_is_not_retrained(self):
        """A finished run, launched again, returns its history."""
        trainer = tuned(self.out.name)
        self.assertEqual(trainer.train(*ids()), self.history)
        for key, value in state(trainer).items():
            self.assertTrue(torch.equal(value, self.weights[key]), key)

    def test_validation_record(self):
        """Damage validation at epochs 2 and 4, with the 11 gauges."""
        self.assertEqual([e["epoch"] for e in self.history["val_r2"]], [2, 4])
        last = self.history["val_r2"][-1]
        self.assertEqual(len(last["r2_gauges"]), 11)
        self.assertAlmostEqual(last["r2_mean"], np.mean(last["r2_gauges"]))
        self.assertEqual(last["r2_top"], last["r2_gauges"][-1])
        self.assertIn(self.history["best_epoch"], (2, 4))

    def test_kept_epoch_and_kwargs(self):
        """The epoch-2 weights are kept; a plain trainer rebuilds the
        searched architecture from the checkpoint."""
        trainer = make(self.out.name, height_targets=True)
        trainer.load_checkpoint(trainer.checkpoint_path(2))
        self.assertEqual(len(trainer.model.blocks), TINY["num_levels"])
        self.assertEqual(trainer.model_kwargs, TINY)
        self.assertFalse(
            all(
                torch.equal(v, self.weights[k])
                for k, v in trainer.model.state_dict().items()))

    def damage_matches_evaluate(self, **kwargs):
        """_damage_scores equals the per-gauge R^2 of `evaluate`."""
        trainer = make(self.out.name, height_targets=True, **kwargs)
        trainer.load_checkpoint()
        rng = (torch.get_rng_state(), np.random.get_state()[1].copy())
        scores = trainer._damage_scores(trainer._damage_dataset(ids()[1]))
        self.assertTrue(torch.equal(rng[0], torch.get_rng_state()))
        self.assertTrue(np.array_equal(rng[1], np.random.get_state()[1]))
        self.assertTrue(trainer.model.training)
        trainer.evaluate(ids()[1], tag="check")
        frame = pd.read_csv(
            os.path.join(self.out.name, "damage_comparison_tcn_fa_check.csv"))
        expected = [
            summarize_damage(frame[f"damage_true_{s}"],
                             frame[f"damage_rec_{s}"])["r2_log_damage"]
            for s in synthetic.STEMS
        ]
        np.testing.assert_allclose(scores["r2_gauges"], expected, atol=1e-5)
        return scores

    def test_damage_score_released_metric(self):
        """Released metric: true damage from damage.parquet."""
        scores = self.damage_matches_evaluate()
        np.testing.assert_allclose(scores["r2_gauges"],
                                   self.history["val_r2"][-1]["r2_gauges"],
                                   atol=1e-12)

    def test_damage_score_other_metric(self):
        """Another low-pass: true damage recomputed from the series."""
        self.damage_matches_evaluate(lowpass_hz=2.5)


class GuardsTest(unittest.TestCase):
    """Parameter cap and NaN guard, in the trainer and as exit codes."""

    def test_parameter_cap(self):
        """A model above max_params_m is refused before training."""
        with tempfile.TemporaryDirectory() as out:
            with self.assertRaises(ModelTooLargeError):
                make(out, max_params_m=1e-4).train(*ids())

    def test_nan_guard(self):
        """A non-finite loss stops the run."""
        with tempfile.TemporaryDirectory() as out:
            trainer = make(out, model_kwargs=TINY)
            build = trainer._build_model

            def poisoned(*args):
                model = build(*args)
                torch.nn.init.constant_(model.output_proj.bias, float("nan"))
                return model

            trainer._build_model = poisoned
            with self.assertRaises(DivergedError):
                trainer.train(*ids())

    def run_script(self, *flags):
        """Exit code of scripts/train/run.py on the synthetic tower."""
        with tempfile.TemporaryDirectory() as out:
            return subprocess.run([
                sys.executable,
                os.path.join(REPO, "scripts", "train",
                             "run.py"), f"--dataset_dir={TMP.name}",
                f"--tower={synthetic.TOWER}", f"--output_dir={out}",
                "--models=tcn", "--num_epochs=1", "--batch_size=4",
                "--num_workers=0", "--run_evaluation=False", *flags
            ],
                                  capture_output=True,
                                  text=True,
                                  env={
                                      **os.environ, "CUDA_VISIBLE_DEVICES": ""
                                  },
                                  check=False).returncode

    def test_exit_codes(self):
        """Distinct exit codes for a too large model and a divergence."""
        self.assertEqual(self.run_script("--max_params_m=0.0001"),
                         C.EXIT_TOO_LARGE)
        self.assertEqual(self.run_script("--learning_rate=1e30"),
                         C.EXIT_DIVERGED)
        self.assertEqual(
            self.run_script("--model_kwargs=hidden_channels=8,num_levels=2"), 0)


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(os.environ.get("FLOATSENSE_DATA"),
                     "set FLOATSENSE_DATA to a released dataset")
class ReleasedDataTest(unittest.TestCase):
    """The damage validation on the released dataset."""

    def test_true_damage_from_release(self):
        """damage.parquet gives the same validation R^2 as the true damage
        recomputed from the series by the path of `evaluate`."""
        tower = load_tower(os.environ["FLOATSENSE_DATA"], "opt2")
        trainer = SequenceModelTrainer(release=tower,
                                       output_dir="",
                                       model_name="tcn",
                                       height_targets=True,
                                       model_kwargs=TINY,
                                       seed=0,
                                       device="cpu")
        train_ids = tower.split_ids("val/train")[:8]
        val_ids = tower.split_ids("val/val")[:4]
        dataset = trainer._make_dataset(train_ids, None)
        trainer.norm_stats = compute_norm_stats(
            tower, train_ids, [dataset.accel_channel] +
            dataset.condition_channels + dataset.height_channels)
        trainer._eval_length = C.INPUT_LENGTH + 1
        torch.manual_seed(0)
        trainer.model = trainer._build_model(4096, len(dataset.input_channels),
                                             len(dataset.condition_channels))
        released = trainer._damage_scores(trainer._damage_dataset(val_ids))
        trainer._val_true = None
        with mock.patch.object(SequenceModelTrainer, "_metric_is_released",
                               lambda self: False):
            recomputed = trainer._damage_scores(
                trainer._damage_dataset(val_ids))
        np.testing.assert_allclose(released["r2_gauges"],
                                   recomputed["r2_gauges"],
                                   rtol=1e-5)
