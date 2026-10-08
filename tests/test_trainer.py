# pylint: disable=wrong-import-position,protected-access
# pylint: disable=use-dict-literal,too-many-public-methods
"""Tests of the training options of the validation-tuned track.

Run from the repository root with `python -m unittest discover tests`.
CPU only, on a tiny synthetic tower (tests/synthetic.py): the published
defaults are bit-identical to the published trainer (fingerprints of its
runs in tests/data/trainer_fingerprints.json), AdamW without weight decay
is Adam, the schedule, the damage validation (equal to `evaluate`), the
resume (equal to an uninterrupted run, refused for another run
configuration), the kept epochs, the parameter cap and the NaN guard with
their exit codes, the SIGUSR1 handler of scripts/train/run.py, and the
loading of a checkpoint written by the published trainer
(tests/data/main_dlinear_fa.pt, with the predictions, damage CSV and
summary the published code computed from it).
"""

import hashlib

import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
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
from floatsense.trainer import ConfigMismatchError
from floatsense.trainer import DivergedError
from floatsense.trainer import ModelTooLargeError
from floatsense.trainer import STOP_REQUESTED
from floatsense.trainer import StoppedError
from floatsense.trainer import SequenceModelTrainer
from floatsense.trainer import _atomic_save
from floatsense.trainer import _rng_state
from floatsense.trainer import _set_rng_state
from floatsense.trainer import lr_factor

FINGERPRINTS = os.path.join(os.path.dirname(__file__), "data",
                            "trainer_fingerprints.json")
REPO = os.path.join(os.path.dirname(__file__), "..")
MAIN_CHECKPOINT = os.path.join(os.path.dirname(__file__), "data",
                               "main_dlinear_fa")
OTHER_TOWER = synthetic.TOWER + "2"  # a copy of the synthetic tower
TINY = {"hidden_channels": 8, "num_levels": 3, "dropout": 0.1}
TMP = None
TOWER = None


def setUpModule():  # pylint: disable=invalid-name
    """Writes the synthetic tower once."""
    global TMP, TOWER  # pylint: disable=global-statement
    TMP = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
    synthetic.write_tower(TMP.name)
    shutil.copytree(os.path.join(TMP.name, synthetic.TOWER),
                    os.path.join(TMP.name, OTHER_TOWER))
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


def state_sha256(state_dict) -> str:
    """Digest of the names, dtypes, shapes and bytes of a state_dict."""
    digest = hashlib.sha256()
    for key, value in state_dict.items():
        value = value.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def dead_pid() -> int:
    """The pid of a process that has exited."""
    process = subprocess.run(
        [sys.executable, "-c", "import os; print(os.getpid())"],
        capture_output=True,
        text=True,
        check=True)
    return int(process.stdout)


class DefaultsTest(unittest.TestCase):
    """The published recipe is unchanged."""

    def test_bit_identical_to_published(self):
        """Same losses, weights and torch random state after training as
        the published trainer, run for run (Adam, no clipping, constant
        rate, validation by loss)."""
        # The sha256 values (weights and torch random state) are specific to
        # the CPU and the torch build they were generated on: on another
        # platform, regenerate tests/data/trainer_fingerprints.json with the
        # trainer of the main branch before comparing.
        with open(FINGERPRINTS, encoding="utf-8") as file:
            reference = json.load(file)
        train_ids, val_ids = ids()
        for model, expected in reference.items():
            with self.subTest(model=model), \
                    tempfile.TemporaryDirectory() as out:
                trainer = make(out, model_name=model, num_epochs=2, val_every=1)
                history = trainer.train(train_ids, val_ids)
                rng = hashlib.sha256(
                    torch.get_rng_state().numpy().tobytes()).hexdigest()
                self.assertEqual(rng, expected["torch_rng_sha256"])
                self.assertEqual(state_sha256(trainer.model.state_dict()),
                                 expected["state_sha256"])
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

    def interrupted(self, default_seconds: float, **kwargs) -> str:
        """Output directory of the run, killed after its first resume
        save."""
        out = tempfile.mkdtemp(dir=self.out.name)
        with mock.patch("floatsense.trainer.CHECKPOINT_SECONDS",
                        default_seconds):
            first = tuned(out, **kwargs)
            first.__class__ = InterruptedTrainer
            with self.assertRaises(Interrupt):
                first.train(*ids())
        return out

    def resumed(self, default_seconds: float, **kwargs) -> SequenceModelTrainer:
        """The same run, killed after its first resume save, resumed."""
        out = self.interrupted(default_seconds, **kwargs)
        trainer = tuned(out, **kwargs)
        saved = torch.load(trainer.resume_path(), weights_only=False)
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

    def test_checkpoint_seconds_argument(self):
        """checkpoint_seconds=0 saves after epoch 1 whatever the default;
        the resumed run is identical."""
        trainer, history, epoch = self.resumed(1e9, checkpoint_seconds=0.0)
        self.assertEqual(epoch, 1)
        self.assert_same_run(trainer, history)

    def test_resume_state_loads_on_cpu(self):
        """The resume state is loaded on the CPU whatever the device, so
        the random generator states can be restored."""
        trainer = tuned(self.interrupted(1e9))
        trainer.device = "meta"  # anything but the CPU
        saved = trainer._load_resume()
        self.assertEqual(saved["rng"]["torch"].device.type, "cpu")
        before = _rng_state()
        try:
            _set_rng_state(saved["rng"])
        finally:
            _set_rng_state(before)

    def test_set_rng_state_after_device_round_trip(self):
        """Generator states moved to a device and back still restore."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        saved = _rng_state()
        expected = torch.rand(4)
        moved = dict(saved, torch=saved["torch"].to(device))
        if saved["cuda"] is not None:
            moved["cuda"] = [s.to(device) for s in saved["cuda"]]
        _set_rng_state(moved)
        self.assertTrue(torch.equal(torch.rand(4), expected))

    @unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
    def test_resume_on_cuda(self):
        """Killed and resumed on CUDA: identical to the uninterrupted CUDA
        run."""
        with tempfile.TemporaryDirectory() as out:
            trainer = tuned(out, device="cuda")
            history = trainer.train(*ids())
            weights = state(trainer)
        resumed, resumed_history, epoch = self.resumed(1e9, device="cuda")
        self.assertEqual(epoch, 2)
        self.assertEqual(resumed_history, history)
        for key, value in state(resumed).items():
            self.assertTrue(torch.equal(value, weights[key]), key)

    def test_config_mismatch_in_progress(self):
        """A resume state in progress is refused for another learning rate
        or other training simulations."""
        out = self.interrupted(1e9)
        with self.assertRaisesRegex(ConfigMismatchError, "learning_rate"):
            tuned(out, learning_rate=2e-3).train(*ids())
        train_ids, val_ids = ids()
        with self.assertRaisesRegex(ConfigMismatchError, "ids_hash"):
            tuned(out).train(train_ids[:-1], val_ids)
        self.assert_same_run(*self.resumed_in(out))

    def resumed_in(self, out: str):
        """(trainer, history) of the run resumed in `out`."""
        trainer = tuned(out)
        return trainer, trainer.train(*ids())

    def test_config_mismatch_completed(self):
        """A completed run is refused for another seed or model knobs."""
        with self.assertRaisesRegex(ConfigMismatchError, "seed"):
            tuned(self.out.name, seed=4).train(*ids())
        with self.assertRaisesRegex(ConfigMismatchError, "model_kwargs"):
            tuned(self.out.name, model_kwargs=dict(TINY,
                                                   dropout=0.2)).train(*ids())

    def test_config_mismatch_task(self):
        """A completed run is refused for more epochs (a longer run goes to
        a new directory), another tower, inputs, scored window or damage
        proxy."""
        other = load_tower(TMP.name, OTHER_TOWER)
        for kwargs, key in [
            (dict(num_epochs=6), "num_epochs"),
            (dict(release=other), "tower"),
            (dict(input_channels=["tower_top_afa_mod", "rotor_speed"]),
             "input_channels"),
            (dict(min_time=100.0), "min_time"),
            (dict(damage_m=3.0), "damage_m"),
        ]:
            with self.subTest(key=key):
                with self.assertRaisesRegex(ConfigMismatchError, key):
                    tuned(self.out.name, **kwargs).train(*ids())

    def test_older_config_accepted(self):
        """A resume state stored before a setting was added (or before the
        configuration was stored) still resumes."""
        for drop in (["tower", "damage_m", "min_time"], None):
            with self.subTest(drop=drop):
                out = self.interrupted(1e9)
                path = tuned(out).resume_path()
                saved = torch.load(path, weights_only=False)
                if drop is None:
                    del saved["config"]
                else:
                    for key in drop:
                        del saved["config"][key]
                torch.save(saved, path)
                self.assert_same_run(*self.resumed_in(out))

    def init_copy(self) -> str:
        """A copy of the final checkpoint of the run, to start from."""
        path = os.path.join(tempfile.mkdtemp(dir=self.out.name), "init.pt")
        shutil.copy(self.trainer.checkpoint_path(), path)
        return path

    def test_resume_with_init_checkpoint(self):
        """With an initial checkpoint, a resumed run keeps the normalization
        stats of its resume state; the same path with other contents is
        refused."""
        init = self.init_copy()
        out = self.interrupted(1e9, init_checkpoint=init)
        path = tuned(out).resume_path()
        saved = torch.load(path, weights_only=False)
        with open(init, "rb") as file:
            digest = hashlib.sha256(file.read()).hexdigest()
        self.assertEqual(saved["config"]["init_checkpoint_sha256"], digest)
        stats = {
            k: [v[0] + 1.0, v[1] * 2.0] for k, v in saved["norm_stats"].items()
        }
        saved["norm_stats"] = stats
        torch.save(saved, path)
        trainer = tuned(out, init_checkpoint=init)
        trainer.train(*ids())
        self.assertEqual(trainer.norm_stats, stats)
        checkpoint = torch.load(init, weights_only=False)
        checkpoint["norm_stats"] = stats
        torch.save(checkpoint, init)
        with self.assertRaisesRegex(ConfigMismatchError,
                                    "init_checkpoint_sha256"):
            tuned(out, init_checkpoint=init).train(*ids())

    def test_relative_paths_resume(self):
        """./init.pt and init.pt are the same initial checkpoint: the real
        path is stored, and a state that stored the path as given still
        resumes."""
        init = self.init_copy()
        cwd = os.getcwd()
        os.chdir(os.path.dirname(init))
        try:
            out = self.interrupted(1e9, init_checkpoint="./init.pt")
            path = tuned(out).resume_path()
            saved = torch.load(path, weights_only=False)
            self.assertEqual(saved["config"]["init_checkpoint"],
                             os.path.realpath(init))
            self.assertIsNone(saved["config"]["calibration_path"])
            saved["config"]["init_checkpoint"] = "./init.pt"
            torch.save(saved, path)
            trainer = tuned(out, init_checkpoint="init.pt")
            history = trainer.train(*ids())
        finally:
            os.chdir(cwd)
        self.assertEqual(len(history["train_loss"]), 4)

    def test_calibration_digest(self):
        """The calibration is stored by real path and by the digest of its
        contents: the same file with other contents is refused."""
        folder = tempfile.mkdtemp(dir=self.out.name)
        path = os.path.join(folder, "calibration_fa.json")
        configs = []
        for text in ('{"gain": 1}', '{"gain": 2}'):
            with open(path, "w", encoding="utf-8") as file:
                file.write(text)
            trainer = tuned(folder, calibration_path=path)
            configs.append(trainer._run_config(*ids()))
        self.assertEqual(configs[0]["calibration_path"], os.path.realpath(path))
        self.assertEqual(configs[0]["calibration_sha256"],
                         hashlib.sha256(b'{"gain": 1}').hexdigest())
        trainer._config = configs[1]
        with self.assertRaisesRegex(ConfigMismatchError, "calibration_sha256"):
            trainer._check_run_config({"config": configs[0]})
        trainer._check_run_config({"config": configs[1]})

    def test_model_kwargs_mismatch_is_config_mismatch(self):
        """A state stored before the configuration held the model knobs is
        refused for other knobs with ConfigMismatchError."""
        out = self.interrupted(1e9)
        path = tuned(out).resume_path()
        saved = torch.load(path, weights_only=False)
        del saved["config"]["model_kwargs"]
        saved["model_kwargs"] = dict(TINY, dropout=0.2)
        torch.save(saved, path)
        with self.assertRaisesRegex(ConfigMismatchError, "Resume state of"):
            tuned(out).train(*ids())

    def test_stop_in_last_epoch(self):
        """SIGUSR1 during the last epoch: the run completes and records the
        request in `stop_requested`; the request does not outlive it."""
        out = tempfile.mkdtemp(dir=self.out.name)
        STOP_REQUESTED.set()
        try:
            trainer = tuned(out, num_epochs=1, val_every=1, save_epochs=[])
            trainer.train(*ids())
            self.assertTrue(trainer.stop_requested)
            self.assertFalse(STOP_REQUESTED.is_set())
        finally:
            STOP_REQUESTED.clear()
        saved = torch.load(trainer.resume_path(), weights_only=False)
        self.assertTrue(saved["completed"])
        trainer = tuned(out, num_epochs=1, val_every=1, save_epochs=[])
        trainer.train(*ids())
        self.assertFalse(trainer.stop_requested)

    def test_temporary_name(self):
        """A temporary is named <path>.tmp.<host>.<pid>.<uuid8>."""
        folder = tempfile.mkdtemp(dir=self.out.name)
        path = os.path.join(folder, "tcn_fa.pt")
        with mock.patch("floatsense.trainer.os.replace") as replace:
            _atomic_save({}, path)
        tmp = replace.call_args[0][0]
        self.assertEqual(replace.call_args[0][1], path)
        pattern = (re.escape(f"{path}.tmp.{socket.gethostname()}."
                             f"{os.getpid()}.") + "[0-9a-f]{8}")
        self.assertRegex(tmp, "^" + pattern + "$")
        self.assertTrue(os.path.isfile(tmp))

    def test_temporaries_across_hosts(self):
        """A temporary is kept only if written on this host by another live
        process less than max(3 checkpoint intervals, 1 h) ago, in the new
        and the older naming."""
        out = tempfile.mkdtemp(dir=self.out.name)
        host, live = socket.gethostname(), os.getppid()
        stem = make(out).checkpoint_path()
        old = time.time() - 2 * 3600.0
        files = {
            f"{stem}.tmp.{host}.{live}.0123abcd": (None, True),
            f"{stem}.tmp.{host}.{dead_pid()}.0123abcd": (None, False),
            f"{stem}.tmp.other.host.{live}.0123abcd": (None, False),
            f"{stem}.tmp.{host}.{live}.4567cdef": (old, False),
            f"{stem}.tmp.{live}": (old, False),
            f"{stem}.tmp.{host}.{live}.0123": (None, False),
        }
        for path, (mtime, _) in files.items():
            with open(path, "w", encoding="utf-8") as file:
                file.write("partial")
            if mtime is not None:
                os.utime(path, (mtime, mtime))
        live_pids = {live}
        kill = os.kill

        def kill_some(pid, sig):
            if pid in live_pids:
                return None
            return kill(pid, sig)

        with mock.patch("floatsense.trainer.os.kill", kill_some):
            make(out)._remove_stale_temporaries()
        for path, (_, kept) in files.items():
            self.assertEqual(os.path.exists(path), kept, path)
        # Longer checkpoint intervals keep a temporary longer.
        path = f"{stem}.tmp.{host}.{live}.89abcdef"
        with open(path, "w", encoding="utf-8") as file:
            file.write("partial")
        os.utime(path, (old, old))
        with mock.patch("floatsense.trainer.os.kill", kill_some):
            make(out, checkpoint_seconds=7200.0)._remove_stale_temporaries()
        self.assertTrue(os.path.exists(path))

    def test_stale_temporaries_removed(self):
        """Temporary files of a killed save of this run are removed; other
        files are kept."""
        out = tempfile.mkdtemp(dir=self.out.name)
        trainer = tuned(out, num_epochs=1, val_every=1, save_epochs=[])
        pid = dead_pid()
        stale = [
            trainer.checkpoint_path() + f".tmp.{pid}",
            trainer.resume_path() + f".tmp.{os.getpid()}",
            trainer.checkpoint_path(7) + f".tmp.{pid}",
        ]
        kept = [
            os.path.join(out, "lstm_fa.pt.tmp.14"),
            os.path.join(out, "tcn_ss.pt.tmp.15"),
            os.path.join(out, "tcn_fa.pt.tmp"),
            os.path.join(out, "notes.tmp.16"),
            os.path.join(out, "sub", "tcn_fa.pt.tmp.17"),
        ]
        for path in stale + kept:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as file:
                file.write("partial")
        trainer.train(*ids())
        for path in stale:
            self.assertFalse(os.path.exists(path), path)
        for path in kept:
            self.assertTrue(os.path.exists(path), path)

    def test_live_writer_temporaries_kept(self):
        """A temporary file whose pid is another live process (a concurrent
        writer's save in flight) is kept, also when the process belongs to
        another user; one of an exited process is removed."""
        out = tempfile.mkdtemp(dir=self.out.name)
        trainer = tuned(out, num_epochs=1, val_every=1, save_epochs=[])
        live = trainer.checkpoint_path() + f".tmp.{os.getppid()}"
        other_user = trainer.resume_path() + ".tmp.1"
        stale = trainer.resume_path() + f".tmp.{dead_pid()}"
        for path in (live, other_user, stale):
            with open(path, "w", encoding="utf-8") as file:
                file.write("partial")
        kill = os.kill

        def kill_as_other_user(pid, sig):
            if pid == 1:
                raise PermissionError
            return kill(pid, sig)

        with mock.patch("floatsense.trainer.os.kill", kill_as_other_user):
            trainer._remove_stale_temporaries()
        self.assertTrue(os.path.exists(live))
        self.assertTrue(os.path.exists(other_user))
        self.assertFalse(os.path.exists(stale))

    def test_stop_request_saves_and_resumes(self):
        """SIGUSR1: the epoch ends, the state is saved, the run stops; the
        resumed run is identical. The request and the handler do not
        outlive the run."""
        out = tempfile.mkdtemp(dir=self.out.name)
        handler = signal.getsignal(signal.SIGUSR1)
        STOP_REQUESTED.set()
        try:
            with mock.patch("floatsense.trainer.CHECKPOINT_SECONDS", 1e9):
                with self.assertRaises(StoppedError):
                    tuned(out).train(*ids())
            self.assertFalse(STOP_REQUESTED.is_set())
            self.assertIs(signal.getsignal(signal.SIGUSR1), handler)
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

    def run_script(self, *flags, out=None):
        """Exit code of scripts/train/run.py on the synthetic tower."""
        with tempfile.TemporaryDirectory() as tmp:
            return subprocess.run([
                sys.executable,
                os.path.join(REPO, "scripts", "train",
                             "run.py"), f"--dataset_dir={TMP.name}",
                f"--tower={synthetic.TOWER}", f"--output_dir={out or tmp}",
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
        """Distinct exit codes for a too large model, a divergence and a
        resume state of another run configuration."""
        self.assertEqual(self.run_script("--max_params_m=0.0001"),
                         C.EXIT_TOO_LARGE)
        self.assertEqual(self.run_script("--learning_rate=1e30"),
                         C.EXIT_DIVERGED)
        small = "--model_kwargs=hidden_channels=8,num_levels=2"
        self.assertEqual(self.run_script(small), 0)
        with tempfile.TemporaryDirectory() as out:
            self.assertEqual(self.run_script(small, "--resume", out=out), 0)
            self.assertEqual(
                self.run_script(small,
                                "--resume",
                                "--learning_rate=2e-3",
                                out=out), C.EXIT_CONFIG_MISMATCH)
            self.assertEqual(
                self.run_script(small,
                                "--resume",
                                f"--tower={OTHER_TOWER}",
                                out=out), C.EXIT_CONFIG_MISMATCH)


# Runs main() of scripts/train/run.py with a recording trainer: prints the
# trainer arguments, the SIGUSR1 handler while the tower loads and during
# each train(), the evaluated models and the handler after main(). With
# 'kill', SIGUSR1 arrives while the tower loads; with 'kill_train' = n,
# during the n-th train(), which then completes as in a last epoch (the
# default action of SIGUSR1 would end the process).
RUN_MAIN = """
import json, os, signal, sys
sys.path.insert(0, os.path.join({repo!r}, "scripts", "train"))
import run
from floatsense import trainer as T
seen = {{}}
load = run.load_tower
def load_tower(*args):
    seen["loading"] = signal.getsignal(signal.SIGUSR1) is T.request_stop
    if {kill!r}:
        os.kill(os.getpid(), signal.SIGUSR1)
    return load(*args)
class Recorder:
    def __init__(self, **kwargs):
        seen["checkpoint_seconds"] = kwargs["checkpoint_seconds"]
        self.name = kwargs["model_name"] + "_" + kwargs["direction"]
        self.stop_requested = False
    def train(self, *args):
        seen.setdefault("training", []).append(
            signal.getsignal(signal.SIGUSR1) is T.request_stop)
        if len(seen["training"]) == {kill_train!r}:
            os.kill(os.getpid(), signal.SIGUSR1)
            seen["stop_requested"] = T.STOP_REQUESTED.is_set()
        seen.setdefault("completed", []).append(self.name)
        self.stop_requested = T.STOP_REQUESTED.is_set()
        T.STOP_REQUESTED.clear()
    def evaluate(self, *args, **kwargs):
        seen.setdefault("evaluated", []).append(self.name)
        return {{}}
run.load_tower, run.SequenceModelTrainer = load_tower, Recorder
run.FLAGS(["run"] + sys.argv[1:])
try:
    run.main(None)
except SystemExit as error:
    seen["exit"] = error.code
seen["after"] = signal.getsignal(signal.SIGUSR1) is signal.SIG_DFL
print("SEEN " + json.dumps(seen))
"""


class ScriptSignalTest(unittest.TestCase):
    """--checkpoint_seconds and the SIGUSR1 handler of run.py."""

    def run_main(self, *flags, kill=False, kill_train=0) -> dict:
        """What the recording trainer saw (see RUN_MAIN)."""
        with tempfile.TemporaryDirectory() as out:
            result = subprocess.run([
                sys.executable, "-c",
                RUN_MAIN.format(repo=REPO, kill=kill, kill_train=kill_train),
                f"--dataset_dir={TMP.name}", f"--tower={synthetic.TOWER}",
                f"--output_dir={out}", "--run_evaluation=False", *flags
            ],
                                    capture_output=True,
                                    text=True,
                                    env={
                                        **os.environ, "CUDA_VISIBLE_DEVICES": ""
                                    },
                                    check=True)
        line = [l for l in result.stdout.splitlines() if l.startswith("SEEN")]
        return json.loads(line[-1][5:])

    def test_checkpoint_seconds_flag(self):
        """The flag reaches the trainer; its default is the constant."""
        self.assertEqual(self.run_main()["checkpoint_seconds"],
                         C.CHECKPOINT_SECONDS)
        self.assertEqual(
            self.run_main("--checkpoint_seconds=30")["checkpoint_seconds"],
            30.0)

    def test_handler_from_startup(self):
        """With --resume the handler is installed before the data load and
        restored after training; without it, never installed."""
        seen = self.run_main("--resume")
        self.assertEqual(
            seen, {
                "loading": True,
                "checkpoint_seconds": C.CHECKPOINT_SECONDS,
                "training": [True],
                "completed": ["tcn_fa"],
                "after": True
            })
        seen = self.run_main()
        self.assertFalse(seen["loading"])
        self.assertEqual(seen["training"], [False])

    def test_handler_over_models_and_directions(self):
        """The handler stays installed over every model and direction and
        is restored once, after the last: SIGUSR1 during a later run
        requests a stop instead of ending the process."""
        for flag in ("--models=tcn,lstm", "--directions=fa,ss"):
            with self.subTest(flag=flag):
                seen = self.run_main("--resume", flag, kill_train=2)
                self.assertEqual(seen["training"], [True, True])
                self.assertTrue(seen["stop_requested"])
                self.assertEqual(seen["exit"], C.EXIT_STOPPED)
                self.assertTrue(seen["after"])

    def test_stop_in_last_epoch(self):
        """SIGUSR1 during the last epoch of the first of two models: the
        run completes, then the script exits with EXIT_STOPPED before its
        evaluation and the next model."""
        seen = self.run_main("--resume",
                             "--models=tcn,lstm",
                             "--num_epochs=1",
                             "--run_evaluation=True",
                             kill_train=1)
        self.assertEqual(seen["exit"], C.EXIT_STOPPED)
        self.assertEqual(seen["completed"], ["tcn_fa"])
        self.assertNotIn("evaluated", seen)
        self.assertTrue(seen["after"])
        seen = self.run_main("--resume", "--models=tcn,lstm", "--num_epochs=1",
                             "--run_evaluation=True")
        self.assertNotIn("exit", seen)
        self.assertEqual(seen["evaluated"], ["tcn_fa", "lstm_fa"])

    def test_signal_during_startup(self):
        """SIGUSR1 while the data loads: clean stop (EXIT_STOPPED), no
        training."""
        seen = self.run_main("--resume", kill=True)
        self.assertEqual(seen["exit"], C.EXIT_STOPPED)
        self.assertNotIn("training", seen)


class MainCheckpointTest(unittest.TestCase):
    """A checkpoint written by the published trainer (no model_kwargs)."""

    def test_loads_with_identical_predictions(self):
        """Exactly the normalized predictions, damage CSV and summary the
        published code computes from it (tests/data/main_dlinear_fa.npz,
        .csv and .json)."""
        with open(MAIN_CHECKPOINT + ".json", encoding="utf-8") as file:
            expected = json.load(file)
        predictions = np.load(MAIN_CHECKPOINT + ".npz")
        with tempfile.TemporaryDirectory() as out:
            trainer = make(out, model_name="dlinear", num_epochs=1)
            shutil.copy(MAIN_CHECKPOINT + ".pt", trainer.checkpoint_path())
            self.assertNotIn(
                "model_kwargs",
                torch.load(trainer.checkpoint_path(), weights_only=False))
            trainer.load_checkpoint()
            self.assertEqual(trainer.model_kwargs, {})
            test_ids = ids()[1]
            dataset = trainer._make_dataset(test_ids, None)
            trainer.model.eval()
            with torch.no_grad():
                for index, sim_id in enumerate(dataset.sim_ids):
                    prediction = trainer._predict_window(
                        dataset, index, dataset[index])
                    want = predictions[f"sim_{int(sim_id)}"]
                    self.assertEqual(prediction.dtype, want.dtype)
                    np.testing.assert_array_equal(prediction, want)
            summary = trainer.evaluate(test_ids, tag="main")
            with open(
                    os.path.join(out, "damage_comparison_dlinear_fa_main.csv"),
                    "rb") as file:
                written = file.read()
            with open(MAIN_CHECKPOINT + ".csv", "rb") as file:
                self.assertEqual(written, file.read())
            self.assertEqual(summary, expected)


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


if __name__ == "__main__":
    unittest.main()
