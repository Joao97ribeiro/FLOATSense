# pylint: disable=wrong-import-position,protected-access
# pylint: disable=use-dict-literal
# pylint: disable=too-many-lines
"""Tests of the drivers of the validation-tuned track (hpo/).

Run from the repository root with `python -m unittest discover tests`.
CPU only; nothing is trained: the drivers run with their --dry_run stub
and the failure handling with tiny stand-in commands. The Optuna tests need
optuna >= 4 and are skipped otherwise.
"""

import argparse
import datetime
import os
import random
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

import numpy as np
import optuna
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from floatsense import constants as FC
from hpo import analyze
from hpo import common
from hpo import confirm
from hpo import constants as C
from hpo import final
from hpo import pick
from hpo import search
from hpo import search_space as S
from hpo import status as hpo_status
from hpo import worker


class FakeTrial:
    """The suggest API of an Optuna trial, drawing at random."""

    def __init__(self, seed: int):
        self.rng = random.Random(seed)

    def suggest_float(self, name, low, high, log=False):
        """Uniform (or log-uniform) draw."""
        del name
        if log:
            return float(np.exp(self.rng.uniform(np.log(low), np.log(high))))
        return self.rng.uniform(low, high)

    def suggest_categorical(self, name, choices):
        """One of the choices."""
        del name
        return self.rng.choice(choices)


def add_trials(study, values):
    """Completed trials with a configuration."""
    dist = optuna.distributions.FloatDistribution(1e-4, 3e-3, log=True)
    for value in values:
        study.add_trial(
            optuna.trial.create_trial(params={"lr": 1e-3},
                                      distributions={"lr": dist},
                                      value=value,
                                      user_attrs={"config": {
                                          "lr": 1e-3
                                      }}))


class SearchSpaceTest(unittest.TestCase):
    """search_space.suggest and as_args."""

    def test_draws_inside_grids(self):
        """Every draw is inside its grid; wd only when use_wd is on."""
        for model, space in S.SPACE.items():
            seen_wd = set()
            for seed in range(200):
                cfg = S.suggest(FakeTrial(seed), model)
                self.assertEqual("wd" in cfg, bool(cfg["use_wd"]))
                seen_wd.add(cfg["use_wd"])
                for name, value in cfg.items():
                    spec = space[name]
                    if spec[0] == "categorical":
                        self.assertIn(value, spec[1], (model, name))
                    else:
                        self.assertGreaterEqual(value, spec[1])
                        self.assertLessEqual(value, spec[2])
            self.assertEqual(seen_wd, {False, True}, model)

    def test_published_inside_grids(self):
        """The fixed-budget values are points of the space."""
        for model, published in S.PUBLISHED.items():
            for name, value in {**S.FIXED_RECIPE, **published}.items():
                spec = S.SPACE[model][name]
                if spec[0] == "categorical":
                    self.assertIn(value, spec[1], (model, name))
                else:
                    self.assertTrue(spec[1] <= value <= spec[2], (model, name))
        self.assertEqual(len(S.LEARNED), 20)

    def test_as_args(self):
        """Weight decay 0 when off; warm-up only with the cosine."""
        args = S.as_args(
            dict(lr=1e-3, use_wd=False, schedule="constant", hidden_dim=64))
        self.assertIn("--weight_decay=0", args)
        self.assertIn("--model_kwargs=hidden_dim=64", args)
        self.assertNotIn(f"--warmup_epochs={S.WARMUP_EPOCHS}", args)
        args = S.as_args(dict(lr=1e-3, use_wd=True, wd=1e-4, schedule="cosine"))
        self.assertIn("--weight_decay=0.0001", args)
        self.assertIn(f"--warmup_epochs={S.WARMUP_EPOCHS}", args)
        self.assertIn(f"--grad_clip={C.GRAD_CLIP}", args)

    def test_formatted(self):
        """The stored configuration is the one passed to run.py."""
        cfg = dict(lr=0.00123456789,
                   use_wd=True,
                   wd=3.33333333e-5,
                   schedule="cosine",
                   dropout=0.123456789,
                   hidden_channels=64)
        stored = S.formatted(cfg)
        self.assertEqual(stored["lr"], 0.00123457)
        self.assertEqual(stored["dropout"], 0.123457)
        self.assertEqual(stored["hidden_channels"], 64)
        self.assertEqual(S.as_args(cfg), S.as_args(stored))
        self.assertIn("--model_kwargs=dropout=0.123457,hidden_channels=64",
                      S.as_args(cfg))


class CommonTest(unittest.TestCase):
    """Records, claims and failure handling."""

    def test_parse_val(self):
        """VAL lines of the trainer are parsed; other lines are not."""
        scores = common.parse_val("VAL epoch=20 r2_mean=0.5 r2_top=0.25 "
                                  "r2_base=0.75 r2_gauges=0.1,0.2,nan\n")
        self.assertEqual(scores["epoch"], 20)
        self.assertEqual(scores["r2_top"], 0.25)
        self.assertTrue(np.isnan(scores["r2_gauges"][2]))
        self.assertIsNone(common.parse_val("[tcn/fa] epoch 20/100 loss 0.1"))

    def test_classify(self):
        """Exit codes and logs map to the failure classes."""
        self.assertEqual(common.classify(0, ""), "ok")
        self.assertEqual(common.classify(FC.EXIT_DIVERGED, ""), "diverged")
        self.assertEqual(common.classify(FC.EXIT_TOO_LARGE, ""), "too_large")
        self.assertEqual(common.classify(-signal.SIGTERM, ""), "preempted")
        self.assertEqual(
            common.classify(
                1, "RuntimeError: CUDA error: an illegal memory "
                "access was encountered"), "hardware")
        self.assertEqual(
            common.classify(1, "torch.OutOfMemoryError: CUDA out of memory"),
            "oom")
        self.assertEqual(common.classify(1, "RuntimeError: out of memory"),
                         "oom")
        self.assertEqual(common.classify(1, "ValueError: bad shape"), "crash")
        self.assertEqual(common.classify(FC.EXIT_CONFIG_MISMATCH, ""),
                         "config_mismatch")

    def test_write_once_and_claim(self):
        """A plan is written once; a live claim is exclusive, a stale one
        is taken over."""
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "plan.json")
            self.assertTrue(common.write_once(path, {"a": 1}))
            self.assertFalse(common.write_once(path, {"a": 2}))
            self.assertEqual(common.read_json(path), {"a": 1})
            unit = os.path.join(root, "unit")
            self.assertTrue(common.claim(unit))
            self.assertFalse(common.claim(unit))
            old = time.time() - 2 * C.STALE_MINUTES * 60
            os.utime(os.path.join(unit, "heartbeat"), (old, old))
            self.assertTrue(common.claim(unit))
            common.release(unit)
            self.assertTrue(common.claim(unit))

    def run_fake(self, root: str, script: str, **kwargs) -> dict:
        """run_unit on a stand-in command."""
        run_dir = os.path.join(root, "unit")
        os.makedirs(run_dir, exist_ok=True)
        with mock.patch.object(common, "RETRY_SECONDS", 0.0):
            return common.run_unit("test/unit",
                                   [sys.executable, "-c", script, run_dir],
                                   run_dir, root, "tcn", **kwargs)

    def test_attempts_and_config(self):
        """Every attempt (pruned included) is recorded with its wall times,
        host and GPU; config.json holds the formatted values and the
        command."""
        script = "print('VAL epoch=10 r2_mean=0.5 r2_top=0.4 r2_base=0.6')\n"
        with tempfile.TemporaryDirectory() as root:
            result = self.run_fake(root,
                                   script,
                                   on_val=lambda scores: True,
                                   config=dict(lr=0.00123456789))
            self.assertEqual(result["status"], "pruned")
            run_dir = os.path.join(root, "unit")
            events = common.read_json(os.path.join(run_dir,
                                                   "attempts.json"))["events"]
            self.assertEqual(len(events), 1)
            for key in ("start", "end", "seconds", "host", "gpu", "status"):
                self.assertIn(key, events[0])
            self.assertEqual(events[0]["status"], "pruned")
            self.assertEqual(events[0]["host"], socket.gethostname())
            config = common.read_json(os.path.join(run_dir, "config.json"))
            self.assertEqual(config["config"], {"lr": 0.00123457})
            self.assertEqual(config["command"][0], sys.executable)

    def test_config_mismatch_is_held(self):
        """A refused resume is not retried and not parked: the unit is held
        for an operator (marker and alert) and no worker picks it."""
        with tempfile.TemporaryDirectory() as root:
            result = self.run_fake(
                root, f"import sys; sys.exit({FC.EXIT_CONFIG_MISMATCH})")
            self.assertEqual(result["status"], "config_mismatch")
            run_dir = os.path.join(root, "unit")
            self.assertEqual(len(common.read_attempts(run_dir)["events"]), 1)
            self.assertFalse(os.path.exists(os.path.join(run_dir, "PARKED")))
            self.assertTrue(common.held(run_dir))
            self.assertEqual(len(os.listdir(os.path.join(root, "alerts"))), 1)

    def test_oom(self):
        """Out of memory ends a search unit after one resume as a crash
        (not parked); with oom_ends=False (phases 2 and 3) it is a crash."""
        script = ("raise SystemExit('torch.OutOfMemoryError: CUDA out of "
                  "memory')")
        with tempfile.TemporaryDirectory() as root:
            result = self.run_fake(root, script)
            self.assertEqual(result["status"], "oom")
            run_dir = os.path.join(root, "unit")
            record = common.read_attempts(run_dir)
            self.assertEqual(record["crashes"], 1)  # resumed once
            self.assertFalse(os.path.exists(os.path.join(run_dir, "PARKED")))
            result = self.run_fake(root, script, oom_ends=False)
            self.assertEqual(result["status"], "parked")
            record = common.read_attempts(run_dir)
            self.assertEqual(record["crashes"], C.MAX_ATTEMPTS)

    def test_retry_once(self):
        """A parked unit is retried once, RETRY_SHELVED_AFTER after its
        parking, with fresh counts; a second parking is final; nothing is
        retried once the test is ready."""
        with tempfile.TemporaryDirectory() as root:
            self.run_fake(root, "raise SystemExit(1)")
            run_dir = os.path.join(root, "unit")
            self.assertFalse(common.retry_due(run_dir))  # not an hour yet
            self.assertFalse(common.park_final(run_dir))
            self.assertTrue(common.park_final(run_dir, frozen=True))
            with mock.patch.object(C, "RETRY_SHELVED_AFTER", 0.0):
                self.assertFalse(common.retry_due(run_dir, frozen=True))
                self.assertTrue(common.retry_due(run_dir))
                self.assertTrue(common.retry_unit(root, "u", run_dir))
            self.assertFalse(os.path.exists(os.path.join(run_dir, "PARKED")))
            self.assertEqual(common.read_attempts(run_dir)["crashes"], 0)
            self.run_fake(root, "raise SystemExit(1)")
            self.assertTrue(common.park_final(run_dir))
            with mock.patch.object(C, "RETRY_SHELVED_AFTER", 0.0):
                self.assertFalse(common.retry_due(run_dir))
                self.assertFalse(common.retry_unit(root, "u", run_dir))

    def test_shared_writes(self):
        """Temporary names are unique per host and process; write_once
        tolerates a temporary file removed meanwhile; an attempts record is
        merged by attempt and not written once the unit is not ours."""
        with tempfile.TemporaryDirectory() as root:
            tmp = common.tmp_path(os.path.join(root, "x.json"))
            self.assertIn(f".tmp.{socket.gethostname()}.{os.getpid()}.", tmp)
            remove = os.remove

            def gone(path):
                remove(path)
                raise FileNotFoundError(path)

            with mock.patch.object(common.os, "remove", gone):
                self.assertTrue(common.write_once(tmp, {}))
            other = {"start": "2026-01-01T00:00:00", "host": "b"}
            common.write_json(common.attempts_path(root), {"events": [other]})
            mine = {"events": [{"start": "2026-01-02T00:00:00", "host": "a"}]}
            self.assertFalse(common.save_attempts(root, mine, lambda: False))
            self.assertEqual(common.read_attempts(root)["events"], [other])
            self.assertTrue(common.save_attempts(root, mine, lambda: True))
            self.assertEqual(len(common.read_attempts(root)["events"]), 2)

    def test_gpu_memory(self):
        """GPU memory from nvidia-smi (MiB), None if unknown."""
        out = mock.Mock(stdout="81559\n")
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}), \
                mock.patch.object(common.subprocess, "run", return_value=out):
            self.assertAlmostEqual(common.gpu_memory_gb(), 81559 / 1024)
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}), \
                mock.patch.object(common.subprocess, "run",
                                  side_effect=OSError("no nvidia-smi")):
            self.assertIsNone(common.gpu_memory_gb())

    def test_checkpoint_seconds_in_command(self):
        """Every unit's run.py command saves its resume state often."""
        args = argparse.Namespace(python="python",
                                  dataset_dir="data",
                                  extra="",
                                  checkpoint_seconds=90.0)
        cmd = common.train_command(
            args, "tcn", "opt2", "/run",
            dict(lr=1e-3, use_wd=False, schedule="constant"), "val/train", 100,
            0)
        self.assertIn("--checkpoint_seconds=90", cmd)
        self.assertEqual(
            worker.parse_args(["--models=tcn"]).checkpoint_seconds,
            C.CHECKPOINT_SECONDS)

    def test_crash_is_resumed(self):
        """A crash is retried; the second attempt completes."""
        script = (
            "import json, os, sys\n"
            "d = sys.argv[1]; p = os.path.join(d, 'n')\n"
            "n = int(open(p).read()) if os.path.exists(p) else 0\n"
            "open(p, 'w').write(str(n + 1))\n"
            "if n == 0: sys.exit(1)\n"
            "print('VAL epoch=10 r2_mean=0.5 r2_top=0.4 r2_base=0.6')\n"
            "json.dump({'val_r2': [{'epoch': 10, 'r2_mean': 0.5}]},\n"
            "          open(os.path.join(d, 'history_tcn_fa.json'), 'w'))\n")
        with tempfile.TemporaryDirectory() as root:
            result = self.run_fake(root, script)
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["history"]["val_r2"][0]["r2_mean"], 0.5)
            record = common.read_json(
                os.path.join(root, "unit", "attempts.json"))
            self.assertEqual(record["crashes"], 1)

    def test_preemption_is_free_and_crashes_park(self):
        """Preemptions do not count; MAX_ATTEMPTS crashes park the unit and
        write an alert."""
        script = ("import os, signal, sys\n"
                  "d = sys.argv[1]; p = os.path.join(d, 'n')\n"
                  "n = int(open(p).read()) if os.path.exists(p) else 0\n"
                  "open(p, 'w').write(str(n + 1))\n"
                  "if n < 2: os.kill(os.getpid(), signal.SIGTERM)\n"
                  "raise RuntimeError('boom')\n")
        with tempfile.TemporaryDirectory() as root:
            result = self.run_fake(root, script)
            self.assertEqual(result["status"], "parked")
            record = common.read_json(
                os.path.join(root, "unit", "attempts.json"))
            self.assertEqual((record["free"], record["crashes"]),
                             (2, C.MAX_ATTEMPTS))
            alerts = os.listdir(os.path.join(root, "alerts"))
            self.assertEqual(len(alerts), 1)
            self.assertTrue(os.path.exists(os.path.join(root, "unit",
                                                        "PARKED")))

    def test_exit_codes_are_returned(self):
        """Divergence and the parameter cap are not retried."""
        with tempfile.TemporaryDirectory() as root:
            for code, status in ((FC.EXIT_DIVERGED, "diverged"),
                                 (FC.EXIT_TOO_LARGE, "too_large")):
                result = self.run_fake(root, f"import sys; sys.exit({code})")
                self.assertEqual(result["status"], status)


class DriversTest(unittest.TestCase):
    """search, confirm and final with the dry-run stub."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def search(self, *extra, model="tcn", tower="opt2"):
        """search.main in dry-run mode on the test root."""
        return search.main([
            f"--model={model}", f"--tower={tower}", "--dry_run",
            f"--root={self.root}", "--n_trials=10", *extra
        ])

    def test_search_prunes_and_stops(self):
        """Ten trials, some pruned by the median rule; a second worker
        finds the budget spent."""
        study = self.search()
        states = [t.state.name for t in study.trials]
        self.assertEqual(len(states), 10)
        self.assertGreaterEqual(states.count("PRUNED"), 1)
        self.assertEqual(states[:C.N_STARTUP], ["COMPLETE"] * C.N_STARTUP)
        for trial in study.trials:
            cfg = trial.user_attrs["config"]
            self.assertEqual("wd" in cfg, cfg["use_wd"])
        self.assertEqual(len(self.search().trials), 10)

    def test_no_pruning_models(self):
        """The NO_PRUNING models are never pruned."""
        study = self.search(model="mamba")
        self.assertTrue(all(t.state.name == "COMPLETE" for t in study.trials))

    def test_stale_trial_is_requeued(self):
        """A RUNNING trial of a dead worker is resumed with its config."""
        study = search.open_study(self.root, "tcn", "opt2")
        trial = study.ask()
        cfg = S.suggest(trial, "tcn")
        run_dir = os.path.join(self.root, "trials", "dead")
        trial.set_user_attr("run_dir", run_dir)
        common.touch(run_dir)
        old = time.time() - 2 * C.STALE_MINUTES * 60
        os.utime(os.path.join(run_dir, "heartbeat"), (old, old))
        study = self.search("--max_new=1")
        resumed = study.trials[1]
        self.assertEqual(study.trials[0].state.name, "FAIL")
        self.assertEqual(resumed.user_attrs["resume_from"], run_dir)
        self.assertEqual(resumed.user_attrs["config"], S.formatted(cfg))

    def test_extension_rule(self):
        """A best trial after the EXTEND_AFTER-th extends the model."""
        n_trials = C.EXTEND_AFTER + 5
        study = search.open_study(self.root, "fits", "ref")
        dist = optuna.distributions.CategoricalDistribution([0.5])
        for number in range(n_trials):
            study.add_trial(
                optuna.trial.create_trial(
                    params={"x": 0.5},
                    distributions={"x": dist},
                    value=1.0 if number == C.EXTEND_AFTER + 2 else 0.0))
        trials = study.get_trials()
        self.assertEqual(search.best_position(trials, n_trials),
                         C.EXTEND_AFTER + 3)
        self.assertEqual(search.best_position(trials, C.EXTEND_AFTER), 1)
        self.assertTrue(search.maybe_extend(self.root, "fits", n_trials))
        self.assertEqual(search.target_trials(self.root, "fits", n_trials),
                         n_trials + C.EXTEND_BY)
        self.assertEqual(search.target_trials(self.root, "tcn", n_trials),
                         n_trials)

    def test_confirm_and_final(self):
        """Plan frozen once, winner by the median, test sealed once."""
        self.search()
        args = [
            "--model=tcn", "--tower=opt2", f"--root={self.root}", "--dry_run",
            "--n_trials=10"
        ]
        plan = confirm.main(args + ["--freeze"])
        self.assertEqual(len(plan["configs"]), C.N_TOP)
        self.assertEqual(confirm.main(args + ["--freeze"]), plan)
        self.assertIsNone(confirm.main(args + ["--summarize"]))
        confirm.main(args)
        record = confirm.main(args + ["--summarize"])
        medians = {r["rank"]: r["median"] for r in record["configs"]}
        self.assertEqual(record["winner_median"], max(medians.values()))
        for row in record["configs"]:
            self.assertAlmostEqual(row["median"],
                                   float(np.median(row["scores"])))
        self.assertEqual(record["best_epoch"] % C.BEST_EPOCH_ROUND, 0)
        test = ["--open_test", f"--root={self.root}", "--dry_run"]
        with self.assertRaises(SystemExit):
            final.main(test)  # every other model and tower is missing
        for model in S.LEARNED:
            for tower in C.TOWERS_SEARCHED:
                if (model, tower) != ("tcn", "opt2"):
                    confirm.park_study(self.root, model, tower, "test")
        final.main(
            ["--model=tcn", "--tower=opt2", f"--root={self.root}", "--dry_run"])
        with self.assertRaises(SystemExit) as error:
            final.main(test)  # READY_FOR_TEST.json not written yet
        self.assertIn(common.READY, str(error.exception))
        self.assertTrue(pick.mark_ready(self.root, 10))
        final.main(test)
        marker = os.path.join(
            final.sealed_dir(self.root, "primary_last", "opt2", 0),
            "SEALED_tcn.json")
        self.assertTrue(common.read_json(marker)["label"].startswith("primary"))
        stamp = common.read_json(marker)["time"]
        final.main(test)
        self.assertEqual(common.read_json(marker)["time"], stamp)
        missing = common.read_json(
            os.path.join(self.root, "sealed", "MISSING.json"))
        self.assertEqual(len(missing["parked"]), 20 * 3 - 1)
        self.assertTrue(
            os.path.exists(
                os.path.join(self.root, "final", "tcn_opt2", "s0",
                             "attempts.json")))

    def stale_running_trial(self, marker: bool) -> tuple:
        """A RUNNING trial with a run directory (and a requeue marker)."""
        study = search.open_study(self.root, "tcn", "opt2")
        trial = study.ask()
        cfg = S.suggest(trial, "tcn")
        run_dir = os.path.join(self.root, "trials", "dead")
        trial.set_user_attr("run_dir", run_dir)
        common.touch(run_dir)
        if marker:
            common.write_json(search.requeue_marker(run_dir, trial.number), {})
        return study, cfg, run_dir

    def test_requeue_kill_window(self):
        """A requeue killed between its marker and its tell (earlier
        order) left a RUNNING trial: it is told FAIL and requeued now, even
        with a live heartbeat."""
        _, cfg, run_dir = self.stale_running_trial(marker=True)
        study = self.search("--max_new=1")
        self.assertEqual(study.trials[0].state.name, "FAIL")
        self.assertEqual(study.trials[1].user_attrs["resume_from"], run_dir)
        self.assertEqual(study.trials[1].user_attrs["config"], S.formatted(cfg))
        self.assertNotEqual(study.trials[1].state.name, "RUNNING")

    def test_requeue_run_dir_from_resume_from(self):
        """A requeued trial whose worker died before setting run_dir is
        requeued again from its resume_from."""
        study = search.open_study(self.root, "tcn", "opt2")
        run_dir = os.path.join(self.root, "trials", "first")
        study.enqueue_trial({}, user_attrs={"resume_from": run_dir})
        trial = study.ask()
        self.assertEqual(search.trial_run_dir(study.trials[trial.number]),
                         run_dir)
        self.assertEqual(search.requeue_stale(study, exclusive=True),
                         [trial.number])
        self.assertEqual(study.trials[-1].state.name, "WAITING")
        self.assertEqual(study.trials[-1].user_attrs["resume_from"], run_dir)

    def test_lost_lock_drops_result(self):
        """A trial is not told after its lock is lost, nor when it was
        finished elsewhere (a takeover requeued it)."""
        study = search.open_study(self.root, "tcn", "opt2")
        args = search.parse_args(
            ["--model=tcn", "--tower=opt2", "--dry_run", f"--root={self.root}"])
        trial = study.ask()
        outcome = search.run_trial(study, trial, args, owned=lambda: False)
        self.assertEqual(outcome["status"], "lost")
        self.assertEqual(study.trials[trial.number].state.name, "RUNNING")
        trial = study.ask()

        def taken() -> bool:
            study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
            return True

        outcome = search.run_trial(study, trial, args, owned=taken)
        self.assertEqual(outcome["status"], "lost")
        self.assertEqual(study.trials[trial.number].state.name, "FAIL")

    def test_over_cap_is_failed_and_bounded(self):
        """Over-cap draws are FAIL (not counted, not TPE startup), the
        budget still completes, and a study of over-cap draws stops."""
        stub = common.stub_unit
        calls = {"n": 0}

        def two_over_cap(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] <= 2:
                return {"status": "too_large", "params": 10**9, "tail": ""}
            return stub(*args, **kwargs)

        with mock.patch.object(common, "stub_unit", two_over_cap):
            study = self.search()
        over = [t for t in study.trials if search.over_cap(t)]
        self.assertEqual(len(over), 2)
        self.assertTrue(all(t.state.name == "FAIL" for t in over))
        self.assertEqual(sum(search.counted(t) for t in study.trials), 10)
        too_large = {"status": "too_large", "params": 10**9, "tail": ""}
        with mock.patch.object(common, "stub_unit",
                               lambda *a, **k: too_large), mock.patch.object(
                                   C, "MAX_OVER_CAP", 3):
            study = self.search("--max_new=1", tower="ref")
        self.assertEqual(len(study.trials), 3)

    def test_no_pruning_at_last_epoch_or_replay(self):
        """should_prune is not asked at the last epoch, nor while a
        finished run replays its VAL lines."""
        with mock.patch.object(optuna.trial.Trial,
                               "should_prune",
                               return_value=True):
            study = self.search("--epochs=10")
            self.assertTrue(
                all(t.state.name == "COMPLETE" for t in study.trials))
            other = search.open_study(self.root, "tcn", "ref")
            run_dir = os.path.join(self.root, "trials", "finished")
            common.write_json(common.history_path(run_dir, "tcn"), {})
            other.enqueue_trial({}, user_attrs={"resume_from": run_dir})
            study = self.search("--max_new=1", tower="ref")
            self.assertEqual(study.trials[0].state.name, "COMPLETE")

    def test_freeze_uses_first_target_trials(self):
        """A study with more counted trials than its budget cannot widen
        the pool of the plan."""
        study = search.open_study(self.root, "tcn", "opt2")
        add_trials(study, [0.1 * i for i in range(10)] + [5.0])
        plan = confirm.main([
            "--model=tcn", "--tower=opt2", f"--root={self.root}",
            "--n_trials=10", "--freeze"
        ])
        self.assertEqual([c["trial"] for c in plan["configs"]], [9, 8])
        self.assertEqual((plan["target"], plan["stopped_early"]), (10, None))

    def test_no_finite_median_parks(self):
        """All seeds diverged: no winner, the (model, tower) is parked."""
        plan = {
            "configs": [{
                "rank": 0,
                "trial": 0,
                "value": 0.5,
                "config": {}
            }],
            "n_seeds": C.N_SEEDS,
            "epochs": 10
        }
        for seed in range(C.N_SEEDS):
            common.write_json(
                os.path.join(
                    confirm.unit_dir(self.root, "tcn", "opt2", 0, seed),
                    "result.json"), {
                        "status": "diverged",
                        "score": None,
                        "best_epoch": None
                    })
        args = confirm.parse_args(
            ["--model=tcn", "--tower=opt2", f"--root={self.root}"])
        self.assertIsNone(confirm.summarize(args, plan))
        self.assertTrue(
            os.path.exists(confirm.parked_path(self.root, "tcn", "opt2")))
        self.assertFalse(
            os.path.exists(confirm.winner_path(self.root, "tcn", "opt2")))


class PickTest(unittest.TestCase):
    """Locks, parking, the gate of a model and the worker breaker."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_restart_takes_back_own_lock(self):
        """Another restart count of the same owner is a dead incarnation;
        another owner waits; the default restart is 0 outside Slurm."""
        self.assertTrue(pick.Lock(self.root, "u", "job", 0).acquire())
        self.assertFalse(pick.Lock(self.root, "u", "other", 7).acquire())
        self.assertTrue(pick.Lock(self.root, "u", "job", 1).acquire())
        self.assertTrue(pick.Lock(self.root, "u", "job", 0).acquire())
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SLURM_RESTART_COUNT", None)
            self.assertEqual(worker.parse_args([]).restart, 0)
            os.environ["SLURM_RESTART_COUNT"] = "2"
            self.assertEqual(worker.parse_args([]).restart, 2)

    def test_breaker_removes_only_its_own(self):
        """A breaker gone or replaced meanwhile is left alone."""
        holder = pick.Lock(self.root, "u", "dead", 0)
        self.assertTrue(holder.acquire())
        old = time.time() - 2 * C.STALE_MINUTES * 60
        os.utime(holder.path, (old, old))
        lock = pick.Lock(self.root, "u", "live", 0)
        breaker = f"{lock.path}.break"
        read = pick.read_lock
        calls = []

        def replaced(path):
            calls.append(path)
            if len(calls) == 2:  # inside _break, the breaker is held
                with open(breaker, "w", encoding="utf-8") as file:
                    file.write("another token")
            return read(path)

        with mock.patch.object(pick, "read_lock", replaced):
            self.assertTrue(lock.acquire())
        with open(breaker, encoding="utf-8") as file:
            self.assertEqual(file.read(), "another token")
        os.remove(breaker)

        def vanished(path):
            calls.append(path)
            if len(calls) == 4 and os.path.exists(breaker):
                os.remove(breaker)
            return read(path)

        os.utime(lock.path, (old, old))
        other = pick.Lock(self.root, "u", "third", 0)
        with mock.patch.object(pick, "read_lock", vanished):
            self.assertTrue(other.acquire())

    def full_study(self, model, tower, n_trials):
        """A finished search study."""
        add_trials(search.open_study(self.root, model, tower),
                   [0.1 * i for i in range(n_trials)])

    def test_parked_tower_does_not_block(self):
        """A parked tower is left out of the gate: the other towers of the
        model get their plans; the parked one is terminal."""
        for tower in ("ref", "opt1"):
            self.full_study("fits", tower, 10)
        confirm.park_study(self.root, "fits", "opt2", "test")
        pick.advance(self.root, ["fits"], 10)
        for tower in ("ref", "opt1"):
            self.assertTrue(
                os.path.exists(confirm.plan_path(self.root, "fits", tower)))
        self.assertFalse(
            os.path.exists(confirm.plan_path(self.root, "fits", "opt2")))
        alerts = os.listdir(os.path.join(self.root, "alerts"))
        self.assertTrue(any("fits_opt2" in a for a in alerts))
        self.assertFalse(pick.all_done(self.root, ["fits"], 10))
        for tower in ("ref", "opt1"):
            confirm.park_study(self.root, "fits", tower, "test")
        self.assertTrue(pick.all_done(self.root, ["fits"], 10))

    def test_ready_marker_needs_every_model(self):
        """READY_FOR_TEST is for the whole track, not a worker's models."""
        for tower in C.TOWERS_SEARCHED:
            confirm.park_study(self.root, "tcn", tower, "test")
        pick.advance(self.root, ["tcn"])
        self.assertFalse(os.path.exists(pick.ready_path(self.root)))
        for model in S.LEARNED:
            for tower in C.TOWERS_SEARCHED:
                confirm.park_study(self.root, model, tower, "test")
        pick.advance(self.root, ["tcn"])
        ready = common.read_json(pick.ready_path(self.root))
        self.assertEqual(ready["models"], list(S.LEARNED))

    def test_retry_rule_in_advance(self):
        """advance retries a parked unit and a study parked after its parked
        trials once, after RETRY_SHELVED_AFTER; a second parking is final
        (all_done no longer waits); other parkings are final at once."""
        common.write_json(confirm.plan_path(self.root, "tcn", "opt2"), {
            "configs": [{
                "rank": 0,
                "config": {}
            }],
            "n_seeds": 1,
            "epochs": 10
        })
        unit = pick.Unit("confirm", "tcn", "opt2", 0, 0)
        unit_dir = pick.run_dir(self.root, unit)
        common.park(self.root, "u", unit_dir, "crash", {"events": []}, {})
        confirm.park_study(self.root, "tcn", "ref", "3 trials shelved", True)
        confirm.park_study(self.root, "tcn", "opt1", "no plan")
        self.assertTrue(pick.study_park_final(self.root, "tcn", "opt1"))
        self.assertFalse(pick.study_park_final(self.root, "tcn", "ref"))
        pick.advance(self.root, ["tcn"], 10)  # not an hour yet
        self.assertTrue(pick.unit_parked(self.root, unit))
        self.assertFalse(pick.pending(self.root, unit, 10))
        with mock.patch.object(C, "RETRY_SHELVED_AFTER", 0.0):
            pick.advance(self.root, ["tcn"], 10)
        self.assertFalse(pick.unit_parked(self.root, unit))
        self.assertTrue(pick.pending(self.root, unit, 10))
        self.assertFalse(
            os.path.exists(confirm.parked_path(self.root, "tcn", "ref")))
        self.assertEqual(
            pick.study_state(self.root, "tcn", "ref", 10)["phase"], "search")
        common.park(self.root, "u", unit_dir, "crash", {"events": []}, {})
        confirm.park_study(self.root, "tcn", "ref", "3 trials shelved", True)
        self.assertTrue(pick.study_park_final(self.root, "tcn", "ref"))
        with mock.patch.object(C, "RETRY_SHELVED_AFTER", 0.0):
            pick.advance(self.root, ["tcn"], 10)
        # The unit is shelved for good: so is its study, and the model is
        # done (every study parked for good).
        self.assertTrue(pick.unit_parked(self.root, unit))
        self.assertTrue(pick.study_park_final(self.root, "tcn", "opt2"))
        self.assertTrue(pick.all_done(self.root, ["tcn"], 10))

    def test_pick_skips_finished_units_without_locks(self):
        """Units done, parked, held or busy are skipped before any lock."""
        common.write_json(confirm.plan_path(self.root, "tcn", "opt2"), {
            "configs": [{
                "rank": 0,
                "config": {}
            }],
            "n_seeds": 4,
            "epochs": 10
        })
        dirs = [
            confirm.unit_dir(self.root, "tcn", "opt2", 0, k) for k in range(4)
        ]
        common.write_json(os.path.join(dirs[0], "result.json"), {})
        common.write_json(os.path.join(dirs[1], "PARKED"), {})
        common.hold(self.root, "u", dirs[2], {})
        for tower in ("ref", "opt1"):
            confirm.park_study(self.root, "tcn", tower, "test")
        acquire = pick.Lock.acquire
        names = []

        def counted(lock):
            names.append(os.path.basename(lock.path))
            return acquire(lock)

        with mock.patch.object(pick.Lock, "acquire", counted):
            unit, lock = pick.pick(self.root, "me", 0, ["tcn"], 10)
        lock.release()
        self.assertEqual(unit.seed, 3)
        self.assertEqual(names, ["confirm_tcn_opt2_c0_s3.lock"])

    def worker(self, *extra):
        """worker.main on the test root, without waits."""
        return worker.main([
            f"--root={self.root}", "--models=tcn", "--dry_run", "--owner=w",
            "--poll_seconds=0", *extra
        ])

    def test_worker_breaker_counts_early_studies(self):
        """Units shelved after a resume save do not stop the worker (it
        exits 0 once they are all shelved for good); units of
        WORKER_FAILURES studies shelved early in a row stop it with
        EXIT_BROKEN and an alert, one model being enough."""
        calls = []

        def parked(status):

            def run(args, unit, lock=None):
                del lock
                calls.append(unit)
                confirm.park_study(args.root, unit.model, unit.tower, "test")
                return status

            return run

        with mock.patch.object(worker, "run", parked("parked")):
            self.assertEqual(self.worker(), 0)
        self.assertEqual(len(calls), len(C.TOWERS_SEARCHED))
        calls.clear()
        with mock.patch.object(worker, "run", parked("parked_early")):
            self.assertEqual(self.worker("--models=fits"), C.EXIT_BROKEN)
        self.assertEqual(len({u.tower for u in calls}), C.WORKER_FAILURES)
        run_dir = os.path.join(self.root, "unit")
        early = {"events": [{"saved": False}] * C.MAX_ATTEMPTS}
        common.park(self.root, "u", run_dir, "crash", early, {})
        self.assertEqual(worker._parked(run_dir), "parked_early")
        early["events"][-1] = {"saved": True}
        common.park(self.root, "u", run_dir, "crash", early, {})
        self.assertEqual(worker._parked(run_dir), "parked")
        alerts = os.listdir(os.path.join(self.root, "alerts"))
        self.assertTrue(any("worker_w" in a for a in alerts))
        record = common.read_json(pick.worker_path(self.root, "w"))
        self.assertEqual(record["exit"], C.EXIT_BROKEN)
        self.assertFalse(os.path.exists(os.path.join(self.root, "bad_hosts")))

    def test_driver_errors_set_aside(self):
        """A unit whose driver always raises is set aside after MAX_ATTEMPTS
        errors (an alert) and never starves the others; driver errors never
        stop the worker; a confirmation unit is shelved by them (crashes of
        its attempts.json); the locks are released."""
        calls = []

        def crash(stop_after):

            def run(args, unit, lock=None):
                del lock
                calls.append(pick.unit_id(unit))
                if len(calls) == stop_after:
                    common.write_json(os.path.join(args.root, "STOP"), {})
                raise RuntimeError("driver bug")

            return run

        n_units = len(C.TOWERS_SEARCHED)
        stop_after = n_units * C.MAX_ATTEMPTS
        with mock.patch.object(worker, "run", crash(stop_after)), \
                mock.patch.object(C, "MAX_LOOP_ERRORS", 2):
            self.assertEqual(self.worker(), 0)
        self.assertEqual(len(set(calls)), n_units)
        self.assertEqual(calls[:C.MAX_ATTEMPTS], [calls[0]] * C.MAX_ATTEMPTS)
        self.assertFalse(os.path.exists(os.path.join(self.root, "parked")))
        self.assertEqual([
            n for n in os.listdir(os.path.join(self.root, "locks"))
            if n.endswith(".lock")
        ], [])
        reasons = [
            common.read_json(os.path.join(self.root, "alerts", a))["reason"]
            for a in os.listdir(os.path.join(self.root, "alerts"))
        ]
        self.assertEqual(sum("set aside" in r for r in reasons), n_units)
        os.remove(os.path.join(self.root, "STOP"))
        common.write_json(confirm.plan_path(self.root, "tcn", "opt2"), {
            "configs": [{
                "rank": 0,
                "config": {}
            }],
            "n_seeds": 1,
            "epochs": 10
        })
        for tower in ("ref", "opt1"):
            confirm.park_study(self.root, "tcn", tower, "test")
        calls.clear()
        with mock.patch.object(worker, "run", crash(C.MAX_ATTEMPTS)):
            self.assertEqual(self.worker(), 0)
        unit = pick.Unit("confirm", "tcn", "opt2", 0, 0)
        self.assertEqual(calls.count(pick.unit_id(unit)), C.MAX_ATTEMPTS)
        self.assertTrue(pick.unit_parked(self.root, unit))
        record = common.read_attempts(pick.run_dir(self.root, unit))
        self.assertEqual(record["crashes"], C.MAX_ATTEMPTS)

    def test_worker_errors_do_not_escape(self):
        """An error of the worker's own records is a loop error (the lock
        released); an error outside the loop exits EXIT_REQUEUE with an
        alert."""
        with mock.patch.object(worker, "event", side_effect=OSError("disk")), \
                mock.patch.object(C, "MAX_LOOP_ERRORS", 2):
            self.assertEqual(self.worker(), C.EXIT_REQUEUE)
        self.assertEqual([
            n for n in os.listdir(os.path.join(self.root, "locks"))
            if n.endswith(".lock")
        ], [])
        with mock.patch.object(worker, "_loop", side_effect=KeyError("x")):
            self.assertEqual(self.worker(), C.EXIT_REQUEUE)
        alerts = [
            common.read_json(os.path.join(self.root, "alerts", a))["reason"]
            for a in os.listdir(os.path.join(self.root, "alerts"))
        ]
        self.assertIn("worker stopped: exception", alerts)

    def test_loop_errors_back_off_then_requeue(self):
        """Exceptions of the worker loop are retried; MAX_LOOP_ERRORS in a
        row stop the worker with EXIT_REQUEUE."""
        calls = []

        def failing(*args, **kwargs):
            del args, kwargs
            calls.append(1)
            if len(calls) == 1:
                raise OSError("stale file handle")
            return None, None

        with mock.patch.object(pick, "pick", failing), \
                mock.patch.object(pick, "all_done", return_value=True):
            self.assertEqual(self.worker(), 0)
        self.assertEqual(len(calls), 2)
        self.assertTrue(os.path.exists(pick.ready_path(self.root)))
        with mock.patch.object(pick, "pick", side_effect=OSError("nfs")), \
                mock.patch.object(C, "MAX_LOOP_ERRORS", 3):
            self.assertEqual(self.worker(), C.EXIT_REQUEUE)

    def test_idle_worker_keeps_polling(self):
        """A worker with nothing to pick while work remains alerts once and
        keeps polling until every unit is done; the exit code is in the
        worker record; a record not refreshed is dead."""
        done = iter([False] * 3 + [True] * 100)
        with mock.patch.object(pick, "pick", return_value=(None, None)), \
                mock.patch.object(pick, "all_done",
                                  lambda *a, **k: next(done)):
            self.assertEqual(self.worker("--idle_minutes=0"), 0)
        alerts = [
            common.read_json(os.path.join(self.root, "alerts", a))
            for a in os.listdir(os.path.join(self.root, "alerts"))
        ]
        self.assertEqual(sum("idle" in a["reason"] for a in alerts), 1)
        workers = hpo_status.read_workers(self.root)
        self.assertEqual(workers[0]["state"], "exited (0)")
        path = pick.worker_path(self.root, "x")
        common.write_json(path, {"owner": "x"})
        old = time.time() - 2 * C.STALE_MINUTES * 60
        os.utime(path, (old, old))
        states = {
            w["owner"]: w["state"] for w in hpo_status.read_workers(self.root)
        }
        self.assertEqual(states["x"], "dead")

    def test_ready_written_by_worker(self):
        """READY_FOR_TEST.json is written by a worker that finds every unit
        done, even while the advance lock is held by another process."""
        for model in S.LEARNED:
            for tower in C.TOWERS_SEARCHED:
                confirm.park_study(self.root, model, tower, "test")
        self.assertTrue(pick.Lock(self.root, "advance", "other", 0).acquire())
        self.assertEqual(self.worker(), 0)
        self.assertTrue(os.path.exists(pick.ready_path(self.root)))

    def test_min_gpu_gb(self):
        """A GPU with less memory than --min_gpu_gb stops the worker at
        start with EXIT_BROKEN, one of memory unknown after a few queries
        with EXIT_REQUEUE, each with an alert."""
        with mock.patch.object(common, "gpu_memory_gb", return_value=8.0), \
                mock.patch.object(worker, "_loop") as loop, \
                mock.patch.object(worker, "GPU_QUERY_SECONDS", 0.0):
            self.assertEqual(self.worker("--min_gpu_gb=16"), C.EXIT_BROKEN)
            with mock.patch.object(common, "gpu_memory_gb",
                                   return_value=None) as query:
                self.assertEqual(self.worker("--min_gpu_gb=16"), C.EXIT_REQUEUE)
            self.assertEqual(query.call_count, worker.GPU_QUERY_TRIES)
            loop.assert_not_called()
            loop.return_value = 0
            self.assertEqual(self.worker("--min_gpu_gb=4"), 0)
            self.assertEqual(self.worker(), 0)
        alerts = os.listdir(os.path.join(self.root, "alerts"))
        self.assertEqual(len(alerts), 2)

    def test_config_mismatch_holds_and_goes_on(self):
        """A configuration mismatch holds the unit and the worker goes on
        (no exit code of its own); a held unit is not pending."""
        with mock.patch.object(worker, "run", return_value="config_mismatch"):
            self.assertEqual(self.worker("--max_units=2"), 0)
        unit = pick.Unit("final", "tcn", "opt2", None, 0)
        run_dir = pick.run_dir(self.root, unit)
        self.assertTrue(pick.pending(self.root, unit, 10))
        common.hold(self.root, "u", run_dir, {})
        self.assertFalse(pick.pending(self.root, unit, 10))

    def test_advance_lock_short_staleness(self):
        """The advance lock of a killed holder is taken over after
        ADVANCE_STALE_SECONDS, not STALE_MINUTES."""
        dead = pick.Lock(self.root, "advance", "dead", 0)
        self.assertTrue(dead.acquire())
        old = time.time() - 2 * C.ADVANCE_STALE_SECONDS
        os.utime(dead.path, (old, old))
        pick.advance(self.root, ["tcn"], 10)
        self.assertTrue(
            os.path.exists(os.path.join(self.root, "locks", "takeovers.log")))
        self.assertFalse(os.path.exists(dead.path))

    def test_hours_left_with_zero_cost(self):
        """A model of cost 0 does not divide by zero."""
        state = {
            "model": "tcn",
            "hours_per_trial": 0.5,
            "cost": 0.0,
            "parked": False,
            "target": 10,
            "counted": 4,
            "plan": False,
            "final_units": [],
            "confirm_units": [],
            "work_left": 0.0
        }
        hours = hpo_status.hours_left(state)
        self.assertGreater(hours, 0.5 * 6)


class AnalyzeTest(unittest.TestCase):
    """The leaderboard of the track on synthetic sealed runs."""

    def test_families(self):
        """Every learned model has one of the eight families."""
        self.assertEqual(set(S.FAMILIES), set(S.LEARNED))
        self.assertEqual(len(set(S.FAMILIES.values())), 8)

    def check_gpu_hours(self, out: str) -> None:
        """gpu_hours.csv of the dry run: every phase, consistent totals."""
        hours = pd.read_csv(os.path.join(out, "gpu_hours.csv"))
        self.assertEqual(set(hours["phase"]),
                         {"search", "confirm", "final", "test", "all"})
        total = hours[(hours["model"] == "all") &
                      (hours["phase"] == "all")]["gpu_hours"].iloc[0]
        per_unit = hours[(hours["model"] != "all") &
                         (hours["tower"] != "all")]["gpu_hours"].sum()
        self.assertAlmostEqual(total, per_unit)
        self.assertGreater(total, 0)

    def test_synthetic_leaderboard(self):
        """Medians over seeds, means over towers, rankings and configs."""
        with tempfile.TemporaryDirectory() as out:
            boards = analyze.main(
                ["--dry_run", f"--out={out}", "--num_resamples=0"])
            board = boards["primary_last"]
            self.assertEqual(set(boards), set(analyze.VARIANTS))
            self.assertEqual(analyze.VARIANTS[0], "primary_last")
            self.assertEqual(set(board["model"]), set(S.LEARNED))
            self.assertTrue((board["missing_towers"].fillna("") == "").all())
            for column in ("r2_log_damage_base", "r2_log_damage_z078",
                           "r2_log_damage_top", "r2_log_damage_mean11",
                           "fraction_within_factor2_mean11",
                           "within_condition_correlation_top"):
                self.assertIn(column, board.columns)
            # The synthetic noise grows along LEARNED: tcn is the best.
            self.assertEqual(board.iloc[0]["model"], S.LEARNED[0])
            folder = os.path.join(out, "primary_last")
            towers = pd.read_csv(os.path.join(folder, "per_tower.csv"))
            scores = pd.read_csv(os.path.join(folder, "scores.csv"))
            subset = scores[(scores["model"] == "fits") &
                            (scores["gauge"] == "tower_top") &
                            (scores["group"] == "all")]
            medians = subset.groupby("tower")["r2_log_damage"].median()
            expected = towers[(towers["model"] == "fits") &
                              (towers["position"] == "top")].set_index(
                                  "tower")["r2_log_damage"]
            np.testing.assert_allclose(medians.sort_index(),
                                       expected.sort_index())
            row = board.set_index("model").loc["fits"]
            self.assertAlmostEqual(row["r2_log_damage_top"], medians.mean())
            top3 = pd.read_csv(os.path.join(folder, "top3.csv"))
            self.assertEqual(len(top3), 3 * len(analyze.CRITERIA))
            families = pd.read_csv(os.path.join(folder, "families.csv"))
            self.assertEqual(len(families), 8 * len(analyze.CRITERIA))
            self.assertTrue(os.path.exists(os.path.join(folder,
                                                        "by_group.csv")))
            configs = pd.read_csv(os.path.join(out, "configs.csv"))
            self.assertEqual(len(configs), 20 * 3)
            for column in ("n_trials_counted", "target", "stopped_early"):
                self.assertIn(column, configs.columns)
            self.check_gpu_hours(out)

    def test_only_sealed_runs_are_scored(self):
        """A run without a SEALED marker, or with a skipped one, is not
        scored: its tower has fewer seeds, counts as missing and the model
        is unranked (its score kept)."""
        with tempfile.TemporaryDirectory() as out:
            root = os.path.join(out, "synthetic")
            dataset = analyze.write_synthetic(root, np.random.default_rng(0))
            sealed = os.path.join(root, "sealed", "primary_last", "ref")
            common.write_json(os.path.join(sealed, "seed0", "SEALED_fits.json"),
                              {"skipped": "diverged"})
            os.remove(os.path.join(sealed, "seed1", "SEALED_tcn.json"))
            args = analyze.parse_args([
                f"--root={root}", f"--dataset_dir={dataset}", f"--out={out}",
                "--num_resamples=0"
            ])
            with self.assertRaises(SystemExit) as error:
                analyze.build(args)
            self.assertIn("tcn/ref/s1", str(error.exception))
            common.write_json(os.path.join(sealed, "seed1", "SEALED_tcn.json"),
                              {"skipped": "diverged"})
            board = analyze.build(
                analyze.parse_args([
                    f"--root={root}", f"--dataset_dir={dataset}",
                    f"--out={out}", "--num_resamples=0"
                ]))["primary_last"].set_index("model")
        for model in ("fits", "tcn"):
            self.assertEqual(board.loc[model, "n_seeds_ref"], C.N_SEEDS - 1)
            self.assertEqual(board.loc[model, "missing_towers"], "ref")
            self.assertFalse(board.loc[model, "ranked"])
            self.assertFalse(np.isnan(board.loc[model, "r2_log_damage_top"]))
        self.assertTrue(board.loc["lstm", "ranked"])

    def test_missing_tower_and_ties(self):
        """A parked tower, or one scored with fewer than N_SEEDS seeds, is
        named as missing (the latter keeps its score, unranked); rankings
        are stable, ties broken by the model name."""
        rows = []
        for model, towers in (("tcn", C.TOWERS_SEARCHED),
                              ("fits", ("ref", "opt1")), ("lstm",
                                                          C.TOWERS_SEARCHED)):
            for tower in towers:
                for position in ("base", "z078", "top", "mean11"):
                    rows.append({
                        "model":
                            model,
                        "tower":
                            tower,
                        "group":
                            "all",
                        "position":
                            position,
                        "n_seeds":
                            C.N_SEEDS - ((model, tower) == ("tcn", "ref")),
                        **{
                            m: 0.5 for m in analyze.METRICS
                        }
                    })
        board = analyze.leaderboard(pd.DataFrame(rows))
        board = analyze.with_missing(board)
        row = board.set_index("model")
        self.assertEqual(row.loc["fits", "missing_towers"], "opt2")
        self.assertEqual(row.loc["fits", "n_towers"], 2)
        self.assertEqual(row.loc["mamba", "missing_towers"], "ref,opt1,opt2")
        self.assertEqual(row.loc["tcn", "missing_towers"], "ref")
        self.assertEqual(row.loc["tcn", "n_seeds_ref"], C.N_SEEDS - 1)
        self.assertAlmostEqual(row.loc["tcn", "r2_log_damage_top"], 0.5)
        # Only the models complete on every tower are ranked; the others
        # follow by name.
        self.assertEqual(list(board["model"][:2]),
                         ["lstm", sorted(set(S.LEARNED) - {"lstm"})[0]])
        self.assertEqual(list(board["ranked"][:2]), [True, False])
        self.assertFalse(row.loc["fits", "ranked"])
        self.assertFalse(row.loc["tcn", "ranked"])
        board = board.assign(r2_log_damage_top=0.5)
        board.loc[board["model"] == "tcn", "ranked"] = True
        board.loc[board["model"] == "tcn", "n_towers"] = 3
        top3, families = analyze.rankings(
            board.drop(columns="group", errors="ignore"))
        self.assertEqual(list(top3[top3["criterion"] == "top"]["model"]),
                         ["lstm", "tcn"])
        self.assertTrue((top3["n_towers"] == 3).all())
        self.assertNotIn("fits", set(families["model"]))

    def test_curves_and_gpu_hours_of_a_search(self):
        """Curves use the counted position of counted trials; the dry run
        records an attempt per trial."""
        with tempfile.TemporaryDirectory() as root:
            study = search.main([
                "--model=tcn", "--tower=opt2", "--dry_run", f"--root={root}",
                "--n_trials=8"
            ])
            study.add_trial(
                optuna.trial.create_trial(state=optuna.trial.TrialState.FAIL))
            curves = analyze.curves(root)
            self.assertEqual(list(curves["position"]), list(range(1, 9)))
            hours = analyze.gpu_hours(root)
            row = hours[(hours["model"] == "tcn") &
                        (hours["phase"] == "search")]
            self.assertEqual(int(row["runs"].iloc[0]), 8)
            self.assertEqual(int(row["attempts"].iloc[0]), 8)


class RobustnessTest(unittest.TestCase):  # pylint: disable=too-many-public-methods
    """Lock beats, out of memory, claims, extension decisions, final
    parkings, attempts and the test records."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def beat_until(self, beater, condition, seconds=5.0):
        """Runs `beater` until `condition()` or `seconds`."""
        end = time.time() + seconds
        with beater:
            while not condition() and time.time() < end:
                time.sleep(0.01)

    def test_beater_survives_transient_errors(self):
        """OSErrors of utime and unreadable lock files are retried: the
        lock is not lost; another token loses it at once; a lock unclear
        LOCK_BEAT_FAILURES times in a row is given up."""
        lock = pick.Lock(self.root, "u", "me", 0)
        self.assertTrue(lock.acquire())
        utime, read = os.utime, pick.read_lock
        calls = {"utime": 0, "read": 0}

        def flaky_utime(*args, **kwargs):
            calls["utime"] += 1
            if calls["utime"] <= 3:
                raise OSError("stale file handle")
            return utime(*args, **kwargs)

        def flaky_read(path):
            calls["read"] += 1
            return {} if calls["read"] in (5, 6) else read(path)

        lost = []
        beater = pick.Beater(lock, lambda: lost.append(1), interval=0.01)
        with mock.patch.object(pick.os, "utime", flaky_utime), \
                mock.patch.object(pick, "read_lock", flaky_read):
            self.beat_until(beater, lambda: calls["read"] > 12)
        self.assertFalse(beater.lost)
        self.assertEqual(lost, [])
        self.assertTrue(lock.owned())
        answers = [{}]
        with mock.patch.object(
                pick, "read_lock", lambda path: answers.pop()
                if answers else read(path)):
            self.assertTrue(lock.owned(wait=0.0))
        # Taken over: readable, another token.
        common.write_json(lock.path, {"token": "other"})
        beater = pick.Beater(lock, lambda: lost.append(1), interval=0.01)
        self.beat_until(beater, lambda: beater.lost)
        self.assertTrue(beater.lost)
        self.assertEqual(beater.failures, 0)
        self.assertFalse(lock.owned(wait=0.0))
        # Missing for good: given up after LOCK_BEAT_FAILURES beats.
        os.remove(lock.path)
        beater = pick.Beater(lock, lambda: lost.append(1), interval=0.01)
        self.beat_until(beater, lambda: beater.lost)
        self.assertTrue(beater.lost)
        self.assertEqual(beater.failures, C.LOCK_BEAT_FAILURES)

    def search(self, *extra, tower="opt2"):
        """search.main in dry-run mode on the test root."""
        return search.main([
            "--model=tcn", f"--tower={tower}", "--dry_run",
            f"--root={self.root}", "--n_trials=10", *extra
        ])

    def test_oom_trials(self):
        """An out-of-memory trial is FAIL (oom), not counted; MAX_OOM of
        them stop the study, which is then parked (not on the host)."""
        stub = common.stub_unit
        calls = {"n": 0}

        def two_oom(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] <= 2:
                return {"status": "oom", "params": 10, "tail": ""}
            return stub(*args, **kwargs)

        with mock.patch.object(common, "stub_unit", two_oom):
            study = self.search()
        ooms = [t for t in study.trials if search.oom(t)]
        self.assertEqual(len(ooms), 2)
        self.assertTrue(all(t.state.name == "FAIL" for t in ooms))
        self.assertEqual(sum(search.counted(t) for t in study.trials), 10)
        oom = {"status": "oom", "params": 10, "tail": ""}
        with mock.patch.object(common, "stub_unit", lambda *a, **k: oom):
            study = self.search("--max_new=100", tower="ref")
        self.assertEqual(len(study.trials), C.MAX_OOM)
        state = pick.study_state(self.root, "tcn", "ref", 10)
        self.assertFalse(state["search_pending"])
        reason, retry = pick.park_reason(state)
        self.assertIn("out of memory", reason)
        self.assertTrue(retry)  # retried once, counting new OOMs only
        common.write_json(search.retried_path(self.root, "tcn", "ref"),
                          {"after_trial": C.MAX_OOM - 2})
        draws = search.failed_draws(self.root, "tcn", "ref", study.trials)
        self.assertEqual(draws["oom"], 1)
        # Once the budget is counted, failed draws no longer park (P4).
        self.assertIsNone(pick.park_reason({**state, "counted": 10}))

    def plan(self, n_seeds=1):
        """A frozen one-configuration plan of tcn/opt2."""
        common.write_json(
            confirm.plan_path(self.root, "tcn", "opt2"), {
                "configs": [{
                    "rank": 0,
                    "config": S.FIXED_RECIPE
                }],
                "n_seeds": n_seeds,
                "epochs": 10
            })

    def worker_args(self):
        """Worker options of the test root."""
        return worker.parse_args(
            [f"--root={self.root}", "--models=tcn", "--dry_run", "--owner=w"])

    def test_claims_of_others_are_kept(self):
        """A worker releases only its own claims (any incarnation); a claim
        of a live process by hand makes the unit busy."""
        self.plan()
        unit = pick.Unit("confirm", "tcn", "opt2", 0, 0)
        run_dir = pick.run_dir(self.root, unit)
        os.makedirs(run_dir)
        path = os.path.join(run_dir, "claim")
        with open(path, "w", encoding="utf-8") as file:
            file.write("|elsewhere:1")
        common.touch(run_dir)
        self.assertFalse(common.release(run_dir, owner="w"))
        self.assertTrue(common.busy(run_dir))
        self.assertEqual(worker.run(self.worker_args(), unit), "busy")
        self.assertTrue(os.path.exists(path))
        with open(path, "w", encoding="utf-8") as file:
            file.write("w|elsewhere:1")  # an earlier incarnation
        self.assertTrue(common.release(run_dir, owner="w"))
        self.assertFalse(os.path.exists(path))
        self.assertEqual(worker.run(self.worker_args(), unit), "done")

    def full_study(self, tower, values):
        """A finished fits study with these values."""
        add_trials(search.open_study(self.root, "fits", tower), values)

    def test_extension_decided_once(self):
        """The gate waits while a tower is parked with its retry pending,
        then writes the extension decision once ('not extended' too); a
        tower retried later does not reopen it, and a tower parked for good
        never decides."""
        late = [0.0] * 9 + [1.0]
        with mock.patch.object(C, "EXTEND_AFTER", 5):
            confirm.park_study(self.root, "fits", "opt2", "test", True)
            self.full_study("opt2", late)
            self.assertFalse(search.maybe_extend(self.root, "fits", 10))
            self.full_study("ref", [1.0] + [0.0] * 9)
            self.full_study("opt1", [1.0] + [0.0] * 9)
            pick.advance(self.root, ["fits"], 10)
            self.assertIsNone(search.extension_decision(self.root, "fits"))
            os.remove(confirm.parked_path(self.root, "fits", "opt2"))
            confirm.park_study(self.root, "fits", "opt2", "test")  # final
            pick.advance(self.root, ["fits"], 10)
            decision = common.read_json(search.extension_path(
                self.root, "fits"))
            self.assertFalse(decision["extended"])
            self.assertEqual(decision["towers"], ["ref", "opt1"])
            os.remove(confirm.parked_path(self.root, "fits", "opt2"))
            self.assertFalse(search.maybe_extend(self.root, "fits", 10))
            pick.advance(self.root, ["fits"], 10)
            self.assertEqual(search.target_trials(self.root, "fits", 10), 10)

    def test_parking_final_and_frozen(self):
        """A study parked after its trials is not terminal (all_done waits
        for its retry); once the test is ready, every parking is final and
        nothing is retried."""
        for tower in C.TOWERS_SEARCHED:
            confirm.park_study(self.root, "tcn", tower, "test", True)
        self.assertFalse(pick.all_done(self.root, ["tcn"], 10))
        common.write_json(pick.ready_path(self.root), {})
        self.assertTrue(pick.all_done(self.root, ["tcn"], 10))
        with mock.patch.object(C, "RETRY_SHELVED_AFTER", 0.0):
            pick.advance(self.root, ["tcn"], 10)
        self.assertTrue(
            os.path.exists(confirm.parked_path(self.root, "tcn", "ref")))

    def test_held_trial_is_not_frozen_or_parked(self):
        """A study with a trial held for an operator gets no plan and is not
        parked; the other towers go on."""
        for tower in C.TOWERS_SEARCHED:
            self.full_study(tower, [0.1 * i for i in range(10)])
        study = search.open_study(self.root, "fits", "opt2")
        trial = study.ask()
        run_dir = os.path.join(self.root, "trials", "held")
        trial.set_user_attr("run_dir", run_dir)
        common.hold(self.root, "u", run_dir, {})
        pick.advance(self.root, ["fits"], 10)
        for tower in ("ref", "opt1"):
            self.assertTrue(
                os.path.exists(confirm.plan_path(self.root, "fits", tower)))
        self.assertFalse(
            os.path.exists(confirm.plan_path(self.root, "fits", "opt2")))
        self.assertFalse(
            os.path.exists(confirm.parked_path(self.root, "fits", "opt2")))
        self.assertEqual(
            pick.study_state(self.root, "fits", "opt2", 10)["phase"], "held")

    def test_capped_after_base_budget_freezes(self):
        """A study with its N_TRIALS counted trials that stops drawing (here
        during an extension) is not parked: its plan is frozen from the
        counted trials."""
        common.write_json(search.extension_path(self.root, "fits"),
                          {"extended": True})
        for tower in C.TOWERS_SEARCHED:
            self.full_study(tower, [0.1 * i for i in range(10)])
            study = search.open_study(self.root, "fits", tower)
            for _ in range(2):
                study.add_trial(
                    optuna.trial.create_trial(
                        state=optuna.trial.TrialState.FAIL,
                        user_attrs={"over_cap": True}))
        with mock.patch.object(C, "MAX_OVER_CAP", 2):
            state = pick.study_state(self.root, "fits", "ref", 10)
            self.assertEqual(state["target"], 10 + C.EXTEND_BY)
            self.assertFalse(state["search_pending"])
            self.assertIsNone(pick.park_reason(state))
            pick.advance(self.root, ["fits"], 10)
        plan = common.read_json(confirm.plan_path(self.root, "fits", "ref"))
        self.assertIn("parameter cap", plan["stopped_early"])
        self.assertEqual(plan["n_trials_counted"], 10)
        self.assertFalse(
            os.path.exists(confirm.parked_path(self.root, "fits", "ref")))

    def test_open_test_keeps_missing_pairs_out(self):
        """A pair recorded as parked in MISSING.json is never scored (even
        if retried since); SKIPPED.json lists the diverged seeds."""
        for model in S.LEARNED:
            for tower in C.TOWERS_SEARCHED:
                if (model, tower) not in (("tcn", "opt2"), ("tcn", "ref")):
                    confirm.park_study(self.root, model, tower, "test")
        common.write_json(os.path.join(self.root, "sealed", "MISSING.json"),
                          {"parked": {
                              "tcn/opt2": {}
                          }})
        common.write_json(
            confirm.winner_path(self.root, "tcn", "ref"), {
                "winner_config": S.FIXED_RECIPE,
                "winner_median": 0.5,
                "margin_to_second": None,
                "best_epoch": 10
            })
        for seed in range(C.N_SEEDS):
            common.write_json(
                os.path.join(final.unit_dir(self.root, "tcn", "ref", seed),
                             "result.json"), {
                                 "status": "diverged",
                                 "epochs": 10,
                                 "best_epoch": 10
                             })
        common.write_json(pick.ready_path(self.root), {})
        final.main(["--open_test", f"--root={self.root}", "--dry_run"])
        self.assertFalse(
            os.path.exists(
                os.path.join(
                    final.sealed_dir(self.root, "primary_last", "opt2", 0),
                    "SEALED_tcn.json")))
        skipped = common.read_json(
            os.path.join(self.root, "sealed", "SKIPPED.json"))
        self.assertEqual(skipped["all_seeds_diverged"], ["tcn/ref"])
        self.assertEqual(len(skipped["seeds"]), C.N_SEEDS)
        configs = analyze.configs(self.root).set_index(["model", "tower"])
        self.assertEqual(configs.loc[("tcn", "ref"), "status"],
                         "all_seeds_diverged")
        # A pair shelved after its winner is 'shelved' (parked first).
        confirm.park_study(self.root, "tcn", "ref", "late")
        configs = analyze.configs(self.root).set_index(["model", "tower"])
        self.assertEqual(configs.loc[("tcn", "ref"), "status"], "shelved")

    def test_open_attempts_and_killed(self):
        """An attempt is recorded open ('running') while it runs, updated by
        the heartbeat; an open attempt left by a hard kill is 'killed' and
        its seconds count; no progress over attempts alerts once."""
        run_dir = os.path.join(self.root, "unit")
        script = ("import json, os, sys, time\n"
                  "time.sleep(0.5)\n"
                  "e = json.load(open(os.path.join(sys.argv[1], "
                  "'attempts.json')))['events'][-1]\n"
                  "print(e['status'], e['seconds'])\n")
        with mock.patch.object(common, "HEARTBEAT_SECONDS", 0.1):
            result = common.run_unit("u",
                                     [sys.executable, "-c", script, run_dir],
                                     run_dir, self.root, "tcn")
        status, seconds = result["tail"].split()
        self.assertEqual(status, "running")
        self.assertGreater(float(seconds), 0.0)
        old = datetime.datetime.now() - datetime.timedelta(minutes=2 *
                                                           C.STALE_MINUTES)
        event = {
            "start": old.isoformat(),
            "end": old.isoformat(),
            "seconds": 3600.0,
            "status": "running"
        }
        self.assertEqual(common.event_status(event), "killed")
        common.write_json(
            os.path.join(self.root, "trials", "tcn_opt2", "t000",
                         "attempts.json"), {"events": [event]})
        hours = analyze.gpu_hours(self.root)
        row = hours[(hours["model"] == "tcn") & (hours["phase"] == "search")]
        self.assertEqual(int(row["killed"].iloc[0]), 1)
        self.assertAlmostEqual(row["gpu_hours"].iloc[0], 1.0)
        record = {"events": [{"status": "preempted", "saved": False}] * 3}
        self.assertTrue(common.check_progress(self.root, "u", run_dir, record))
        self.assertFalse(common.check_progress(self.root, "u", run_dir, record))
        record = {"events": [{"status": "preempted", "saved": True}] * 3}
        self.assertFalse(common.check_progress(self.root, "u", run_dir, record))

    def test_test_inference_is_recorded(self):
        """The inference runs of the test are recorded (phase 'test' of
        gpu_hours.csv)."""
        run_dir = final.unit_dir(self.root, "tcn", "opt2", 0)
        common.write_json(os.path.join(run_dir, "result.json"), {
            "status": "ok",
            "epochs": 10,
            "best_epoch": 10
        })
        with open(os.path.join(run_dir, "tcn_fa.pt"), "w",
                  encoding="utf-8") as file:
            file.write("weights")
        args = final.parse_args(["--open_test", f"--root={self.root}"])
        out = final.sealed_dir(self.root, "primary_last", "opt2", 0)
        common.write_text(os.path.join(out, "log_tcn.txt"), "earlier\n")
        test_dir = final.test_run_dir(self.root, "tcn", "opt2", "primary_last",
                                      0)
        seen = []

        def call(*args, **kwargs):
            del args, kwargs
            seen.append(common.read_attempts(test_dir)["events"][-1]["status"])
            return int(len(seen) == 1)  # fails, then succeeds

        with mock.patch.object(final.subprocess, "call", call):
            final.score(args, "tcn", "opt2", 0, "primary_last")
            marker = os.path.join(out, "SEALED_tcn.json")
            self.assertFalse(os.path.exists(marker))
            failures = common.read_json(
                os.path.join(self.root, "sealed", "TEST_FAILURES.json"))
            self.assertEqual(len(failures), 1)
            final.score(args, "tcn", "opt2", 0, "primary_last")
            self.assertTrue(os.path.exists(marker))
        self.assertEqual(seen, ["running", "running"])
        with open(os.path.join(out, "log_tcn.txt"), encoding="utf-8") as file:
            self.assertTrue(file.read().startswith("earlier"))
        self.assertEqual(
            common.read_attempts(test_dir)["events"][-1]["status"], "ok")
        hours = analyze.gpu_hours(self.root)
        row = hours[(hours["model"] == "tcn") & (hours["phase"] == "test")]
        self.assertGreaterEqual(int(row["attempts"].iloc[0]), 1)

    def test_torn_journal_and_unreadable_study(self):
        """A torn last journal line is cut (a backup kept) when the study is
        opened for writing; a study damaged mid-file is skipped with one
        alert while the others go on, and the model is never done."""
        study = search.open_study(self.root, "tcn", "opt2")
        add_trials(study, [0.1, 0.2])
        path = search.journal_path(self.root, "tcn", "opt2")
        with open(path, "ab") as file:
            file.write(b'{"op_code": 5, "work')
        study = search.open_study(self.root, "tcn", "opt2")
        study.ask()
        self.assertEqual(
            len(search.open_study(self.root, "tcn", "opt2", False).trials), 3)
        backups = [
            n for n in os.listdir(os.path.dirname(path)) if ".torn." in n
        ]
        self.assertEqual(len(backups), 1)
        with open(path, "rb") as file:
            data = file.read()
        with open(path, "wb") as file:
            file.write(b"garbage\n" + data)
        with mock.patch.object(pick, "_BROKEN", set()):
            for _ in range(2):
                states = pick.all_states(self.root, ["tcn"], 10)
                self.assertEqual(len(states), len(C.TOWERS_SEARCHED) - 1)
                pick.advance(self.root, ["tcn"], 10)
            for tower in ("ref", "opt1"):
                confirm.park_study(self.root, "tcn", tower, "test")
            self.assertFalse(pick.all_done(self.root, ["tcn"], 10))
        reasons = [
            common.read_json(os.path.join(self.root, "alerts", a))["reason"]
            for a in os.listdir(os.path.join(self.root, "alerts"))
        ]
        self.assertEqual(reasons.count("study unreadable, skipped"), 1)
        self.assertEqual(reasons.count("advance failed, skipped"), 1)

    def test_own_claim_and_dead_process(self):
        """A claim of the same owner (another incarnation) is not busy for
        it; a claim or a lock of a dead process of this host is stale at
        once."""
        self.plan()
        unit = pick.Unit("confirm", "tcn", "opt2", 0, 0)
        run_dir = pick.run_dir(self.root, unit)
        os.makedirs(run_dir)
        path = os.path.join(run_dir, "claim")
        common.write_text(path, "w|elsewhere:1")
        common.touch(run_dir)
        self.assertTrue(common.busy(run_dir))
        self.assertFalse(common.busy(run_dir, "w"))
        self.assertFalse(pick.pending(self.root, unit, 10))
        self.assertTrue(pick.pending(self.root, unit, 10, "w"))
        with subprocess.Popen([sys.executable, "-c", ""]) as process:
            process.wait()
        dead = f"{socket.gethostname()}:{process.pid}"
        self.assertTrue(common.dead_process(dead))
        self.assertFalse(
            common.dead_process(f"{socket.gethostname()}:"
                                f"{os.getpid()}"))
        common.write_text(path, f"x|{dead}")
        self.assertFalse(common.busy(run_dir))
        self.assertTrue(common.claim(run_dir))
        holder = pick.Lock(self.root, "u", "other", 0)
        self.assertTrue(holder.acquire())
        self.assertFalse(pick.Lock(self.root, "u", "me", 0).acquire())
        common.write_json(holder.path, {
            **pick.read_lock(holder.path), "pid": process.pid
        })
        self.assertTrue(pick.Lock(self.root, "u", "me", 0).acquire())

    def test_operator_retry(self):
        """An operator retries a unit or a study at once, even after a final
        parking, with fresh counts; nothing once the test is ready."""
        self.plan()
        unit = pick.Unit("confirm", "tcn", "opt2", 0, 0)
        run_dir = pick.run_dir(self.root, unit)
        record = {"crashes": C.MAX_ATTEMPTS, "free": 0, "events": []}
        common.write_json(common.attempts_path(run_dir), record)
        common.park(self.root, "u", run_dir, "crash", record, {})
        with mock.patch.object(C, "RETRY_SHELVED_AFTER", 0.0):
            self.assertTrue(common.retry_unit(self.root, "u", run_dir))
        common.park(self.root, "u", run_dir, "crash", record, {})
        self.assertTrue(common.park_final(run_dir))
        confirm.park_study(self.root, "tcn", "opt2",
                           f"unit shelved for good: {pick.unit_id(unit)}")
        self.assertTrue(pick.operator_retry(self.root, pick.unit_id(unit)))
        self.assertFalse(pick.unit_parked(self.root, unit))
        self.assertEqual(common.read_attempts(run_dir)["crashes"], 0)
        self.assertFalse(
            os.path.exists(confirm.parked_path(self.root, "tcn", "opt2")))
        confirm.park_study(self.root, "tcn", "ref", "no plan")
        self.assertTrue(pick.operator_retry(self.root, "search/tcn_ref"))
        self.assertFalse(
            os.path.exists(confirm.parked_path(self.root, "tcn", "ref")))
        self.assertFalse(pick.operator_retry(self.root, "search/tcn_ref"))
        common.write_json(pick.ready_path(self.root), {})
        with self.assertRaises(SystemExit):
            pick.operator_retry(self.root, "search/tcn_ref")

    def test_extension_capped_retried_once(self):
        """An extension capped by shelved trials parks the study for one
        delayed retry; capped again after it, the plan is frozen."""
        common.write_json(search.extension_path(self.root, "fits"),
                          {"extended": True})

        def shelved(tower, n):
            study = search.open_study(self.root, "fits", tower)
            for _ in range(n):
                study.add_trial(
                    optuna.trial.create_trial(
                        state=optuna.trial.TrialState.FAIL,
                        user_attrs={"parked": True}))

        for tower in C.TOWERS_SEARCHED:
            self.full_study(tower, [0.1 * i for i in range(10)])
        shelved("ref", C.PARK_STUDY_AFTER)
        pick.advance(self.root, ["fits"], 10)
        marker = common.read_json(confirm.parked_path(self.root, "fits", "ref"))
        self.assertTrue(marker["retry"])
        self.assertFalse(
            os.path.exists(confirm.plan_path(self.root, "fits", "ref")))
        with mock.patch.object(C, "RETRY_SHELVED_AFTER", 0.0):
            pick.advance(self.root, ["fits"], 10)
        state = pick.study_state(self.root, "fits", "ref", 10)
        self.assertEqual((state["phase"], state["capped"]), ("search", None))
        shelved("ref", C.PARK_STUDY_AFTER)
        pick.advance(self.root, ["fits"], 10)
        plan = common.read_json(confirm.plan_path(self.root, "fits", "ref"))
        self.assertIn("trials shelved", plan["stopped_early"])
        self.assertEqual(plan["target"], 10 + C.EXTEND_BY)
        self.assertFalse(
            os.path.exists(confirm.parked_path(self.root, "fits", "ref")))


if __name__ == "__main__":
    unittest.main()
