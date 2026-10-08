# pylint: disable=wrong-import-position,protected-access
# pylint: disable=use-dict-literal
"""Tests of the drivers of the validation-tuned track (hpo/).

Run from the repository root with `python -m unittest discover tests`.
CPU only; nothing is trained: the drivers run with their --dry_run stub
and the failure handling with tiny stand-in commands. The Optuna tests need
optuna >= 4 and are skipped otherwise.
"""

import argparse
import os
import random
import signal
import socket
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
            "crash")

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

    def test_config_mismatch_parks_at_once(self):
        """A refused resume is parked after one attempt and is not
        un-parked (not a fault of the host)."""
        with tempfile.TemporaryDirectory() as root:
            result = self.run_fake(
                root, f"import sys; sys.exit({common.EXIT_CONFIG_MISMATCH})")
            self.assertEqual(result["status"], "parked")
            run_dir = os.path.join(root, "unit")
            self.assertEqual(len(common.read_attempts(run_dir)["events"]), 1)
            self.assertFalse(common.can_unpark(run_dir, "elsewhere"))

    def test_unpark_once_from_another_host(self):
        """A unit parked by crashes on one host is un-parked once, by
        another host only, with fresh attempt counts."""
        with tempfile.TemporaryDirectory() as root:
            self.run_fake(root, "raise SystemExit(1)")
            run_dir = os.path.join(root, "unit")
            marker = common.read_json(os.path.join(run_dir, "PARKED"))
            self.assertEqual(marker["hosts"], [socket.gethostname()])
            self.assertFalse(common.can_unpark(run_dir, socket.gethostname()))
            self.assertTrue(common.unpark(run_dir, "elsewhere"))
            self.assertFalse(os.path.exists(os.path.join(run_dir, "PARKED")))
            self.assertEqual(common.read_attempts(run_dir)["crashes"], 0)
            self.run_fake(root, "raise SystemExit(1)")
            self.assertTrue(os.path.exists(os.path.join(run_dir, "PARKED")))
            self.assertFalse(common.can_unpark(run_dir, "elsewhere"))
            self.assertTrue(pick.park_final(run_dir))

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
                  "raise MemoryError('CUDA out of memory')\n")
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
        with self.assertRaises(SystemExit):
            final.main(test)  # tcn/opt2 not trained yet
        final.main(
            ["--model=tcn", "--tower=opt2", f"--root={self.root}", "--dry_run"])
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

    def test_pick_unparks_single_host_failures(self):
        """A unit and a study parked by failures on one other host are
        un-parked once by the next pick."""
        common.write_json(confirm.plan_path(self.root, "tcn", "opt2"), {
            "configs": [{
                "rank": 0,
                "config": {}
            }],
            "n_seeds": 1,
            "epochs": 10
        })
        unit_dir = confirm.unit_dir(self.root, "tcn", "opt2", 0, 0)
        common.write_json(
            os.path.join(unit_dir, "attempts.json"), {
                "crashes": 3,
                "free": 0,
                "events": [{
                    "host": "elsewhere",
                    "status": "crash"
                }] * 3
            })
        common.write_json(os.path.join(unit_dir, "PARKED"),
                          {"hosts": ["elsewhere"]})
        confirm.park_study(self.root, "tcn", "ref", "3 trials parked",
                           ["elsewhere"])
        lock = pick._try(self.root, pick.Unit("confirm", "tcn", "opt2", 0, 0),
                         "me", 0, 10)
        self.assertIsNotNone(lock)
        lock.release()
        self.assertFalse(os.path.exists(os.path.join(unit_dir, "PARKED")))
        self.assertTrue(common.read_attempts(unit_dir)["unparked"])
        unit, lock = pick.pick(self.root, "me", 0, ["tcn"], 10)
        self.assertEqual(unit.phase, "search")
        lock.release()
        self.assertFalse(
            os.path.exists(confirm.parked_path(self.root, "tcn", "ref")))
        state = pick.study_state(self.root, "tcn", "ref", 10)
        self.assertTrue(state["unparked"])
        self.assertEqual(state["phase"], "search")

    def test_worker_breaker(self):
        """A worker whose units keep parking stops with EXIT_BROKEN and an
        alert, after WORKER_FAILURES units."""
        calls = []

        def parked(args, unit, lock=None):
            del args, lock
            calls.append(unit)
            return "parked"

        with mock.patch.object(worker, "run", parked):
            code = worker.main([
                f"--root={self.root}", "--models=tcn", "--dry_run", "--owner=w"
            ])
        self.assertEqual(code, C.EXIT_BROKEN)
        self.assertEqual(len(calls), C.WORKER_FAILURES)
        alerts = os.listdir(os.path.join(self.root, "alerts"))
        self.assertTrue(any(a.endswith("worker_w.json") for a in alerts))

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
                         {"search", "confirm", "final", "all"})
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
            self.check_gpu_hours(out)

    def test_missing_tower_and_ties(self):
        """A parked tower is named as missing; rankings are stable, ties
        broken by the model name."""
        rows = []
        for model, towers in (("tcn", C.TOWERS_SEARCHED),
                              ("fits", ("ref", "opt1")), ("lstm",
                                                          C.TOWERS_SEARCHED)):
            for tower in towers:
                for position in ("base", "z078", "top", "mean11"):
                    rows.append({
                        "model": model,
                        "tower": tower,
                        "group": "all",
                        "position": position,
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
        self.assertEqual(list(board["model"][:3]), ["fits", "lstm", "tcn"])
        top3, _ = analyze.rankings(board.drop(columns="group", errors="ignore"))
        self.assertEqual(list(top3[top3["criterion"] == "top"]["model"]),
                         ["fits", "lstm", "tcn"])

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


if __name__ == "__main__":
    unittest.main()
