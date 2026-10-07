# pylint: disable=wrong-import-position,protected-access
# pylint: disable=use-dict-literal
"""Tests of the drivers of the validation-tuned track (hpo/).

Run from the repository root with `python -m unittest discover tests`.
CPU only; nothing is trained: the drivers run with their --dry_run stub
and the failure handling with tiny stand-in commands. The Optuna tests need
optuna >= 4 and are skipped otherwise.
"""

import os
import random
import signal
import sys
import tempfile
import time
import unittest
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from floatsense import constants as C
from hpo import analyze
from hpo import common
from hpo import search_space as S

try:
    import optuna
    HAS_OPTUNA = int(optuna.__version__.split(".", maxsplit=1)[0]) >= 4
except ImportError:
    HAS_OPTUNA = False


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
        self.assertIn(f"--grad_clip={S.GRAD_CLIP}", args)


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
        self.assertEqual(common.classify(C.EXIT_DIVERGED, ""), "diverged")
        self.assertEqual(common.classify(C.EXIT_TOO_LARGE, ""), "too_large")
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
            old = time.time() - 2 * S.STALE_MINUTES * 60
            os.utime(os.path.join(unit, "heartbeat"), (old, old))
            self.assertTrue(common.claim(unit))
            common.release(unit)
            self.assertTrue(common.claim(unit))

    def run_fake(self, root: str, script: str) -> dict:
        """run_unit on a stand-in command."""
        run_dir = os.path.join(root, "unit")
        os.makedirs(run_dir, exist_ok=True)
        with mock.patch.object(common, "RETRY_SECONDS", 0.0):
            return common.run_unit("test/unit",
                                   [sys.executable, "-c", script, run_dir],
                                   run_dir, root, "tcn")

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
                             (2, S.MAX_ATTEMPTS))
            alerts = os.listdir(os.path.join(root, "alerts"))
            self.assertEqual(len(alerts), 1)
            self.assertTrue(os.path.exists(os.path.join(root, "unit",
                                                        "PARKED")))

    def test_exit_codes_are_returned(self):
        """Divergence and the parameter cap are not retried."""
        with tempfile.TemporaryDirectory() as root:
            for code, status in ((C.EXIT_DIVERGED, "diverged"),
                                 (C.EXIT_TOO_LARGE, "too_large")):
                result = self.run_fake(root, f"import sys; sys.exit({code})")
                self.assertEqual(result["status"], status)


@unittest.skipUnless(HAS_OPTUNA, "needs optuna >= 4")
class DriversTest(unittest.TestCase):
    """search, confirm and final with the dry-run stub."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def search(self, *extra, model="tcn", tower="opt2"):
        """search.main in dry-run mode on the test root."""
        from hpo import search  # pylint: disable=import-outside-toplevel
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
        self.assertEqual(states[:S.N_STARTUP], ["COMPLETE"] * S.N_STARTUP)
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
        from hpo import search  # pylint: disable=import-outside-toplevel
        study = search.open_study(self.root, "tcn", "opt2")
        trial = study.ask()
        cfg = S.suggest(trial, "tcn")
        run_dir = os.path.join(self.root, "trials", "dead")
        trial.set_user_attr("run_dir", run_dir)
        common.touch(run_dir)
        old = time.time() - 2 * S.STALE_MINUTES * 60
        os.utime(os.path.join(run_dir, "heartbeat"), (old, old))
        study = self.search("--max_new=1")
        resumed = study.trials[1]
        self.assertEqual(study.trials[0].state.name, "FAIL")
        self.assertEqual(resumed.user_attrs["resume_from"], run_dir)
        self.assertEqual(resumed.user_attrs["config"], cfg)

    def test_extension_rule(self):
        """A best trial after the EXTEND_AFTER-th extends the model."""
        from hpo import search  # pylint: disable=import-outside-toplevel
        n_trials = S.EXTEND_AFTER + 5
        study = search.open_study(self.root, "fits", "ref")
        dist = optuna.distributions.CategoricalDistribution([0.5])
        for number in range(n_trials):
            study.add_trial(
                optuna.trial.create_trial(
                    params={"x": 0.5},
                    distributions={"x": dist},
                    value=1.0 if number == S.EXTEND_AFTER + 2 else 0.0))
        trials = study.get_trials()
        self.assertEqual(search.best_position(trials, n_trials),
                         S.EXTEND_AFTER + 3)
        self.assertEqual(search.best_position(trials, S.EXTEND_AFTER), 1)
        self.assertTrue(search.maybe_extend(self.root, "fits", n_trials))
        self.assertEqual(search.target_trials(self.root, "fits", n_trials),
                         n_trials + S.EXTEND_BY)
        self.assertEqual(search.target_trials(self.root, "tcn", n_trials),
                         n_trials)

    def test_confirm_and_final(self):
        """Plan frozen once, winner by the median, test sealed once."""
        from hpo import confirm, final  # pylint: disable=import-outside-toplevel
        self.search()
        args = [
            "--model=tcn", "--tower=opt2", f"--root={self.root}", "--dry_run",
            "--n_trials=10"
        ]
        plan = confirm.main(args + ["--freeze"])
        self.assertEqual(len(plan["configs"]), S.N_TOP)
        self.assertEqual(confirm.main(args + ["--freeze"]), plan)
        self.assertIsNone(confirm.main(args + ["--summarize"]))
        confirm.main(args)
        record = confirm.main(args + ["--summarize"])
        medians = {r["rank"]: r["median"] for r in record["configs"]}
        self.assertEqual(record["winner_median"], max(medians.values()))
        for row in record["configs"]:
            self.assertAlmostEqual(row["median"],
                                   float(np.median(row["scores"])))
        self.assertEqual(record["best_epoch"] % S.BEST_EPOCH_ROUND, 0)
        test = [
            "--open_test", "--models=tcn", "--towers=opt2",
            f"--root={self.root}", "--dry_run"
        ]
        with self.assertRaises(SystemExit):
            final.main(test + ["--partial"])  # not trained yet
        with self.assertRaises(SystemExit):
            final.main(test)  # a subset needs --partial
        final.main(
            ["--model=tcn", "--tower=opt2", f"--root={self.root}", "--dry_run"])
        final.main(test + ["--partial"])
        marker = os.path.join(final.sealed_dir(self.root, "last", "opt2", 0),
                              "SEALED_tcn.json")
        stamp = common.read_json(marker)["time"]
        final.main(test + ["--partial"])
        self.assertEqual(common.read_json(marker)["time"], stamp)


class AnalyzeTest(unittest.TestCase):
    """The leaderboard of the track on synthetic sealed runs."""

    def test_families(self):
        """Every learned model has one of the eight families."""
        self.assertEqual(set(S.FAMILIES), set(S.LEARNED))
        self.assertEqual(len(set(S.FAMILIES.values())), 8)

    def test_synthetic_leaderboard(self):
        """Medians over seeds, means over towers, rankings and configs."""
        with tempfile.TemporaryDirectory() as out:
            boards = analyze.main(
                ["--dry_run", f"--out={out}", "--num_resamples=0"])
            board = boards["last"]
            self.assertEqual(set(boards), set(analyze.VARIANTS))
            self.assertEqual(set(board["model"]),
                             set(S.LEARNED) | {analyze.FLOOR})
            for column in ("r2_log_damage_base", "r2_log_damage_z078",
                           "r2_log_damage_top", "r2_log_damage_mean11",
                           "fraction_within_factor2_mean11",
                           "within_condition_correlation_top"):
                self.assertIn(column, board.columns)
            # The synthetic noise grows along LEARNED: tcn is the best.
            self.assertEqual(board.iloc[0]["model"], S.LEARNED[0])
            folder = os.path.join(out, "last")
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


if __name__ == "__main__":
    unittest.main()
