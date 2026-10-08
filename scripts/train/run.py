# pylint: disable=wrong-import-position
# pylint: disable=too-many-locals
# pylint: disable=too-many-branches
# pylint: disable=too-many-statements
"""Trains and evaluates sequence models on one tower (within tower), with
optional zero-shot evaluation on the other towers and few-shot adaptation.

Examples:
  # within tower on opt2, the benchmark task (fore-aft moment at 11 heights)
  python scripts/train/run.py --flagfile=scripts/train/config.cfg \
      --tower=opt2 --models=tcn

  # the same model, also evaluated zero-shot on the other two towers
  python scripts/train/run.py ... --tower=opt2 --eval_towers=ref,opt1

  # ten-shot: start from the opt2 checkpoint, adapt on 10 ref simulations
  python scripts/train/run.py ... --tower=ref --batch_size=4 \
      --train_split=fewshot/train_10_draw0 \
      --init_checkpoint_dir=outputs/within/opt2/seed0 \
      --output_dir=outputs/fewshot/opt2_to_ref/draw0/seed0

  # sensor ablation: both accelerometer axes and SCADA
  python scripts/train/run.py ... --input_channels=tower_top_afa_mod,\
      tower_top_ass_mod,rotor_speed,blade_pitch,wind_speed

Exit codes:
  0  Done.
  3  Diverged: a non-finite training loss (or damage-validation prediction).
     This guard is on with the default flags too: the published code
     finished such a run and saved a non-finite checkpoint, so the two
     differ only for runs that diverge.
  4  Too large: more trainable parameters than --max_params_m.
  5  Stopped: SIGUSR1 with --resume; the resume state was saved at the end
     of the epoch and a relaunch continues from it (a request during the
     last epoch lets that run complete, then stops before its evaluation
     and the next model or direction).
  6  Config mismatch: with --resume, the resume state in --output_dir was
     written with another run configuration (tower, task, recipe, training
     simulations or --num_epochs). A run is never extended in place: a
     longer run goes to a new --output_dir.
"""

import os
import signal
import sys
from typing import List

from absl import app
from absl import flags
from absl import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from floatsense import constants as C
from floatsense import SequenceModelTrainer
from floatsense import load_tower
from floatsense.models import LENGTH_FIXED_MODELS
from floatsense.models import parse_model_kwargs
from floatsense.trainer import STOP_REQUESTED
from floatsense.trainer import ConfigMismatchError
from floatsense.trainer import DivergedError
from floatsense.trainer import ModelTooLargeError
from floatsense.trainer import StoppedError
from floatsense.trainer import request_stop
from floatsense.release import split_tag
from floatsense.heights import calibrate_profile

FLAGS = flags.FLAGS

flags.DEFINE_string("dataset_dir", None,
                    "Released FLOATSense dataset, one folder per tower.")
flags.DEFINE_string("tower", "opt2", "Training tower (ref, opt1 or opt2).")
flags.DEFINE_string("train_split", "train",
                    "Training split: train, val/train or fewshot/<name>.")
flags.DEFINE_string("test_split", "test", "Evaluation split: test or val/val.")
flags.DEFINE_string("val_split", "",
                    "Optional split whose loss is logged during training.")
flags.DEFINE_list("eval_towers", [],
                  "Other towers evaluated zero-shot with the trained model.")
flags.DEFINE_string(
    "output_dir", None, "Output directory; defaults to <output_root>/<tower>/"
    "seed<seed>.")
flags.DEFINE_string("output_root", "outputs/within",
                    "Root of the default outputs.")
flags.DEFINE_list("models", ["tcn"], "Models to train.")
flags.DEFINE_list("directions", ["fa"], "Directions to process.")
flags.DEFINE_integer("seed", 0, "Training seed.")
flags.DEFINE_bool(
    "deterministic", True,
    "Deterministic cuDNN/CUDA kernels: reruns of a seed are "
    "identical (the paper runs used False).")
flags.DEFINE_integer("max_train_sims", 0, "If > 0, cap the training sims.")
flags.DEFINE_integer("max_eval_sims", 0, "If > 0, cap the evaluated sims.")

flags.DEFINE_float("min_time", C.MIN_TIME, "Start of the scored window (s).")
flags.DEFINE_float("max_time", C.MAX_TIME, "End of the scored window (s).")
flags.DEFINE_bool(
    "lowpass", True, "Zero-phase Butterworth low-pass of the true and "
    "reconstructed moments before the damage metric (the "
    "damage of these towers lies below 3 Hz).")
flags.DEFINE_float("lowpass_hz", C.LOWPASS_HZ, "Cutoff of --lowpass [Hz].")
flags.DEFINE_integer("lowpass_order", C.LOWPASS_ORDER,
                     "Butterworth order of one pass (sosfiltfilt runs two).")
flags.DEFINE_integer("crop_length", 4096, "Training crop length.")
flags.DEFINE_integer("batch_size", 16, "Training batch size.")
flags.DEFINE_float("learning_rate", 1e-3, "Adam (or AdamW) learning rate.")
flags.DEFINE_float(
    "weight_decay", None, "AdamW weight decay; unset keeps Adam (the "
    "published recipe). AdamW with 0 equals Adam.")
flags.DEFINE_enum("schedule", "constant", ["constant", "cosine"],
                  "Learning-rate schedule (cosine: to zero at the last step).")
flags.DEFINE_integer("warmup_epochs", 0, "Linear learning-rate warm-up.")
flags.DEFINE_float("grad_clip", 0.0, "Maximum gradient norm (0 = off).")
flags.DEFINE_string(
    "model_kwargs", "", "Architecture knobs of the models, 'k=v,k=v' "
    "(e.g. hidden_channels=96,num_levels=7,dropout=0.1); empty = published.")
flags.DEFINE_enum(
    "val_score", "loss", ["loss", "damage"],
    "Score of --val_split: loss (squared error on normalized crops) or "
    "damage (R^2 of log10 damage over the 11 gauges, printed as VAL lines).")
flags.DEFINE_bool(
    "resume", False, "Keep a resume state (every validation and every "
    "--checkpoint_seconds) and continue from it if present; SIGUSR1 saves "
    "it at the end of the epoch and exits with code 5. A resume state in "
    "--output_dir written with another run configuration, another "
    "--num_epochs included, exits with code 6: extend a run in a new "
    "--output_dir.")
flags.DEFINE_float(
    "checkpoint_seconds", C.CHECKPOINT_SECONDS,
    "With --resume, wall time between two saves of the resume state, "
    "checked at the end of each epoch [s] (0 = after every epoch).")
flags.DEFINE_float(
    "max_params_m", 0.0, "Exit (code 4) if the model has more trainable "
    "parameters, in millions (0 = no limit).")
flags.DEFINE_list(
    "save_epochs", [], "Epochs whose weights are also kept as "
    "<model>_<direction>_epoch<e>.pt.")
flags.DEFINE_integer("num_epochs", 50, "Training epochs.")
flags.DEFINE_integer("val_every", 0,
                     "Validation cadence in epochs (0 = every num_epochs/10).")
flags.DEFINE_integer(
    "early_stopping_patience", 0,
    "Stop after this many validations without improvement "
    "and restore the best weights (0 = off).")
flags.DEFINE_integer("num_workers", 4, "DataLoader workers.")
flags.DEFINE_bool("run_training", True, "Train (otherwise load checkpoint).")
flags.DEFINE_bool("run_evaluation", True, "Evaluate on the test split.")
flags.DEFINE_string("init_checkpoint_dir", None,
                    "Directory with <model>_<direction>.pt to fine-tune from.")
flags.DEFINE_string(
    "calibration_dir", None,
    "Directory with the physics calibration_<direction>.json "
    "(hybrid models); defaults to outputs/physics/<tower>[_<split tag>]"
    "[_ss], where scripts/physics/run.py writes the calibration of the "
    "same split and direction.")
flags.DEFINE_float("condition_bound", 0.5,
                   "Tanh bound of the hybrid correction (0 disables it).")
flags.DEFINE_string("target_channel", None,
                    "Overrides the target channel (e.g. tower_5_mfa).")
flags.DEFINE_list("input_channels", None,
                  "Replaces the default input stack (sensor ablations).")
flags.DEFINE_bool(
    "height_targets", True,
    "Height-conditioned task: one model reconstructs the "
    "moment at the 11 instrumented heights.")
flags.DEFINE_enum("loss", "mse", ["mse", "damage"], "Training loss.")
flags.DEFINE_float("damage_loss_weight", 1.0, "Weight of the damage term.")
flags.DEFINE_list("sn_intercepts_log10",
                  [str(v) for v in C.SN_INTERCEPTS_LOG10],
                  "SN curve log10 intercepts.")
flags.DEFINE_list("sn_slopes", [str(v) for v in C.SN_SLOPES],
                  "SN curve slopes.")


def floats(values: List[str]) -> List[float]:
    """Converts a list of strings to floats."""
    return [float(value) for value in values]


def main(_):
    """Trains and evaluates the requested models and directions."""
    # A resumable run stops cleanly on SIGUSR1 from the start to the end,
    # also while the data loads and between models and directions.
    previous_handler = None
    if FLAGS.resume and FLAGS.run_training:
        previous_handler = (signal.signal(signal.SIGUSR1, request_stop),)
    try:
        train_and_evaluate()
    finally:
        if previous_handler is not None:
            signal.signal(signal.SIGUSR1, previous_handler[0])


def train_and_evaluate():
    """`main` with the SIGUSR1 handler of a resumable run installed."""
    output_dir = FLAGS.output_dir or os.path.join(
        FLAGS.output_root, FLAGS.tower, f"seed{FLAGS.seed}")
    # The hybrids read the physics calibrated on the same split as the model
    # (outputs/physics/<tower>[_<split tag>], as scripts/physics/run.py
    # writes it), so a few-shot hybrid never sees the full training split.
    tag = split_tag(FLAGS.train_split)

    def calibration_dir(direction: str) -> str:
        if FLAGS.calibration_dir:
            return FLAGS.calibration_dir
        return os.path.join(
            "outputs", "physics", FLAGS.tower + (f"_{tag}" if tag else "") +
            ("_ss" if direction == "ss" else ""))

    for model_name in FLAGS.models:
        for direction in FLAGS.directions:
            if FLAGS.init_checkpoint_dir:
                path = os.path.join(FLAGS.init_checkpoint_dir,
                                    f"{model_name}_{direction}.pt")
                if not os.path.exists(path):
                    raise SystemExit(
                        f"--init_checkpoint_dir: {path} not found. Train the "
                        "source model first (e.g. the within-tower run of the "
                        "source tower) or point the flag to its folder.")
            if model_name.startswith("hybrid"):
                path = os.path.join(calibration_dir(direction),
                                    f"calibration_{direction}.json")
                if not os.path.exists(path):
                    raise SystemExit(
                        f"{model_name} needs the physics calibration {path}: "
                        "run scripts/physics/run.py on the same tower and "
                        "--train_split first, or pass --calibration_dir.")

    source = load_tower(FLAGS.dataset_dir, FLAGS.tower)
    # Single-height task: the damage is scored at the gauge of the target.
    damage_gauge = 0
    if FLAGS.height_targets and FLAGS.target_channel:
        raise ValueError("--target_channel applies to the single-height task: "
                         "add --height_targets=False.")
    if FLAGS.target_channel:
        gauges = {
            f"{stem}_m{d}": gauge
            for gauge, stem in enumerate(source.geometry.channels)
            for d in ("fa", "ss")
        }
        if FLAGS.target_channel not in gauges:
            raise ValueError(f"Unknown target channel {FLAGS.target_channel}.")
        damage_gauge = gauges[FLAGS.target_channel]
    hybrids = [m for m in FLAGS.models if m.startswith("hybrid")]
    if hybrids and not FLAGS.height_targets and damage_gauge:
        raise ValueError("The hybrid models anchor to the physics at the "
                         "requested height only in the 11-height task.")
    train_ids = source.split_ids(FLAGS.train_split)
    test_ids = source.split_ids(FLAGS.test_split)
    if FLAGS.max_train_sims:
        train_ids = train_ids[:FLAGS.max_train_sims]
    if FLAGS.run_training:
        missing = sorted(set(train_ids) - set(source.sim_ids))
        if missing:
            raise SystemExit(
                f"{len(missing)} of the {len(train_ids)} simulations of split "
                f"'{FLAGS.train_split}' are not in {source.tower_dir} (the "
                "review subset holds only splits/review/test): training needs "
                "the full dataset; use --run_training=False to evaluate "
                "released checkpoints.")
    if FLAGS.max_eval_sims:
        test_ids = test_ids[:FLAGS.max_eval_sims]
    logging.info(
        "Tower %s | train %s | test %d | out %s", FLAGS.tower,
        len(train_ids) if FLAGS.run_training else "- (evaluation only)",
        len(test_ids), output_dir)

    model_kwargs = parse_model_kwargs(FLAGS.model_kwargs)
    hybrid = any(m.startswith("hybrid") for m in FLAGS.models)
    height_factors = ({
        d:
            calibrate_profile(FLAGS.dataset_dir, FLAGS.tower, direction=d)
            ["factors"] for d in FLAGS.directions
    } if FLAGS.height_targets and hybrid else {})
    for model_name in FLAGS.models:
        crop_length = (0 if model_name in LENGTH_FIXED_MODELS else
                       FLAGS.crop_length)
        for direction in FLAGS.directions:
            trainer = SequenceModelTrainer(
                release=source,
                output_dir=output_dir,
                direction=direction,
                model_name=model_name,
                min_time=FLAGS.min_time,
                max_time=FLAGS.max_time,
                apply_lowpass=FLAGS.lowpass,
                lowpass_hz=FLAGS.lowpass_hz,
                lowpass_order=FLAGS.lowpass_order,
                crop_length=crop_length,
                batch_size=FLAGS.batch_size,
                learning_rate=FLAGS.learning_rate,
                num_epochs=FLAGS.num_epochs,
                val_every=FLAGS.val_every,
                early_stopping_patience=FLAGS.early_stopping_patience,
                num_workers=FLAGS.num_workers,
                sn_intercepts_log10=floats(FLAGS.sn_intercepts_log10),
                sn_slopes=floats(FLAGS.sn_slopes),
                loss_name=FLAGS.loss,
                damage_loss_weight=FLAGS.damage_loss_weight,
                init_checkpoint=(os.path.join(FLAGS.init_checkpoint_dir,
                                              f"{model_name}_{direction}.pt")
                                 if FLAGS.init_checkpoint_dir else None),
                calibration_path=(os.path.join(calibration_dir(direction),
                                               f"calibration_{direction}.json")
                                  if model_name.startswith("hybrid") else None),
                condition_bound=FLAGS.condition_bound,
                target_channel=FLAGS.target_channel,
                damage_gauge=damage_gauge,
                input_channels=FLAGS.input_channels,
                height_targets=FLAGS.height_targets,
                height_factors=height_factors.get(direction),
                seed=FLAGS.seed,
                deterministic=FLAGS.deterministic,
                weight_decay=FLAGS.weight_decay,
                schedule=FLAGS.schedule,
                warmup_epochs=FLAGS.warmup_epochs,
                grad_clip=FLAGS.grad_clip,
                model_kwargs=model_kwargs,
                val_score=FLAGS.val_score,
                resume=FLAGS.resume,
                max_params_m=FLAGS.max_params_m,
                save_epochs=[int(e) for e in FLAGS.save_epochs],
                checkpoint_seconds=FLAGS.checkpoint_seconds)
            if FLAGS.run_training:
                logging.info("Training %s (%s).", model_name, direction)
                val_ids = (source.split_ids(FLAGS.val_split)
                           if FLAGS.val_split else None)
                if val_ids and FLAGS.max_eval_sims:
                    val_ids = val_ids[:FLAGS.max_eval_sims]
                try:
                    if STOP_REQUESTED.is_set():
                        raise StoppedError("Stopped before training.")
                    trainer.train(train_ids, val_ids)
                except ConfigMismatchError as error:
                    logging.error("Config mismatch: %s", error)
                    sys.exit(C.EXIT_CONFIG_MISMATCH)
                except DivergedError as error:
                    logging.error("Diverged: %s", error)
                    sys.exit(C.EXIT_DIVERGED)
                except ModelTooLargeError as error:
                    logging.error("Too large: %s", error)
                    sys.exit(C.EXIT_TOO_LARGE)
                except StoppedError as error:
                    logging.warning("Stopped: %s", error)
                    sys.exit(C.EXIT_STOPPED)
                if trainer.stop_requested:
                    # SIGUSR1 in the last epoch: this run completed and its
                    # state is saved; a relaunch evaluates it and goes on.
                    logging.warning("Stopped after %s (%s) completed.",
                                    model_name, direction)
                    sys.exit(C.EXIT_STOPPED)
            else:
                trainer.load_checkpoint()
            if not FLAGS.run_evaluation:
                continue
            logging.info("Summary: %s", trainer.evaluate(test_ids))
            for target in FLAGS.eval_towers:
                release = load_tower(FLAGS.dataset_dir, target)
                target_ids = release.split_ids(FLAGS.test_split)
                if FLAGS.max_eval_sims:
                    target_ids = target_ids[:FLAGS.max_eval_sims]
                logging.info(
                    "Zero-shot %s -> %s: %s", FLAGS.tower, target,
                    trainer.evaluate(target_ids,
                                     release=release,
                                     tag=f"zs_{target}"))
    logging.info("Done.")


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    flags.mark_flag_as_required("dataset_dir")
    app.run(main)
