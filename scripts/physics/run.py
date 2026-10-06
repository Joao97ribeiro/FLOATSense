"""Calibrates the physics baseline and scores it at the 11 gauges.

C1 comes from the parked runs of the tower; the other five constants are
the median of per-simulation estimates over the calibration split. The
moment at every gauge is the base reconstruction times the height factor
of the tower (floatsense.heights).

Examples:
  # within tower: calibrate on the train split of opt2, score its test split
  python scripts/physics/run.py --flagfile=scripts/physics/config.cfg \
      --tower=opt2

  # zero-shot ref -> opt2: constants and height profile of ref
  python scripts/physics/run.py ... --tower=opt2 --source=ref

  # ten-shot: constants recalibrated on 10 opt2 simulations
  python scripts/physics/run.py ... --tower=opt2 \
      --train_split=fewshot/train_10_draw0
"""

import json
import os
import sys

from absl import app
from absl import flags
from absl import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from floatsense import constants as C  # noqa: E402  pylint: disable=wrong-import-position
from floatsense import Calibration  # noqa: E402  pylint: disable=wrong-import-position
from floatsense import PhysicsReconstruction  # noqa: E402  pylint: disable=wrong-import-position
from floatsense import load_tower  # noqa: E402  pylint: disable=wrong-import-position
from floatsense import parked_c_theta  # noqa: E402  pylint: disable=wrong-import-position
from floatsense.heights import calibrate_profile  # noqa: E402  pylint: disable=wrong-import-position
from floatsense.heights import evaluate_heights  # noqa: E402  pylint: disable=wrong-import-position
from floatsense.metrics import summarize_damage  # noqa: E402  pylint: disable=wrong-import-position
from floatsense.release import split_tag  # noqa: E402  pylint: disable=wrong-import-position

FLAGS = flags.FLAGS

flags.DEFINE_string("dataset_dir", None,
                    "Released FLOATSense dataset, one folder per tower.")
flags.DEFINE_string("tower", "opt2", "Tower scored (ref, opt1 or opt2).")
flags.DEFINE_string("source", None,
                    "Zero-shot: apply the constants and the height profile "
                    "of this tower instead of calibrating on --tower.")
flags.DEFINE_string("train_split", "train", "Calibration split.")
flags.DEFINE_string("test_split", "test", "Evaluation split.")
flags.DEFINE_string("direction", "fa", "Direction ('fa' or 'ss').")
flags.DEFINE_string("tag", None,
                    "Output suffix; defaults to the calibration split "
                    "(none for train, fs10_draw0 for fewshot/train_10_draw0, "
                    "val for val/train).")
flags.DEFINE_string("output_root", "outputs/physics", "Root of the outputs.")
flags.DEFINE_string("mass_csv", None,
                    "Tower mass profile; defaults to towers/<tower>_mass.csv.")
flags.DEFINE_float("min_time", C.MIN_TIME, "Start of the scored window (s).")
flags.DEFINE_float("max_time", C.MAX_TIME, "End of the scored window (s).")
flags.DEFINE_float("pad_seconds", C.PAD_SECONDS,
                   "Margin around the scored window.")
flags.DEFINE_integer("segment_length", C.SEGMENT_LENGTH,
                     "Welch segment length.")
flags.DEFINE_list("lf_fit_band", [str(v) for v in C.LF_FIT_BAND],
                  "Bins fitting the low-frequency polynomial (Hz).")
flags.DEFINE_float("operating_power_kw", C.OPERATING_POWER_KW,
                   "Power above which a sample counts as operating.")
flags.DEFINE_float("band_hz", C.BAND_HZ,
                   "Upper edge of the reconstruction band (gain is zero "
                   "above it).")
flags.DEFINE_bool("lowpass", True,
                  "Zero-phase Butterworth low-pass of the true and "
                  "reconstructed moments before the damage metric (the "
                  "damage of these towers lies below 3 Hz).")
flags.DEFINE_float("lowpass_hz", C.LOWPASS_HZ, "Cutoff of --lowpass [Hz].")
flags.DEFINE_integer("lowpass_order", C.LOWPASS_ORDER,
                     "Butterworth order of one pass (sosfiltfilt runs two).")
flags.DEFINE_integer("max_eval_sims", 0, "If > 0, cap the evaluated sims.")
flags.DEFINE_list("sn_intercepts_log10",
                  [str(v) for v in C.SN_INTERCEPTS_LOG10],
                  "SN curve log10 intercepts.")
flags.DEFINE_list("sn_slopes", [str(v) for v in C.SN_SLOPES],
                  "SN curve slopes.")


def main(_):
    """Calibrates (or loads) the gains and scores the 11 gauges."""
    # <tower>[_zs_<source>][_<tag>][_ss]: a run never overwrites another.
    tag = FLAGS.tag if FLAGS.tag is not None else (
        "" if FLAGS.source else split_tag(FLAGS.train_split))
    suffix = "_ss" if FLAGS.direction == "ss" else ""
    name = (FLAGS.tower + (f"_zs_{FLAGS.source}" if FLAGS.source else "") +
            (f"_{tag}" if tag else "") + suffix)
    output_dir = os.path.join(FLAGS.output_root, name)
    release = load_tower(FLAGS.dataset_dir, FLAGS.tower)
    physics = PhysicsReconstruction(
        release=release,
        output_dir=output_dir,
        parked_c_theta=parked_c_theta(FLAGS.dataset_dir, FLAGS.tower),
        min_time=FLAGS.min_time,
        max_time=FLAGS.max_time,
        pad_seconds=FLAGS.pad_seconds,
        segment_length=FLAGS.segment_length,
        lf_fit_band=tuple(float(v) for v in FLAGS.lf_fit_band),
        operating_power_kw=FLAGS.operating_power_kw,
        band_hz=FLAGS.band_hz,
        apply_lowpass=FLAGS.lowpass,
        lowpass_hz=FLAGS.lowpass_hz,
        lowpass_order=FLAGS.lowpass_order,
        sn_intercepts_log10=[float(v) for v in FLAGS.sn_intercepts_log10],
        sn_slopes=[float(v) for v in FLAGS.sn_slopes])

    profile_tower = FLAGS.source or FLAGS.tower
    if FLAGS.source:
        calibration = Calibration.from_json(
            os.path.join(FLAGS.output_root, FLAGS.source + suffix,
                         f"calibration_{FLAGS.direction}.json"))
    else:
        calibration_ids = release.split_ids(FLAGS.train_split)
        missing = sorted(set(calibration_ids) - set(release.sim_ids))
        if missing:
            raise SystemExit(
                f"{len(missing)} of the {len(calibration_ids)} simulations of "
                f"split '{FLAGS.train_split}' are not in {release.tower_dir} "
                "(the review subset holds only splits/review/test): the physics "
                "calibration needs the full dataset.")
        calibration = physics.calibrate(FLAGS.direction, calibration_ids)
    logging.info("%s: %s", FLAGS.direction, calibration)

    profile = calibrate_profile(FLAGS.dataset_dir, profile_tower,
                                FLAGS.direction,
                                mass_csv=FLAGS.mass_csv)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "profile.json"), "w",
              encoding="utf-8") as file:
        json.dump({"profile_tower": profile_tower, **profile}, file, indent=4)
    logging.info("%s: h_rna %.2f -> %.2f m", profile_tower,
                 profile["h_rna_nominal"], profile["h_rna_effective"])

    test_ids = release.split_ids(FLAGS.test_split)
    if FLAGS.max_eval_sims:
        test_ids = test_ids[:FLAGS.max_eval_sims]
    scores = evaluate_heights(physics, test_ids, calibration,
                              profile["factors"], FLAGS.direction,
                              os.path.join(output_dir, "damage_heights.csv"))
    for height in (release.geometry.channels[0],
                   release.geometry.channels[-1]):  # base and top
        if f"damage_true_{height}" in scores:
            s = summarize_damage(scores[f"damage_true_{height}"].to_numpy(),
                                 scores[f"damage_rec_{height}"].to_numpy())
            logging.info("%s %s: R2 log damage %.3f | median ratio %.3f | "
                         "within 2 %.3f (%d sims)", FLAGS.tower, height,
                         s["r2_log_damage"], s["median_damage_ratio"],
                         s["fraction_within_factor2"], len(scores))
    logging.info("Done: %s (all heights and metrics: scripts/benchmark/run.py)",
                 output_dir)


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    flags.mark_flag_as_required("dataset_dir")
    app.run(main)
