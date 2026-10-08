# pylint: disable=wrong-import-position
# pylint: disable=use-dict-literal
"""Tests of the model builder of the validation-tuned track.

Run from the repository root with `python -m unittest discover tests`.
CPU only. The defaults must rebuild the published models exactly: the
seeded state_dict shapes, parameter sums and outputs (in eval mode, and in
train mode with the torch random state after it, which pins the dropout
defaults) are compared with the fingerprints of the published code
(tests/data/model_fingerprints.json).
The pretrained encoders (Chronos, MOMENT, TimesFM) load Hugging Face
weights and run only with FLOATSENSE_PRETRAINED=1.
"""

import hashlib
import itertools
import json
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from floatsense.constants import INPUT_LENGTH
from floatsense.models import LENGTH_FIXED_MODELS
from floatsense.models import build_model
from floatsense.models import count_parameters
from floatsense.models import parse_model_kwargs
from hpo import search_space

FINGERPRINTS = os.path.join(os.path.dirname(__file__), "data",
                            "model_fingerprints.json")
PRETRAINED = ("chronos", "moment", "moment_ft", "timesfm", "timesfm_ft")
MODELS = ("spectral", "hybrid", "tcn", "dlinear", "naive", "hybrid_tcn", "s4",
          "mamba", "unet", "timesnet", "fits", "itransformer", "fno",
          "prob_tcn", "lstm", "transformer") + PRETRAINED
CHANNELS = 5
CONDITION = 4
LENGTH = 512  # input and num_samples of the fingerprints
CROP = 4096  # training crop of the protocol


def run_pretrained() -> bool:
    """The pretrained encoders are tested only on request."""
    return os.environ.get("FLOATSENSE_PRETRAINED") == "1"


def forward(model: torch.nn.Module, length: int, seed: int = 1):
    """Output of `model` on a seeded random batch of two items."""
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn(2, CHANNELS, length, generator=generator)
    condition = torch.randn(2, CONDITION, generator=generator)
    with torch.no_grad():
        if getattr(model, "needs_physics_gain", False):
            gain = torch.complex(
                torch.randn(2, length // 2 + 1, generator=generator),
                torch.randn(2, length // 2 + 1, generator=generator))
            return model(inputs, condition, gain)
        return model(inputs, condition)


def fingerprint(name: str, **model_kwargs) -> dict:
    """Seeded state_dict shapes, parameter sums and output of a model, in
    eval mode and then in train mode (with the random state it leaves)."""
    torch.manual_seed(0)
    model = build_model(name, LENGTH, CHANNELS, CONDITION,
                        **model_kwargs).eval()
    state = model.state_dict()
    floats = [v.double() for v in state.values() if v.is_floating_point()]
    output = forward(model, LENGTH).double()
    model.train()
    torch.manual_seed(2)
    train_output = forward(model, LENGTH).double()
    rng = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
    return {
        "shapes": {
            k: list(v.shape) for k, v in state.items()
        },
        "param_sum": float(sum(v.sum() for v in floats)),
        "param_abs_sum": float(sum(v.abs().sum() for v in floats)),
        "out_sum": float(output.sum()),
        "out_abs_sum": float(output.abs().sum()),
        "out_head": output.flatten()[:8].tolist(),
        "train_out_sum": float(train_output.sum()),
        "train_out_abs_sum": float(train_output.abs().sum()),
        "train_rng": rng,
    }


def corners(model: str):
    """Every corner of the architecture grid of `model` (first and last
    value of each categorical knob, both ends of a uniform one)."""
    knobs = {
        name: ([spec[1][0], spec[1][-1]]
               if spec[0] == "categorical" else [spec[1], spec[2]])
        for name, spec in search_space.SPACE[model].items()
        if name not in ("lr", "use_wd", "wd", "schedule")
    }
    names = sorted(knobs)
    for values in itertools.product(*(knobs[n] for n in names)):
        yield dict(zip(names, values))


class DefaultsTest(unittest.TestCase):
    """The defaults rebuild the published models."""

    @classmethod
    def setUpClass(cls):
        with open(FINGERPRINTS, encoding="utf-8") as file:
            cls.reference = json.load(file)

    def test_defaults_match_published(self):
        """Same parameter shapes, parameter sums and outputs as the
        published code, seed for seed."""
        for name in MODELS:
            if name in PRETRAINED and not run_pretrained():
                continue
            with self.subTest(model=name):
                expected = self.reference[name]
                actual = fingerprint(name)
                self.assertEqual(actual["shapes"], expected["shapes"])
                for key in ("param_sum", "param_abs_sum", "out_sum",
                            "out_abs_sum", "train_out_sum",
                            "train_out_abs_sum"):
                    self.assertAlmostEqual(actual[key],
                                           expected[key],
                                           delta=1e-6 *
                                           max(1.0, abs(expected[key])))
                for got, want in zip(actual["out_head"], expected["out_head"]):
                    self.assertAlmostEqual(got, want, delta=1e-6)
                # Same random draws in train mode (dropout and the like).
                self.assertEqual(actual["train_rng"], expected["train_rng"])

    def test_published_values_as_kwargs(self):
        """Passing the published knobs explicitly changes nothing."""
        for name, published in search_space.PUBLISHED.items():
            if name in PRETRAINED and not run_pretrained():
                continue
            kwargs = {k: v for k, v in published.items() if k != "lr"}
            with self.subTest(model=name):
                self.assertEqual(fingerprint(name, **kwargs), fingerprint(name))

    def test_unknown_kwarg_raises(self):
        """A misspelt knob is an error, not silently ignored."""
        with self.assertRaises(TypeError):
            build_model("tcn", LENGTH, CHANNELS, CONDITION, hidden=32)


class KnobsTest(unittest.TestCase):
    """The searched knobs reach the models."""

    def test_tcn_levels_and_dropout(self):
        """num_levels sets the dilations 2**i; dropout adds no parameters."""
        model = build_model("tcn",
                            LENGTH,
                            CHANNELS,
                            CONDITION,
                            num_levels=6,
                            dropout=0.1)
        self.assertEqual([b.conv1.dilation[0] for b in model.blocks],
                         [1, 2, 4, 8, 16, 32])
        self.assertEqual(
            count_parameters(
                build_model("tcn", LENGTH, CHANNELS, CONDITION, dropout=0.2)),
            count_parameters(build_model("tcn", LENGTH, CHANNELS, CONDITION)))

    def test_dropout_is_active_in_training_only(self):
        """A model with dropout is stochastic in train mode only."""
        for name, kwargs in (("tcn", {}), ("prob_tcn",
                                           {}), ("lstm", dict(num_layers=2)),
                             ("transformer", {}), ("itransformer", {})):
            with self.subTest(model=name):
                model = build_model(name,
                                    LENGTH,
                                    CHANNELS,
                                    CONDITION,
                                    dropout=0.2,
                                    **kwargs)
                model.eval()
                self.assertTrue(
                    torch.equal(forward(model, LENGTH), forward(model, LENGTH)))
                model.train()
                torch.manual_seed(0)
                first = forward(model, LENGTH)
                self.assertFalse(torch.equal(first, forward(model, LENGTH)))

    def test_hybrid_tcn_bound(self):
        """condition_bound reaches the Hybrid-TCN; its TCN is unchanged."""
        model = build_model("hybrid_tcn",
                            LENGTH,
                            CHANNELS,
                            CONDITION,
                            condition_bound=1.0)
        self.assertEqual(model.bound, 1.0)
        self.assertEqual(len(model.residual.blocks), 8)

    def test_unet_kernel(self):
        """kernel_size reaches every U-Net convolution."""
        model = build_model("unet", LENGTH, CHANNELS, CONDITION, kernel_size=7)
        kernels = {
            m.kernel_size[0]
            for m in model.modules()
            if isinstance(m, torch.nn.Conv1d) and m.kernel_size[0] > 1
        }
        self.assertEqual(kernels, {7})


class GridCornersTest(unittest.TestCase):
    """Every corner of every grid builds, runs, and fits the parameter cap."""

    def test_corners(self):
        """Builds at the protocol length; the trainable count is printed."""
        for model in sorted(search_space.SPACE):
            if model in PRETRAINED and not run_pretrained():
                continue
            fixed = model in LENGTH_FIXED_MODELS
            num_samples = INPUT_LENGTH + 1 if fixed else CROP
            for kwargs in corners(model):
                with self.subTest(model=model, **kwargs):
                    torch.manual_seed(0)
                    net = build_model(model, num_samples, CHANNELS, CONDITION,
                                      **kwargs)
                    millions = count_parameters(net) / 1e6
                    print(f"\n  {model:13s} {kwargs} {millions:8.3f} M", end="")
                    if model not in PRETRAINED:
                        self.assertLess(millions, search_space.MAX_PARAMS_M)
                    output = forward(net, num_samples if fixed else LENGTH)
                    self.assertTrue(torch.isfinite(output).all())


class ParseKwargsTest(unittest.TestCase):
    """--model_kwargs parsing."""

    def test_types(self):
        """bool, int, float and str values are typed."""
        self.assertEqual(
            parse_model_kwargs("a=1,b=0.5,c=True,d=false,e=cosine,f=1e-3,"
                               "g=-2"),
            dict(a=1, b=0.5, c=True, d=False, e="cosine", f=1e-3, g=-2))
        self.assertIsInstance(parse_model_kwargs("a=1")["a"], int)
        self.assertIsInstance(parse_model_kwargs("a=1.0")["a"], float)

    def test_empty_and_errors(self):
        """Empty means no kwargs; a malformed or repeated pair raises."""
        self.assertEqual(parse_model_kwargs(""), {})
        self.assertEqual(parse_model_kwargs(None), {})
        for bad in ("a", "a=1,a=2", "=1"):
            with self.assertRaises(ValueError):
                parse_model_kwargs(bad)

    def test_round_trip_of_as_args(self):
        """The string written by search_space.as_args parses back."""
        cfg = dict(lr=1e-3,
                   use_wd=False,
                   schedule="constant",
                   hidden_channels=64,
                   dropout=0.123456789,
                   kernel_size=5)
        arg = [
            a for a in search_space.as_args(cfg)
            if a.startswith("--model_kwargs=")
        ][0]
        parsed = parse_model_kwargs(arg.split("=", 1)[1])
        self.assertEqual(parsed["hidden_channels"], 64)
        # as_args writes 6 significant digits.
        self.assertAlmostEqual(parsed["dropout"], 0.123456789, places=5)


if __name__ == "__main__":
    unittest.main()
