# pylint: disable=wrong-import-position,protected-access
# pylint: disable=too-few-public-methods
"""Tests of the evaluation input length (Trainer._predict_window).

Run from the repository root with `python -m unittest discover tests`.
CPU only; the models are untrained and the dataset is synthetic.
"""

import os
import sys
import unittest

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from floatsense.constants import INPUT_LENGTH
from floatsense.models import build_model
from floatsense.trainer import SequenceModelTrainer

WINDOW = INPUT_LENGTH + 1  # scored window, 400 to 1,000 s inclusive
CHANNELS = 5
CONDITION = 4


class WindowDataset:
    """The window_length / window_offset slicing of SequenceDataset."""

    def __init__(self, length: int = WINDOW):
        rng = np.random.default_rng(0)
        self.inputs = torch.tensor(rng.standard_normal((CHANNELS, length)),
                                   dtype=torch.float32)
        self.window_length = None
        self.window_offset = 0

    def __getitem__(self, index: int):
        del index
        inputs = self.inputs
        if self.window_length:
            inputs = inputs[:, self.window_offset:self.window_offset +
                            self.window_length]
        return {
            "inputs": inputs,
            "condition": torch.zeros(CONDITION),
            "target": inputs[:1]
        }


class Recorder(nn.Module):
    """Returns the first input channel and records the input lengths."""

    def __init__(self):
        super().__init__()
        self.lengths = []

    def forward(self, inputs, condition):
        """First input channel; `condition` is ignored."""
        del condition
        self.lengths.append(inputs.shape[-1])
        return inputs[:, :1]


def make_trainer(model_name: str,
                 model: nn.Module,
                 eval_length: int,
                 crop_length: int = 4096) -> SequenceModelTrainer:
    """Trainer with a model and the eval length of its checkpoint."""
    trainer = SequenceModelTrainer(release=None,
                                   output_dir="",
                                   model_name=model_name,
                                   crop_length=crop_length,
                                   device="cpu")
    trainer.model = model.eval()
    trainer._eval_length = eval_length
    return trainer


def predict(trainer: SequenceModelTrainer, dataset: WindowDataset):
    """_predict_window of item 0 over the full window."""
    with torch.no_grad():
        return trainer._predict_window(dataset, 0, dataset[0])


class PredictWindowTest(unittest.TestCase):
    """Which input lengths each kind of model sees at evaluation."""

    def test_length_sensitive_models_see_input_length(self):
        """A crop-trained length-sensitive model gets two 6,000-sample
        inputs, either eval length stored in its checkpoint, and returns the
        full window."""
        for eval_length in (INPUT_LENGTH, WINDOW):
            dataset = WindowDataset()
            trainer = make_trainer("transformer", Recorder(), eval_length)
            output = predict(trainer, dataset)
            self.assertEqual(trainer.model.lengths, [INPUT_LENGTH] * 2)
            self.assertEqual(output.shape, (WINDOW,))
            np.testing.assert_allclose(output,
                                       dataset.inputs[0].numpy(),
                                       atol=1e-6)
            self.assertIsNone(dataset.window_length)
            self.assertEqual(dataset.window_offset, 0)

    def test_other_crop_models_predict_directly(self):
        """A crop-trained model that does not depend on the input length
        (here the TCN) predicts the full window in one pass."""
        for eval_length in (INPUT_LENGTH, WINDOW):
            trainer = make_trainer("tcn", Recorder(), eval_length)
            output = predict(trainer, WindowDataset())
            self.assertEqual(trainer.model.lengths, [WINDOW])
            self.assertEqual(output.shape, (WINDOW,))

    def test_length_fixed_models_keep_their_size(self):
        """A length-fixed model trained on the full window is predicted
        directly; one trained on 6,000 samples is stitched."""
        for eval_length, lengths in ((WINDOW, [WINDOW]), (INPUT_LENGTH,
                                                          [INPUT_LENGTH] * 2)):
            trainer = make_trainer("itransformer", Recorder(), eval_length)
            output = predict(trainer, WindowDataset())
            self.assertEqual(trainer.model.lengths, lengths)
            self.assertEqual(output.shape, (WINDOW,))

    def test_shorter_window_is_predicted_directly(self):
        """A window of at most INPUT_LENGTH samples is not stitched."""
        trainer = make_trainer("tcn", Recorder(), WINDOW)
        output = predict(trainer, WindowDataset(INPUT_LENGTH))
        self.assertEqual(trainer.model.lengths, [INPUT_LENGTH])
        self.assertEqual(output.shape, (INPUT_LENGTH,))

    def test_length_sensitive_models(self):
        """PatchTST, TimesNet and U-Net predict the first 6,000 samples
        exactly as from a 6,000-sample input, over 6,001 samples."""
        for name in ("transformer", "timesnet", "unet"):
            with self.subTest(model=name):
                torch.manual_seed(0)
                model = build_model(name, INPUT_LENGTH, CHANNELS, CONDITION)
                dataset = WindowDataset()
                output = predict(make_trainer(name, model, WINDOW), dataset)
                self.assertEqual(output.shape, (WINDOW,))
                self.assertTrue(np.all(np.isfinite(output)))
                dataset.window_length = INPUT_LENGTH
                with torch.no_grad():
                    first = model(dataset[0]["inputs"][None],
                                  torch.zeros(1, CONDITION))[0, 0].numpy()
                np.testing.assert_allclose(output[:INPUT_LENGTH],
                                           first,
                                           atol=1e-5)


if __name__ == "__main__":
    unittest.main()
