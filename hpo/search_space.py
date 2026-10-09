# pylint: disable=use-dict-literal
"""The search space of the validation-tuned track, in one place, so the
search effort is comparable across models.

What is held equal is the SEARCH EFFORT (the same number of trials per model
over a space of comparable complexity), not the shape of a grid.

What is NOT searched, and is fixed by the protocol for every model: the
dataset, the split, the validation split (splits/val), the inputs (task
inputs), the targets (fore-aft moment at the 11 heights), the 3 Hz low-pass,
the crop length (training; evaluation stitches 6,000-sample inputs), the
loss (MSE; Gaussian NLL for Prob-TCN), AdamW, gradient clipping, the batch
size, the epoch budget, the validation frequency, the damage model (radius
at the gauge) and the selection metric. A search that is allowed to change
those is not comparing architectures.

The protocol numbers (trials, epochs, seeds, splits, failures, the
parameter cap and the warm-up) are in hpo/constants.py.
"""

from hpo import constants as C

LR = ("loguniform", 1e-4, 3e-3)
LR_PRETRAINED = ("loguniform", 1e-5, 3e-3)  # published 1e-3 inside
WD = ("loguniform", 1e-6, 1e-1)
# Weight decay is switched on/off (use_wd); wd is drawn only when on, so the
# published recipe (no weight decay) is a point of the space.
USE_WD = ("categorical", [False, True])
SCHEDULE = ("categorical", ["constant", "cosine"])  # cosine with warm-up
TCN_SPACE = dict(hidden_channels=("categorical", [32, 64, 96, 128]),
                 kernel_size=("categorical", [3, 5, 7]),
                 num_levels=("categorical", [6, 7, 8, 9]),
                 dropout=("uniform", 0.0, 0.2))
BOUND = ("categorical", [0.25, 0.5, 1.0])
CONTEXT = ("categorical", [256, 512, 1024])  # TimesFM
CONTEXT_CHRONOS = ("categorical", [256, 512])  # T5 encoder: 512 tokens max
CONTEXT_MOMENT = ("categorical", [512])  # fixed by the HF config


def _space(lr=LR, **knobs):
    return dict(lr=lr, use_wd=USE_WD, wd=WD, schedule=SCHEDULE, **knobs)


# All 20 learned entries of the leaderboard (physics and floor are
# calibrated, not trained).
SPACE = {
    "tcn":
        _space(**TCN_SPACE),
    "prob_tcn":
        _space(**TCN_SPACE),
    "hybrid_tcn":
        _space(condition_bound=BOUND),  # inner TCN at the published size
    "lstm":
        _space(hidden_size=("categorical", [64, 96, 128, 192]),
               num_layers=("categorical", [1, 2, 3]),
               dropout=("uniform", 0.0, 0.2)),  # bidirectional as published
    "transformer":
        _space(embed_dim=("categorical", [64, 128, 192, 256]),
               num_layers=("categorical", [2, 4, 6]),
               patch_size=("categorical", [8, 16, 32]),
               dropout=("uniform", 0.0, 0.2)),  # 4 heads as published
    "itransformer":
        _space(embed_dim=("categorical", [128, 192, 256]),
               num_layers=("categorical", [2, 3, 4]),
               dropout=("uniform", 0.0, 0.2)),  # 4 heads
    "s4":
        _space(hidden_channels=("categorical", [32, 64, 96]),
               state_dim=("categorical", [16, 32, 64]),
               num_layers=("categorical", [2, 4, 6])),
    # Same lr range as the others: with gradient clipping for every model,
    # the divergence of Mamba at 1e-3 is left to the search to find.
    "mamba":
        _space(hidden_channels=("categorical", [32, 64, 96]),
               state_dim=("categorical", [4, 8, 16]),
               num_layers=("categorical", [2, 4, 6])),
    "fits":
        _space(cutoff_bins=("categorical", [384, 768, 1536, 1800])
              ),  # 0.64-3.0 Hz
    "unet":
        _space(base_channels=("categorical", [16, 32, 48]),
               num_stages=("categorical", [3, 4, 5]),
               kernel_size=("categorical", [3, 5, 7])),
    "timesnet":
        _space(hidden_channels=("categorical", [32, 48, 64]),
               num_blocks=("categorical", [1, 2, 3]),
               num_periods=("categorical", [2, 3, 5])),
    "fno":
        _space(hidden_channels=("categorical", [32, 48, 64]),
               num_modes=("categorical", [64, 128, 256]),
               num_layers=("categorical", [2, 4])),
    "dlinear":
        _space(kernel_size=("categorical", [51, 101, 201]),
               context=("categorical", [129, 257, 513])),
    "spectral":
        _space(hidden_dim=("categorical", [32, 64, 128])),
    "hybrid":
        _space(hidden_dim=("categorical", [32, 64, 128]),
               condition_bound=BOUND),
    # Pretrained encoders: frozen (trained head) or fine-tuned; the backbone
    # is fixed; the optimizer, schedule and context length are searched.
    "chronos":
        _space(lr=LR_PRETRAINED, context_length=CONTEXT_CHRONOS),
    "moment":
        _space(lr=LR_PRETRAINED, context_length=CONTEXT_MOMENT),
    "moment_ft":
        _space(lr=LR_PRETRAINED, context_length=CONTEXT_MOMENT),
    "timesfm":
        _space(lr=LR_PRETRAINED, context_length=CONTEXT),
    "timesfm_ft":
        _space(lr=LR_PRETRAINED, context_length=CONTEXT),
}
LEARNED = tuple(SPACE)  # the 20 ranked entries
PRETRAINED = ("chronos", "moment", "moment_ft", "timesfm", "timesfm_ft")

# The fixed-budget (paper) optimizer values. Not enqueued in the search; kept
# to check that every published value lies inside its grid.
FIXED_RECIPE = dict(lr=1e-3, use_wd=False, schedule="constant")
# The FULL published configuration per model (models.py defaults, README);
# Mamba was published at lr 3e-4. Not enqueued (independent search).
PUBLISHED = {
    "tcn":
        dict(hidden_channels=64, kernel_size=5, num_levels=8, dropout=0.0),
    "prob_tcn":
        dict(hidden_channels=64, kernel_size=5, num_levels=8, dropout=0.0),
    "hybrid_tcn":
        dict(condition_bound=0.5),
    "lstm":
        dict(hidden_size=96, num_layers=2, dropout=0.0),
    "transformer":
        dict(embed_dim=128, num_layers=4, patch_size=16, dropout=0.1),
    "itransformer":
        dict(embed_dim=256, num_layers=3, dropout=0.1),
    "s4":
        dict(hidden_channels=64, state_dim=32, num_layers=4),
    "mamba":
        dict(lr=3e-4, hidden_channels=64, state_dim=4, num_layers=4),
    "fits":
        dict(cutoff_bins=768),
    "unet":
        dict(base_channels=32, num_stages=4, kernel_size=5),
    "timesnet":
        dict(hidden_channels=48, num_blocks=2, num_periods=3),
    "fno":
        dict(hidden_channels=48, num_modes=128, num_layers=4),
    "dlinear":
        dict(kernel_size=101, context=257),
    "spectral":
        dict(hidden_dim=64),
    "hybrid":
        dict(hidden_dim=64, condition_bound=0.5),
    "chronos":
        dict(context_length=512),
    "moment":
        dict(context_length=512),
    "moment_ft":
        dict(context_length=512),
    "timesfm":
        dict(context_length=1024),
    "timesfm_ft":
        dict(context_length=1024),
}

# --- Leaderboard (analyze.py) -----------------------------------------------
# The eight families of the paper (best model of each is reported).
FAMILIES = {
    "tcn": "convolutional",
    "prob_tcn": "convolutional",
    "unet": "convolutional",
    "dlinear": "convolutional",
    "lstm": "recurrent",
    "transformer": "attention",
    "itransformer": "attention",
    "s4": "state-space",
    "mamba": "state-space",
    "spectral": "spectral",
    "fits": "spectral",
    "fno": "spectral",
    "timesnet": "period-folding",
    "chronos": "pretrained",
    "moment": "pretrained",
    "moment_ft": "pretrained",
    "timesfm": "pretrained",
    "timesfm_ft": "pretrained",
    "hybrid": "physics-anchored",
    "hybrid_tcn": "physics-anchored",
}


def suggest(trial, model):
    """Draws one configuration for `model` from its space."""
    out = {}
    for name, spec in SPACE[model].items():
        if name == "wd" and not out.get("use_wd"):
            continue
        kind = spec[0]
        if kind == "loguniform":
            out[name] = trial.suggest_float(name, spec[1], spec[2], log=True)
        elif kind == "uniform":
            out[name] = trial.suggest_float(name, spec[1], spec[2])
        elif kind == "categorical":
            out[name] = trial.suggest_categorical(name, spec[1])
        else:
            raise ValueError(kind)
    return out


def formatted(cfg):
    """The configuration with every float at 6 significant digits: the
    values passed to scripts/train/run.py, stored as the trial's 'config'
    and in the run's config.json (Optuna keeps the unrounded draw in the
    trial's params; formatting it again gives the same values)."""
    return {
        k: float(f"{v:.6g}") if isinstance(v, float) else v
        for k, v in cfg.items()
    }


def as_args(cfg):
    """The configuration as scripts/train/run.py arguments (formatted)."""
    cfg = formatted(cfg)
    wd = cfg["wd"] if cfg.get("use_wd") else 0.0
    args = [
        f"--learning_rate={cfg['lr']:.6g}", f"--weight_decay={wd:.6g}",
        f"--schedule={cfg['schedule']}", f"--grad_clip={C.GRAD_CLIP}"
    ]
    if cfg["schedule"] == "cosine":
        args.append(f"--warmup_epochs={C.WARMUP_EPOCHS}")
    kwargs = {
        k: v
        for k, v in cfg.items()
        if k not in ("lr", "use_wd", "wd", "schedule")
    }
    if kwargs:
        args.append("--model_kwargs=" +
                    ",".join(f"{k}={v}" for k, v in kwargs.items()))
    return args
