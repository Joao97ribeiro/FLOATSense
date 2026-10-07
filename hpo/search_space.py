# pylint: disable=use-dict-literal
"""The search space and the numbers of the validation-tuned track, in one
place, so the budget is comparable across models.

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

Objective: VALIDATION R^2 of log10 damage, mean over the 11 heights, at the
last epoch of each run. Phase 2 confirms the top-2 configurations of each
(model, tower) over N_SEEDS seeds (winner = highest median over the seeds);
phase 3 retrains the winner on the full training split and opens the test
split once. The search has no enqueued published configuration; PUBLISHED
below only records the fixed-budget values (all inside the grids).
"""

# A parameter ceiling, so that "more capacity" cannot be bought indefinitely
# by one family (millions of trainable parameters); the pretrained encoders
# are exempt (their backbone is fixed).
MAX_PARAMS_M = 60.0

LR = ("loguniform", 1e-4, 3e-3)
LR_PRETRAINED = ("loguniform", 1e-5, 3e-3)  # published 1e-3 inside
WD = ("loguniform", 1e-6, 1e-1)
# Weight decay is switched on/off (use_wd); wd is drawn only when on, so the
# published recipe (no weight decay) is a point of the space.
USE_WD = ("categorical", [False, True])
SCHEDULE = ("categorical", ["constant", "cosine"])  # cosine with warm-up
WARMUP_EPOCHS = 5  # linear warm-up of the cosine schedule
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

# --- Phase 1: search -------------------------------------------------------
N_TRIALS = 30  # per model and tower (completed + pruned trials count)
N_STARTUP = 7  # random trials, then TPE; no enqueued trial
# Trials of 100 epochs (one third of the confirmation budget), scored at
# their last validation epoch; validation every VAL_EVERY epochs;
# MedianPruner after PRUNE_WARMUP epochs.
VAL_EVERY = 10
EPOCHS_TRIAL = 100
PRUNE_WARMUP = 50
NO_PRUNING = ("mamba", "moment_ft", "timesfm_ft")
TRIAL_SEED = 0  # training seed of every trial
# Extension rule: if a study's best eligible trial comes after the
# EXTEND_AFTER-th, the three studies of that model get EXTEND_BY more.
EXTEND_AFTER = 20
EXTEND_BY = 20
TOWERS_SEARCHED = ("ref", "opt1", "opt2")
SEARCH_TRAIN_SPLIT = "val/train"  # 1,380 simulations
SEARCH_VAL_SPLIT = "val/val"  # 348 simulations

# --- Phase 2: confirmation, phase 3: test ----------------------------------
EPOCHS_FINAL = 300  # confirmation and retraining
N_TOP = 2  # configurations confirmed per study
N_SEEDS = 3
FINAL_TRAIN_SPLIT = "train"  # 1,728 simulations
TEST_SPLIT = "test"
BEST_EPOCH_ROUND = 10  # the secondary test epoch is rounded to this

# --- Fixed for every run -----------------------------------------------------
GRAD_CLIP = 1.0

# --- Failures ----------------------------------------------------------------
MAX_ATTEMPTS = 3  # crashes (OOM included) before a unit is parked
MAX_FREE_RETRIES = 10  # preemptions and hardware faults (not counted)
STALE_MINUTES = 15  # a unit whose heartbeat is older belongs to a dead worker
DIVERGED_FALLBACK = -1.0  # score of a diverged trial before any completes

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


def as_args(cfg):
    """The configuration as scripts/train/run.py arguments."""
    wd = cfg["wd"] if cfg.get("use_wd") else 0.0
    args = [
        f"--learning_rate={cfg['lr']:.6g}", f"--weight_decay={wd:.6g}",
        f"--schedule={cfg['schedule']}", f"--grad_clip={GRAD_CLIP}"
    ]
    if cfg["schedule"] == "cosine":
        args.append(f"--warmup_epochs={WARMUP_EPOCHS}")
    kwargs = {
        k: v
        for k, v in cfg.items()
        if k not in ("lr", "use_wd", "wd", "schedule")
    }
    if kwargs:
        args.append("--model_kwargs=" +
                    ",".join(f"{k}={v}" for k, v in kwargs.items()))
    return args
