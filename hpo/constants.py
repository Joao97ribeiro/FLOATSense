"""The numbers of the validation-tuned track: trials, epochs, seeds, splits,
failures, workers and their records. The search grids are in
hpo/search_space.py.

Objective: VALIDATION R^2 of log10 damage, mean over the 11 heights, at the
last epoch of each run. Phase 2 confirms the top-2 configurations of each
(model, tower) over N_SEEDS seeds (winner = highest median over the seeds);
phase 3 retrains the winner on the full training split and opens the test
split once. The search has no enqueued published configuration; PUBLISHED
in hpo/search_space.py only records the fixed-budget values (all inside the
grids).
"""

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
# A parameter ceiling, so that "more capacity" cannot be bought indefinitely
# by one family (millions of trainable parameters); the pretrained encoders
# are exempt (their backbone is fixed).
MAX_PARAMS_M = 60.0
WARMUP_EPOCHS = 5  # linear warm-up of the cosine schedule

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
MAX_ATTEMPTS = 3  # crashes before a unit is shelved
MAX_FREE_RETRIES = 10  # preemptions and hardware faults (not counted)
RETRY_SECONDS = 5.0  # pause before resuming a failed run
STALE_SECONDS = 900.0  # a unit whose heartbeat is older has a dead worker
DIVERGED_FALLBACK = -1.0  # score of a diverged trial before any completes
# Search draws above the parameter cap, or out of memory (after one resume
# as a crash), are failed trials (not counted toward N_TRIALS, ignored by
# TPE). A study with MAX_OVER_CAP of one kind, or SHELVE_STUDY_AFTER
# shelved trials, stops drawing: it is shelved if it has fewer than
# N_TRIALS counted trials, else its plan is frozen from the counted ones
# (during an extension, after one retry if it stopped for a kind in
# RETRIED_KINDS). A confirmation or final unit out of memory is a crash like
# any other.
MAX_OVER_CAP = 20
MAX_OOM = MAX_OVER_CAP
SHELVE_STUDY_AFTER = 3
RETRIED_KINDS = ("shelved_trials", "oom")
# A shelved unit, or a study shelved for a kind in RETRIED_KINDS, is
# retried once, RETRY_SHELVED_AFTER seconds after its shelving, with its
# failure counts reset (and the extension decision of its model waits for
# it); a second shelving is for good. Nothing is retried once
# READY_FOR_TEST.json or sealed/ exists. Every other shelving is for good.
# An operator retries at any time: python hpo/pick.py --retry=<unit>. A
# unit whose driver raises MAX_ATTEMPTS times in one worker is set aside by
# that worker for RETRY_SHELVED_AFTER (a confirmation or final unit is
# shelved by then: each exception is a crash).
RETRY_SHELVED_AFTER = 3600.0
# Attempts in a row that end stopped or preempted without a new resume save
# before an alert (a unit that makes no progress: the checkpoint interval is
# longer than the time the workers get).
NO_PROGRESS_ATTEMPTS = 3

# --- Records of the runs (common.py) ----------------------------------------
HEARTBEAT_SECONDS = 30.0  # a running unit touches its heartbeat this often
TAIL_LINES = 60  # last lines of a run kept to classify its exit
ALERT_TAIL_CHARS = 4000  # end of the log kept in an alert
TRACEBACK_CHARS = 1500  # end of a traceback kept in a log or an alert
GPU_QUERY_TIMEOUT = 30.0  # seconds of one nvidia-smi query
MIN_DURATIONS = 3  # completed trials before a study has hours per trial

# --- Workers (pick.py, worker.py, status.py) --------------------------------
LOCK_BEAT_SECONDS = 60.0  # a worker rewrites the lock of its unit this often
# Beats in a row that fail (lock unreadable or missing, utime error) before
# the worker gives the unit up (well under STALE_SECONDS).
LOCK_BEAT_FAILURES = 5
BREAK_LOCK_SECONDS = 120.0  # a stale lock breaker older than this is removed
LOCK_TRIES = 3  # tries at taking a lock (a stale one is taken over)
# Reads of a lock before an unclear answer counts as lost (the safe side).
OWNED_TRIES = 3
OWNED_WAIT_SECONDS = 1.0
# The lock of `pick.advance` (one caller at a time): beaten this often and
# stale after ADVANCE_STALE_SECONDS, so a killed holder blocks little.
ADVANCE_BEAT_SECONDS = 20.0
ADVANCE_STALE_SECONDS = 120.0
# Distinct studies whose units one worker shelved in a row after failing
# early (no attempt saved a resume state: before the first validation or
# resume save) before it stops with EXIT_BROKEN (a broken machine: bad
# environment, full disk, missing mount). Any unit that ran (done, a
# trial told, shelved after a save) clears the count; out of memory, a
# configuration mismatch and exceptions of the drivers never count.
WORKER_FAILURES = 3
# Exceptions in a row of the worker loop itself (advance, pick; not of a
# driver): an alert after LOOP_ERRORS_ALERT, a stop with EXIT_RESTART after
# MAX_LOOP_ERRORS (each is followed by a wait of POLL_SECONDS).
LOOP_ERRORS_ALERT = 3
MAX_LOOP_ERRORS = 10
IDLE_MINUTES = 30.0  # nothing to pick for this long: one alert
POLL_SECONDS = 60.0  # wait between two picks of an idle worker
GPU_QUERY_TRIES = 3  # nvidia-smi queries of the GPU memory (--min_gpu_gb)
GPU_QUERY_SECONDS = 10.0  # wait between two of them
# Worker exit codes besides 0 (every unit of its models done): EXIT_BROKEN
# a broken machine (do not restart it on the same machine); EXIT_RESTART
# stopped (signal, lock lost, repeated errors, GPU memory unknown), to be
# restarted.
EXIT_BROKEN = 98
EXIT_RESTART = 99

# --- Status and leaderboard (status.py, analyze.py) -------------------------
MAX_ALERTS_SHOWN = 20  # most recent alerts listed by the status
STATUS_REFRESH_SECONDS = 600  # reload period of status.html
TOP_K = 3  # best ranked models listed per criterion (top<TOP_K>.csv)
NUM_RESAMPLES = 1000  # bootstrap resamples of the benchmark scorer

# --- Defaults of the command lines -------------------------------------------
DEFAULT_ROOT = "outputs/hpo"
DEFAULT_DATASET = "data/FLOATSense"

# --- Dry-run stub (common.stub_unit) -----------------------------------------
STUB_SECONDS = 0.0  # seconds per validation (0: instantaneous)
STUB_PARAMS = 1000  # trainable parameters reported by the stub
# Curve key of a unit (its level): a search trial uses its number, a
# confirmation unit STUB_KEY_CONFIRM + STUB_KEY_RANK * rank + seed, a final
# unit STUB_KEY_FINAL + seed.
STUB_KEY_CONFIRM = 100
STUB_KEY_RANK = 10
STUB_KEY_FINAL = 200
STUB_LEVEL_STEP = 0.618034  # golden-ratio step: distinct keys, spread levels
STUB_HALF_EPOCHS = 30.0  # epochs to reach half of the level
