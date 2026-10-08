"""The numbers of the validation-tuned track: trials, epochs, seeds, splits,
failures and workers. The search space is in hpo/search_space.py.

Objective: VALIDATION R^2 of log10 damage, mean over the 11 heights, at the
last epoch of each run. Phase 2 confirms the top-2 configurations of each
(model, tower) over N_SEEDS seeds (winner = highest median over the seeds);
phase 3 retrains the winner on the full training split and opens the test
split once. The search has no enqueued published configuration; PUBLISHED
in hpo/search_space.py only records the fixed-budget values (all inside the
grids).

Naming: a unit or study that stopped after failures is 'parked' in the code
and in the files (PARKED markers, <root>/parked/) and 'shelved' in the
alerts, the status and the README (the README keeps 'parked' for the 22
calibration runs of the dataset).
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
MAX_ATTEMPTS = 3  # crashes before a unit is parked
MAX_FREE_RETRIES = 10  # preemptions and hardware faults (not counted)
STALE_MINUTES = 15  # a unit whose heartbeat is older belongs to a dead worker
DIVERGED_FALLBACK = -1.0  # score of a diverged trial before any completes
# Search draws above the parameter cap, or out of memory, are failed trials
# (not counted toward N_TRIALS, ignored by TPE). A study with MAX_OVER_CAP
# of one kind, or PARK_STUDY_AFTER parked trials, stops drawing: it is
# parked if it has fewer than N_TRIALS counted trials, else its plan is
# frozen from the counted ones. A confirmation or final unit out of memory
# is a crash like any other.
MAX_OVER_CAP = 20
MAX_OOM = MAX_OVER_CAP
PARK_STUDY_AFTER = 3
# A parked unit, or a study parked after its parked trials, is retried
# once, RETRY_SHELVED_AFTER seconds after its parking, with its failure
# counts reset; a second parking is final. Nothing is retried once
# READY_FOR_TEST.json or sealed/ exists. Every other parking is final.
RETRY_SHELVED_AFTER = 3600.0
# Attempts in a row that end stopped or preempted without a new resume save
# before an alert (a unit that makes no progress: the checkpoint interval is
# longer than the time the workers get).
NO_PROGRESS_ATTEMPTS = 3

# --- Workers (pick.py, worker.py, status.py) --------------------------------
LOCK_BEAT_SECONDS = 60.0  # a worker rewrites the lock of its unit this often
# Beats in a row that fail (lock unreadable or missing, utime error) before
# the worker gives the unit up (well under STALE_MINUTES).
LOCK_BEAT_FAILURES = 5
BREAK_LOCK_SECONDS = 120.0  # a stale lock breaker older than this is removed
# The lock of `pick.advance` (one caller at a time): beaten this often and
# stale after ADVANCE_STALE_SECONDS, so a killed holder blocks little.
ADVANCE_BEAT_SECONDS = 20.0
ADVANCE_STALE_SECONDS = 120.0
# Units in a row of one worker parked after their failures, from at least
# two different models, before the worker stops with EXIT_BROKEN (a broken
# machine: bad environment, full disk, missing mount). Out of memory, a
# configuration mismatch and exceptions of the drivers never count.
WORKER_FAILURES = 2
# Exceptions in a row of the worker loop or of a driver: an alert after
# LOOP_ERRORS_ALERT, a stop with EXIT_REQUEUE after MAX_LOOP_ERRORS (each
# is followed by a wait of POLL_SECONDS).
LOOP_ERRORS_ALERT = 3
MAX_LOOP_ERRORS = 10
IDLE_MINUTES = 30.0  # nothing to pick for this long: one alert
POLL_SECONDS = 60.0  # wait between two picks of an idle worker
# Worker exit codes besides 0 (every unit of its models done): EXIT_BROKEN
# a broken machine (do not restart it on the same machine); EXIT_REQUEUE
# stopped (signal, lock lost, repeated errors), to be restarted.
EXIT_BROKEN = 98
EXIT_REQUEUE = 99
# Wall time between two resume saves of a run (run.py --checkpoint_seconds;
# short, so a preempted run loses little).
CHECKPOINT_SECONDS = 90.0
