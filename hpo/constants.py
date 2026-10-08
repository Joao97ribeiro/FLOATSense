"""The numbers of the validation-tuned track: trials, epochs, seeds, splits,
failures and workers. The search space is in hpo/search_space.py.

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
# Draws above the parameter cap are failed trials (not counted toward
# N_TRIALS, ignored by TPE); a study with this many is parked.
MAX_OVER_CAP = 20
# A parked unit (or study) whose failures all came from one host is un-parked
# once, by the first worker of another host that picks it.

# --- Workers (pick.py, worker.py, status.py) --------------------------------
LOCK_BEAT_SECONDS = 60.0  # a worker rewrites the lock of its unit this often
BREAK_LOCK_SECONDS = 120.0  # a stale lock breaker older than this is removed
PARK_STUDY_AFTER = 3  # parked trials before a search study is parked
# Consecutive units of one worker ending parked or crashed before it stops
# picking (a broken node: bad environment, full disk, missing mount).
WORKER_FAILURES = 2
IDLE_MINUTES = 30.0  # a worker with nothing to pick exits after this
POLL_SECONDS = 60.0  # wait between two picks of an idle worker
EXIT_REQUEUE = 99  # worker exit code: stopped mid-unit, to be restarted
EXIT_BROKEN = 98  # worker exit code: WORKER_FAILURES failed units in a row
# Wall time between two resume saves of a run (run.py --checkpoint_seconds;
# short, so a preempted run loses little).
CHECKPOINT_SECONDS = 90.0
