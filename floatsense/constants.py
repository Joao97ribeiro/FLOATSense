"""Fixed values of the FLOATSense benchmark, defined once.

Every module and the defaults of the script flags read these values; the
configs in scripts/*/config.cfg repeat them as the record of a run. The
gauges of each tower (name, FLOATBench section, height, radius, thickness)
and its mass per length are data (sections.parquet and towers/); the
training recipe is in the trainer and the configs.
"""

# --- Data and scored window ------------------------------------------------
SAMPLING_FREQUENCY = 10.0  # [Hz] of the released series
MIN_TIME = 400.0  # start of the scored window [s]
MAX_TIME = 1000.0  # end of the scored window [s], inclusive (6,001 samples)
# Input length of every crop-trained model at evaluation: the length the
# paper checkpoints were scored at (a multiple of the PatchTST and U-Net patch
# and pooling sizes); the 6,001-sample window is predicted from the inputs at
# offsets 0 and 1, stitched (see Trainer._predict_window).
INPUT_LENGTH = 6000
# Channel stems of the 11 gauges, base to top (the moments are
# <stem>_mfa and <stem>_mss; sections.parquet gives their heights).
GAUGE_STEMS = (("tower_bottom",) + tuple(f"tower_{i}" for i in range(1, 10)) +
               ("tower_top",))

# --- Damage metric ---------------------------------------------------------
LOWPASS_HZ = 3.0  # zero-phase Butterworth cutoff on true and predicted [Hz]
LOWPASS_ORDER = 4  # order of one pass (sosfiltfilt runs two)
SN_INTERCEPTS_LOG10 = (12.010, 15.350)  # DNV-RP-C203 bilinear S-N curve
SN_SLOPES = (3.0, 5.0)
THICKNESS_REFERENCE_MM = 25.0  # S-N thickness correction
THICKNESS_EXPONENT = 0.2
FATIGUE_LIFE_THRESHOLD = 1e7  # cycles at the S-N slope change
DAMAGE_WORKERS = 8  # processes of the rainflow damage pool (at most)

# --- Physics baseline (Pimenta et al. 2024, as adapted here) ---------------
BAND_HZ = 3.0  # upper edge of the reconstruction band [Hz]
HARMONIC_ORDERS = (3, 6, 9)  # rotor harmonics with their own gain
PAD_SECONDS = 50.0  # margin around the scored window [s]
SEGMENT_LENGTH = 4096  # Welch segment of the calibration
LF_FIT_BAND = (0.01, 0.05)  # bins fitting the low-frequency gain [Hz]
OPERATING_POWER_KW = 100.0  # power above which a sample counts as operating

# --- Rotor-nacelle assembly of the IEA 22 MW (ElastoDyn inputs) ------------
NAC_MASS = 821239.8004933242  # nacelle mass [kg]
NAC_CM_Z = 4.2647901842947595  # nacelle centre of mass above the yaw [m]
HUB_MASS = 120447.70224890654  # [kg]
YAW_MASS = 28740.99049474962  # [kg]
BLADE_MASS = 82427.5  # one blade [kg]
TWR2SHFT = 4.142540706280534  # tower top to shaft [m]
OVERHANG = -14.07711591388923  # [m]
SHFT_TILT = -6.0  # shaft tilt [deg]

# --- Training runs (validation-tuned track) --------------------------------
CHECKPOINT_SECONDS = 300.0  # resume state at least this often [s wall time]
# A temporary of a save (killed writer) is removed once older than
# max(STALE_TEMPORARY_INTERVALS checkpoint intervals, STALE_TEMPORARY_SECONDS);
# a younger one may be the save in flight of a concurrent writer.
STALE_TEMPORARY_INTERVALS = 3.0
STALE_TEMPORARY_SECONDS = 3600.0  # [s]
TEMPORARY_TAG_LENGTH = 8  # hex characters of the random tag of a temporary
EXIT_DIVERGED = 3  # exit code: non-finite loss or validation prediction
EXIT_TOO_LARGE = 4  # exit code: more trainable parameters than --max_params_m
EXIT_STOPPED = 5  # exit code: SIGUSR1, resume state saved at the epoch end
EXIT_CONFIG_MISMATCH = 6  # exit code: resume state of another run config
