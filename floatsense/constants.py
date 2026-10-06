"""Fixed values of the FLOATSense benchmark, defined once.

Every module and the defaults of the script flags read these values; the
configs in scripts/*/config.cfg repeat them as the record of a run. The
geometry of each tower (gauge heights, radius, thickness, mass per length)
is data and comes with the dataset (sections.parquet) and towers/; the
training recipe is in the trainer and the configs.
"""

# --- Data and scored window ------------------------------------------------
SAMPLING_FREQUENCY = 10.0  # [Hz] of the released series
MIN_TIME = 400.0  # start of the scored window [s]
MAX_TIME = 1000.0  # end of the scored window [s], inclusive (6,001 samples)
NUM_SECTIONS = 30  # FLOATBench tower sections

# The 11 moment gauges, base to top: channel stem, zero-based FLOATBench
# section scored (section_id - 1) and gauge height above the base as a
# fraction of the tower height (model input of the 11-height task; the exact
# heights are in sections.parquet).
HEIGHT_TARGETS = ([("tower_bottom", 0, 0.0)] + [
    (f"tower_{k}", 3 * k - 1, z / 149.386)
    for k, z in zip(range(1, 10), (12.4488, 27.3874, 42.3260, 57.2646, 72.2032,
                                   87.1418, 102.0804, 117.0190, 131.9576))
] + [("tower_top", 29, 1.0)])

# --- Damage metric ---------------------------------------------------------
LOWPASS_HZ = 3.0  # zero-phase Butterworth cutoff on true and predicted [Hz]
LOWPASS_ORDER = 4  # order of one pass (sosfiltfilt runs two)
SN_INTERCEPTS_LOG10 = (12.010, 15.350)  # DNV-RP-C203 bilinear S-N curve
SN_SLOPES = (3.0, 5.0)
THICKNESS_REFERENCE_MM = 25.0  # S-N thickness correction
THICKNESS_EXPONENT = 0.2
FATIGUE_LIFE_THRESHOLD = 1e7  # cycles at the S-N slope change

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
