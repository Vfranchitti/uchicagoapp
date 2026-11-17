"""Project-wide paths and constants.

Every file path and every number with a physical or policy meaning is
defined here once and imported elsewhere (Sullivan, "Don't Repeat
Yourself" and "No magic numbers"). Sections follow the pipeline order in
README.md: data -> quality checks -> exploratory analysis -> Model 1 ->
Model 2 -> robustness and sanity checks.
"""
from pathlib import Path


# ===========================================================================
# PATHS
# ===========================================================================
REPO_ROOT = Path(__file__).resolve().parents[1]

# Inputs (read-only)
DATA_DIR = REPO_ROOT / 'data'
METADATA_PATH = DATA_DIR / 'metadata.csv'
PREDICTIONS_TEMPLATE_PATH = DATA_DIR / 'predictions_template.csv'
LABELED_CLIP_DIR = DATA_DIR / 'labeled'
UNLABELED_CLIP_DIR = DATA_DIR / 'unlabeled'

# Output folders, one per pipeline stage
OUTPUT_DIR = REPO_ROOT / 'outputs'
DELIVERABLES_DIR = OUTPUT_DIR / 'deliverables'
TABLE_DIR = OUTPUT_DIR / 'tables'
QUALITY_TABLE_DIR = TABLE_DIR / '1_quality'
EDA_TABLE_DIR = TABLE_DIR / '2_eda'
MODEL1_TABLE_DIR = TABLE_DIR / '3_model1'
MODEL2_TABLE_DIR = TABLE_DIR / '4_model2'
CHECKS_TABLE_DIR = TABLE_DIR / '5_checks'
FIGURE_DIR = OUTPUT_DIR / 'figures'
EDA_FIGURE_DIR = FIGURE_DIR / '1_eda'
MODEL_FIGURE_DIR = FIGURE_DIR / '2_model'
CHECKS_FIGURE_DIR = FIGURE_DIR / '3_checks'
OUTPUT_SUBDIRS = (DELIVERABLES_DIR, QUALITY_TABLE_DIR, EDA_TABLE_DIR,
                  MODEL1_TABLE_DIR, MODEL2_TABLE_DIR, CHECKS_TABLE_DIR,
                  EDA_FIGURE_DIR, MODEL_FIGURE_DIR, CHECKS_FIGURE_DIR)

# Deliverable. Rename with the application reference number before
# submitting (e.g., 1234567_predictions.csv).
PREDICTIONS_OUTPUT_PATH = DELIVERABLES_DIR / 'predictions.csv'

# Stage 1: data quality
DATA_CHECKS_CSV_PATH = QUALITY_TABLE_DIR / 'data_checks.csv'
DATA_CHECKS_MARKDOWN_PATH = QUALITY_TABLE_DIR / 'data_checks.md'
COVERAGE_TABLE_PATH = QUALITY_TABLE_DIR / 'coverage_labeled_vs_unlabeled.csv'
QUALITY_AUDIT_CSV_PATH = QUALITY_TABLE_DIR / 'quality_audit.csv'
QUALITY_AUDIT_MARKDOWN_PATH = QUALITY_TABLE_DIR / 'quality_audit.md'
DATA_TYPE_INVENTORY_PATH = QUALITY_TABLE_DIR / 'data_type_inventory.csv'
BASE_RATE_TABLE_PATH = QUALITY_TABLE_DIR / 'base_rate_ppv_scenarios.csv'
BASE_RATE_SUMMARY_PATH = QUALITY_TABLE_DIR / 'base_rate_summary.md'

# Stage 2: exploratory analysis and band measures
CLIP_FEATURES_PATH = EDA_TABLE_DIR / 'clip_features.csv'
NOISE_FLOOR_SPECTRA_PATH = EDA_TABLE_DIR / 'noise_floor_spectra_by_logger.csv'
NOISE_FLOOR_SUMMARY_PATH = EDA_TABLE_DIR / 'noise_floor_summary.md'
EXCESS_BAND_TABLE_PATH = EDA_TABLE_DIR / 'excess_over_noise_floor_by_band.csv'
BAND_GEOMETRY_PATH = EDA_TABLE_DIR / 'band_geometry.csv'
BAND_EVENNESS_PATH = EDA_TABLE_DIR / 'band_evenness.csv'
CLUSTER_TABLE_PATH = EDA_TABLE_DIR / 'cluster_assignments.csv'
CLUSTER_SUMMARY_PATH = EDA_TABLE_DIR / 'cluster_summary.md'
CLUSTER_FEATURE_TABLE_PATH = EDA_TABLE_DIR / 'cluster_features.csv'

# Stage 3: Model 1 (knowledge-based score) and its probabilities
MODEL1_SCORES_PATH = MODEL1_TABLE_DIR / 'model1_scores.csv'
MODEL1_LOO_PATH = MODEL1_TABLE_DIR / 'model1_leave_one_out.csv'
MODEL1_SUMMARY_PATH = MODEL1_TABLE_DIR / 'model1_summary.md'
MODEL1_PROBABILITY_PATH = MODEL1_TABLE_DIR / 'model1_probabilities.csv'
MODEL1_PROBABILITY_SUMMARY_PATH = (MODEL1_TABLE_DIR
                                   / 'model1_probability_summary.md')

# Stage 4: Model 2 (Firth logistic regression)
MODEL2_PROBABILITY_PATH = MODEL2_TABLE_DIR / 'model2_probabilities.csv'
MODEL2_FITS_PATH = MODEL2_TABLE_DIR / 'model2_fits.csv'
MODEL2_SUMMARY_PATH = MODEL2_TABLE_DIR / 'model2_summary.md'

# Stage 5: robustness and sanity checks (outside the model)
SANITY_PVC_PREDICTION_TABLE_PATH = (CHECKS_TABLE_DIR
                                    / 'sanity_predicted_pvc_leak.csv')
SANITY_PVC_SITE_TABLE_PATH = (CHECKS_TABLE_DIR
                              / 'sanity_pvc_sites_vs_prediction.csv')
SANITY_PVC_VS_MODEL_PATH = CHECKS_TABLE_DIR / 'sanity_pvc_vs_model_calls.csv'
SANITY_PVC_FIGURE_PATH = CHECKS_FIGURE_DIR / 'sanity_predicted_pvc_leak.png'
SELF_TRAINING_TABLE_PATH = CHECKS_TABLE_DIR / 'check_self_training.csv'
SELF_TRAINING_SUMMARY_PATH = CHECKS_TABLE_DIR / 'check_self_training.md'


# ===========================================================================
# DATA FORMAT AND CONTEXT (from the Candidate Instructions)
# ===========================================================================
SAMPLE_RATE_HZ = 4000
CLIP_DURATION_SECONDS = 5
SAMPLES_PER_CLIP = SAMPLE_RATE_HZ * CLIP_DURATION_SECONDS
AMPLITUDE_LIMIT = 1.0          # amplitudes are scaled to [-1, 1]
WAV_INT16_FULL_SCALE = 32768   # int16 WAV -> float: divide by 2**15

N_CLIPS_EXPECTED = 46
N_LABELED_EXPECTED = 6
N_UNLABELED_EXPECTED = 40

LOGGER_MODELS = ('A', 'B')
PIPE_MATERIALS = ('cast_iron', 'PVC')
LEAK_LABEL = 'leak'
NO_LEAK_LABEL = 'no_leak'
VALID_LABELS = (LEAK_LABEL, NO_LEAK_LABEL)
DUG_LABELS = VALID_LABELS      # label groups of the six dug sites

HISTORICAL_LEAK_RATE = 1 / 8   # ~1 in 8 surveyed sites has a leak
MAX_DIGS_THIS_WEEK = 5
COST_PER_DIG_DOLLARS = 8000

# Recordings are overnight. Hours after this cutoff are treated as the
# previous evening (e.g., 22:08 -> -1.87 h) so the night reads left to right.
EVENING_CUTOFF_HOUR = 12
HOURS_PER_DAY = 24
MINUTES_PER_HOUR = 60
# The crew chief notes some loggers were still recording when "the morning
# commute started." 6:00 a.m. is used as an approximate start, not a
# measured one.
MORNING_COMMUTE_START_HOUR = 6.0


# ===========================================================================
# SIGNAL PROCESSING
# ===========================================================================
POWER_TO_DB = 10               # decibels = 10 x log10(power ratio)
# Short-window loudness envelope: 50 ms windows resolve passing vehicles and
# impacts (which last tenths of a second) while averaging out the hiss.
ENVELOPE_WINDOW_SECONDS = 0.05
ENVELOPE_WINDOW_SAMPLES = int(ENVELOPE_WINDOW_SECONDS * SAMPLE_RATE_HZ)
# A window counts as a "transient" if it is this many times louder than the
# clip's median window (a robust baseline for the steady background).
TRANSIENT_LOUDNESS_RATIO = 3.0
# Welch spectrum segment length: 0.25 s gives 4 Hz frequency resolution.
WELCH_SEGMENT_SAMPLES = 1000
# Spectrogram frames: 64 ms windows with 50% overlap (about 155 frames per
# clip) resolve how band power changes over the 5 seconds.
SPECTROGRAM_SEGMENT_SAMPLES = 256
SPECTROGRAM_OVERLAP_SAMPLES = 128
# Frequency bands (Hz) for band-energy shares; upper edge is Nyquist.
NYQUIST_HZ = SAMPLE_RATE_HZ / 2
FREQUENCY_BAND_EDGES_HZ = (0, 250, 500, 1000, NYQUIST_HZ)
# Traffic and heavy impacts are mostly low-frequency rumble; a hiss spreads
# energy to higher pitches. Share of energy above this cutoff summarizes
# "how high-pitched" a recording is.
HIGH_PITCH_CUTOFF_HZ = 500
# Narrow electrical-hum lines at 120 Hz multiples (U02, U15, U16, U22).
HUM_FUNDAMENTAL_HZ = 120
HUM_HALF_WIDTH_HZ = 8


# ===========================================================================
# STAGE 1: DATA QUALITY
# ===========================================================================
# Saturated samples pile up at 0.98999, just under full scale, so the
# effective sensor ceiling (not 1.0) defines clipping.
CLIPPING_THRESHOLD = 0.98          # |amplitude| at/above this = clipped
CSV_WAV_TOLERANCE = 1e-4           # CSV stores 5 decimals; allow rounding
TIME_GRID_TOLERANCE_SECONDS = 1e-4  # CSV stores 4 decimals of time
MAX_ALLOWED_FLATLINE_SAMPLES = 20  # identical consecutive samples (5 ms)
DUPLICATE_CORRELATION_THRESHOLD = 0.95
NEAR_SILENT_RMS = 1e-3
OUTLIER_ROBUST_Z = 3.5             # |robust z| above this = outlier
# Below this RMS a clip is at the sensor's own noise floor (nothing loud
# enough in the pipe to register); see the quiet cluster in the EDA.
NOISE_FLOOR_RMS = 0.01
# Documented formats (Candidate Instructions, "The Data").
SITE_ID_PATTERN = r'^HYD-\d{4}$'
CLIP_ID_PATTERN = r'^[LU]\d{2}$'
CLOCK_TIME_PATTERN = r'^([01]\d|2[0-3]):[0-5]\d$'
VALID_LABELED_FLAGS = ('yes', 'no')
# Hand-typed times tend to "heap" on round minutes (:00, :05, ...). With
# no heaping, about 2 in 10 minute values end in 0 or 5.
ROUND_MINUTE_DIVISOR = 5
EXPECTED_ROUND_MINUTE_SHARE = 0.2
SIGNIFICANCE_LEVEL = 0.05
# Overnight window (hours from midnight; evening negative): 8 pm-8 am.
OVERNIGHT_WINDOW_HOURS = (-4.0, 8.0)
# A clip with more than 1% of samples at the sensor ceiling is overloaded.
MAX_CLIPPED_SHARE = 0.01

# Recordings removed from clustering and modeling. U40 is a low-quality
# recording, not fit for analysis: signal saturation (15% of samples
# clipped at sensor full scale) and a 1.2 s dropout (constant zero-valued
# samples) make its amplitude and spectral features invalid.
ANALYSIS_EXCLUDED_CLIPS = ('U40',)
ANALYSIS_EXCLUSION_REASON = (
    'U40 excluded (insufficient data quality): signal saturation (15% of '
    'samples clipped) and a 1.2 s dropout make its features invalid.')


# ===========================================================================
# STAGE 2: EXPLORATORY ANALYSIS AND BAND MEASURES
# ===========================================================================
# Bands (Hz) for "power above the logger's own noise floor." 250-400 Hz is
# the edge zone between low-frequency rumble and the leak band; 400-1,300 Hz
# is the band all three dug leaks share (EDA Figs 5-6).
EXCESS_BANDS_HZ = {
    'low_0_250': (0, 250),
    'edge_250_400': (250, 400),
    'leak_band_400_1300': (400, 1300),
    'high_1300_2000': (1300, NYQUIST_HZ),
}
# A band counts as "present" when its median power is this far above the
# logger's noise floor (3 dB = twice the power).
BAND_PRESENT_EXCESS_DB = 3.0
# Quiet clips left out of the noise-floor baseline even though they fall
# below NOISE_FLOOR_RMS. U34 (PVC) carries a steady 250-400 Hz band about
# 19 dB above the floor, so it is not pure device noise.
EXCLUDED_FROM_NOISE_FLOOR = ('U34',)

CLUSTER_COUNTS = (2, 3, 4)
KMEANS_RANDOM_SEED = 20261117
KMEANS_RESTARTS = 50


# ===========================================================================
# STAGE 3: MODEL 1 (knowledge-based score) AND ITS PROBABILITIES
# ===========================================================================
# Two versions of every spectral measure: 'floor' (each logger's noise floor
# subtracted; the main analysis) and 'raw' (as recorded; robustness check).
MAIN_VERSION = 'floor'
MODEL_VERSIONS = ('floor', 'raw')
# A frame "has the band" when the band stands out at least this many dB
# above the average of its two neighboring bands (2x the power).
PERSISTENCE_CONTRAST_DB = 3.0
# Stated before scoring, from the crew chief's "it doesn't stop": a leak
# band should be present in at least 90% of 64 ms frames. Sites below this
# have their score scaled down in proportion (persistence / 0.9).
PERSISTENCE_CUTOFF = 0.9
# Neighbor bands for the PVC low-band contrast: rumble below, leak band
# above. The PVC band itself is the fixed 250-400 Hz edge band.
PVC_BAND_KEY = 'edge_250_400'
PVC_NEIGHBOR_KEYS = ('low_0_250', 'leak_band_400_1300')
# A site "has a persistent band" when the band is present in at least
# PERSISTENCE_CUTOFF of frames and stands out at least this much overall.
PERSISTENT_BAND_MIN_CONTRAST_DB = 3.0

# Pseudo-counts added to each cell of the 2 x 2 dug-site table, because 3/3
# and 0/3 would otherwise imply certainty: Jeffreys (0.5) and Laplace (1).
SMOOTHING_PSEUDO_COUNTS = (0.5, 1.0)
# Fifth dig chosen as an informative dig (approved 2026-11-17): the
# strongest site with the unconfirmed PVC band. Its result would give the
# first labeled PVC example.
INFORMATIVE_DIG_SITE = 'U34'

# Graded confidence within each tier (approved 2026-11-17). Presence sets
# the tier; quality places a site inside the tier's range (on log-odds).
# quality = steadiness^STEADINESS_WEIGHT x intenseness^INTENSENESS_WEIGHT
#   steadiness  = persistence over time x coverage across pitch, each
#                 relative to the dug leaks (capped at 1)
#   intenseness = contrast in dB relative to the weakest dug leak (capped
#                 at 1: more intense than a confirmed leak earns no extra)
STEADINESS_WEIGHT = 0.6
INTENSENESS_WEIGHT = 0.4
# PVC signature-match discount: share of these leak-signature properties a
# PVC band site shares (w), then p = w x graded p + (1 - w) x base rate.
#   persistent over time       persistence >= PERSISTENCE_CUTOFF
#   stands out                 contrast >= PERSISTENT_BAND_MIN_CONTRAST_DB
#   even across its band       coverage >= PERSISTENCE_CUTOFF
#   leak-like width            |width - median leak width| <= tolerance
#   gap above background       gap >= fraction x smallest leak gap
SIGNATURE_WIDTH_TOLERANCE_OCTAVES = 0.5
SIGNATURE_GAP_FRACTION = 0.5


# ===========================================================================
# STAGE 4: MODEL 2 (Firth-penalized logistic regression on the M1 score)
# ===========================================================================
FIRTH_MAX_ITERATIONS = 200
FIRTH_TOLERANCE = 1e-9


# ===========================================================================
# STAGE 5: ROBUSTNESS AND SANITY CHECKS
# ===========================================================================
# Literature-based comparisons. Nothing in feature, clustering or modeling
# code may import from src/sanity_* modules: the model uses only the pilot
# data provided.
# Label spreading: k nearest neighbors in standardized feature space, and
# how much each labeled site may move away from its own label (alpha).
LABEL_SPREADING_NEIGHBORS = 5
LABEL_SPREADING_ALPHA = 0.2
# Self-training: per round, add undug sites whose sample-scale probability
# is above / below these cut-offs as pseudo-labels, then refit.
SELF_TRAINING_LEAK_CUTOFF = 0.9
SELF_TRAINING_NO_LEAK_CUTOFF = 0.1
SELF_TRAINING_MAX_ROUNDS = 10


# ===========================================================================
# OUTPUT FORMATTING
# ===========================================================================
TABLE_DECIMALS = 4             # rounding for saved tables
SUMMARY_TABLE_ROWS = 12        # rows shown in Markdown summary tables
FIGURE_DPI = 300
DATA_SOURCE_NOTE = ('Data source: city water department acoustic logger '
                    'pilot, 46 overnight recordings.')
