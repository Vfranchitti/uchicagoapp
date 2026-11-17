"""Project-wide paths and constants.

Every file path and every number with a physical or policy meaning is
defined here once and imported elsewhere (Sullivan, "Don't Repeat
Yourself" and "No magic numbers").
"""
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = REPO_ROOT / 'data'
METADATA_PATH = DATA_DIR / 'metadata.csv'
PREDICTIONS_TEMPLATE_PATH = DATA_DIR / 'predictions_template.csv'
LABELED_CLIP_DIR = DATA_DIR / 'labeled'
UNLABELED_CLIP_DIR = DATA_DIR / 'unlabeled'

OUTPUT_DIR = REPO_ROOT / 'outputs'
TABLE_DIR = OUTPUT_DIR / 'tables'
EDA_FIGURE_DIR = OUTPUT_DIR / 'figures' / 'eda'

DATA_CHECKS_CSV_PATH = TABLE_DIR / 'data_checks.csv'
DATA_CHECKS_MARKDOWN_PATH = TABLE_DIR / 'data_checks.md'
CLIP_FEATURES_PATH = TABLE_DIR / 'clip_features.csv'
COVERAGE_TABLE_PATH = TABLE_DIR / 'coverage_labeled_vs_unlabeled.csv'

# ---------------------------------------------------------------------------
# Recording format (from the Candidate Instructions)
# ---------------------------------------------------------------------------
SAMPLE_RATE_HZ = 4000
CLIP_DURATION_SECONDS = 5
SAMPLES_PER_CLIP = SAMPLE_RATE_HZ * CLIP_DURATION_SECONDS
AMPLITUDE_LIMIT = 1.0          # amplitudes are scaled to [-1, 1]
WAV_INT16_FULL_SCALE = 32768   # int16 WAV -> float: divide by 2**15

N_CLIPS_EXPECTED = 46
N_LABELED_EXPECTED = 6
N_UNLABELED_EXPECTED = 40

# ---------------------------------------------------------------------------
# Metadata vocabularies
# ---------------------------------------------------------------------------
LOGGER_MODELS = ('A', 'B')
PIPE_MATERIALS = ('cast_iron', 'PVC')
LEAK_LABEL = 'leak'
NO_LEAK_LABEL = 'no_leak'
VALID_LABELS = (LEAK_LABEL, NO_LEAK_LABEL)

# ---------------------------------------------------------------------------
# Decision context (from the Candidate Instructions)
# ---------------------------------------------------------------------------
HISTORICAL_LEAK_RATE = 1 / 8   # ~1 in 8 surveyed sites has a leak
MAX_DIGS_THIS_WEEK = 5
COST_PER_DIG_DOLLARS = 8000

# ---------------------------------------------------------------------------
# Clock-time handling
# ---------------------------------------------------------------------------
# Recordings are overnight. Hours after this cutoff are treated as the
# previous evening (e.g., 22:08 -> -1.87 h) so the night reads left to right.
EVENING_CUTOFF_HOUR = 12
HOURS_PER_DAY = 24
MINUTES_PER_HOUR = 60
# The crew chief notes some loggers were still recording when "the morning
# commute started." 6:00 a.m. is used as an approximate start, not a
# measured one; sensitivity to this choice should be checked.
MORNING_COMMUTE_START_HOUR = 6.0

# ---------------------------------------------------------------------------
# Signal-summary settings
# ---------------------------------------------------------------------------
# Short-window loudness envelope: 50 ms windows resolve passing vehicles and
# impacts (which last tenths of a second) while averaging out the hiss.
ENVELOPE_WINDOW_SECONDS = 0.05
ENVELOPE_WINDOW_SAMPLES = int(ENVELOPE_WINDOW_SECONDS * SAMPLE_RATE_HZ)
# A window counts as a "transient" if it is this many times louder than the
# clip's median window (a robust baseline for the steady background).
TRANSIENT_LOUDNESS_RATIO = 3.0
# Welch spectrum segment length: 0.25 s gives 4 Hz frequency resolution.
WELCH_SEGMENT_SAMPLES = 1000
# Frequency bands (Hz) for band-energy shares; upper edge is Nyquist.
NYQUIST_HZ = SAMPLE_RATE_HZ / 2
FREQUENCY_BAND_EDGES_HZ = (0, 250, 500, 1000, NYQUIST_HZ)
# Traffic and heavy impacts are mostly low-frequency rumble; a hiss spreads
# energy to higher pitches. Share of energy above this cutoff summarizes
# "how high-pitched" a recording is.
HIGH_PITCH_CUTOFF_HZ = 500

# ---------------------------------------------------------------------------
# Data-quality thresholds
# ---------------------------------------------------------------------------
# Saturated samples pile up at 0.98999, just under full scale, so the
# effective sensor ceiling (not 1.0) defines clipping.
CLIPPING_THRESHOLD = 0.98          # |amplitude| at/above this = clipped
CSV_WAV_TOLERANCE = 1e-4           # CSV stores 5 decimals; allow rounding
TIME_GRID_TOLERANCE_SECONDS = 1e-4  # CSV stores 4 decimals of time
MAX_ALLOWED_FLATLINE_SAMPLES = 20  # identical consecutive samples (5 ms)
DUPLICATE_CORRELATION_THRESHOLD = 0.95
NEAR_SILENT_RMS = 1e-3
OUTLIER_ROBUST_Z = 3.5             # |robust z| above this = outlier

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
FIGURE_DPI = 300
DATA_SOURCE_NOTE = ('Data source: city water department acoustic logger '
                    'pilot, 46 overnight recordings.')
