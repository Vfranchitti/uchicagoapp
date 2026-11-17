"""Load the pilot's metadata and per-clip recordings.

All other scripts read data through these functions so that file layout
and parsing rules live in exactly one place.
"""
import numpy as np
import pandas as pd
from scipy.io import wavfile

from src.config import (
    EVENING_CUTOFF_HOUR,
    HOURS_PER_DAY,
    LABELED_CLIP_DIR,
    LEAK_LABEL,
    METADATA_PATH,
    MINUTES_PER_HOUR,
    MORNING_COMMUTE_START_HOUR,
    UNLABELED_CLIP_DIR,
    WAV_INT16_FULL_SCALE,
)


def load_metadata():
    """Read metadata.csv and add analysis-ready columns.

    Added columns
    -------------
    is_labeled : bool
        Site has been dug and its outcome is known.
    is_leak : boolean (nullable)
        True/False for dug sites, <NA> for undug sites.
    overnight_hour : float
        Clock time in hours relative to midnight, with evening times
        negative (22:08 -> -1.87), so the night is a continuous axis.
    is_commute_hours : bool
        Recorded at or after the approximate morning-commute start.
    """
    df = pd.read_csv(METADATA_PATH, dtype=str, keep_default_na=False)
    df['is_labeled'] = df['labeled'].str.lower().eq('yes')
    df['is_leak'] = pd.array(
        [(label == LEAK_LABEL) if is_labeled else pd.NA
         for label, is_labeled in zip(df['label'], df['is_labeled'])],
        dtype='boolean')
    df['overnight_hour'] = df['recorded_at'].map(clock_to_overnight_hour)
    df['is_commute_hours'] = (df['overnight_hour']
                              >= MORNING_COMMUTE_START_HOUR)
    df['label_group'] = np.where(df['is_labeled'], df['label'],
                                 'unlabeled')
    return df


def clock_to_overnight_hour(clock_text):
    """Convert 'HH:MM' to hours from midnight, evenings negative."""
    hours_text, minutes_text = clock_text.split(':')
    hour = int(hours_text) + int(minutes_text) / MINUTES_PER_HOUR
    if hour >= EVENING_CUTOFF_HOUR:
        hour -= HOURS_PER_DAY
    return hour


def clip_directory(is_labeled):
    """Return the folder holding a clip's files."""
    return LABELED_CLIP_DIR if is_labeled else UNLABELED_CLIP_DIR


def load_clip_csv(clip_id, is_labeled):
    """Read one clip's CSV (columns time_s, amplitude)."""
    return pd.read_csv(clip_directory(is_labeled) / f'{clip_id}.csv')


def load_clip_wav(clip_id, is_labeled):
    """Read one clip's WAV; return (sample_rate_hz, amplitude in [-1, 1])."""
    sample_rate_hz, raw_samples = wavfile.read(
        clip_directory(is_labeled) / f'{clip_id}.wav')
    return sample_rate_hz, raw_samples.astype(float) / WAV_INT16_FULL_SCALE


def load_all_amplitudes(metadata):
    """Return {clip_id: amplitude array} from the CSV files."""
    return {
        clip_id: load_clip_csv(clip_id, is_labeled)['amplitude'].to_numpy()
        for clip_id, is_labeled in zip(metadata['clip_id'],
                                       metadata['is_labeled'])
    }


if __name__ == '__main__':
    metadata = load_metadata()
    print(metadata.head())
