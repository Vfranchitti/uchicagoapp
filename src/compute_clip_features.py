"""Summarize each 5-second recording with interpretable signal features.

Features are chosen to mirror the crew chief's field notes: a leak is a
"steady hiss, like a radio between stations" that "doesn't stop," while
street noise "comes and goes." Each feature therefore measures loudness,
steadiness over time, or how noise-like (hiss-like) the sound is.

Output: outputs/tables/2_eda/clip_features.csv (one row per clip, joined to
metadata).
"""
import numpy as np
import pandas as pd
from scipy import signal, stats

from src.config import (
    CLIP_FEATURES_PATH,
    ENVELOPE_WINDOW_SAMPLES,
    EXCESS_BANDS_HZ,
    FREQUENCY_BAND_EDGES_HZ,
    HIGH_PITCH_CUTOFF_HZ,
    HUM_FUNDAMENTAL_HZ,
    HUM_HALF_WIDTH_HZ,
    POWER_TO_DB,
    SAMPLE_RATE_HZ,
    SPECTROGRAM_OVERLAP_SAMPLES,
    SPECTROGRAM_SEGMENT_SAMPLES,
    TRANSIENT_LOUDNESS_RATIO,
    WELCH_SEGMENT_SAMPLES,
)
from src.load_recordings import load_all_amplitudes, load_metadata
from src.utils import make_output_dirs

ENVELOPE_LOW_PERCENTILE = 10
ENVELOPE_HIGH_PERCENTILE = 90

# Human-readable names used in tables and figure axes.
FEATURE_LABELS = {
    'rms_loudness': 'Overall loudness (RMS)',
    'peak_amplitude': 'Peak amplitude',
    'crest_factor': 'Crest factor (peak / RMS)',
    'excess_kurtosis': 'Spikiness (excess kurtosis)',
    'loudness_variability': 'Loudness variability (CV of 50 ms RMS)',
    'loudness_range_ratio': 'Loud-to-quiet ratio (P90 / P10 of 50 ms RMS)',
    'transient_share': 'Share of time with loud bursts',
    'spectral_flatness': 'Hiss-likeness (spectral flatness)',
    'spectral_centroid_hz': 'Spectral centroid (Hz)',
    'dominant_frequency_hz': 'Dominant frequency (Hz)',
    'band_share_0_250_hz': 'Energy share 0-250 Hz',
    'band_share_250_500_hz': 'Energy share 250-500 Hz',
    'band_share_500_1000_hz': 'Energy share 500-1,000 Hz',
    'band_share_1000_2000_hz': 'Energy share 1,000-2,000 Hz',
    'high_pitch_share': 'Energy share above 500 Hz',
}
FEATURE_COLUMNS = list(FEATURE_LABELS)
# Added for clustering; kept out of FEATURE_COLUMNS so earlier EDA outputs
# (correlations, PCA, outlier checks) stay as published.
BAND_DYNAMICS_LABELS = {
    'typical_window_rms': 'Typical loudness (median of 50 ms RMS)',
    'leak_band_level_variability_db':
        'Leak-band steadiness (SD over time of 400-1,300 Hz power, dB)',
}


def compute_rms_envelope(amplitude):
    """Return (window_center_seconds, rms) over non-overlapping windows."""
    n_windows = len(amplitude) // ENVELOPE_WINDOW_SAMPLES
    windows = amplitude[:n_windows * ENVELOPE_WINDOW_SAMPLES].reshape(
        n_windows, ENVELOPE_WINDOW_SAMPLES)
    window_rms = np.sqrt(np.mean(windows ** 2, axis=1))
    window_centers = ((np.arange(n_windows) + 0.5)
                      * ENVELOPE_WINDOW_SAMPLES / SAMPLE_RATE_HZ)
    return window_centers, window_rms


def find_longest_constant_run(amplitude):
    """Return (start_index, length) of the longest run of identical samples.

    Real sound never repeats exactly for long; a long run means the signal
    dropped out (or was pinned at a limit).
    """
    is_change = np.diff(amplitude) != 0
    run_boundaries = np.flatnonzero(np.concatenate(([True], is_change,
                                                    [True])))
    run_lengths = np.diff(run_boundaries)
    longest = int(np.argmax(run_lengths))
    return int(run_boundaries[longest]), int(run_lengths[longest])


def remove_hum_lines(frequencies_hz, values):
    """Interpolate across narrow bins at 120 Hz multiples (electrical hum).

    Works on any spectrum-like array whose first axis matches
    frequencies_hz (1-D spectrum or 2-D frequency x time).
    """
    distance_to_line = np.abs(
        (frequencies_hz + HUM_FUNDAMENTAL_HZ / 2) % HUM_FUNDAMENTAL_HZ
        - HUM_FUNDAMENTAL_HZ / 2)
    is_hum = (distance_to_line <= HUM_HALF_WIDTH_HZ) & (frequencies_hz > 0)
    cleaned = np.array(values, dtype=float, copy=True)
    if cleaned.ndim == 1:
        cleaned[is_hum] = np.interp(frequencies_hz[is_hum],
                                    frequencies_hz[~is_hum],
                                    cleaned[~is_hum])
        return cleaned
    for frame in range(cleaned.shape[1]):
        cleaned[is_hum, frame] = np.interp(frequencies_hz[is_hum],
                                           frequencies_hz[~is_hum],
                                           cleaned[~is_hum, frame])
    return cleaned


def compute_welch_psd(amplitude):
    """Return (frequencies_hz, power spectral density) via Welch's method."""
    return signal.welch(amplitude, fs=SAMPLE_RATE_HZ,
                        nperseg=WELCH_SEGMENT_SAMPLES)


def summarize_loudness(amplitude):
    """Loudness and spikiness features from the raw waveform."""
    rms_loudness = np.sqrt(np.mean(amplitude ** 2))
    peak_amplitude = np.max(np.abs(amplitude))
    return {
        'rms_loudness': rms_loudness,
        'peak_amplitude': peak_amplitude,
        'crest_factor': peak_amplitude / rms_loudness,
        'excess_kurtosis': stats.kurtosis(amplitude, fisher=True),
    }


def summarize_steadiness(amplitude):
    """How much loudness changes across the 5 seconds."""
    _, window_rms = compute_rms_envelope(amplitude)
    median_rms = np.median(window_rms)
    low_rms, high_rms = np.percentile(
        window_rms, [ENVELOPE_LOW_PERCENTILE, ENVELOPE_HIGH_PERCENTILE])
    # A dropout (silent stretch) makes the quiet percentile zero; the ratio
    # is then undefined rather than infinite.
    loudness_range_ratio = high_rms / low_rms if low_rms > 0 else np.nan
    return {
        'loudness_variability': np.std(window_rms) / np.mean(window_rms),
        'loudness_range_ratio': loudness_range_ratio,
        'transient_share': np.mean(
            window_rms > TRANSIENT_LOUDNESS_RATIO * median_rms),
    }


def summarize_spectrum(amplitude):
    """Pitch-content features from the Welch power spectrum."""
    frequencies_hz, psd = compute_welch_psd(amplitude)
    # Drop the 0 Hz bin: it measures DC offset, not sound, and its value
    # can be ~0, which breaks the geometric mean.
    is_audible_bin = frequencies_hz > 0
    frequencies_hz = frequencies_hz[is_audible_bin]
    psd = psd[is_audible_bin]
    features = {
        'spectral_flatness': stats.gmean(psd) / np.mean(psd),
        'spectral_centroid_hz': np.sum(frequencies_hz * psd) / np.sum(psd),
        'dominant_frequency_hz': frequencies_hz[np.argmax(psd)],
    }
    total_power = np.sum(psd)
    band_edges = FREQUENCY_BAND_EDGES_HZ
    for low_hz, high_hz in zip(band_edges[:-1], band_edges[1:]):
        is_in_band = (frequencies_hz > low_hz) & (frequencies_hz <= high_hz)
        column = f'band_share_{low_hz:.0f}_{high_hz:.0f}_hz'
        features[column] = np.sum(psd[is_in_band]) / total_power
    is_high_pitch = frequencies_hz > HIGH_PITCH_CUTOFF_HZ
    features['high_pitch_share'] = np.sum(psd[is_high_pitch]) / total_power
    return features


def summarize_band_dynamics(amplitude):
    """Burst-resistant loudness and how steady the leak band is over time.

    typical_window_rms: median 50 ms loudness, so brief traffic bursts do
    not inflate it. leak_band_level_variability_db: standard deviation
    over time of 400-1,300 Hz power in dB. The crew chief's "it doesn't
    stop" applied to the band: a steady leak gives a small value. dB is
    used rather than a raw coefficient of variation so one loud burst does
    not dominate.
    """
    _, window_rms = compute_rms_envelope(amplitude)
    frequencies_hz, _, power = signal.spectrogram(
        amplitude, fs=SAMPLE_RATE_HZ, nperseg=SPECTROGRAM_SEGMENT_SAMPLES,
        noverlap=SPECTROGRAM_OVERLAP_SAMPLES)
    low_hz, high_hz = EXCESS_BANDS_HZ['leak_band_400_1300']
    is_leak_band = (frequencies_hz > low_hz) & (frequencies_hz <= high_hz)
    band_power_db = POWER_TO_DB * np.log10(
        power[is_leak_band].sum(axis=0) + np.finfo(float).tiny)
    return {
        'typical_window_rms': np.median(window_rms),
        'leak_band_level_variability_db': np.std(band_power_db),
    }


def summarize_clip(amplitude):
    """Combine all per-clip features into one dict."""
    return {**summarize_loudness(amplitude),
            **summarize_steadiness(amplitude),
            **summarize_spectrum(amplitude),
            **summarize_band_dynamics(amplitude)}


def build_clip_feature_table(metadata, amplitudes):
    """One row per clip: metadata columns plus signal features."""
    feature_rows = pd.DataFrame(
        [{'clip_id': clip_id, **summarize_clip(amplitudes[clip_id])}
         for clip_id in metadata['clip_id']])
    return metadata.merge(feature_rows, on='clip_id', validate='one_to_one')


if __name__ == '__main__':
    metadata = load_metadata()
    amplitudes = load_all_amplitudes(metadata)
    clip_features = build_clip_feature_table(metadata, amplitudes)
    make_output_dirs()
    clip_features.to_csv(CLIP_FEATURES_PATH, index=False)
    print(f'Wrote {len(clip_features)} rows to {CLIP_FEATURES_PATH}')
