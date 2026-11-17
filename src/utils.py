"""Small helpers shared across the pipeline (defined once, imported)."""
import numpy as np

from src.config import OUTPUT_SUBDIRS, POWER_TO_DB


def make_output_dirs():
    """Create every output folder (no-op if they exist)."""
    for folder in OUTPUT_SUBDIRS:
        folder.mkdir(parents=True, exist_ok=True)


def to_db(power):
    """Power to decibels, guarding against log(0)."""
    return POWER_TO_DB * np.log10(np.asarray(power)
                                  + np.finfo(float).tiny)


def in_band(frequencies_hz, band_hz):
    """Boolean mask for frequencies in (low, high]."""
    low_hz, high_hz = band_hz
    return (frequencies_hz > low_hz) & (frequencies_hz <= high_hz)


def moving_average(values, n_bins):
    """Centered moving average over n_bins (edges padded with zeros)."""
    kernel = np.ones(n_bins) / n_bins
    return np.convolve(values, kernel, mode='same')
