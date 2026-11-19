"""Whole-clip and frame-level spectra shared by Model 1 and its checks.

Two versions of every spectrum: 'raw' (as recorded) and 'floor' (minus
each logger's own noise floor, measured on its quiet clips).
"""
import numpy as np
from scipy import signal

from src.analyze_noise_floor import (
    compute_clip_spectra_db,
    compute_excess_spectra,
    compute_logger_baselines,
    identify_floor_clips,
)
from src.compute_clip_features import build_clip_feature_table
from src.config import (
    SAMPLE_RATE_HZ,
    SPECTROGRAM_OVERLAP_SAMPLES,
    SPECTROGRAM_SEGMENT_SAMPLES,
)
from src.utils import to_db


def compute_spectrogram_db(amplitude):
    """(frequencies_hz, dB array of shape frequency x frame)."""
    frequencies_hz, _, power = signal.spectrogram(
        amplitude, fs=SAMPLE_RATE_HZ, nperseg=SPECTROGRAM_SEGMENT_SAMPLES,
        noverlap=SPECTROGRAM_OVERLAP_SAMPLES)
    return frequencies_hz, to_db(power)


def build_spectral_inputs(metadata, amplitudes):
    """Whole-clip and frame-level spectra for both versions."""
    clip_features = build_clip_feature_table(metadata, amplitudes)
    welch_frequencies_hz, welch_db = compute_clip_spectra_db(metadata,
                                                             amplitudes)
    floor_clip_ids = identify_floor_clips(clip_features)
    welch_baselines = compute_logger_baselines(metadata, welch_db,
                                               floor_clip_ids)
    welch_excess = compute_excess_spectra(metadata, welch_db,
                                          welch_baselines)
    frames = {clip_id: compute_spectrogram_db(amplitudes[clip_id])
              for clip_id in metadata['clip_id']}
    frame_frequencies_hz = next(iter(frames.values()))[0]
    logger_by_clip = metadata.set_index('clip_id')['logger_model']
    frame_baselines = {
        logger: np.median(np.hstack([
            frames[clip_id][1] for clip_id in floor_clip_ids
            if logger_by_clip[clip_id] == logger]), axis=1)
        for logger in logger_by_clip.unique()}
    frame_db = {
        'raw': {clip_id: values for clip_id, (_, values) in frames.items()},
        'floor': {clip_id: values
                  - frame_baselines[logger_by_clip[clip_id]][:, None]
                  for clip_id, (_, values) in frames.items()},
    }
    clip_db = {'raw': welch_db, 'floor': welch_excess}
    return {'clip_frequencies_hz': welch_frequencies_hz,
            'frame_frequencies_hz': frame_frequencies_hz,
            'clip_db': clip_db, 'frame_db': frame_db,
            'welch_excess': welch_excess, 'floor_clip_ids': floor_clip_ids}
