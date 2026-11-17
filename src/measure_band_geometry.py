"""Compare where each contrasting band sits relative to the background band.

The background band is the logger's own low-pitch shelf: its noise floor is
about 14 dB louder below roughly 300 Hz (EDA Fig. 18). For each recording
with a contrasting band (the three dug cast-iron leaks and the two PVC
sites U34 and U15), this script measures the band's edges and level, then
compares them with the top of the background shelf: the gap between the
two, the ratio of band edges to the shelf edge, and the band's width in
octaves. Only pilot data are used.

Output: outputs/tables/2_eda/band_geometry.csv
"""
import numpy as np
import pandas as pd

from src.analyze_noise_floor import (
    compute_clip_spectra_db,
    compute_excess_spectra,
    compute_logger_baselines,
    identify_floor_clips,
)
from src.compute_clip_features import (
    build_clip_feature_table,
    remove_hum_lines,
)
from src.config import BAND_GEOMETRY_PATH, LEAK_LABEL, NYQUIST_HZ
from src.load_recordings import load_all_amplitudes, load_metadata
from src.utils import make_output_dirs, moving_average

PVC_BAND_CLIPS = ('U34', 'U15')
# Search above the logger's low shelf for the contrasting band.
BAND_SEARCH_HZ = (250, NYQUIST_HZ)
# Band edges: where power falls below half the band's level (in dB).
EDGE_LEVEL_FRACTION = 0.5
# Shelf edge: where the floor crosses midway between its two plateaus.
SHELF_LOW_PLATEAU_HZ = (50, 250)
SHELF_HIGH_PLATEAU_HZ = (500, 1500)
SHELF_SEARCH_HZ = (200, 500)
EDGE_SMOOTHING_BINS = 3


def _band_mask(frequencies_hz, band_hz):
    low_hz, high_hz = band_hz
    return (frequencies_hz >= low_hz) & (frequencies_hz <= high_hz)


def find_shelf_edge_hz(baseline_db):
    """Pitch where the logger floor steps down from its low shelf."""
    frequencies_hz = baseline_db.index.to_numpy()
    values = moving_average(baseline_db.to_numpy(), EDGE_SMOOTHING_BINS)
    midpoint = (np.median(values[_band_mask(frequencies_hz,
                                            SHELF_LOW_PLATEAU_HZ)])
                + np.median(values[_band_mask(frequencies_hz,
                                              SHELF_HIGH_PLATEAU_HZ)])) / 2
    in_search = _band_mask(frequencies_hz, SHELF_SEARCH_HZ)
    below = np.flatnonzero(in_search & (values < midpoint))
    return float(frequencies_hz[below[0]])


def find_band(excess_row):
    """Edges (Hz) and median level (dB) of the strongest contiguous band."""
    frequencies_hz = excess_row.index.to_numpy()
    hum_free = remove_hum_lines(frequencies_hz, excess_row.to_numpy())
    values = moving_average(hum_free, EDGE_SMOOTHING_BINS)
    in_search = _band_mask(frequencies_hz, BAND_SEARCH_HZ)
    peak_index = np.flatnonzero(in_search)[np.argmax(values[in_search])]
    threshold = values[peak_index] * EDGE_LEVEL_FRACTION
    low_index = peak_index
    while low_index > 0 and values[low_index - 1] >= threshold:
        low_index -= 1
    high_index = peak_index
    while (high_index < len(values) - 1
           and values[high_index + 1] >= threshold):
        high_index += 1
    level_db = float(np.median(values[low_index:high_index + 1]))
    return (float(frequencies_hz[low_index]),
            float(frequencies_hz[high_index]), level_db)


def measure_band_geometry():
    """Band edges, level, and position relative to each logger's shelf."""
    metadata = load_metadata()
    amplitudes = load_all_amplitudes(metadata)
    clip_features = build_clip_feature_table(metadata, amplitudes)
    _, spectra_db = compute_clip_spectra_db(metadata, amplitudes)
    baselines_db = compute_logger_baselines(
        metadata, spectra_db, identify_floor_clips(clip_features))
    excess_db = compute_excess_spectra(metadata, spectra_db, baselines_db)
    shelf_edges = {logger: find_shelf_edge_hz(baselines_db[logger])
                   for logger in baselines_db.columns}
    leak_ids = metadata.loc[metadata['label_group'] == LEAK_LABEL,
                            'clip_id'].tolist()
    rows = []
    for clip_id in leak_ids + list(PVC_BAND_CLIPS):
        record = metadata.set_index('clip_id').loc[clip_id]
        low_hz, high_hz, level_db = find_band(excess_db.loc[clip_id])
        shelf_hz = shelf_edges[record['logger_model']]
        rows.append({
            'clip_id': clip_id,
            'group': ('cast-iron leak' if clip_id in leak_ids
                      else 'PVC band site'),
            'logger_model': record['logger_model'],
            'shelf_top_hz': shelf_hz,
            'band_low_hz': low_hz,
            'band_high_hz': high_hz,
            'band_level_db_above_floor': round(level_db, 1),
            'gap_above_shelf_hz': low_hz - shelf_hz,
            'band_low_to_shelf_ratio': round(low_hz / shelf_hz, 2),
            'band_high_to_shelf_ratio': round(high_hz / shelf_hz, 2),
            'band_width_ratio': round(high_hz / low_hz, 2),
            'band_width_octaves': round(np.log2(high_hz / low_hz), 2),
        })
    geometry = pd.DataFrame(rows)
    make_output_dirs()
    geometry.to_csv(BAND_GEOMETRY_PATH, index=False)
    return geometry


if __name__ == '__main__':
    with pd.option_context('display.width', 220):
        print(measure_band_geometry().to_string(index=False))
