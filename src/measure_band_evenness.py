"""How evenly does each site's band cover its pitch range?

Model 1 measures a band's intenseness (median contrast in dB over the band)
and its persistence over time, but not its evenness across pitches. This
script adds three evenness measures, on the noise-floor-subtracted
whole-clip spectrum, for every scorable site:

  bin_coverage           share of the band's frequency bins (4 Hz each)
                         whose power is at least PERSISTENCE_CONTRAST_DB
                         above the mean of the two neighboring bands'
                         medians (the across-pitch counterpart of
                         persistence)
  within_band_sd_db      standard deviation across the band's bins of
                         power above the floor, in dB (lower = more even)
  within_band_flatness   spectral flatness inside the band: geometric
                         mean / arithmetic mean of linear power above the
                         floor (1 = perfectly flat)

Cast-iron sites use the 400-1,300 Hz band; PVC sites the 250-400 Hz band
(120 Hz hum lines removed), as in Model 1. Check only; not a model input.

Output: outputs/tables/2_eda/band_evenness.csv
"""
import numpy as np
import pandas as pd
from scipy import stats

from src.compute_clip_features import remove_hum_lines
from src.config import (
    ANALYSIS_EXCLUDED_CLIPS,
    BAND_EVENNESS_PATH,
    MODEL1_PROBABILITY_PATH,
    MODEL1_SCORES_PATH,
    PERSISTENCE_CONTRAST_DB,
    POWER_TO_DB,
    TABLE_DECIMALS,
)
from src.load_recordings import load_all_amplitudes, load_metadata
from src.run_model_knowledge_score import (
    build_spectral_inputs,
    cast_iron_bands,
    derive_leak_band,
    pvc_bands,
)
from src.utils import in_band, make_output_dirs


def evenness(spectrum_db, frequencies_hz, band_hz, neighbors_hz):
    """Three evenness measures for one site's band."""
    band_db = spectrum_db[in_band(frequencies_hz, band_hz)]
    neighbor_level_db = np.mean([
        np.median(spectrum_db[in_band(frequencies_hz, edges)])
        for edges in neighbors_hz])
    band_power = np.maximum(10 ** (band_db / POWER_TO_DB), np.finfo(
        float).tiny)
    return {
        'bin_coverage': float(np.mean(
            band_db - neighbor_level_db >= PERSISTENCE_CONTRAST_DB)),
        'within_band_sd_db': float(np.std(band_db)),
        'within_band_flatness': float(stats.gmean(band_power)
                                      / np.mean(band_power)),
    }


def compute_band_evenness():
    """Evenness measures for every scorable site (no other joins)."""
    metadata = load_metadata()
    metadata = metadata[~metadata['clip_id'].isin(ANALYSIS_EXCLUDED_CLIPS)]
    amplitudes = load_all_amplitudes(metadata)
    inputs = build_spectral_inputs(metadata, amplitudes)
    frequencies_hz = inputs['clip_frequencies_hz']
    leak_ids = metadata.loc[metadata['label_group'] == 'leak',
                            'clip_id'].tolist()
    bands = {'cast_iron': (cast_iron_bands(
        derive_leak_band(inputs['welch_excess'], leak_ids)), False),
             'PVC': (pvc_bands(), True)}
    rows = []
    for record in metadata.itertuples():
        (band_hz, neighbors_hz), is_hum_removed = bands[record.pipe_material]
        spectrum = inputs['clip_db']['floor'].loc[record.clip_id].to_numpy()
        if is_hum_removed:
            spectrum = remove_hum_lines(frequencies_hz, spectrum)
        rows.append({'clip_id': record.clip_id,
                     'label_group': record.label_group,
                     'pipe_material': record.pipe_material,
                     **evenness(spectrum, frequencies_hz, band_hz,
                                neighbors_hz)})
    return pd.DataFrame(rows)


def measure_band_evenness():
    """Evenness for every scorable site, joined to Model 1 outputs."""
    table = compute_band_evenness()
    scores = pd.read_csv(MODEL1_SCORES_PATH)
    scores = scores[scores['version'] == 'floor'][
        ['clip_id', 'contrast_db', 'persistence', 'score']]
    predictions = pd.read_csv(MODEL1_PROBABILITY_PATH)[
        ['clip_id', 'prediction']]
    table = table.merge(scores, on='clip_id').merge(
        predictions, on='clip_id', how='left')
    make_output_dirs()
    table.round(TABLE_DECIMALS).to_csv(BAND_EVENNESS_PATH, index=False)
    return table


if __name__ == '__main__':
    result = measure_band_evenness()
    focus = result[(result['label_group'] != 'unlabeled')
                   | (result['prediction'] == 1)
                   | result['clip_id'].isin(['U16', 'U22', 'U32', 'U35'])]
    with pd.option_context('display.width', 200):
        print(focus.sort_values(['label_group', 'score'],
                                ascending=[True, False])
              .round(2).to_string(index=False))
