"""SANITY CHECK ONLY. Not used by any feature, clustering or model code.

Predicts what a leak at a PVC site might look like in this pilot's data,
by passing the observed cast-iron leak signature through published
plastic-pipe attenuation. Then compares the 12 PVC recordings with that
prediction. The model itself works only from the provided pilot data; this
literature-based prediction is held aside to sanity-check the model's PVC
calls at the end.

Method (sources in notes/research/pvc_attenuation_research.md):
- Cast-iron leak signature: median of the three dug leaks' power above
  their own logger's noise floor, inside 400-1,300 Hz (zero elsewhere).
- Plastic-pipe loss per metre rises in step with pitch:
  loss_dB_per_m = 8.686 * beta * 2 * pi * f (Almeida et al. 2014 [4]).
  beta spans 1e-4 to 5e-4 s/m across plastic pipes and soils ([1], [3],
  [4]; range assembled in the research note).
- Plastic loses at least 5x more than comparable metal pipe (Hunaidi et
  al. 2000 [2]), so the extra loss of PVC over iron is taken as 4/5 of the
  PVC loss.
- Leak power is attenuated and re-added to the logger floor, so predicted
  values are "dB above the floor," the same scale as EDA Fig. 19.

Outputs
-------
outputs/tables/5_checks/sanity_predicted_pvc_leak.csv
outputs/tables/5_checks/sanity_pvc_sites_vs_prediction.csv
outputs/figures/3_checks/sanity_predicted_pvc_leak.png
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analyze_noise_floor import (
    compute_clip_spectra_db,
    compute_excess_spectra,
    compute_logger_baselines,
    identify_floor_clips,
)
from src.compute_clip_features import build_clip_feature_table
from src.config import (
    BAND_PRESENT_EXCESS_DB,
    EXCESS_BANDS_HZ,
    LEAK_LABEL,
    NYQUIST_HZ,
    POWER_TO_DB,
    SANITY_PVC_FIGURE_PATH,
    SANITY_PVC_PREDICTION_TABLE_PATH,
    SANITY_PVC_SITE_TABLE_PATH,
    TABLE_DECIMALS,
)
from src.load_recordings import load_all_amplitudes, load_metadata
from src.plot_style import (
    ANNOTATION_FONT_SIZE,
    CRIME_LAB_GOLD,
    DISPLAY_SMOOTHING_BINS,
    HEALTH_LAB_LIGHT_TEAL,
    HEALTH_LAB_NAVY,
    HEALTH_LAB_TEAL,
    LIGHT_GRID_GREY,
    PANEL_TITLE_FONT_SIZE,
    TEXT_DARK,
    TEXT_GREY,
    UCHICAGO_MAROON,
    UNLABELED_GREY,
    add_title_and_note,
    apply_urban_labs_style,
    save_figure,
)
from src.utils import make_output_dirs, moving_average

# --- Literature inputs (see notes/research/pvc_attenuation_research.md) ---
NEPERS_TO_DB = 8.686              # 20 * log10(e)
BETA_LOW_S_PER_M = 1e-4           # least lossy plastic pipe
BETA_HIGH_S_PER_M = 5e-4          # most lossy plastic pipe
PVC_TO_IRON_LOSS_RATIO = 5        # "at least five" times metal [2]
LEAK_DISTANCES_M = (2, 5, 10, 25)
# Pitch ranges where published PVC/plastic leak energy was measured.
PUBLISHED_PVC_BAND_HZ = (5, 150)  # [1], [2], [4]
PUBLISHED_IRON_BAND_HZ = (220, 1400)  # field average, iron mains [5]

DISTANCE_COLORS = dict(zip(LEAK_DISTANCES_M, (
    UCHICAGO_MAROON, CRIME_LAB_GOLD, HEALTH_LAB_TEAL, HEALTH_LAB_NAVY)))
HIGHLIGHT_PVC_SITES = {'U34': UCHICAGO_MAROON, 'U15': CRIME_LAB_GOLD,
                       'U32': HEALTH_LAB_NAVY}
SANITY_FIGURE_SIZE = (7.5, 4.6)
LOWER_PITCH_BAND_HZ = (300, 600)
UPPER_PITCH_BAND_HZ = (900, 1300)


def plastic_extra_loss_db(frequencies_hz, beta, distance_m):
    """Extra loss of PVC over iron (dB) between leak and hydrant."""
    pvc_loss = NEPERS_TO_DB * beta * 2 * np.pi * frequencies_hz * distance_m
    return pvc_loss * (1 - 1 / PVC_TO_IRON_LOSS_RATIO)


def attenuate_excess(excess_db, loss_db):
    """Attenuate the leak's power above the floor, then re-add the floor."""
    leak_power = 10 ** (excess_db / POWER_TO_DB) - 1
    return POWER_TO_DB * np.log10(1 + leak_power
                                  * 10 ** (-loss_db / POWER_TO_DB))


def cast_iron_leak_signature(metadata, excess_db):
    """Median dug-leak excess inside the leak band; zero outside it."""
    frequencies_hz = excess_db.columns.to_numpy()
    leak_ids = metadata.loc[metadata['label_group'] == LEAK_LABEL,
                            'clip_id']
    signature = excess_db.loc[leak_ids].median(axis=0).to_numpy()
    low_hz, high_hz = EXCESS_BANDS_HZ['leak_band_400_1300']
    is_in_band = (frequencies_hz > low_hz) & (frequencies_hz <= high_hz)
    return frequencies_hz, np.where(is_in_band, signature, 0.0)


def predict_pvc_spectra(frequencies_hz, signature_db):
    """Predicted dB above floor for each distance and beta bound."""
    rows = []
    for distance_m in LEAK_DISTANCES_M:
        for bound, beta in (('low_loss', BETA_LOW_S_PER_M),
                            ('high_loss', BETA_HIGH_S_PER_M)):
            predicted = attenuate_excess(
                signature_db,
                plastic_extra_loss_db(frequencies_hz, beta, distance_m))
            rows.append(pd.DataFrame({
                'distance_m': distance_m, 'loss_bound': bound,
                'frequency_hz': frequencies_hz,
                'predicted_db_above_floor': predicted}))
    return pd.concat(rows, ignore_index=True)


def summarize_prediction(predictions):
    """Detectable band (>= threshold above floor) per distance and bound."""
    rows = []
    for (distance_m, bound), group in predictions.groupby(
            ['distance_m', 'loss_bound'], sort=False):
        detectable = group[group['predicted_db_above_floor']
                           >= BAND_PRESENT_EXCESS_DB]
        rows.append({
            'distance_m': distance_m, 'loss_bound': bound,
            'detectable_from_hz': (detectable['frequency_hz'].min()
                                   if len(detectable) else np.nan),
            'detectable_to_hz': (detectable['frequency_hz'].max()
                                 if len(detectable) else np.nan),
            'peak_db_above_floor':
                group['predicted_db_above_floor'].max(),
        })
    return pd.DataFrame(rows)


def _band_median(excess_row, band_hz):
    """Median dB above floor within a pitch band."""
    frequencies_hz = excess_row.index.to_numpy()
    low_hz, high_hz = band_hz
    return float(excess_row[(frequencies_hz > low_hz)
                            & (frequencies_hz <= high_hz)].median())


def compare_pvc_sites(metadata, excess_db):
    """How each PVC site's spectrum relates to the predicted shape.

    A near-hydrant PVC leak (prediction) is strongest at the bottom of the
    cast-iron band and fades toward 1,300 Hz. A far leak shows nothing in
    this range.
    """
    pvc = metadata[metadata['pipe_material'] == 'PVC']
    rows = []
    for record in pvc.itertuples():
        excess_row = excess_db.loc[record.clip_id]
        lower = _band_median(excess_row, LOWER_PITCH_BAND_HZ)
        upper = _band_median(excess_row, UPPER_PITCH_BAND_HZ)
        edge = _band_median(excess_row, EXCESS_BANDS_HZ['edge_250_400'])
        is_present = lower >= BAND_PRESENT_EXCESS_DB
        if not is_present and edge < BAND_PRESENT_EXCESS_DB:
            reading = 'at floor: consistent with no leak or a distant leak'
        elif not is_present:
            reading = ('energy only below 400 Hz: not produced by '
                       'filtering the cast-iron band')
        elif upper < lower - BAND_PRESENT_EXCESS_DB:
            reading = 'fades with pitch: shape of a near-hydrant leak'
        else:
            reading = ('flat or rising to 1,300 Hz: unlike a PVC leak '
                       '(plastic removes high pitches)')
        rows.append({'clip_id': record.clip_id,
                     'logger_model': record.logger_model,
                     'recorded_at': record.recorded_at,
                     'db_above_floor_250_400': round(edge, 1),
                     'db_above_floor_300_600': round(lower, 1),
                     'db_above_floor_900_1300': round(upper, 1),
                     'reading': reading})
    return pd.DataFrame(rows)


def plot_prediction(frequencies_hz, signature_db, predictions, metadata,
                    excess_db):
    """Sanity figure: predicted PVC leak vs. the 12 observed PVC sites."""
    apply_urban_labs_style()
    fig, (predicted_ax, observed_ax) = plt.subplots(
        1, 2, figsize=SANITY_FIGURE_SIZE, sharey=True)
    for ax in (predicted_ax, observed_ax):
        ax.axvspan(*PUBLISHED_PVC_BAND_HZ, color=HEALTH_LAB_LIGHT_TEAL,
                   alpha=0.25, linewidth=0)
    predicted_ax.text(PUBLISHED_PVC_BAND_HZ[1] + 15, 25,
                      'published PVC\nleak energy',
                      fontsize=ANNOTATION_FONT_SIZE,
                      color=HEALTH_LAB_TEAL, va='top')
    predicted_ax.plot(frequencies_hz, signature_db, color=TEXT_DARK,
                      linewidth=1.2, linestyle='--',
                      label='Cast-iron leak (observed)')
    for distance_m in LEAK_DISTANCES_M:
        bounds = {bound: group['predicted_db_above_floor'].to_numpy()
                  for bound, group in predictions[
                      predictions['distance_m'] == distance_m
                  ].groupby('loss_bound')}
        color = DISTANCE_COLORS[distance_m]
        is_silent = bounds['low_loss'].max() < BAND_PRESENT_EXCESS_DB
        label = (f'PVC leak {distance_m} m away'
                 + (' (nothing above floor)' if is_silent else ''))
        predicted_ax.fill_between(frequencies_hz, bounds['high_loss'],
                                  bounds['low_loss'], color=color,
                                  alpha=0.35, linewidth=0, label=label)
    predicted_ax.set_title('Predicted: cast-iron leak heard through PVC',
                           loc='left', fontsize=PANEL_TITLE_FONT_SIZE)
    handles, labels = predicted_ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left',
               bbox_to_anchor=(0.06,
                               0.915), ncol=3, fontsize=ANNOTATION_FONT_SIZE)
    for clip_id in metadata.loc[metadata['pipe_material'] == 'PVC',
                                'clip_id']:
        color = HIGHLIGHT_PVC_SITES.get(clip_id, UNLABELED_GREY)
        is_highlight = clip_id in HIGHLIGHT_PVC_SITES
        observed_ax.plot(frequencies_hz,
                         moving_average(excess_db.loc[clip_id].to_numpy(),
                                        DISPLAY_SMOOTHING_BINS),
                         color=color, linewidth=1.1 if is_highlight else 0.5,
                         alpha=1 if is_highlight else 0.7,
                         label=clip_id if is_highlight else None,
                         zorder=3 if is_highlight else 2)
    observed_ax.set_title('Observed: the 12 PVC recordings', loc='left',
                          fontsize=PANEL_TITLE_FONT_SIZE)
    observed_ax.legend(loc='upper right', fontsize=ANNOTATION_FONT_SIZE)
    for ax in (predicted_ax, observed_ax):
        ax.axhline(BAND_PRESENT_EXCESS_DB, color=TEXT_GREY, linewidth=0.6,
                   linestyle=':')
        ax.set_xlabel('Pitch (Hz)')
        ax.set_xlim(0, NYQUIST_HZ)
        ax.grid(axis='x', color=LIGHT_GRID_GREY)
    predicted_ax.set_ylabel('dB above own logger\'s noise floor')
    fig.subplots_adjust(wspace=0.08)
    bottom = add_title_and_note(
        fig, 'S1', 'Sanity check (not used by the model): what might a PVC '
        'leak look like?',
        'Left: the three dug cast-iron leaks\' 400-1,300 Hz band passed '
        'through published plastic-pipe losses (shaded range = least to '
        'most lossy pipe; Almeida et al. 2014; Hunaidi et al. 2000). '
        'Right: observed PVC sites on the same scale (smoothed over 20 Hz). '
        'Dotted line: 3 dB above floor. Caveats: filtering cannot create '
        'the low-pitch energy (below about 150 Hz, light teal) where '
        'published PVC leaks mostly sit; the logger\'s low-pitch response '
        'and leak distances are unknown; the clips appear synthesized.')
    fig.subplots_adjust(top=0.76, bottom=bottom)
    return save_figure(fig, SANITY_PVC_FIGURE_PATH.parent,
                       SANITY_PVC_FIGURE_PATH.stem)


def run_pvc_sanity_check():
    """Build the prediction, compare PVC sites, save tables and figure."""
    metadata = load_metadata()
    amplitudes = load_all_amplitudes(metadata)
    clip_features = build_clip_feature_table(metadata, amplitudes)
    _, spectra_db = compute_clip_spectra_db(metadata, amplitudes)
    baselines_db = compute_logger_baselines(
        metadata, spectra_db, identify_floor_clips(clip_features))
    excess_db = compute_excess_spectra(metadata, spectra_db, baselines_db)
    frequencies_hz, signature_db = cast_iron_leak_signature(metadata,
                                                            excess_db)
    predictions = predict_pvc_spectra(frequencies_hz, signature_db)
    make_output_dirs()
    summary = summarize_prediction(predictions)
    summary.round(TABLE_DECIMALS).to_csv(SANITY_PVC_PREDICTION_TABLE_PATH,
                                         index=False)
    sites = compare_pvc_sites(metadata, excess_db)
    sites.to_csv(SANITY_PVC_SITE_TABLE_PATH, index=False)
    figure_path = plot_prediction(frequencies_hz, signature_db, predictions,
                                  metadata, excess_db)
    return summary, sites, figure_path


if __name__ == '__main__':
    prediction_summary, site_table, path = run_pvc_sanity_check()
    with pd.option_context('display.width', 200,
                           'display.max_colwidth', 90):
        print(prediction_summary.round(1).to_string(index=False))
        print(site_table.to_string(index=False))
    print(f'Saved {path}')
