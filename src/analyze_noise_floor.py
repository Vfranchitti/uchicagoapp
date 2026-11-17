"""Measure each logger's noise floor and the sound above it at every site.

The quietest recordings (RMS below NOISE_FLOOR_RMS, all undug) contain no
loud events and near-normal amplitude distributions: most likely the
logger's own electronic noise while the pipe is silent. Their median power
spectrum, computed separately for logger A and logger B, is each device's
baseline. Every clip's spectrum minus its own logger's baseline (in dB) is
then "sound above the floor," comparable across devices.

This also lets us look for a band that sits above the floor at some PVC
sites, possibly fainter than the cast-iron leak band (e.g., U34).

Outputs
-------
outputs/tables/noise_floor_spectra_by_logger.csv
outputs/tables/noise_floor_summary.md
outputs/tables/excess_over_noise_floor_by_band.csv
outputs/figures/eda/eda_18_logger_noise_floors.png
outputs/figures/eda/eda_19_excess_over_noise_floor.png
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy import stats

from src.compute_clip_features import (
    build_clip_feature_table,
    compute_welch_psd,
)
from src.config import (
    BAND_PRESENT_EXCESS_DB,
    EXCESS_BAND_TABLE_PATH,
    EXCESS_BANDS_HZ,
    EXCLUDED_FROM_NOISE_FLOOR,
    LEAK_LABEL,
    LOGGER_MODELS,
    NOISE_FLOOR_RMS,
    NOISE_FLOOR_SPECTRA_PATH,
    NOISE_FLOOR_SUMMARY_PATH,
    NYQUIST_HZ,
    TABLE_DIR,
)
from src.load_recordings import load_all_amplitudes, load_metadata
from src.plot_style import (
    HEALTH_LAB_TEAL,
    LABEL_COLORS,
    LOGGER_COLORS,
    TEXT_DARK,
    TEXT_GREY,
    UCHICAGO_MAROON,
    add_title_and_note,
    apply_urban_labs_style,
    finish_and_save,
)

POWER_TO_DB = 10
# Light smoothing across frequency for display only (5 bins = 20 Hz).
DISPLAY_SMOOTHING_BINS = 5
HEATMAP_DB_RANGE = (-6, 30)
GROUP_ORDER = (('A', 'cast_iron'), ('A', 'PVC'), ('B', 'cast_iron'),
               ('B', 'PVC'))
NOISE_FLOOR_FIGURE = (7.5, 3.8)
EXCESS_FIGURE = (7.5, 9.6)
EXCESS_COLORMAP = LinearSegmentedColormap.from_list(
    'teal_white_maroon', [HEALTH_LAB_TEAL, '#FFFFFF', UCHICAGO_MAROON])


def _to_db(power):
    """Power to decibels, guarding against log(0)."""
    return POWER_TO_DB * np.log10(power + np.finfo(float).tiny)


def compute_clip_spectra_db(metadata, amplitudes):
    """Return (frequencies_hz, DataFrame of dB spectra, one row per clip)."""
    spectra = {}
    for clip_id in metadata['clip_id']:
        frequencies_hz, psd = compute_welch_psd(amplitudes[clip_id])
        spectra[clip_id] = _to_db(psd)
    return frequencies_hz, pd.DataFrame(spectra, index=frequencies_hz).T


def identify_floor_clips(clip_features):
    """Clips quiet enough to represent the logger's own noise.

    Quiet clips that still carry a distinct band (EXCLUDED_FROM_NOISE_FLOOR)
    are left out so the baseline reflects the device alone.
    """
    is_quiet = clip_features['rms_loudness'] < NOISE_FLOOR_RMS
    is_excluded = clip_features['clip_id'].isin(EXCLUDED_FROM_NOISE_FLOOR)
    return clip_features.loc[is_quiet & ~is_excluded, 'clip_id'].tolist()


def compute_logger_baselines(metadata, spectra_db, floor_clip_ids):
    """Median dB spectrum of the floor clips, separately for each logger."""
    is_floor = metadata['clip_id'].isin(floor_clip_ids)
    return pd.DataFrame({
        logger: spectra_db.loc[metadata.loc[
            is_floor & (metadata['logger_model'] == logger), 'clip_id']
        ].median(axis=0)
        for logger in LOGGER_MODELS})


def compute_excess_spectra(metadata, spectra_db, baselines_db):
    """Each clip's spectrum minus its own logger's floor (dB)."""
    logger_by_clip = metadata.set_index('clip_id')['logger_model']
    return spectra_db.apply(
        lambda row: row - baselines_db[logger_by_clip[row.name]], axis=1)


def summarize_excess_by_band(metadata, excess_db):
    """Median dB above the floor in each band, per clip."""
    frequencies_hz = excess_db.columns.to_numpy()
    band_medians = pd.DataFrame({
        band: excess_db.loc[:, (frequencies_hz > low)
                            & (frequencies_hz <= high)].median(axis=1)
        for band, (low, high) in EXCESS_BANDS_HZ.items()})
    band_medians.index.name = 'clip_id'
    return metadata.set_index('clip_id')[
        ['label_group', 'logger_model', 'pipe_material', 'recorded_at']
    ].join(band_medians).reset_index()


def compare_floors(clip_features, floor_clip_ids, baselines_db):
    """Level and shape differences between the two loggers' floors."""
    floors = clip_features[clip_features['clip_id'].isin(floor_clip_ids)]
    rms = {logger: floors.loc[floors['logger_model'] == logger,
                              'rms_loudness'] for logger in LOGGER_MODELS}
    level_gap_db = (baselines_db['B'] - baselines_db['A']).median()
    shape_a = baselines_db['A'] - baselines_db['A'].mean()
    shape_b = baselines_db['B'] - baselines_db['B'].mean()
    return {
        'n_floor_clips': {k: len(v) for k, v in rms.items()},
        'floor_clip_ids': {logger: floors.loc[floors['logger_model']
                                              == logger, 'clip_id'].tolist()
                           for logger in LOGGER_MODELS},
        'median_rms': {k: float(v.median()) for k, v in rms.items()},
        'rms_test_p': stats.mannwhitneyu(*rms.values()).pvalue,
        'median_level_gap_db': float(level_gap_db),
        'shape_correlation': float(np.corrcoef(shape_a, shape_b)[0, 1]),
        'max_shape_gap_db': float(np.max(np.abs(shape_a - shape_b))),
    }


def write_summary(comparison, band_table):
    """Plain-language summary of the floor check and band search."""
    leaks = band_table[band_table['label_group'] == LEAK_LABEL]
    has_leak_band = (band_table['leak_band_400_1300']
                     >= BAND_PRESENT_EXCESS_DB)
    pvc = band_table[band_table['pipe_material'] == 'PVC']
    pvc_above = pvc[(pvc[list(EXCESS_BANDS_HZ)] >= BAND_PRESENT_EXCESS_DB
                     ).any(axis=1)]
    lines = [
        '# Logger noise floors and sound above the floor',
        '',
        f'- Floor clips (RMS < {NOISE_FLOOR_RMS}, excluding '
        f'{", ".join(EXCLUDED_FROM_NOISE_FLOOR)}): '
        f'{comparison["floor_clip_ids"]}.',
        f'- Median RMS of floor clips: A '
        f'{comparison["median_rms"]["A"]:.4f}, B '
        f'{comparison["median_rms"]["B"]:.4f} (Mann-Whitney p = '
        f'{comparison["rms_test_p"]:.2f}). Logger B\'s floor spectrum sits '
        f'a median {comparison["median_level_gap_db"]:+.1f} dB relative to '
        f'A\'s.',
        f'- Floor shapes: correlation {comparison["shape_correlation"]:.2f}'
        f' between A and B after removing overall level; largest gap '
        f'{comparison["max_shape_gap_db"]:.1f} dB at any frequency.',
        '- Dug leaks, median dB above floor in 400-1,300 Hz: '
        + ', '.join(f'{row.clip_id} {row.leak_band_400_1300:+.1f}'
                    for row in leaks.itertuples()) + '.',
        f'- Clips with the 400-1,300 Hz band at least '
        f'{BAND_PRESENT_EXCESS_DB:.0f} dB above their floor: '
        f'{band_table.loc[has_leak_band, "clip_id"].tolist()}.',
        f'- PVC clips with any band at least {BAND_PRESENT_EXCESS_DB:.0f} dB '
        f'above floor: '
        + (', '.join(
            f'{row.clip_id} (' + ', '.join(
                f'{band} {getattr(row, band):+.1f}'
                for band in EXCESS_BANDS_HZ
                if getattr(row, band) >= BAND_PRESENT_EXCESS_DB) + ')'
            for row in pvc_above.itertuples()) or 'none') + '.',
        '',
        'Band-level table: `excess_over_noise_floor_by_band.csv`.',
    ]
    NOISE_FLOOR_SUMMARY_PATH.write_text('\n'.join(lines) + '\n',
                                        encoding='utf-8')


def _smooth(values):
    """Moving average across frequency, for display."""
    kernel = np.ones(DISPLAY_SMOOTHING_BINS) / DISPLAY_SMOOTHING_BINS
    return np.convolve(values, kernel, mode='same')


def plot_noise_floors(metadata, spectra_db, baselines_db, floor_clip_ids,
                      comparison):
    """Fig 18 (ACCURATE). Each logger's noise floor, level and shape."""
    frequencies_hz = baselines_db.index.to_numpy()
    fig, (level_ax, shape_ax) = plt.subplots(1, 2, figsize=NOISE_FLOOR_FIGURE,
                                             sharex=True)
    logger_by_clip = metadata.set_index('clip_id')['logger_model']
    for clip_id in floor_clip_ids:
        level_ax.plot(frequencies_hz, _smooth(spectra_db.loc[clip_id]),
                      color=LOGGER_COLORS[logger_by_clip[clip_id]],
                      linewidth=0.4, alpha=0.35)
    for logger in LOGGER_MODELS:
        baseline = baselines_db[logger]
        n_clips = comparison['n_floor_clips'][logger]
        level_ax.plot(frequencies_hz, _smooth(baseline),
                      color=LOGGER_COLORS[logger], linewidth=1.8,
                      label=f'Logger {logger} floor (median of {n_clips})')
        shape_ax.plot(frequencies_hz, _smooth(baseline - baseline.mean()),
                      color=LOGGER_COLORS[logger], linewidth=1.8)
    level_ax.set_title('Absolute level', loc='left')
    shape_ax.set_title('Shape only (each floor centered on its mean)',
                       loc='left')
    level_ax.set_ylabel('Power (dB)')
    shape_ax.set_ylabel('Relative power (dB)')
    for ax in (level_ax, shape_ax):
        ax.set_xlabel('Pitch (Hz)')
    level_ax.set_xlim(0, NYQUIST_HZ)
    level_ax.legend(loc='upper right', fontsize=7)
    fig.subplots_adjust(wspace=0.3)
    bottom = add_title_and_note(
        fig, 18, 'Each logger\'s own noise floor, from its quietest '
        'recordings',
        f'Floor clips are the {sum(comparison["n_floor_clips"].values())} '
        f'recordings with RMS below {NOISE_FLOOR_RMS} (thin lines; all '
        f'undug), excluding {", ".join(EXCLUDED_FROM_NOISE_FLOOR)}, which '
        f'carries a distinct 250-400 Hz band; thick lines are each '
        f'logger\'s median. Logger B\'s floor '
        f'is {comparison["median_level_gap_db"]:+.1f} dB relative to A\'s, '
        f'and the shapes correlate at {comparison["shape_correlation"]:.2f}.'
        f' Logger B\'s floor rests on only '
        f'{comparison["n_floor_clips"]["B"]} clips, so treat it as '
        f'approximate. Display smoothed over 20 Hz.')
    return finish_and_save(fig, 'eda_18_logger_noise_floors', bottom,
                           top=0.84)


def plot_excess_heatmap(metadata, excess_db):
    """Fig 19 (ACCURATE). Sound above each logger's floor, all 46 clips."""
    ordered = pd.concat([
        metadata[(metadata['logger_model'] == logger)
                 & (metadata['pipe_material'] == pipe)
                 ].sort_values('overnight_hour')
        for logger, pipe in GROUP_ORDER])
    frequencies_hz = excess_db.columns.to_numpy()
    matrix = np.vstack([_smooth(excess_db.loc[clip_id].to_numpy())
                        for clip_id in ordered['clip_id']])
    fig, ax = plt.subplots(figsize=EXCESS_FIGURE)
    image = ax.imshow(
        matrix, aspect='auto', cmap=EXCESS_COLORMAP,
        norm=TwoSlopeNorm(vmin=HEATMAP_DB_RANGE[0], vcenter=0,
                          vmax=HEATMAP_DB_RANGE[1]),
        extent=[frequencies_hz[0], frequencies_hz[-1], len(ordered), 0],
        interpolation='nearest')
    labels = [f'{r.clip_id}  {r.logger_model} | '
              f'{r.pipe_material.replace("_", " ")} | {r.recorded_at}'
              for r in ordered.itertuples()]
    ax.set_yticks(np.arange(len(ordered)) + 0.5)
    ax.set_yticklabels(labels, fontsize=6)
    for tick, group in zip(ax.get_yticklabels(), ordered['label_group']):
        if group != 'unlabeled':
            tick.set_color(LABEL_COLORS[group])
            tick.set_fontweight('bold')
    group_sizes = [((ordered['logger_model'] == logger)
                    & (ordered['pipe_material'] == pipe)).sum()
                   for logger, pipe in GROUP_ORDER]
    for boundary in np.cumsum(group_sizes)[:-1]:
        ax.axhline(boundary, color=TEXT_DARK, linewidth=0.8)
    for band, (low, high) in EXCESS_BANDS_HZ.items():
        ax.axvline(high, color=TEXT_GREY, linewidth=0.5, linestyle=':')
    ax.set_xlabel('Pitch (Hz)')
    ax.grid(False)
    top_axis = ax.secondary_xaxis('top')
    top_axis.set_xticks([np.mean(edges)
                         for edges in EXCESS_BANDS_HZ.values()])
    top_axis.set_xticklabels(['rumble', 'edge', 'leak band', 'high'],
                             fontsize=7)
    top_axis.tick_params(length=0)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.5, pad=0.02)
    colorbar.set_label('dB above own logger\'s noise floor')
    colorbar.outline.set_visible(False)
    bottom = add_title_and_note(
        fig, 19, 'Sound above each logger\'s noise floor, all 46 recordings',
        'Each row is one recording, grouped (top to bottom) logger A cast '
        'iron, A PVC, B cast iron, B PVC, in clock order within group. '
        'Color is power above the median spectrum of that logger\'s '
        'quietest clips: white is at the floor, maroon is above it, teal '
        'below. Bold labels mark dug sites (maroon = leak, teal = no '
        'leak). Dotted lines mark 250, 400, and 1,300 Hz. Floor clips sit '
        'near 0 by construction. Display smoothed over 20 Hz.')
    return finish_and_save(fig, 'eda_19_excess_over_noise_floor', bottom,
                           top=0.93, left=0.27)


def run_noise_floor_analysis():
    """Compute floors, excess spectra, tables, and figures."""
    apply_urban_labs_style()
    metadata = load_metadata()
    amplitudes = load_all_amplitudes(metadata)
    clip_features = build_clip_feature_table(metadata, amplitudes)
    _, spectra_db = compute_clip_spectra_db(metadata, amplitudes)
    floor_clip_ids = identify_floor_clips(clip_features)
    baselines_db = compute_logger_baselines(metadata, spectra_db,
                                            floor_clip_ids)
    excess_db = compute_excess_spectra(metadata, spectra_db, baselines_db)
    band_table = summarize_excess_by_band(metadata, excess_db)
    comparison = compare_floors(clip_features, floor_clip_ids, baselines_db)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    baselines_db.rename_axis('frequency_hz').round(3).to_csv(
        NOISE_FLOOR_SPECTRA_PATH)
    band_table.round(2).to_csv(EXCESS_BAND_TABLE_PATH, index=False)
    write_summary(comparison, band_table)
    return [plot_noise_floors(metadata, spectra_db, baselines_db,
                              floor_clip_ids, comparison),
            plot_excess_heatmap(metadata, excess_db)]


if __name__ == '__main__':
    for figure_path in run_noise_floor_analysis():
        print(f'Saved {figure_path}')
    print(NOISE_FLOOR_SUMMARY_PATH.read_text(encoding='utf-8'))
