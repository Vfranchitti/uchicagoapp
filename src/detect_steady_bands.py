"""Steady-band detection: the presence test of the final Model 1.

Where do steady bands appear, without fixing a range first? First run as a
descriptive scan (2026-11-18), then adopted (2026-11-19) as Model 1's
presence test for both pipe materials. Rules are fixed in config.py.

Steps, separately for each pipe material
1. Steady spectrum per recording: for each pitch, the level the sound
   stays above for 90% of the recording (10th percentile over 250 ms
   frames; the L90 used for background sound in environmental acoustics),
   with 120 Hz hum lines removed, smoothed over 48 Hz, minus the
   logger's noise floor measured the same way. Brief
   traffic bursts drop out without discarding any frames.
2. Shared background: median steady spectrum of the OTHER recordings of
   the same material (the recording itself is left out).
3. A recording's own band: a run of pitches at least
   SCAN_EXCESS_OVER_SHARED_DB above the shared background (split where
   the level steps or dips by SCAN_SPLIT_CHANGE_DB or more), at least
   SCAN_MIN_BAND_WIDTH_HZ wide,
   standing at least SCAN_FLANK_CONTRAST_DB above the SCAN_FLANK_WIDTH_HZ
   of pitches on both sides.
4. Presence: cast iron needs a detected band covering at least half of the
   confirmed-leak band (400-1,300 Hz); PVC counts any detected band.

Validation on cast iron, stated before running: a band covering 400-1,300
Hz in all three dug leaks and in none of the three dug non-leaks.

Logger cross-check (from the parked standardization step): logger B minus
logger A in the steady background of cast-iron recordings without a band,
compared with the 5.8 dB measured on the quiet clips. Reported only.

Revisions (mechanics only, thresholds unchanged): after run 1, smoothing,
dip splitting and a percentile-basis floor; after run 2, dips measured
from medians; after run 3, wider hum removal. Runs 1-2 are kept as
band_scan_run1_* and band_scan_run2_*.

Outputs
-------
outputs/tables/3_model1/band_scan_bands.csv
outputs/tables/3_model1/band_scan_map.csv
outputs/tables/3_model1/band_scan_summary.md
outputs/figures/2_model/band_scan_map.png
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import uniform_filter1d

from src.compute_clip_features import remove_hum_lines
from src.config import (
    ANALYSIS_EXCLUDED_CLIPS,
    ANALYSIS_EXCLUSION_REASON,
    BAND_SCAN_FIGURE_PATH,
    BAND_SCAN_MAP_PATH,
    BAND_SCAN_SUMMARY_PATH,
    BAND_SCAN_TABLE_PATH,
    CAST_IRON_BAND_MIN_OVERLAP,
    DUG_LABELS,
    EXCESS_BANDS_HZ,
    LEAK_LABEL,
    NO_LEAK_LABEL,
    NYQUIST_HZ,
    PIPE_MATERIALS,
    POWER_TO_DB,
    SCAN_EXCESS_OVER_SHARED_DB,
    SCAN_FLANK_CONTRAST_DB,
    SCAN_FLANK_WIDTH_HZ,
    SAMPLE_RATE_HZ,
    SCAN_MAX_GAP_HZ,
    SCAN_MIN_BAND_WIDTH_HZ,
    SCAN_SMOOTHING_HZ,
    SCAN_SPLIT_CHANGE_DB,
    STEADY_FRAME_OVERLAP_SAMPLES,
    STEADY_FRAME_SAMPLES,
    STEADY_LEVEL_PERCENTILE,
    TABLE_DECIMALS,
)
from src.load_recordings import load_all_amplitudes, load_metadata
from src.plot_style import (
    ANNOTATION_FONT_SIZE,
    FRESNO_BAND_BLUE,
    LABEL_COLORS,
    PANEL_TITLE_FONT_SIZE,
    SMALL_FONT_SIZE,
    TEXT_GREY,
    add_title_and_note,
    apply_urban_labs_style,
    save_figure,
)
from src.spectral_inputs import build_spectral_inputs
from src.utils import in_band, make_output_dirs, to_db

VALIDATION_BAND_HZ = EXCESS_BANDS_HZ['leak_band_400_1300']
MATERIAL_NAMES = {'cast_iron': 'Cast iron', 'PVC': 'PVC'}
FIGURE_SIZE = (7.5, 8.2)
FIGURE_NUMBER = 'M3'
BAR_HEIGHT = 0.6
LOGGER_CHECK_BAND_HZ = (250, NYQUIST_HZ)   # above the low rumble shelf


# ---------------------------------------------------------------------------
# Spectra
# ---------------------------------------------------------------------------
def _smooth_db(values_db, bin_width_hz):
    """Average neighboring frequency bins on the power scale."""
    n_bins = max(1, int(round(SCAN_SMOOTHING_HZ / bin_width_hz)))
    power = uniform_filter1d(10 ** (values_db / POWER_TO_DB), size=n_bins,
                             mode='nearest')
    return to_db(power)


def _steady_level(amplitude):
    """(frequencies_hz, 10th-percentile dB per pitch over 250 ms frames)."""
    frequencies_hz, _, power = signal.spectrogram(
        amplitude, fs=SAMPLE_RATE_HZ, nperseg=STEADY_FRAME_SAMPLES,
        noverlap=STEADY_FRAME_OVERLAP_SAMPLES)
    return frequencies_hz, np.percentile(to_db(power),
                                         STEADY_LEVEL_PERCENTILE, axis=1)


def raw_steady_spectra(metadata, amplitudes):
    """Steady level per pitch as recorded, hum removed, smoothed."""
    rows = {}
    for clip_id in metadata['clip_id']:
        frequencies_hz, level = _steady_level(amplitudes[clip_id])
        rows[clip_id] = _smooth_db(remove_hum_lines(frequencies_hz, level),
                                   frequencies_hz[1] - frequencies_hz[0])
    return pd.DataFrame.from_dict(rows, orient='index',
                                  columns=frequencies_hz)


def steady_spectra(metadata, amplitudes, floor_clip_ids):
    """Steady level per pitch above the logger floor (clip x Hz).

    The floor is the median steady spectrum of the logger's quiet clips,
    so recording and floor share the same percentile basis.
    """
    raw_steady = raw_steady_spectra(metadata, amplitudes)
    logger_by_clip = metadata.set_index('clip_id')['logger_model']
    floor_ids = [clip_id for clip_id in floor_clip_ids
                 if clip_id in raw_steady.index]
    floors = raw_steady.loc[floor_ids].groupby(
        logger_by_clip[floor_ids]).median()
    return raw_steady - floors.loc[logger_by_clip[raw_steady.index]
                                   ].to_numpy()


def shared_background(steady, clip_id, material_ids):
    """Median steady spectrum of the other recordings of the material."""
    others = [other for other in material_ids if other != clip_id]
    return steady.loc[others].median(axis=0).to_numpy()


# ---------------------------------------------------------------------------
# Band detection
# ---------------------------------------------------------------------------
def find_runs(mask, max_gap):
    """(first, last) index pairs of True runs, joining short gaps."""
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []
    runs, start, previous = [], indices[0], indices[0]
    for index in indices[1:]:
        if index - previous > max_gap + 1:
            runs.append((start, previous))
            start = index
        previous = index
    runs.append((start, previous))
    return runs


def split_at_changes(level, first, last, min_bins):
    """Split a run where its level steps or dips by SCAN_SPLIT_CHANGE_DB.

    Binary segmentation on medians: find the cut that maximizes the gap
    between the median level on each side (each side at least min_bins
    wide); cut there if the gap is at least SCAN_SPLIT_CHANGE_DB, then
    check each side again. Catches a band joined to rumble below it, to a
    raised shelf, or to high-pitch traffic above it.
    """
    if last - first + 1 < 2 * min_bins:
        return [(first, last)]
    cuts = range(first + min_bins, last - min_bins + 2)
    gaps = [abs(np.median(level[first:cut]) - np.median(level[cut:last + 1]))
            for cut in cuts]
    best = int(np.argmax(gaps))
    if gaps[best] < SCAN_SPLIT_CHANGE_DB:
        return [(first, last)]
    cut = first + min_bins + best
    return (split_at_changes(level, first, cut - 1, min_bins)
            + split_at_changes(level, cut, last, min_bins))


def flank_contrast(level, frequencies_hz, low_hz, high_hz, band_level):
    """Band level minus the median level just below and just above it.

    Returns (below, above); NaN where a flank falls outside the spectrum.
    """
    below = in_band(frequencies_hz, (low_hz - SCAN_FLANK_WIDTH_HZ,
                                     low_hz)) & (frequencies_hz < low_hz)
    above = in_band(frequencies_hz, (high_hz, high_hz + SCAN_FLANK_WIDTH_HZ))
    is_full_below = low_hz - SCAN_FLANK_WIDTH_HZ >= frequencies_hz[0]
    is_full_above = high_hz + SCAN_FLANK_WIDTH_HZ <= frequencies_hz[-1]
    contrast_below = (band_level - np.median(level[below])
                      if is_full_below and below.any() else np.nan)
    contrast_above = (band_level - np.median(level[above])
                      if is_full_above and above.any() else np.nan)
    return contrast_below, contrast_above


def candidate_runs(level, background, frequencies_hz):
    """Every run above the shared background, with each rule's result."""
    bin_width_hz = frequencies_hz[1] - frequencies_hz[0]
    max_gap_bins = int(round(SCAN_MAX_GAP_HZ / bin_width_hz))
    min_bins = int(np.ceil(SCAN_MIN_BAND_WIDTH_HZ / bin_width_hz))
    runs = [piece for first, last in find_runs(
                level - background >= SCAN_EXCESS_OVER_SHARED_DB,
                max_gap_bins)
            for piece in split_at_changes(level, first, last, min_bins)]
    rows = []
    for first, last in runs:
        low_hz, high_hz = frequencies_hz[first], frequencies_hz[last]
        band_level = np.median(level[first:last + 1])
        below, above = flank_contrast(level, frequencies_hz, low_hz,
                                      high_hz, band_level)
        width_hz = (last - first + 1) * bin_width_hz
        is_wide = width_hz >= SCAN_MIN_BAND_WIDTH_HZ
        is_distinct = (np.nanmin([below, above]) >= SCAN_FLANK_CONTRAST_DB
                       if not (np.isnan(below) or np.isnan(above))
                       else False)
        rows.append({
            'low_hz': low_hz, 'high_hz': high_hz, 'width_hz': width_hz,
            'level_db_above_floor': band_level,
            'excess_over_shared_db': float(np.median(
                level[first:last + 1] - background[first:last + 1])),
            'contrast_below_db': below, 'contrast_above_db': above,
            'is_wide_enough': is_wide, 'is_distinct_both_sides': is_distinct,
            'is_band': is_wide and is_distinct})
    return rows


def scan_material(metadata, steady, material):
    """Candidate runs for every recording of one pipe material."""
    records = metadata[metadata['pipe_material'] == material]
    material_ids = records['clip_id'].tolist()
    frequencies_hz = steady.columns.to_numpy()
    rows = []
    for record in records.itertuples():
        background = shared_background(steady, record.clip_id, material_ids)
        level = steady.loc[record.clip_id].to_numpy()
        for run in candidate_runs(level, background, frequencies_hz):
            rows.append({'clip_id': record.clip_id, 'pipe_material': material,
                         'logger_model': record.logger_model,
                         'label_group': record.label_group, **run})
    return pd.DataFrame(rows)


def detect_all_bands(metadata, steady):
    """Candidate runs (with rule results) for every recording."""
    return pd.concat([scan_material(metadata, steady, material)
                      for material in PIPE_MATERIALS], ignore_index=True)


def _covered_share(row, band_hz):
    """Share of band_hz covered by a detected band."""
    overlap = min(row['high_hz'], band_hz[1]) - max(row['low_hz'],
                                                    band_hz[0])
    return max(overlap, 0) / (band_hz[1] - band_hz[0])


def band_presence(bands, metadata):
    """{clip_id: True/False}: Model 1's presence test.

    Cast iron: a detected band covering at least CAST_IRON_BAND_MIN_OVERLAP
    of the confirmed-leak band. PVC: any detected band.
    """
    found = bands[bands['is_band']]
    present = set()
    for row in found.to_dict('records'):
        if row['pipe_material'] == 'PVC' or (
                _covered_share(row, VALIDATION_BAND_HZ)
                >= CAST_IRON_BAND_MIN_OVERLAP):
            present.add(row['clip_id'])
    return {clip_id: clip_id in present for clip_id in metadata['clip_id']}


def corrected_band_level(level, frequencies_hz, band_hz, neighbors_hz):
    """Band's steady power minus its neighbors' (dB above floor).

    Medians in dB per range, converted to power; the neighbor power is the
    mean over the given neighbor ranges. NaN when the band is no louder
    than its neighbors.
    """
    def power(edges):
        return 10 ** (np.median(level[in_band(frequencies_hz, edges)])
                      / POWER_TO_DB)
    excess = power(band_hz) - np.mean([power(edges)
                                       for edges in neighbors_hz])
    return float(to_db(excess)) if excess > 0 else np.nan


def logger_cross_check(metadata, raw_steady, presence):
    """Logger B minus A: steady background of cast iron without a band."""
    records = metadata[(metadata['pipe_material'] == 'cast_iron')
                       & ~metadata['clip_id'].map(presence)]
    mask = in_band(raw_steady.columns.to_numpy(), LOGGER_CHECK_BAND_HZ)
    medians = {logger: raw_steady.loc[group['clip_id'], mask].median(axis=0)
               for logger, group in records.groupby('logger_model')}
    difference = medians['B'] - medians['A']
    return {'offset_db': float(np.median(difference)),
            'shape_correlation': float(np.corrcoef(medians['A'],
                                                   medians['B'])[0, 1]),
            'n_recordings': records['logger_model'].value_counts()
            .to_dict()}


# ---------------------------------------------------------------------------
# Map and validation
# ---------------------------------------------------------------------------
def coverage_map(bands, metadata, frequencies_hz):
    """Per pitch and material: recordings with a band covering it."""
    rows = []
    for material in PIPE_MATERIALS:
        n_recordings = int((metadata['pipe_material'] == material).sum())
        subset = bands[(bands['pipe_material'] == material)
                       & bands['is_band']]
        for frequency in frequencies_hz:
            covering = subset[(subset['low_hz'] <= frequency)
                              & (subset['high_hz'] >= frequency)]
            rows.append({'pipe_material': material, 'frequency_hz': frequency,
                         'n_recordings_with_band': covering['clip_id']
                         .nunique(), 'n_recordings': n_recordings})
    return pd.DataFrame(rows)


def validate_on_cast_iron(presence, metadata):
    """Pre-stated test: band in all 3 dug leaks, in no dug non-leak."""
    cast_iron = metadata[metadata['pipe_material'] == 'cast_iron']
    hits = {clip_id for clip_id in cast_iron['clip_id'] if presence[clip_id]}
    dug = metadata[metadata['label_group'].isin(DUG_LABELS)
                   & (metadata['pipe_material'] == 'cast_iron')]
    leaks = set(dug.loc[dug['label_group'] == LEAK_LABEL, 'clip_id'])
    non_leaks = set(dug.loc[dug['label_group'] == NO_LEAK_LABEL, 'clip_id'])
    return {'leaks_found': sorted(leaks & hits),
            'leaks_missed': sorted(leaks - hits),
            'non_leaks_flagged': sorted(non_leaks & hits),
            'passes': leaks <= hits and not (non_leaks & hits),
            'undug_with_band': sorted(hits - leaks - non_leaks)}


def band_ranges(bands, material):
    """Group overlapping detected bands of one material into ranges."""
    subset = bands[(bands['pipe_material'] == material)
                   & bands['is_band']].sort_values('low_hz')
    ranges = []
    for row in subset.to_dict('records'):
        if ranges and row['low_hz'] <= ranges[-1]['high_hz']:
            current = ranges[-1]
            current['high_hz'] = max(current['high_hz'], row['high_hz'])
            current['clips'].append(row['clip_id'])
        else:
            ranges.append({'low_hz': row['low_hz'],
                           'high_hz': row['high_hz'],
                           'clips': [row['clip_id']]})
    return ranges


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
def _band_lines(bands, material):
    """Markdown rows for the bands found in one material."""
    found = bands[(bands['pipe_material'] == material)
                  & bands['is_band']].sort_values(['low_hz', 'clip_id'])
    lines = ['| Recording | Logger | Status | Band (Hz) | Width (Hz) | dB '
             'above floor | dB above shared | dB above sides (below / '
             'above) |', '|---|---|---|---|---|---|---|---|']
    lines += [f'| {r.clip_id} | {r.logger_model} | {r.label_group} | '
              f'{r.low_hz:.0f}-{r.high_hz:.0f} | {r.width_hz:.0f} | '
              f'{r.level_db_above_floor:.1f} | {r.excess_over_shared_db:.1f}'
              f' | {r.contrast_below_db:.1f} / {r.contrast_above_db:.1f} |'
              for r in found.itertuples()]
    return lines if len(found) else ['No recording met all three rules.']


def write_summary(bands, metadata, validation, logger_check):
    """Plain-language Markdown summary of the scan."""
    lines = ['# Steady-band detection (Model 1 presence test)', '',
             f'- {ANALYSIS_EXCLUSION_REASON}',
             f'- Rules (fixed before running): steady level = '
             f'{STEADY_LEVEL_PERCENTILE}th percentile over 64 ms frames; '
             f'band = at least {SCAN_EXCESS_OVER_SHARED_DB:.0f} dB above '
             f'the shared background of its material, at least '
             f'{SCAN_MIN_BAND_WIDTH_HZ} Hz wide, and at least '
             f'{SCAN_FLANK_CONTRAST_DB:.0f} dB above the '
             f'{SCAN_FLANK_WIDTH_HZ} Hz on both sides.',
             f'- Presence: cast iron needs a detected band covering at '
             f'least {CAST_IRON_BAND_MIN_OVERLAP:.0%} of 400-1,300 Hz; PVC '
             f'counts any detected band.',
             f'- Revisions (mechanics only, thresholds unchanged): after '
             f'run 1, spectra smoothed over {SCAN_SMOOTHING_HZ} Hz, runs '
             f'split at dips, and the noise floor measured on the same '
             f'percentile basis; after run 2, dips measured from medians, '
             f'not peaks (peaks split U08); '
             f'after run 3, steady levels measured on 250 ms frames (4 Hz '
             f'resolution) so the hum removal clears the hum lines (240 Hz '
             f'hum hid U15 and U02 at 16 Hz resolution); after run 4, runs '
             f'split wherever the median level steps or dips by '
             f'{SCAN_SPLIT_CHANGE_DB:.0f} dB or more (steps joined U08 to '
             f'high-pitch traffic and U34 to a low shelf). '
             f'Runs 1-2: band_scan_run1_summary.md and '
             f'band_scan_run2_summary.md.',
             '', '## Logger cross-check (reported only)', '',
             f'- Logger B minus logger A, steady background of cast-iron '
             f'recordings without a band ({logger_check["n_recordings"]}): '
             f'{logger_check["offset_db"]:+.1f} dB (median over 250-2,000 '
             f'Hz); shape correlation {logger_check["shape_correlation"]:.2f}.'
             f' Quiet-clip estimate: +5.8 dB.', '',
             '## Validation on cast iron (stated before running)', '',
             f'- Passes: **{"yes" if validation["passes"] else "no"}**.',
             f'- Dug leaks with a band covering 400-1,300 Hz: '
             f'{validation["leaks_found"]} (missed: '
             f'{validation["leaks_missed"]}).',
             f'- Dug non-leaks flagged: {validation["non_leaks_flagged"]}.',
             f'- Undug cast-iron sites with such a band: '
             f'{validation["undug_with_band"]}.', '']
    for material in PIPE_MATERIALS:
        n = int((metadata['pipe_material'] == material).sum())
        lines += [f'## {MATERIAL_NAMES[material]} ({n} recordings)', '']
        for found_range in band_ranges(bands, material):
            clips = sorted(set(found_range['clips']))
            lines.append(f'- **{found_range["low_hz"]:.0f}-'
                         f'{found_range["high_hz"]:.0f} Hz**: {len(clips)} '
                         f'of {n} recordings ({", ".join(clips)}).')
        rejected = bands[(bands['pipe_material'] == material)
                         & ~bands['is_band']]
        lines += ['', *_band_lines(bands, material), '',
                  f'- Runs above the shared background that failed a rule: '
                  f'{len(rejected)} (too narrow: '
                  f'{int((~rejected["is_wide_enough"]).sum())}; not distinct '
                  f'from both sides: '
                  f'{int((~rejected["is_distinct_both_sides"]).sum())}).',
                  '']
    BAND_SCAN_SUMMARY_PATH.write_text('\n'.join(lines), encoding='utf-8')


def plot_band_map(bands, metadata):
    """One row per recording; bars where a band was found."""
    apply_urban_labs_style()
    fig, axes = plt.subplots(2, 1, figsize=FIGURE_SIZE,
                             gridspec_kw={'height_ratios': [33, 12]})
    for ax, material in zip(axes, PIPE_MATERIALS):
        records = metadata[metadata['pipe_material'] == material].sort_values(
            ['label_group', 'clip_id'])
        positions = {clip_id: index for index, clip_id
                     in enumerate(records['clip_id'])}
        if material == 'cast_iron':
            ax.axvspan(*VALIDATION_BAND_HZ, color=FRESNO_BAND_BLUE, zorder=0,
                       linewidth=0)
        found = bands[(bands['pipe_material'] == material) & bands['is_band']]
        for row in found.itertuples():
            ax.barh(positions[row.clip_id], row.high_hz - row.low_hz,
                    left=row.low_hz, height=BAR_HEIGHT,
                    color=LABEL_COLORS[row.label_group])
        ax.set_yticks(list(positions.values()))
        ax.set_yticklabels(list(positions), fontsize=SMALL_FONT_SIZE - 1)
        ax.invert_yaxis()
        ax.set_xlim(0, NYQUIST_HZ)
        ax.grid(axis='x', color=TEXT_GREY, alpha=0.15)
        ax.set_title(f'{MATERIAL_NAMES[material]} ({len(records)} '
                     'recordings)', loc='left',
                     fontsize=PANEL_TITLE_FONT_SIZE)
    axes[-1].set_xlabel('Pitch (Hz)')
    axes[0].text(VALIDATION_BAND_HZ[1] + 20, -0.9, 'confirmed-leak band',
                 fontsize=ANNOTATION_FONT_SIZE, color=TEXT_GREY, va='bottom')
    note = ('Bars mark steady bands found by the scan (maroon = dug leak, '
            'teal = dug no leak, grey = not yet dug). A band must sit at '
            'least 3 dB above the shared background of its material, be at '
            'least 50 Hz wide, and stand 3 dB above the pitches on both '
            'sides. Steady level = 10th percentile over 64 ms frames, so '
            'brief traffic bursts drop out. U40 excluded.')
    bottom = add_title_and_note(fig, FIGURE_NUMBER,
                                'Where steady bands appear, by pipe material',
                                note)
    fig.subplots_adjust(top=0.92, bottom=bottom, hspace=0.25)
    return save_figure(fig, BAND_SCAN_FIGURE_PATH.parent,
                       BAND_SCAN_FIGURE_PATH.stem)


def run_band_scan():
    """Detect bands, validate on cast iron, cross-check loggers; save."""
    metadata = load_metadata()
    metadata = metadata[~metadata['clip_id'].isin(ANALYSIS_EXCLUDED_CLIPS)]
    metadata = metadata.reset_index(drop=True)
    amplitudes = load_all_amplitudes(metadata)
    floor_ids = build_spectral_inputs(metadata, amplitudes)['floor_clip_ids']
    steady = steady_spectra(metadata, amplitudes, floor_ids)
    bands = detect_all_bands(metadata, steady)
    presence = band_presence(bands, metadata)
    validation = validate_on_cast_iron(presence, metadata)
    logger_check = logger_cross_check(
        metadata, raw_steady_spectra(metadata, amplitudes), presence)
    make_output_dirs()
    bands.round(TABLE_DECIMALS).to_csv(BAND_SCAN_TABLE_PATH, index=False)
    coverage_map(bands, metadata, steady.columns.to_numpy()).to_csv(
        BAND_SCAN_MAP_PATH, index=False)
    write_summary(bands, metadata, validation, logger_check)
    plot_band_map(bands, metadata)
    return bands, validation


if __name__ == '__main__':
    run_band_scan()
    print(BAND_SCAN_SUMMARY_PATH.read_text(encoding='utf-8'))
