"""Exploratory figures for the acoustic logger pilot, in Urban Labs style.

Figures are ordered from the broadest view of the data (who was recorded,
when, and with what) to the most technical (correlations, principal
components). Each figure is tagged in its docstring as:
- LAY: readable by a non-technical audience,
- INTERPRETABLE: each axis has a plain physical meaning,
- ACCURATE: information-dense but needs training to read.

Run from the repo root:  python -m src.plot_eda_figures
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import signal, stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.compute_clip_features import (
    FEATURE_COLUMNS,
    FEATURE_LABELS,
    build_clip_feature_table,
    compute_rms_envelope,
    compute_welch_psd,
    find_longest_constant_run,
)
from src.config import (
    CLIP_DURATION_SECONDS,
    CLIPPING_THRESHOLD,
    HIGH_PITCH_CUTOFF_HZ,
    LEAK_LABEL,
    MORNING_COMMUTE_START_HOUR,
    NO_LEAK_LABEL,
    NYQUIST_HZ,
    SAMPLE_RATE_HZ,
    TRANSIENT_LOUDNESS_RATIO,
)
from src.load_recordings import load_all_amplitudes, load_metadata
from src.plot_style import (
    CRIME_LAB_GOLD,
    FRESNO_BAND_BLUE,
    HEALTH_LAB_LIGHT_TEAL,
    HEALTH_LAB_NAVY,
    HEALTH_LAB_TEAL,
    LABEL_COLORS,
    LABEL_DISPLAY_NAMES,
    LOGGER_MARKERS,
    PIPE_COLORS,
    PIPE_MARKERS,
    TEXT_DARK,
    TEXT_GREY,
    UCHICAGO_MAROON,
    UNLABELED_GREY,
    add_title_and_note,
    apply_urban_labs_style,
    finish_and_save,
    format_overnight_axis,
    shade_commute_window,
)

# Axis span for overnight plots: 9:30 pm to 7:00 am.
FIRST_AXIS_HOUR = -2.5
LAST_AXIS_HOUR = 7.0
JITTER_SEED = 20261117
JITTER_WIDTH = 0.18
MIN_LABEL_SEPARATION_HOURS = 0.6    # closer dug-site labels alternate
MIN_SEGMENT_FOR_COUNT_LABEL = 3     # smaller bar segments go unlabeled
POINTS_PER_INCH = 72
# Bounding box of a 7 pt clip ID at 100 dpi; labels closer than this in
# both directions would overlap.
LABEL_WIDTH_PIXELS = 30
LABEL_HEIGHT_PIXELS = 12
# Dug sites get fixed sub-rows in the dot plot (one clip per sub-row) so
# their labels never collide.
DUG_SUBROW_OFFSETS = (-0.27, 0.0, 0.27)
# Candidate label offsets (points), tried in order until one is clear.
SCATTER_LABEL_OFFSETS = [(5, 3), (5, -10), (-24, 3), (-24, -10), (5, 12),
                         (-24, 12), (5, -19), (-24, -19)]
HISTOGRAM_MIN_DENSITY = 1e-5        # floor for the log-density axis
# Extra bottom margin (figure fraction) for rotated heatmap labels.
CORRELATION_TICK_LABEL_ALLOWANCE = 0.16
SPECTROGRAM_SEGMENT_SAMPLES = 256   # 64 ms windows
SPECTROGRAM_OVERLAP_SAMPLES = 128
SPECTROGRAM_DB_FLOOR_PERCENTILE = 5
SPECTROGRAM_DB_CEILING_PERCENTILE = 99.5
POWER_TO_DB = 10
HISTOGRAM_BINS = 81
HISTOGRAM_STANDARD_UNITS = 8        # x-range of standardized amplitude
SMALL_FIGURE = (7.5, 3.6)
MEDIUM_FIGURE = (7.5, 4.6)
TALL_FIGURE = (7.5, 6.6)
FULL_SCALE_AMPLITUDE = 1.0
PCA_COMPONENTS = 2
# Clips singled out in the checks (outliers, unusual spectra, dropout).
NOTABLE_UNDUG_CLIPS = ('U16', 'U22', 'U32', 'U35', 'U40')
# Features that span orders of magnitude are log-scaled for display/PCA.
LOG_SCALED_FEATURES = ('rms_loudness', 'peak_amplitude', 'crest_factor',
                       'loudness_range_ratio')
# loudness_range_ratio is undefined for the clip with a dropout (U40).
PCA_FEATURES = [feature for feature in FEATURE_COLUMNS
                if feature != 'loudness_range_ratio']
SPECTROGRAM_COLORMAP = LinearSegmentedColormap.from_list(
    'health_lab_blues',
    ['#FFFFFF', HEALTH_LAB_LIGHT_TEAL, HEALTH_LAB_TEAL, HEALTH_LAB_NAVY])
CORRELATION_COLORMAP = LinearSegmentedColormap.from_list(
    'teal_white_maroon', [HEALTH_LAB_TEAL, '#FFFFFF', UCHICAGO_MAROON])


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _format_clock(recorded_at):
    """'22:08' -> '10:08 pm'."""
    hours_text, minutes_text = recorded_at.split(':')
    hour = int(hours_text)
    suffix = 'am' if hour < 12 else 'pm'
    return f'{hour % 12 or 12}:{minutes_text} {suffix}'


def _labeled_records(metadata):
    """Dug sites ordered leaks first, then by recording time."""
    labeled = metadata[metadata['is_labeled']].copy()
    labeled['is_no_leak'] = labeled['label'].eq(NO_LEAK_LABEL)
    return labeled.sort_values(['is_no_leak', 'overnight_hour'])


def _labeled_panel_grid(metadata, figsize, sharey=True):
    """3 x 2 grid: leaks in the left column, no-leaks in the right."""
    fig, axes = plt.subplots(3, 2, figsize=figsize, sharex=True,
                             sharey=sharey)
    labeled = _labeled_records(metadata)
    panels = []
    for column, label in enumerate((LEAK_LABEL, NO_LEAK_LABEL)):
        rows = labeled[labeled['label'] == label]
        for row_index, record in enumerate(rows.itertuples()):
            panels.append((axes[row_index, column], record))
    axes[0, 0].annotate('LEAK FOUND', xy=(0.5, 1.32),
                        xycoords='axes fraction', ha='center',
                        fontweight='bold', color=UCHICAGO_MAROON)
    axes[0, 1].annotate('NO LEAK', xy=(0.5, 1.32),
                        xycoords='axes fraction', ha='center',
                        fontweight='bold', color=HEALTH_LAB_TEAL)
    return fig, axes, panels


def _panel_title(record):
    """'L06 | logger A | 10:08 pm'."""
    return (f'{record.clip_id}  |  logger {record.logger_model}  |  '
            f'{_format_clock(record.recorded_at)}')


def _jitter(n_points, rng):
    """Uniform vertical jitter so overlapping dots stay visible."""
    return rng.uniform(-JITTER_WIDTH, JITTER_WIDTH, n_points)


def _label_legend_handles():
    """Legend entries for dug-leak, dug-no-leak, and undug sites."""
    return [Line2D([], [], marker='o', linestyle='', markersize=7,
                   markerfacecolor=LABEL_COLORS[group],
                   markeredgecolor='white', label=LABEL_DISPLAY_NAMES[group])
            for group in (LEAK_LABEL, NO_LEAK_LABEL, 'unlabeled')]


def _scatter_by_group(ax, df, x_column, y_values, marker_map, marker_column):
    """Scatter undug (grey, behind) then dug sites (colored, on top)."""
    for group in ('unlabeled', NO_LEAK_LABEL, LEAK_LABEL):
        subset = df[df['label_group'] == group]
        y_subset = y_values[subset.index]
        size = 30 if group == 'unlabeled' else 70
        for level, marker in marker_map.items():
            is_level = subset[marker_column] == level
            ax.scatter(subset.loc[is_level, x_column], y_subset[is_level],
                       s=size, marker=marker, color=LABEL_COLORS[group],
                       edgecolor='white', linewidth=0.6,
                       zorder=3 if group == 'unlabeled' else 4)


def _place_labels(ax, labeled_points, candidate_offsets, horizontal_align):
    """Annotate points, choosing for each the first offset that keeps its
    label clear of labels already placed.

    labeled_points: iterable of (text, x, y, color).
    candidate_offsets: (dx, dy) in points, tried in order.
    """
    points_to_pixels = ax.figure.dpi / POINTS_PER_INCH
    placed_positions = []
    for text, x, y, color in labeled_points:
        anchor = ax.transData.transform((x, y))
        chosen_offset = candidate_offsets[0]
        for offset in candidate_offsets:
            candidate = anchor + np.array(offset) * points_to_pixels
            if all(abs(candidate[0] - placed[0]) > LABEL_WIDTH_PIXELS
                   or abs(candidate[1] - placed[1]) > LABEL_HEIGHT_PIXELS
                   for placed in placed_positions):
                chosen_offset = offset
                break
        placed_positions.append(
            anchor + np.array(chosen_offset) * points_to_pixels)
        ax.annotate(text, (x, y), xytext=chosen_offset,
                    textcoords='offset points', ha=horizontal_align,
                    fontsize=7, color=color, zorder=5)


def _annotate_clips(ax, df, x_column, y_column, clip_ids, color=TEXT_DARK):
    """Write clip IDs beside selected scatter points without overlaps."""
    selected = df[df['clip_id'].isin(clip_ids)]
    _place_labels(ax, [(record.clip_id, getattr(record, x_column),
                        getattr(record, y_column), color)
                       for record in selected.itertuples()],
                  SCATTER_LABEL_OFFSETS, 'left')


def _annotate_row_without_overlap(ax, labeled_in_row, row_position):
    """Label dug sites above their dots, dropping below when crowded."""
    previous_hour = None
    for record in labeled_in_row.sort_values('overnight_hour').itertuples():
        is_crowded = (previous_hour is not None and record.overnight_hour
                      - previous_hour < MIN_LABEL_SEPARATION_HOURS)
        ax.annotate(record.clip_id, (record.overnight_hour, row_position),
                    xytext=(0, -15 if is_crowded else 9),
                    textcoords='offset points', ha='center', fontsize=7,
                    color=LABEL_COLORS[record.label_group],
                    fontweight='bold')
        previous_hour = record.overnight_hour


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_coverage_timeline(metadata):
    """Fig 1 (LAY). Every recording by clock time, logger, and pipe."""
    row_definitions = [('A', 'cast_iron'), ('A', 'PVC'),
                       ('B', 'cast_iron'), ('B', 'PVC')]
    fig, ax = plt.subplots(figsize=SMALL_FIGURE)
    shade_commute_window(ax, MORNING_COMMUTE_START_HOUR, LAST_AXIS_HOUR,
                         label_y=-0.55)
    tick_labels = []
    for row_position, (logger, pipe) in enumerate(reversed(row_definitions)):
        in_row = metadata[(metadata['logger_model'] == logger)
                          & (metadata['pipe_material'] == pipe)]
        n_dug = in_row['is_labeled'].sum()
        tick_labels.append(f'Logger {logger}, {pipe.replace("_", " ")}\n'
                           f'({len(in_row)} sites, {n_dug} dug)')
        for group in ('unlabeled', NO_LEAK_LABEL, LEAK_LABEL):
            subset = in_row[in_row['label_group'] == group]
            ax.scatter(subset['overnight_hour'],
                       np.full(len(subset), row_position),
                       s=40 if group == 'unlabeled' else 110,
                       color=LABEL_COLORS[group], edgecolor='white',
                       linewidth=0.8, zorder=3)
        _annotate_row_without_overlap(ax, in_row[in_row['is_labeled']],
                                      row_position)
    ax.set_yticks(range(len(row_definitions)))
    ax.set_yticklabels(tick_labels)
    ax.set_ylim(-0.6, len(row_definitions) - 0.3)
    ax.grid(axis='x', color='#EFEFEF')
    ax.grid(axis='y', visible=False)
    format_overnight_axis(ax, FIRST_AXIS_HOUR, LAST_AXIS_HOUR)
    ax.set_xlabel('Clock time the recording was made')
    ax.legend(handles=_label_legend_handles(), loc='upper center',
              bbox_to_anchor=(0.5, 1.13), ncol=3)
    fig.subplots_adjust(left=0.2)
    bottom = add_title_and_note(
        fig, 1, 'When and where each hydrant was recorded (N=46)',
        'Each dot is one hydrant recording. All six dug sites are cast '
        'iron; none of the 12 PVC sites has been dug. The three leaks were '
        'recorded before 3:30 am and the three non-leaks after 4:15 am, so '
        'in the dug sites, recording time and leak status are entangled. '
        'Shading marks an approximate morning-commute start of 6:00 am.')
    return finish_and_save(fig, 'eda_01_coverage_timeline', bottom,
                           top=0.80)


def plot_coverage_bars(metadata):
    """Fig 2 (LAY). Dug vs. undug counts by logger, pipe, and timing."""
    timing = np.where(metadata['is_commute_hours'], 'At/after 6 am',
                      'Before 6 am')
    factors = [('Logger model', 'Logger ' + metadata['logger_model']),
               ('Pipe material',
                metadata['pipe_material'].str.replace('_', ' ')
                .str.replace('cast iron', 'Cast iron')),
               ('Recording time', pd.Series(timing, index=metadata.index))]
    groups = (LEAK_LABEL, NO_LEAK_LABEL, 'unlabeled')
    fig, axes = plt.subplots(1, 3, figsize=SMALL_FIGURE, sharex=True)
    for ax, (factor_title, levels) in zip(axes, factors):
        counts = pd.crosstab(levels, metadata['label_group']).reindex(
            columns=groups, fill_value=0)
        left = np.zeros(len(counts))
        for group in groups:
            values = counts[group].to_numpy()
            ax.barh(counts.index, values, left=left,
                    color=LABEL_COLORS[group], edgecolor='white')
            for y_position, (value, start) in enumerate(zip(values, left)):
                if value >= MIN_SEGMENT_FOR_COUNT_LABEL:
                    ax.text(start + value / 2, y_position, str(value),
                            ha='center', va='center', fontsize=7,
                            color='white' if group != 'unlabeled'
                            else TEXT_DARK, fontweight='bold')
            left += values
        ax.set_yticks(range(len(counts)))
        ax.set_yticklabels([
            f'{level}\n{row[LEAK_LABEL]} leak, {row[NO_LEAK_LABEL]} no leak,'
            f'\n{row["unlabeled"]} not dug'
            for level, row in counts.iterrows()], fontsize=7)
        ax.set_title(factor_title, loc='left')
        ax.grid(axis='y', visible=False)
        ax.grid(axis='x', color='#EFEFEF')
        ax.set_xlabel('Number of sites')
        ax.invert_yaxis()
    fig.legend(handles=[Patch(color=LABEL_COLORS[g],
                              label=LABEL_DISPLAY_NAMES[g]) for g in groups],
               loc='upper center', bbox_to_anchor=(0.5, 0.91), ncol=3)
    fig.subplots_adjust(wspace=0.75, left=0.12)
    bottom = add_title_and_note(
        fig, 2, 'Which kinds of sites have been dug? (N=46)',
        'Dug sites are the only ones with known outcomes, so they are the '
        'only evidence about what a leak sounds like. Their mix (half '
        'leaks, no PVC) differs from both the undug sites and the '
        'historical 1-in-8 leak rate.')
    return finish_and_save(fig, 'eda_02_coverage_bars', bottom, top=0.76)


def plot_labeled_waveforms(metadata, amplitudes):
    """Fig 3 (LAY). Raw 5-second waveforms of the six dug sites."""
    fig, axes, panels = _labeled_panel_grid(metadata, MEDIUM_FIGURE)
    time_seconds = np.arange(len(next(iter(amplitudes.values())))) \
        / SAMPLE_RATE_HZ
    for ax, record in panels:
        ax.plot(time_seconds, amplitudes[record.clip_id], linewidth=0.3,
                color=LABEL_COLORS[record.label])
        ax.set_title(_panel_title(record), loc='left', fontsize=8)
        ax.grid(axis='y', visible=False)
    for ax in axes[-1]:
        ax.set_xlabel('Seconds into recording')
    for ax in axes[:, 0]:
        ax.set_ylabel('Amplitude')
    axes[0, 0].set_xlim(0, CLIP_DURATION_SECONDS)
    fig.subplots_adjust(hspace=0.55, wspace=0.12)
    bottom = add_title_and_note(
        fig, 3, 'What the six dug sites sound like: raw recordings',
        'Each panel is one 5-second recording on a common scale. Tall '
        'spikes are brief loud sounds (e.g., a passing vehicle); an even '
        'band is constant sound. Even the leak sites contain bursts, and '
        'one no-leak site (L04) is mostly quiet with a few large spikes.')
    return finish_and_save(fig, 'eda_03_labeled_waveforms', bottom,
                           top=0.83)


def plot_labeled_loudness_envelopes(metadata, amplitudes):
    """Fig 4 (LAY, INTERPRETABLE). Loudness over time, relative to typical.

    This is the most direct picture of the crew chief's rule: a leak hiss
    "doesn't stop" (flat line near 1), traffic "comes and goes" (peaks).
    """
    fig, axes, panels = _labeled_panel_grid(metadata, MEDIUM_FIGURE)
    for ax, record in panels:
        window_centers, window_rms = compute_rms_envelope(
            amplitudes[record.clip_id])
        relative_loudness = window_rms / np.median(window_rms)
        color = LABEL_COLORS[record.label]
        ax.fill_between(window_centers, relative_loudness, step='mid',
                        color=color, alpha=0.18, linewidth=0)
        ax.step(window_centers, relative_loudness, where='mid',
                color=color, linewidth=0.9)
        ax.axhline(TRANSIENT_LOUDNESS_RATIO, color=CRIME_LAB_GOLD,
                   linestyle='--', linewidth=0.9)
        ax.axhline(1, color=TEXT_GREY, linewidth=0.6)
        ax.set_title(_panel_title(record), loc='left', fontsize=8)
    axes[0, 1].annotate('loud-burst line (3x typical)',
                        xy=(CLIP_DURATION_SECONDS, TRANSIENT_LOUDNESS_RATIO),
                        xytext=(-2, 3), textcoords='offset points',
                        ha='right', fontsize=7, color=CRIME_LAB_GOLD)
    for ax in axes[-1]:
        ax.set_xlabel('Seconds into recording')
    for ax in axes[:, 0]:
        ax.set_ylabel('Loudness vs.\ntypical')
    axes[0, 0].set_xlim(0, CLIP_DURATION_SECONDS)
    axes[0, 0].set_ylim(bottom=0)
    fig.subplots_adjust(hspace=0.55, wspace=0.12)
    bottom = add_title_and_note(
        fig, 4, 'Steady hiss or passing noise? Loudness across each '
        'recording',
        'Loudness is the root-mean-square amplitude in 50-millisecond '
        'windows divided by the clip\'s median window, so 1 = the clip\'s '
        'typical level. A steady hiss is a flat line near 1; sounds that '
        '"come and go" rise above the dashed line. L03 (leak) is the '
        'flattest; L06 (leak, 10 pm) has bursts like the no-leak sites.')
    return finish_and_save(fig, 'eda_04_labeled_loudness_envelopes',
                           bottom, top=0.83)


def _spectrogram_db(amplitude):
    """Return (times, frequencies, power in dB) for one clip."""
    frequencies_hz, times_seconds, power = signal.spectrogram(
        amplitude, fs=SAMPLE_RATE_HZ, nperseg=SPECTROGRAM_SEGMENT_SAMPLES,
        noverlap=SPECTROGRAM_OVERLAP_SAMPLES)
    power_db = POWER_TO_DB * np.log10(power + np.finfo(float).tiny)
    return times_seconds, frequencies_hz, power_db


def plot_labeled_spectrograms(metadata, amplitudes):
    """Fig 5 (ACCURATE). Time-by-pitch power for the six dug sites."""
    fig, axes, panels = _labeled_panel_grid(metadata, TALL_FIGURE)
    spectrograms = {record.clip_id: _spectrogram_db(
        amplitudes[record.clip_id]) for _, record in panels}
    all_db = np.concatenate([db.ravel() for _, _, db in
                             spectrograms.values()])
    color_floor, color_ceiling = np.percentile(
        all_db, [SPECTROGRAM_DB_FLOOR_PERCENTILE,
                 SPECTROGRAM_DB_CEILING_PERCENTILE])
    for ax, record in panels:
        times_seconds, frequencies_hz, power_db = spectrograms[
            record.clip_id]
        image = ax.pcolormesh(times_seconds, frequencies_hz, power_db,
                              cmap=SPECTROGRAM_COLORMAP, vmin=color_floor,
                              vmax=color_ceiling, shading='auto',
                              rasterized=True)
        ax.set_title(_panel_title(record), loc='left', fontsize=8)
        ax.grid(False)
    for ax in axes[-1]:
        ax.set_xlabel('Seconds into recording')
    for ax in axes[:, 0]:
        ax.set_ylabel('Pitch (Hz)')
    axes[0, 0].set_ylim(0, NYQUIST_HZ)
    fig.subplots_adjust(hspace=0.45, wspace=0.12, right=0.86)
    colorbar_axis = fig.add_axes([0.89, 0.25, 0.02, 0.55])
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label('Power (dB, darker = louder)')
    colorbar.outline.set_visible(False)
    bottom = add_title_and_note(
        fig, 5, 'Which pitches are present, moment by moment '
        '(spectrograms)',
        'Each panel shows sound power by pitch (vertical) and time '
        '(horizontal) on a shared color scale. Vertical stripes are brief '
        'broadband events; horizontal bands are sustained tones; an even '
        'wash over many pitches is hiss. All three leaks, including L06, '
        'show a continuous dark band between about 400 and 1,300 Hz for '
        'the full 5 seconds; none of the no-leak sites do. Traffic appears '
        'as dark patches below 300 Hz.')
    return finish_and_save(fig, 'eda_05_labeled_spectrograms', bottom,
                           top=0.86)


def plot_power_spectra(metadata, amplitudes):
    """Fig 6 (ACCURATE). Average pitch content, absolute and shape-only."""
    fig, (absolute_ax, shape_ax) = plt.subplots(1, 2, figsize=SMALL_FIGURE,
                                                sharex=True)
    unlabeled_ids = metadata.loc[~metadata['is_labeled'], 'clip_id']
    for clip_id in unlabeled_ids:
        frequencies_hz, psd = compute_welch_psd(amplitudes[clip_id])
        shape_ax.plot(frequencies_hz,
                      POWER_TO_DB * np.log10(psd / psd.sum()),
                      color=UNLABELED_GREY, linewidth=0.4, alpha=0.6,
                      zorder=1)
    for record in _labeled_records(metadata).itertuples():
        frequencies_hz, psd = compute_welch_psd(amplitudes[record.clip_id])
        line_style = '-' if record.logger_model == 'A' else '--'
        line_kwargs = dict(color=LABEL_COLORS[record.label],
                           linestyle=line_style, linewidth=1.1, zorder=3)
        absolute_ax.plot(frequencies_hz, POWER_TO_DB * np.log10(psd),
                         label=record.clip_id, **line_kwargs)
        shape_ax.plot(frequencies_hz,
                      POWER_TO_DB * np.log10(psd / psd.sum()),
                      **line_kwargs)
    absolute_ax.set_title('Absolute power (depends on loudness and device)',
                          loc='left')
    shape_ax.set_title('Shape only (each clip scaled to equal total)',
                       loc='left')
    absolute_ax.set_ylabel('Power (dB)')
    shape_ax.set_ylabel('Relative power (dB)')
    for ax in (absolute_ax, shape_ax):
        ax.set_xlabel('Pitch (Hz)')
        ax.axvline(HIGH_PITCH_CUTOFF_HZ, color=TEXT_GREY, linewidth=0.6,
                   linestyle=':')
    absolute_ax.set_xlim(0, NYQUIST_HZ)
    legend_handles = [
        Line2D([], [], color=UCHICAGO_MAROON, label='Dug: leak'),
        Line2D([], [], color=HEALTH_LAB_TEAL, label='Dug: no leak'),
        Line2D([], [], color=UNLABELED_GREY, label='Not yet dug'),
        Line2D([], [], color=TEXT_GREY, linestyle='-', label='Logger A'),
        Line2D([], [], color=TEXT_GREY, linestyle='--', label='Logger B')]
    fig.legend(handles=legend_handles, loc='upper center', ncol=5,
               bbox_to_anchor=(0.5, 0.92))
    fig.subplots_adjust(wspace=0.3)
    bottom = add_title_and_note(
        fig, 6, 'Average pitch content of each recording (power spectra)',
        'Welch power spectra (0.25-second segments). The left panel mixes '
        'loudness, pipe, and device gain; the right panel removes overall '
        'loudness to compare shape. Dotted line marks 500 Hz. All three '
        'leak spectra rise into a plateau from about 400 to 1,300 Hz and '
        'drop sharply at both edges; the no-leak spectra have no such '
        'plateau. The plateau is weakest for L06.')
    return finish_and_save(fig, 'eda_06_power_spectra', bottom, top=0.76)


def plot_amplitude_distributions(metadata, amplitudes):
    """Fig 7 (ACCURATE). Amplitude histograms vs. a normal (pure hiss)."""
    fig, axes, panels = _labeled_panel_grid(metadata, MEDIUM_FIGURE)
    bin_edges = np.linspace(-HISTOGRAM_STANDARD_UNITS,
                            HISTOGRAM_STANDARD_UNITS, HISTOGRAM_BINS)
    reference_x = np.linspace(-HISTOGRAM_STANDARD_UNITS,
                              HISTOGRAM_STANDARD_UNITS, 400)
    for ax, record in panels:
        amplitude = amplitudes[record.clip_id]
        standardized = (amplitude - amplitude.mean()) / amplitude.std()
        ax.hist(standardized, bins=bin_edges, density=True,
                color=LABEL_COLORS[record.label], alpha=0.75)
        ax.plot(reference_x, stats.norm.pdf(reference_x), color=TEXT_DARK,
                linestyle='--', linewidth=0.8)
        ax.set_yscale('log')
        ax.set_ylim(bottom=HISTOGRAM_MIN_DENSITY)
        ax.set_title(_panel_title(record), loc='left', fontsize=8)
        ax.text(0.98, 0.85,
                f'spikiness = {stats.kurtosis(amplitude):.1f}',
                transform=ax.transAxes, ha='right', fontsize=7,
                color=TEXT_GREY)
    for ax in axes[-1]:
        ax.set_xlabel('Amplitude (standard deviations from mean)')
    for ax in axes[:, 0]:
        ax.set_ylabel('Density (log)')
    fig.subplots_adjust(hspace=0.55, wspace=0.12)
    bottom = add_title_and_note(
        fig, 7, 'How spiky is each recording? Amplitude distribution vs. '
        'pure hiss',
        'Dashed curve is the normal distribution that pure, steady hiss '
        'would follow. Heavy tails (bars far outside the curve) mean rare, '
        'very loud moments; spikiness is excess kurtosis (0 for pure hiss)'
        '. Only L03 matches pure hiss.')
    return finish_and_save(fig, 'eda_07_amplitude_distributions', bottom,
                           top=0.83)


def _display_values(df, feature):
    """Values on the scale they are plotted (log for wide-ranging ones)."""
    values = df[feature]
    if feature in LOG_SCALED_FEATURES:
        return np.log10(values)
    if feature == 'excess_kurtosis':
        return np.sign(values) * np.log10(1 + np.abs(values))
    return values


def _display_label(feature):
    """Axis label, noting any log transform."""
    if feature in LOG_SCALED_FEATURES:
        return f'{FEATURE_LABELS[feature]}, log10'
    if feature == 'excess_kurtosis':
        return f'{FEATURE_LABELS[feature]}, signed log10(1+|x|)'
    return FEATURE_LABELS[feature]


def plot_feature_dot_plots(clip_features):
    """Fig 8 (INTERPRETABLE). Key features: dug leak vs. no leak vs. undug."""
    features = ('rms_loudness', 'loudness_variability', 'transient_share',
                'excess_kurtosis', 'spectral_flatness', 'high_pitch_share')
    row_order = [('Dug: leak', clip_features['label_group'] == LEAK_LABEL),
                 ('Dug: no leak',
                  clip_features['label_group'] == NO_LEAK_LABEL),
                 ('Not dug, logger A',
                  (clip_features['label_group'] == 'unlabeled')
                  & (clip_features['logger_model'] == 'A')),
                 ('Not dug, logger B',
                  (clip_features['label_group'] == 'unlabeled')
                  & (clip_features['logger_model'] == 'B'))]
    rng = np.random.default_rng(JITTER_SEED)
    fig, axes = plt.subplots(2, 3, figsize=TALL_FIGURE, sharey=True)
    for ax, feature in zip(axes.ravel(), features):
        values = _display_values(clip_features, feature)
        for row_position, (_, is_in_row) in enumerate(row_order):
            subset = clip_features[is_in_row]
            is_dug_row = row_position <= 1
            if is_dug_row:
                y_positions = row_position + np.array(
                    DUG_SUBROW_OFFSETS[:len(subset)])
            else:
                y_positions = row_position + _jitter(len(subset), rng)
            ax.scatter(values[is_in_row], y_positions,
                       s=60 if is_dug_row else 34,
                       color=[LABEL_COLORS[g] for g in
                              subset['label_group']],
                       marker='o', edgecolor='white', linewidth=0.5,
                       zorder=3)
            if is_dug_row:
                for clip_id, value, y, group in zip(
                        subset['clip_id'], values[is_in_row], y_positions,
                        subset['label_group']):
                    ax.annotate(clip_id, (value, y), xytext=(6, 0),
                                textcoords='offset points', va='center',
                                fontsize=6.5, color=LABEL_COLORS[group])
        ax.set_xlabel(_display_label(feature), fontsize=7.5)
        ax.grid(axis='y', visible=False)
        ax.grid(axis='x', color='#EFEFEF')
    axes[0, 0].set_yticks(range(len(row_order)))
    axes[0, 0].set_yticklabels([name for name, _ in row_order])
    axes[0, 0].invert_yaxis()
    fig.subplots_adjust(hspace=0.45, wspace=0.12)
    bottom = add_title_and_note(
        fig, 8, 'How do dug leak and no-leak sites compare on six '
        'recording features?',
        'Each dot is one recording; undug sites are split by logger model. '
        'None of these whole-clip summaries cleanly separates all three '
        'leaks from all three non-leaks: L06 sits with or right beside the '
        'non-leaks on every one. Logger B undug sites are louder overall '
        'than logger A. '
        'A large cluster of undug sites is very quiet, very steady, and '
        'noise-like, likely the sensor\'s own background noise.')
    return finish_and_save(fig, 'eda_08_feature_dot_plots', bottom,
                           top=0.92)


def plot_features_by_clock_time(clip_features):
    """Fig 9 (INTERPRETABLE). Do traffic-type features rise toward dawn?"""
    features = ('rms_loudness', 'loudness_variability', 'transient_share')
    fig, axes = plt.subplots(1, 3, figsize=SMALL_FIGURE, sharex=True)
    for ax, feature in zip(axes, features):
        values = _display_values(clip_features, feature)
        shade_commute_window(ax, MORNING_COMMUTE_START_HOUR, LAST_AXIS_HOUR)
        _scatter_by_group(ax, clip_features, 'overnight_hour', values,
                          LOGGER_MARKERS, 'logger_model')
        rho, p_value = stats.spearmanr(clip_features['overnight_hour'],
                                       clip_features[feature])
        ax.set_title(f'Spearman rho = {rho:+.2f} (p = {p_value:.2f})',
                     loc='left', fontsize=7.5, fontweight='normal',
                     color=TEXT_GREY)
        ax.set_ylabel(_display_label(feature), fontsize=7.5)
        format_overnight_axis(ax, FIRST_AXIS_HOUR, LAST_AXIS_HOUR)
        ax.set_xticks([-2, 0, 3, 6])
        ax.set_xticklabels(['10 pm', '12 am', '3 am', '6 am'])
    legend_handles = _label_legend_handles() + [
        Line2D([], [], marker=marker, linestyle='', color=TEXT_GREY,
               label=f'Logger {logger}')
        for logger, marker in LOGGER_MARKERS.items()]
    fig.legend(handles=legend_handles, loc='upper center', ncol=5,
               bbox_to_anchor=(0.5, 0.91))
    fig.subplots_adjust(wspace=0.42)
    bottom = add_title_and_note(
        fig, 9, 'Do recordings get noisier toward the morning commute?',
        'Each dot is one recording (all 46). Recordings get louder later in '
        'the night; burstiness shows no clear trend. Because the three '
        'non-leak dug sites were all recorded late, any loudness-based rule '
        'learned from them may partly be a clock-time rule.')
    return finish_and_save(fig, 'eda_09_features_by_clock_time', bottom,
                           top=0.76)


def plot_device_and_pipe_effects(clip_features):
    """Fig 10 (INTERPRETABLE). Logger and pipe differences, undug sites."""
    features = ('rms_loudness', 'loudness_variability', 'high_pitch_share')
    unlabeled = clip_features[~clip_features['is_labeled']]
    comparisons = [('logger_model', ('A', 'B'), 'Logger', {
        'A': HEALTH_LAB_NAVY, 'B': CRIME_LAB_GOLD}),
        ('pipe_material', ('cast_iron', 'PVC'), 'Pipe', PIPE_COLORS)]
    rng = np.random.default_rng(JITTER_SEED)
    fig, axes = plt.subplots(2, 3, figsize=MEDIUM_FIGURE)
    for row_axes, (column, levels, row_name, colors) in zip(axes,
                                                            comparisons):
        for ax, feature in zip(row_axes, features):
            values = _display_values(unlabeled, feature)
            grouped = [values[unlabeled[column] == level].to_numpy()
                       for level in levels]
            ax.boxplot(grouped, orientation='horizontal', widths=0.5,
                       showfliers=False,
                       medianprops=dict(color=TEXT_DARK),
                       boxprops=dict(color=TEXT_GREY),
                       whiskerprops=dict(color=TEXT_GREY),
                       capprops=dict(color=TEXT_GREY))
            for position, (level, level_values) in enumerate(
                    zip(levels, grouped), start=1):
                ax.scatter(level_values,
                           position + _jitter(len(level_values), rng),
                           s=22, color=colors[level], edgecolor='white',
                           linewidth=0.4, zorder=3)
            p_value = stats.mannwhitneyu(*grouped).pvalue
            ax.set_title(f'Mann-Whitney p = {p_value:.2f}', loc='left',
                         fontsize=7.5, fontweight='normal', color=TEXT_GREY)
            ax.set_yticks([1, 2])
            ax.set_yticklabels([
                f'{row_name} {level.replace("_", " ")} (n={len(g)})'
                for level, g in zip(levels, grouped)], fontsize=7.5)
            ax.set_xlabel(_display_label(feature), fontsize=7.5)
            ax.grid(axis='y', visible=False)
            ax.grid(axis='x', color='#EFEFEF')
    for ax in axes[:, 1:].ravel():
        ax.set_yticklabels([])
    fig.subplots_adjust(hspace=0.75, wspace=0.12, left=0.17)
    bottom = add_title_and_note(
        fig, 10, 'Do the device or the pipe change the sound? (40 undug '
        'sites)',
        'Boxes show the middle half of undug sites; dots are individual '
        'sites. Logger B recordings are about 3x louder than logger A '
        '(p = 0.01), so raw loudness partly measures the device. Pipe '
        'material shows no clear difference, but no PVC leak has been '
        'observed, so a PVC leak signature is unknown.')
    return finish_and_save(fig, 'eda_10_device_and_pipe_effects', bottom,
                           top=0.89)


def plot_feature_correlations(clip_features):
    """Fig 11 (ACCURATE). Spearman correlations among candidate features."""
    correlations = clip_features[FEATURE_COLUMNS].corr(method='spearman')
    fig, ax = plt.subplots(figsize=TALL_FIGURE)
    image = ax.imshow(correlations, cmap=CORRELATION_COLORMAP, vmin=-1,
                      vmax=1)
    short_labels = [FEATURE_LABELS[f].split(' (')[0] for f in FEATURE_COLUMNS]
    ax.set_xticks(range(len(FEATURE_COLUMNS)))
    ax.set_yticks(range(len(FEATURE_COLUMNS)))
    ax.set_xticklabels(short_labels, rotation=55, ha='right', fontsize=7)
    ax.set_yticklabels(short_labels, fontsize=7)
    for row_index in range(len(FEATURE_COLUMNS)):
        for column_index in range(len(FEATURE_COLUMNS)):
            value = correlations.iloc[row_index, column_index]
            ax.text(column_index, row_index, f'{value:.1f}', ha='center',
                    va='center', fontsize=6,
                    color='white' if abs(value) > 0.6 else TEXT_DARK)
    ax.grid(False)
    ax.spines[:].set_visible(False)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.7)
    colorbar.set_label('Spearman correlation')
    colorbar.outline.set_visible(False)
    bottom = add_title_and_note(
        fig, 11, 'Many candidate features measure the same thing '
        '(all 46 recordings)',
        f'Rank correlations between the {len(FEATURE_COLUMNS)} candidate '
        'features. Two blocks emerge: loudness/burstiness (top left) and '
        'pitch content (bottom right). Features within a block are near-'
        'duplicates; with only six dug sites, a model should use at most '
        'one feature from each block.')
    return finish_and_save(fig, 'eda_11_feature_correlations',
                           bottom + CORRELATION_TICK_LABEL_ALLOWANCE,
                           top=0.93, left=0.24)


def plot_principal_components(clip_features):
    """Fig 12 (ACCURATE, low interpretability). PCA map of all clips."""
    display_table = pd.DataFrame({
        feature: _display_values(clip_features, feature)
        for feature in PCA_FEATURES})
    standardized = StandardScaler().fit_transform(display_table)
    pca = PCA(n_components=PCA_COMPONENTS).fit(standardized)
    scores = pca.transform(standardized)
    plot_table = clip_features.assign(pc1=scores[:, 0], pc2=scores[:, 1])
    fig, ax = plt.subplots(figsize=SMALL_FIGURE)
    _scatter_by_group(ax, plot_table, 'pc1', plot_table['pc2'],
                      LOGGER_MARKERS, 'logger_model')
    labeled_ids = plot_table.loc[plot_table['is_labeled'], 'clip_id']
    _annotate_clips(ax, plot_table, 'pc1', 'pc2', labeled_ids)
    _annotate_clips(ax, plot_table, 'pc1', 'pc2', NOTABLE_UNDUG_CLIPS,
                    color=TEXT_GREY)
    explained = pca.explained_variance_ratio_
    ax.set_xlabel(f'Component 1 ({explained[0]:.0%} of variation)')
    ax.set_ylabel(f'Component 2 ({explained[1]:.0%} of variation)')
    ax.grid(axis='x', color='#EFEFEF')
    legend_handles = _label_legend_handles() + [
        Line2D([], [], marker=marker, linestyle='', color=TEXT_GREY,
               label=f'Logger {logger}')
        for logger, marker in LOGGER_MARKERS.items()]
    ax.legend(handles=legend_handles, loc='upper center', ncol=5,
              bbox_to_anchor=(0.5, 1.14))
    bottom = add_title_and_note(
        fig, 12, 'A compressed overview of all 46 recordings (principal '
        'components)',
        f'Two components summarize {explained.sum():.0%} of the variation '
        f'in {len(PCA_FEATURES)} standardized features. This view is '
        'information-dense but its axes have no direct physical meaning. '
        'The tight cluster of quiet, steady undug sites stands apart from '
        'every dug site.')
    return finish_and_save(fig, 'eda_12_principal_components', bottom,
                           top=0.80)


def plot_steadiness_vs_pitch(clip_features):
    """Fig 13 (LAY, INTERPRETABLE). The field note as a two-axis map."""
    fig, ax = plt.subplots(figsize=MEDIUM_FIGURE)
    x_values = np.log10(clip_features['loudness_variability'])
    plot_table = clip_features.assign(log_variability=x_values)
    _scatter_by_group(ax, plot_table, 'log_variability',
                      plot_table['high_pitch_share'], PIPE_MARKERS,
                      'pipe_material')
    labeled_ids = plot_table.loc[plot_table['is_labeled'], 'clip_id']
    _annotate_clips(ax, plot_table, 'log_variability', 'high_pitch_share',
                    labeled_ids)
    _annotate_clips(ax, plot_table, 'log_variability', 'high_pitch_share',
                    NOTABLE_UNDUG_CLIPS, color=TEXT_GREY)
    tick_values = [0.05, 0.1, 0.2, 0.5, 1, 2]
    ax.set_xticks(np.log10(tick_values))
    ax.set_xticklabels([str(v) for v in tick_values])
    ax.set_xlabel('Loudness variability (log scale)  '
                  '← steady    comes and goes →')
    ax.set_ylabel(f'Share of energy above {HIGH_PITCH_CUTOFF_HZ} Hz\n'
                  '← low rumble    high-pitched hiss →')
    ax.grid(axis='x', color='#EFEFEF')
    legend_handles = _label_legend_handles() + [
        Line2D([], [], marker=marker, linestyle='', color=TEXT_GREY,
               label=pipe.replace('_', ' '))
        for pipe, marker in PIPE_MARKERS.items()]
    ax.legend(handles=legend_handles, loc='upper center', ncol=5,
              bbox_to_anchor=(0.5, 1.1))
    bottom = add_title_and_note(
        fig, 13, 'Two field-note features: how steady and how high-pitched '
        'is each recording?',
        'The crew chief describes a leak as a steady hiss. Leaks L01 and '
        'L03 sit toward the steady, high-pitched corner; L06 sits with the '
        'no-leak sites. U16 and U22 are steady and high-pitched, but their '
        'spectra are flat (even hiss at all pitches) rather than the 400-'
        '1,300 Hz band seen at the leaks, and U35\'s high value comes from '
        'one impact. The dense cluster near 0.1 is quiet background.')
    return finish_and_save(fig, 'eda_13_steadiness_vs_pitch', bottom,
                           top=0.86)


def plot_quality_spotlight(metadata, amplitudes, clip_features):
    """Fig 14 (LAY). Recordings flagged by the data checks."""
    fig, (dropout_ax, impulse_ax) = plt.subplots(2, 1, figsize=MEDIUM_FIGURE,
                                                 sharex=True)
    time_seconds = np.arange(len(amplitudes['U40'])) / SAMPLE_RATE_HZ
    dropout_amplitude = amplitudes['U40']
    dropout_start, dropout_length = find_longest_constant_run(
        dropout_amplitude)
    dropout_start_seconds = dropout_start / SAMPLE_RATE_HZ
    dropout_seconds = dropout_length / SAMPLE_RATE_HZ
    dropout_ax.axvspan(dropout_start_seconds,
                       dropout_start_seconds + dropout_seconds,
                       color=FRESNO_BAND_BLUE, zorder=0)
    dropout_ax.plot(time_seconds, dropout_amplitude, linewidth=0.3,
                    color=UNLABELED_GREY)
    is_clipped = np.abs(dropout_amplitude) >= CLIPPING_THRESHOLD
    dropout_ax.scatter(time_seconds[is_clipped],
                       dropout_amplitude[is_clipped], s=1.5,
                       color=UCHICAGO_MAROON, zorder=3, linewidth=0)
    for limit in (-FULL_SCALE_AMPLITUDE, FULL_SCALE_AMPLITUDE):
        dropout_ax.axhline(limit, color=CRIME_LAB_GOLD, linestyle='--',
                           linewidth=0.8)
    dropout_ax.text(dropout_start_seconds + dropout_seconds / 2, 0.15,
                    f'signal drops out\nfor {dropout_seconds:.1f} s',
                    ha='center', fontsize=7.5, color=HEALTH_LAB_TEAL,
                    fontweight='bold')
    u40 = metadata.set_index('clip_id').loc['U40']
    dropout_ax.set_title(
        f'U40  |  logger {u40.logger_model}  |  '
        f'{_format_clock(u40.recorded_at)}  |  {is_clipped.mean():.0%} of '
        f'samples clipped at the sensor ceiling (maroon)',
        loc='left', fontsize=8)
    impulse_amplitude = amplitudes['U35']
    impulse_ax.plot(time_seconds, impulse_amplitude, linewidth=0.3,
                    color=UNLABELED_GREY)
    peak_index = np.argmax(np.abs(impulse_amplitude))
    impulse_ax.annotate('single very loud impact',
                        (time_seconds[peak_index],
                         impulse_amplitude[peak_index]),
                        xytext=(15, -5), textcoords='offset points',
                        fontsize=7.5, color=HEALTH_LAB_TEAL,
                        fontweight='bold')
    u35 = metadata.set_index('clip_id').loc['U35']
    u35_kurtosis = clip_features.set_index('clip_id').loc[
        'U35', 'excess_kurtosis']
    impulse_ax.set_title(
        f'U35  |  logger {u35.logger_model}  |  '
        f'{_format_clock(u35.recorded_at)}  |  spikiness '
        f'{u35_kurtosis:.0f} (pure hiss = 0)', loc='left', fontsize=8)
    for ax in (dropout_ax, impulse_ax):
        ax.set_ylabel('Amplitude')
        ax.grid(axis='y', visible=False)
    impulse_ax.set_xlabel('Seconds into recording')
    impulse_ax.set_xlim(0, CLIP_DURATION_SECONDS)
    fig.subplots_adjust(hspace=0.45)
    bottom = add_title_and_note(
        fig, 14, 'Two recordings that need a closer listen before modeling',
        'U40, the loudest clip, is overloaded: its loudest stretches are '
        'cut off at the sensor ceiling just under full scale (dashed '
        'lines), and the signal drops to exactly zero for 1.2 seconds. Its '
        'loudness and pitch summaries are unreliable. U35 is dominated by '
        'one impact (e.g., a dropped manhole cover) that swamps whole-clip '
        'averages; burst-resistant features are needed for clips like it.')
    return finish_and_save(fig, 'eda_14_quality_spotlight', bottom,
                           top=0.87)


def make_all_eda_figures():
    """Load data once and draw every exploratory figure."""
    apply_urban_labs_style()
    metadata = load_metadata()
    amplitudes = load_all_amplitudes(metadata)
    clip_features = build_clip_feature_table(metadata, amplitudes)
    return [
        plot_coverage_timeline(metadata),
        plot_coverage_bars(metadata),
        plot_labeled_waveforms(metadata, amplitudes),
        plot_labeled_loudness_envelopes(metadata, amplitudes),
        plot_labeled_spectrograms(metadata, amplitudes),
        plot_power_spectra(metadata, amplitudes),
        plot_amplitude_distributions(metadata, amplitudes),
        plot_feature_dot_plots(clip_features),
        plot_features_by_clock_time(clip_features),
        plot_device_and_pipe_effects(clip_features),
        plot_feature_correlations(clip_features),
        plot_principal_components(clip_features),
        plot_steadiness_vs_pitch(clip_features),
        plot_quality_spotlight(metadata, amplitudes, clip_features),
    ]


if __name__ == '__main__':
    for figure_path in make_all_eda_figures():
        print(f'Saved {figure_path}')
