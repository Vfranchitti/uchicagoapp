"""Exploratory figures: what the recordings themselves look like.

Fig 3  raw waveforms of the six dug sites
Fig 4  loudness envelopes (steady hiss vs. passing noise)
Fig 5  spectrograms of the six dug sites
Fig 6  average power spectra
Fig 7  amplitude distributions vs. pure hiss
Fig 14 two recordings flagged by the quality checks

Figures are tagged in their docstrings as LAY (readable by a non-technical
audience), INTERPRETABLE (axes with a plain physical meaning) or ACCURATE
(information-dense, needs training to read).

Run from the repo root:  python -m src.plot_eda_signals
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy import stats

from src.compute_clip_features import (
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
    NO_LEAK_LABEL,
    NYQUIST_HZ,
    POWER_TO_DB,
    SAMPLE_RATE_HZ,
    TRANSIENT_LOUDNESS_RATIO,
)
from src.load_recordings import load_all_amplitudes, load_metadata
from src.plot_style import (
    CRIME_LAB_GOLD,
    FRESNO_BAND_BLUE,
    HEALTH_LAB_TEAL,
    LABEL_COLORS,
    LABEL_FONT_SIZE,
    MEDIUM_FIGURE,
    PANEL_TITLE_FONT_SIZE,
    SMALL_FIGURE,
    SMALL_FONT_SIZE,
    SPECTROGRAM_COLORMAP,
    SPECTROGRAM_DB_CEILING_PERCENTILE,
    SPECTROGRAM_DB_FLOOR_PERCENTILE,
    TALL_FIGURE,
    TEXT_DARK,
    TEXT_GREY,
    UCHICAGO_MAROON,
    UNLABELED_GREY,
    add_title_and_note,
    apply_urban_labs_style,
    finish_and_save,
    format_clock,
    spectrogram_db,
)

HISTOGRAM_MIN_DENSITY = 1e-5        # floor for the log-density axis
HISTOGRAM_BINS = 81
HISTOGRAM_STANDARD_UNITS = 8        # x-range of standardized amplitude
FULL_SCALE_AMPLITUDE = 1.0


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
            f'{format_clock(record.recorded_at)}')


def plot_labeled_waveforms(metadata, amplitudes):
    """Fig 3 (LAY). Raw 5-second waveforms of the six dug sites."""
    fig, axes, panels = _labeled_panel_grid(metadata, MEDIUM_FIGURE)
    time_seconds = np.arange(len(next(iter(amplitudes.values())))) \
        / SAMPLE_RATE_HZ
    for ax, record in panels:
        ax.plot(time_seconds, amplitudes[record.clip_id], linewidth=0.3,
                color=LABEL_COLORS[record.label])
        ax.set_title(_panel_title(record), loc='left',
                     fontsize=PANEL_TITLE_FONT_SIZE)
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
        ax.set_title(_panel_title(record), loc='left',
                     fontsize=PANEL_TITLE_FONT_SIZE)
    axes[0, 1].annotate('loud-burst line (3x typical)',
                        xy=(CLIP_DURATION_SECONDS, TRANSIENT_LOUDNESS_RATIO),
                        xytext=(-2, 3), textcoords='offset points',
                        ha='right', fontsize=SMALL_FONT_SIZE,
                        color=CRIME_LAB_GOLD)
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


def plot_labeled_spectrograms(metadata, amplitudes):
    """Fig 5 (ACCURATE). Time-by-pitch power for the six dug sites."""
    fig, axes, panels = _labeled_panel_grid(metadata, TALL_FIGURE)
    spectrograms = {record.clip_id: spectrogram_db(
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
        ax.set_title(_panel_title(record), loc='left',
                     fontsize=PANEL_TITLE_FONT_SIZE)
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
        ax.set_title(_panel_title(record), loc='left',
                     fontsize=PANEL_TITLE_FONT_SIZE)
        ax.text(0.98, 0.85,
                f'spikiness = {stats.kurtosis(amplitude):.1f}',
                transform=ax.transAxes, ha='right', fontsize=SMALL_FONT_SIZE,
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
                    ha='center', fontsize=LABEL_FONT_SIZE,
                    color=HEALTH_LAB_TEAL,
                    fontweight='bold')
    u40 = metadata.set_index('clip_id').loc['U40']
    dropout_ax.set_title(
        f'U40  |  logger {u40.logger_model}  |  '
        f'{format_clock(u40.recorded_at)}  |  {is_clipped.mean():.0%} of '
        f'samples clipped at the sensor ceiling (maroon)',
        loc='left', fontsize=PANEL_TITLE_FONT_SIZE)
    impulse_amplitude = amplitudes['U35']
    impulse_ax.plot(time_seconds, impulse_amplitude, linewidth=0.3,
                    color=UNLABELED_GREY)
    peak_index = np.argmax(np.abs(impulse_amplitude))
    impulse_ax.annotate('single very loud impact',
                        (time_seconds[peak_index],
                         impulse_amplitude[peak_index]),
                        xytext=(15, -5), textcoords='offset points',
                        fontsize=LABEL_FONT_SIZE, color=HEALTH_LAB_TEAL,
                        fontweight='bold')
    u35 = metadata.set_index('clip_id').loc['U35']
    u35_kurtosis = clip_features.set_index('clip_id').loc[
        'U35', 'excess_kurtosis']
    impulse_ax.set_title(
        f'U35  |  logger {u35.logger_model}  |  '
        f'{format_clock(u35.recorded_at)}  |  spikiness '
        f'{u35_kurtosis:.0f} (pure hiss = 0)', loc='left',
        fontsize=PANEL_TITLE_FONT_SIZE)
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


def make_plot_eda_signals_figures():
    """Load data once and draw every figure in this file."""
    apply_urban_labs_style()
    metadata = load_metadata()
    amplitudes = load_all_amplitudes(metadata)
    clip_features = build_clip_feature_table(metadata, amplitudes)
    return [
        plot_labeled_waveforms(metadata, amplitudes),
        plot_labeled_loudness_envelopes(metadata, amplitudes),
        plot_labeled_spectrograms(metadata, amplitudes),
        plot_power_spectra(metadata, amplitudes),
        plot_amplitude_distributions(metadata, amplitudes),
        plot_quality_spotlight(metadata, amplitudes, clip_features),
    ]


if __name__ == '__main__':
    for figure_path in make_plot_eda_signals_figures():
        print(f'Saved {figure_path}')
