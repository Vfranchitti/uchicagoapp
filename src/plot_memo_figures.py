"""Figures for the two-page memo, styled after the EPIC Fresno brief.

Each figure is one memo column wide, drawn on the light grey panel the
Fresno brief uses; the figure title and note are set in the memo itself.

Fig 1  When and where the pilot recorded (data and design checks)
Fig 2  What a leak sounds like: steady sound by pitch (exploration)
Fig 3  Chance of a leak at each candidate site (results)
Fig 4  Which candidates hold up under each check (results)

Reads outputs of earlier steps (Model 1, Model 2, the semi-supervised
checks); run after them.

Outputs: outputs/figures/4_memo/memo_fig{1..4}_*.png
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.text import Text
from matplotlib.transforms import blended_transform_factory

from src.config import (
    ANALYSIS_EXCLUDED_CLIPS,
    EXCESS_BANDS_HZ,
    HISTORICAL_LEAK_RATE,
    LEAK_LABEL,
    MEMO_FIGURE_DIR,
    MODEL1_PROBABILITY_PATH,
    MODEL1_SCORES_PATH,
    MODEL2_PROBABILITY_PATH,
    MORNING_COMMUTE_START_HOUR,
    NO_LEAK_LABEL,
    NYQUIST_HZ,
    SELF_TRAINING_TABLE_PATH,
)
from src.detect_steady_bands import steady_spectra
from src.load_recordings import load_all_amplitudes, load_metadata
from src.plot_model1_results import CLUSTER_COLUMNS, build_robustness_grid
from src.plot_style import save_figure as _save_figure
from src.run_model_knowledge_score import (
    leak_clusters_by_setting,
    leave_one_out,
)
from src.spectral_inputs import build_spectral_inputs

# Fresno brief palette and type (Gotham is not installed; Century Gothic is
# the closest geometric sans available).
FRESNO_FONT = 'Century Gothic'
FRESNO_MAROON = '#8B0021'
FRESNO_BLUE = '#3F7FA8'
FRESNO_BAND = '#D6E8F3'
FRESNO_PANEL = '#F2F2F2'
FRESNO_TEXT = '#37434A'
FRESNO_GREY = '#9A9A9A'
FRESNO_LIGHT_GREY = '#CFCFCF'
FONT_SIZE = 6.5
SMALL_SIZE = 5.5
COLUMN_FIGURE = (3.3, 2.15)
TALL_COLUMN_FIGURE = (3.3, 2.3)
FIRST_HOUR, LAST_HOUR = -2.25, 7.45
# Text must stay this many pixels (at 100 dpi) inside the canvas.
EDGE_PAD_PIXELS = 2
SMOOTHING_BINS = 9
ROW_JITTER = (-0.12, 0.0, 0.12)
BAND_SITES_ORDER = ('U07', 'U17', 'U08', 'U36', 'U34', 'U15', 'U03')


def _style():
    """Fresno look: small grey text, no top/right spines, grey panel."""
    plt.rcParams.update({
        'font.family': FRESNO_FONT, 'font.size': FONT_SIZE,
        'axes.edgecolor': FRESNO_GREY, 'axes.linewidth': 0.5,
        'axes.labelcolor': FRESNO_TEXT, 'axes.labelsize': FONT_SIZE,
        'axes.spines.top': False, 'axes.spines.right': False,
        'xtick.color': FRESNO_TEXT, 'ytick.color': FRESNO_TEXT,
        'xtick.labelsize': SMALL_SIZE, 'ytick.labelsize': SMALL_SIZE,
        'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
        'axes.facecolor': FRESNO_PANEL, 'figure.facecolor': FRESNO_PANEL,
        'savefig.facecolor': FRESNO_PANEL, 'legend.frameon': False,
        'legend.fontsize': SMALL_SIZE, 'axes.grid': False,
    })


def _band_label(ax, x, y, text):
    """Small all-caps label inside a shaded band (Fresno 'PILOT')."""
    ax.text(x, y, text, fontsize=SMALL_SIZE - 0.5, color=FRESNO_BLUE,
            ha='center', va='top')


def save_figure(fig, folder, file_stem):
    """Save after checking that no text runs off the canvas."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.bbox.width, fig.bbox.height
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text().strip():
            continue
        box = text.get_window_extent(renderer)
        if (box.x0 < EDGE_PAD_PIXELS or box.y0 < EDGE_PAD_PIXELS
                or box.x1 > width - EDGE_PAD_PIXELS
                or box.y1 > height - EDGE_PAD_PIXELS):
            raise ValueError(f'{file_stem}: text {text.get_text()!r} runs '
                             'off the canvas')
    return _save_figure(fig, folder, file_stem)


def _clock(hour):
    hour_of_day = hour % 24
    suffix = 'am' if hour_of_day < 12 else 'pm'
    return f'{int(hour_of_day % 12 or 12)} {suffix}'


# ---------------------------------------------------------------------------
# Figure 1: when and where the pilot recorded
# ---------------------------------------------------------------------------
def plot_recording_timeline(metadata):
    """Each recording as a dot by clock time, one row per pipe material."""
    fig, ax = plt.subplots(figsize=COLUMN_FIGURE)
    ax.axvspan(MORNING_COMMUTE_START_HOUR, LAST_HOUR, color=FRESNO_BAND,
               linewidth=0, zorder=0)
    _band_label(ax, (MORNING_COMMUTE_START_HOUR + LAST_HOUR) / 2, 1.8,
                'MORNING\nCOMMUTE')
    rows = {'cast_iron': 1.0, 'PVC': 0.0}
    for index, record in enumerate(metadata.sort_values(
            'overnight_hour').itertuples()):
        y = rows[record.pipe_material] + ROW_JITTER[index % 3]
        if record.clip_id in ANALYSIS_EXCLUDED_CLIPS:
            ax.scatter(record.overnight_hour, y, marker='x', s=14,
                       color=FRESNO_GREY, linewidth=0.8, zorder=3)
            continue
        style = {LEAK_LABEL: dict(s=34, color=FRESNO_MAROON, zorder=4),
                 NO_LEAK_LABEL: dict(s=34, color=FRESNO_BLUE, zorder=4)}.get(
            record.label_group, dict(s=10, color=FRESNO_LIGHT_GREY,
                                     zorder=2))
        ax.scatter(record.overnight_hour, y, edgecolor='white',
                   linewidth=0.4, **style)
    ax.text(0.4, 1.33, 'All 3 leaks:\nbefore 3:30 am', color=FRESNO_MAROON,
            fontsize=SMALL_SIZE, ha='center', va='bottom')
    ax.text(4.7, 1.33, 'All 3 non-leaks:\nafter 4:15 am', color=FRESNO_BLUE,
            fontsize=SMALL_SIZE, ha='center', va='bottom')
    ax.text(2.3, -0.42, 'No PVC site has been dug', color=FRESNO_TEXT,
            fontsize=SMALL_SIZE, ha='center', va='top', style='italic')
    ax.set_yticks(list(rows.values()))
    ax.set_yticklabels(['Cast iron\n(34 sites)', 'PVC\n(12 sites)'])
    ax.set_ylim(-0.7, 1.85)
    ticks = range(-2, 7)
    ax.set_xticks(list(ticks))
    ax.set_xticklabels([_clock(t) for t in ticks])
    ax.set_xlim(FIRST_HOUR, LAST_HOUR)
    ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)
    handles = [plt.Line2D([], [], marker='o', linestyle='', color=c,
                          markersize=m, label=label)
               for c, m, label in ((FRESNO_MAROON, 4.5, 'Dug: leak'),
                                   (FRESNO_BLUE, 4.5, 'Dug: no leak'),
                                   (FRESNO_LIGHT_GREY, 3, 'Not dug'))]
    handles.append(plt.Line2D([], [], marker='x', linestyle='',
                              color=FRESNO_GREY, markersize=3.5,
                              label='Unusable (U40)'))
    ax.legend(handles=handles, loc='upper center', ncol=4,
              bbox_to_anchor=(0.45, -0.13), handletextpad=0.2,
              columnspacing=0.8)
    fig.subplots_adjust(left=0.17, right=0.97, top=0.97, bottom=0.25)
    return save_figure(fig, MEMO_FIGURE_DIR, 'memo_fig1_recording_timeline')


# ---------------------------------------------------------------------------
# Figure 2: what a leak sounds like
# ---------------------------------------------------------------------------
def plot_leak_signature(metadata, steady):
    """Median steady sound by pitch: confirmed leaks vs. non-leaks."""
    fig, ax = plt.subplots(figsize=COLUMN_FIGURE)
    band_hz = EXCESS_BANDS_HZ['leak_band_400_1300']
    ax.axvspan(*band_hz, color=FRESNO_BAND, linewidth=0, zorder=0)
    frequencies_hz = steady.columns.to_numpy()
    kernel = np.ones(SMOOTHING_BINS) / SMOOTHING_BINS
    for group, color, label in ((LEAK_LABEL, FRESNO_MAROON,
                                 'Confirmed leaks (3)'),
                                (NO_LEAK_LABEL, FRESNO_BLUE,
                                 'Confirmed non-leaks (3)')):
        ids = metadata.loc[metadata['label_group'] == group, 'clip_id']
        median = steady.loc[ids].median(axis=0).to_numpy()
        smooth = np.convolve(median, kernel, mode='same')
        ax.plot(frequencies_hz, smooth, color=color, linewidth=1.2,
                label=label, zorder=3)
        marks = slice(SMOOTHING_BINS, None, 25)
        ax.scatter(frequencies_hz[marks], smooth[marks], s=6, color=color,
                   zorder=4)
    ax.set_ylim(-2, 25)
    ax.set_yticks([0, 5, 10, 15, 20])
    _band_label(ax, np.mean(band_hz), 24.6, 'LEAK BAND')
    ax.set_xlim(0, NYQUIST_HZ)
    ax.set_xlabel('Pitch (Hz)')
    ax.set_ylabel('Steady sound above\nlogger\'s own noise (dB)')
    ax.legend(loc='center', bbox_to_anchor=(0.42, 0.52), handlelength=1.0)
    fig.subplots_adjust(left=0.17, right=0.95, top=0.95, bottom=0.2)
    return save_figure(fig, MEMO_FIGURE_DIR, 'memo_fig2_leak_signature')


# ---------------------------------------------------------------------------
# Figure 3: chance of a leak at each candidate site
# ---------------------------------------------------------------------------
def plot_leak_chances(probabilities):
    """Main estimate and range for each band site; all others pooled."""
    fig, ax = plt.subplots(figsize=TALL_COLUMN_FIGURE)
    table = probabilities.set_index('clip_id')
    others = probabilities[probabilities['prediction'] == 0]
    rows = [(clip_id, table.loc[clip_id]) for clip_id in BAND_SITES_ORDER]
    positions = np.arange(len(rows) + 1)[::-1]
    for position, (clip_id, record) in zip(positions, rows):
        color = FRESNO_MAROON if record['ranking'] == 'cast_iron' \
            else FRESNO_BLUE
        ax.plot([record['probability_low'], record['probability_high']],
                [position] * 2, color=color, alpha=0.3, linewidth=4,
                solid_capstyle='round')
        ax.scatter(record['confidence'], position, s=26, color=color,
                   edgecolor='white', linewidth=0.5, zorder=3)
        dig = record['dig_priority']
        prefix = f'Dig {int(dig)}' if dig == dig else 'Not on list'
        pipe = 'cast iron' if record['ranking'] == 'cast_iron' else 'PVC'
        ax.text(-0.015, position, f'{clip_id}  ', ha='right', va='center',
                fontsize=FONT_SIZE, fontweight='bold', color=FRESNO_TEXT,
                transform=ax.get_yaxis_transform())
        ax.text(record['probability_high'] + 0.015, position,
                f'{record["confidence"]:.0%}  ({prefix}, {pipe})',
                va='center', fontsize=SMALL_SIZE, color=FRESNO_TEXT)
    ax.plot([others['probability_low'].min(),
             others['probability_high'].max()], [positions[-1]] * 2,
            color=FRESNO_GREY, alpha=0.4, linewidth=4,
            solid_capstyle='round')
    ax.scatter(others['confidence'].median(), positions[-1], s=26,
               color=FRESNO_GREY, edgecolor='white', linewidth=0.5,
               zorder=3)
    ax.text(-0.015, positions[-1], f'{len(others)} other  ', ha='right',
            va='center', fontsize=FONT_SIZE, color=FRESNO_TEXT,
            transform=ax.get_yaxis_transform())
    ax.text(others['probability_high'].max() + 0.015, positions[-1],
            f'{others["confidence"].median():.0%}  (no steady band)',
            va='center', fontsize=SMALL_SIZE, color=FRESNO_TEXT)
    ax.axvline(HISTORICAL_LEAK_RATE, color=FRESNO_TEXT, linestyle='--',
               linewidth=0.6)
    ax.text(HISTORICAL_LEAK_RATE + 0.012, positions[0] + 0.8,
            'Typical site (1 in 8)', ha='left', va='center',
            fontsize=SMALL_SIZE, color=FRESNO_TEXT)
    ax.set_xlim(0, 0.8)
    ax.set_ylim(-0.6, len(rows) + 1.25)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_xticklabels([f'{t:.0%}' for t in ax.get_xticks()])
    ax.spines['bottom'].set_bounds(0, 0.5)
    ax.spines['left'].set_visible(False)
    ax.set_yticks([])
    ax.set_xlabel('Chance of a leak (dot) and its range (bar)', x=0.31)
    fig.subplots_adjust(left=0.17, right=0.97, top=0.97, bottom=0.18)
    return save_figure(fig, MEMO_FIGURE_DIR, 'memo_fig3_leak_chances')


# ---------------------------------------------------------------------------
# Figure 4: which candidates hold up under each check
# ---------------------------------------------------------------------------
LOO_COLUMN = 'Each dug\nsite left\nout (6)'
CLUSTER_COLUMN = 'Grouped\nwithout\noutcomes (4)'
CHECK_COLUMNS = ('Main\nanalysis', 'Without\nnoise\nadjustment',
                 LOO_COLUMN, CLUSTER_COLUMN, 'Similar-\nsite\ncheck',
                 'Other\nmodel above\n1 in 8')
N_CAST_IRON_SITES = 4


def robustness_fractions(grid, spreading, model2):
    """Share of runs flagging each band site, per check (NaN = n/a)."""
    loo = [column for column in grid.columns if column.startswith('Drop ')]
    rows = {}
    for clip_id in BAND_SITES_ORDER:
        rows[clip_id] = [
            grid.loc[clip_id, 'Main version'],
            grid.loc[clip_id, 'Raw version'],
            grid.loc[clip_id, loo].mean(skipna=False),
            grid.loc[clip_id, list(CLUSTER_COLUMNS.values())].mean(),
            float(spreading.loc[clip_id, 'spread_is_leak']),
            float(model2.loc[clip_id, 'confidence'] > HISTORICAL_LEAK_RATE),
        ]
    counts = {LOO_COLUMN: len(loo), CLUSTER_COLUMN: len(CLUSTER_COLUMNS)}
    return pd.DataFrame(rows, index=CHECK_COLUMNS).T, counts


def plot_robustness(fractions, counts):
    """Dot matrix: filled = flagged in every run; label = runs flagged."""
    fig, ax = plt.subplots(figsize=TALL_COLUMN_FIGURE)
    n_rows = len(fractions)
    for row, clip_id in enumerate(fractions.index):
        y = n_rows - 1 - row
        for column, check in enumerate(fractions.columns):
            share = fractions.loc[clip_id, check]
            if np.isnan(share):
                ax.text(column, y, 'n/a', ha='center', va='center',
                        fontsize=SMALL_SIZE - 0.5, color=FRESNO_GREY)
                continue
            face = (FRESNO_MAROON if share == 1 else 'white' if share == 0
                    else '#C98A97')
            ax.scatter(column, y, s=95, facecolor=face,
                       edgecolor=FRESNO_MAROON if share else FRESNO_GREY,
                       linewidth=0.6, zorder=3)
            if check in counts and 0 < share < 1:
                ax.text(column, y, f'{round(share * counts[check])}/'
                        f'{counts[check]}', ha='center', va='center',
                        fontsize=SMALL_SIZE - 1, color='white', zorder=4)
    ax.set_xticks(range(len(fractions.columns)))
    ax.set_xticklabels(fractions.columns, fontsize=SMALL_SIZE - 0.5,
                       linespacing=1.0)
    ax.xaxis.tick_top()
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(list(fractions.index[::-1]), fontsize=FONT_SIZE,
                       fontweight='bold')
    divider = n_rows - N_CAST_IRON_SITES - 0.5
    ax.axhline(divider, color=FRESNO_GREY, linewidth=0.5, linestyle=':')
    group_axis = blended_transform_factory(ax.transAxes, ax.transData)
    for label, middle in (('CAST IRON', (divider + n_rows - 1) / 2),
                          ('PVC', (divider - 0.5) / 2)):
        ax.text(-0.14, middle, label, rotation=90, ha='center',
                va='center', fontsize=SMALL_SIZE - 0.5, color=FRESNO_BLUE,
                transform=group_axis)
    ax.set_xlim(-0.6, len(fractions.columns) - 0.4)
    ax.set_ylim(-0.6, n_rows - 0.4)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    fig.subplots_adjust(left=0.17, right=0.98, top=0.79, bottom=0.03)
    return save_figure(fig, MEMO_FIGURE_DIR, 'memo_fig4_robustness')


def make_memo_figures():
    """Build the four memo figures; return their paths."""
    _style()
    metadata = load_metadata()
    usable = metadata[~metadata['clip_id'].isin(ANALYSIS_EXCLUDED_CLIPS)]
    amplitudes = load_all_amplitudes(usable)
    inputs = build_spectral_inputs(usable, amplitudes)
    steady = steady_spectra(usable, amplitudes, inputs['floor_clip_ids'])
    probabilities = pd.read_csv(MODEL1_PROBABILITY_PATH)
    scores = pd.read_csv(MODEL1_SCORES_PATH)
    _, _, folds = leave_one_out(usable, inputs,
                                usable[usable['pipe_material']
                                       == 'cast_iron'])
    grid = build_robustness_grid(scores, probabilities, folds,
                                 leak_clusters_by_setting(usable))
    fractions, counts = robustness_fractions(
        grid, pd.read_csv(SELF_TRAINING_TABLE_PATH).set_index('clip_id'),
        pd.read_csv(MODEL2_PROBABILITY_PATH).set_index('clip_id'))
    return [plot_recording_timeline(metadata),
            plot_leak_signature(usable, steady),
            plot_leak_chances(probabilities),
            plot_robustness(fractions, counts)]


if __name__ == '__main__':
    for path in make_memo_figures():
        print(path)
