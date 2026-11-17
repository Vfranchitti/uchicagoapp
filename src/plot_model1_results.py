"""Memo figures for Model 1 (knowledge-based score).

Fig 1 (LAY)            Dig list with leak-probability ranges.
Fig 2 (INTERPRETABLE)  How the score works: band contrast and persistence.
Fig 3 (INTERPRETABLE)  Decision map: contrast vs. persistence, all sites.
Fig 4 (ACCURATE)       Robustness grid: which sites stay flagged across
                       versions, leave-one-out folds, and clustering.

Reads Model 1 outputs (run src/run_model_knowledge_score.py and
src/compute_model1_probabilities.py first).

Outputs: outputs/figures/model/model1_fig{1..4}_*.png
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from src.compute_model1_probabilities import has_persistent_band
from src.config import (
    ANALYSIS_EXCLUDED_CLIPS,
    ANALYSIS_EXCLUSION_REASON,
    FIGURE_DPI,
    HISTORICAL_LEAK_RATE,
    INFORMATIVE_DIG_SITE,
    LEAK_LABEL,
    MODEL1_PROBABILITY_PATH,
    MODEL1_SCORES_PATH,
    MODEL_FIGURE_DIR,
    NO_LEAK_LABEL,
    NYQUIST_HZ,
    PERSISTENCE_CONTRAST_DB,
    PERSISTENCE_CUTOFF,
    PERSISTENT_BAND_MIN_CONTRAST_DB,
    SAMPLE_RATE_HZ,
    SPECTROGRAM_OVERLAP_SAMPLES,
    SPECTROGRAM_SEGMENT_SAMPLES,
)
from src.load_recordings import load_all_amplitudes, load_metadata
from src.plot_style import (
    FRESNO_BAND_BLUE,
    HEALTH_LAB_LIGHT_TEAL,
    HEALTH_LAB_NAVY,
    HEALTH_LAB_TEAL,
    LABEL_COLORS,
    PANEL_GREY,
    TEXT_DARK,
    TEXT_GREY,
    UCHICAGO_MAROON,
    UNLABELED_GREY,
    add_title_and_note,
    apply_urban_labs_style,
)
from src.run_model_knowledge_score import (
    band_contrast,
    build_spectral_inputs,
    cast_iron_bands,
    derive_leak_band,
    leak_cluster_members,
    leave_one_out,
)

PIPE_COLORS = {'cast_iron': HEALTH_LAB_NAVY, 'pvc': HEALTH_LAB_LIGHT_TEAL}
PIPE_MARKERS = {'cast_iron': 'o', 'pvc': '^'}
PIPE_NAMES = {'cast_iron': 'cast iron', 'pvc': 'PVC'}
DUG_LABELS = (LEAK_LABEL, NO_LEAK_LABEL)
EXPLAINER_CLIPS = (('L03', LABEL_COLORS[LEAK_LABEL], 'L03 (dug: leak)'),
                   ('U17', HEALTH_LAB_NAVY, 'U17 (not yet dug)'),
                   ('L02', LABEL_COLORS[NO_LEAK_LABEL], 'L02 (dug: no leak)'))
PERSISTENCE_EXAMPLES = (('L06', LABEL_COLORS[LEAK_LABEL],
                         'L06: leak with traffic on top'),
                        ('U35', UNLABELED_GREY,
                         'U35: one loud impact'))
CLUSTER_COLUMNS = {'kmeans_k3': 'k-means, 3 groups',
                   'kmeans_k4': 'k-means, 4 groups',
                   'ward_k3': 'Ward, 3 groups', 'ward_k4': 'Ward, 4 groups'}
EXTRA_GRID_ROWS = 4
# Hand-placed label offsets (points) where default labels collide.
DECISION_MAP_LABEL_OFFSETS = {'U34': (4, -12), 'L01': (-6, 6),
                              'L05': (6, -12), 'L02': (5, 5),
                              'U15': (5, -9)}
SMOOTHING_BINS = 5
SECONDS_PER_FRAME = (SPECTROGRAM_SEGMENT_SAMPLES
                     - SPECTROGRAM_OVERLAP_SAMPLES) / SAMPLE_RATE_HZ


def _save(fig, file_stem):
    MODEL_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = MODEL_FIGURE_DIR / f'{file_stem}.png'
    fig.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(fig)
    return output_path


def _smooth(values):
    kernel = np.ones(SMOOTHING_BINS) / SMOOTHING_BINS
    return np.convolve(values, kernel, mode='same')


def _reason(record, is_dig):
    """One-line reason for a dig-list site."""
    if record.ranking == 'pvc':
        lead = 'Informative dig' if is_dig else 'Second PVC band'
        return (f'{lead}: steady 250-400 Hz band (+{record.score:.1f} dB,\n'
                f'present {record.persistence:.0%}); PVC signature '
                f'unconfirmed')
    return (f'Steady 400-1,300 Hz leak band (+{record.score:.1f} dB,\n'
            f'present {record.persistence:.0%}), like the dug leaks')


# ---------------------------------------------------------------------------
# Figure 1
# ---------------------------------------------------------------------------
def plot_dig_list(probabilities):
    """Fig 1 (LAY). Recommended digs with probability ranges."""
    dig = probabilities.dropna(subset=['dig_priority']).sort_values(
        'dig_priority')
    next_up = probabilities[(probabilities['prediction'] == 1)
                            & probabilities['dig_priority'].isna()]
    rows = pd.concat([dig, next_up])
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    positions = np.arange(len(rows))[::-1]
    for position, record in zip(positions, rows.itertuples()):
        is_dig = not np.isnan(record.dig_priority)
        color = PIPE_COLORS[record.ranking] if is_dig else UNLABELED_GREY
        ax.plot([record.probability_low, record.probability_high],
                [position, position], color=color, linewidth=5,
                solid_capstyle='round', alpha=0.45)
        ax.plot(record.confidence, position,
                marker=PIPE_MARKERS[record.ranking], color=color,
                markersize=9, markeredgecolor='white')
        prefix = f'{int(record.dig_priority)}. ' if is_dig else 'Next: '
        ax.text(-0.02, position, f'{prefix}{record.clip_id}',
                ha='right', va='center', fontsize=9, fontweight='bold',
                color=TEXT_DARK if is_dig else TEXT_GREY,
                transform=ax.get_yaxis_transform())
        ax.text(0.62, position + 0.22,
                f'{PIPE_NAMES[record.ranking]}, logger {record.logger_model}'
                f'  |  {record.probability_low:.0%}-'
                f'{record.probability_high:.0%}',
                fontsize=7, color=TEXT_DARK, fontweight='bold', va='center')
        ax.text(0.62, position - 0.17, _reason(record, is_dig),
                fontsize=6.2, color=TEXT_GREY, va='center',
                linespacing=1.05)
    ax.axvline(HISTORICAL_LEAK_RATE, color=UCHICAGO_MAROON, linestyle='--',
               linewidth=0.9)
    ax.text(HISTORICAL_LEAK_RATE, len(rows) - 0.35,
            'typical site\n(1 in 8)', ha='center', va='bottom', fontsize=6.5,
            color=UCHICAGO_MAROON)
    ax.set_xlim(0, 1.25)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    ax.set_xticklabels([f'{tick:.0%}' for tick in ax.get_xticks()])
    ax.set_yticks([])
    ax.set_ylim(-0.7, len(rows) - 0.1)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_bounds(0, 0.6)
    ax.grid(axis='y', visible=False)
    ax.set_xlabel('Estimated chance of a leak', x=0.24)
    bottom = add_title_and_note(
        fig, 1, 'Where to dig this week, and how sure we are',
        'Presence of a persistent band sets the range (36-50% with one, '
        '2-3% without); the marker sits within it by band quality '
        '(steadiness over time and across pitch, weighted 0.6, and '
        'intenseness, contrast in dB, weighted 0.4, each relative to the dug '
        'leaks). PVC sites are discounted toward the 1-in-8 base rate by the '
        'share of five leak-signature properties their band shares (U34: '
        '3 of 5). Bars span analysis choices; a wide bar means the band '
        'falls below a cut-off in the raw-spectrum version. '
        f'{ANALYSIS_EXCLUSION_REASON}')
    fig.subplots_adjust(top=0.88, bottom=bottom, left=0.1, right=0.98)
    return _save(fig, 'model1_fig1_dig_list')


# ---------------------------------------------------------------------------
# Figure 2
# ---------------------------------------------------------------------------
def plot_how_score_works(inputs, leak_band_hz):
    """Fig 2 (INTERPRETABLE). Contrast (left) and persistence (right)."""
    band_hz, neighbors_hz = cast_iron_bands(leak_band_hz)
    fig, (spectrum_ax, time_ax) = plt.subplots(
        1, 2, figsize=(7.5, 3.8), gridspec_kw={'width_ratios': [1.25, 1]})
    spectrum_ax.axvspan(*band_hz, color=UCHICAGO_MAROON, alpha=0.08,
                        linewidth=0)
    for low_hz, high_hz in neighbors_hz:
        spectrum_ax.axvspan(low_hz, high_hz, color=PANEL_GREY, linewidth=0)
    frequencies_hz = inputs['clip_frequencies_hz']
    for clip_id, color, label in EXPLAINER_CLIPS:
        spectrum_ax.plot(frequencies_hz,
                         _smooth(inputs['clip_db']['floor']
                                 .loc[clip_id].to_numpy()),
                         color=color, linewidth=1.1, label=label)
    spectrum_ax.text(np.mean(band_hz), 27, 'leak band', ha='center',
                     fontsize=7, color=UCHICAGO_MAROON, fontweight='bold')
    for low_hz, high_hz in neighbors_hz:
        spectrum_ax.text((low_hz + high_hz) / 2, 27, 'neighbor', ha='center',
                         fontsize=6.5, color=TEXT_GREY)
    spectrum_ax.set_xlim(0, NYQUIST_HZ)
    spectrum_ax.set_ylim(-3, 30)
    spectrum_ax.set_xlabel('Pitch (Hz)')
    spectrum_ax.set_ylabel('dB above logger\'s noise floor')
    spectrum_ax.set_title('A. Contrast: does the band stand out?',
                          loc='left', fontsize=8)
    spectrum_ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.24),
                       ncol=3, fontsize=6.5)
    frame_frequencies_hz = inputs['frame_frequencies_hz']
    for clip_id, color, label in PERSISTENCE_EXAMPLES:
        contrast = band_contrast(inputs['frame_db']['floor'][clip_id],
                                 frame_frequencies_hz, band_hz, neighbors_hz)
        share = np.mean(contrast >= PERSISTENCE_CONTRAST_DB)
        times = (np.arange(len(contrast)) + 0.5) * SECONDS_PER_FRAME
        time_ax.plot(times, contrast, color=color, linewidth=1,
                     label=f'{label} (present {share:.0%})')
    time_ax.axhline(PERSISTENCE_CONTRAST_DB, color=TEXT_GREY,
                    linestyle='--', linewidth=0.8)
    time_ax.text(3.6, PERSISTENCE_CONTRAST_DB + 0.6, '"present" line (3 dB)',
                 fontsize=6.5, color=TEXT_GREY)
    time_ax.set_xlabel('Seconds into recording')
    time_ax.set_ylabel('Band contrast (dB)')
    time_ax.set_title('B. Persistence: does it ever stop?', loc='left',
                      fontsize=8)
    time_ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.24),
                   ncol=1, fontsize=6.5)
    fig.subplots_adjust(wspace=0.32)
    bottom = add_title_and_note(
        fig, 2, 'How the score works: a band that stands out and never stops',
        'A: the dug leak (L03) and top undug site (U17) carry a block of '
        f'sound at {band_hz[0]:.0f}-{band_hz[1]:.0f} Hz that rises above the '
        'neighboring pitches (grey); the dug non-leak (L02) does not. '
        'Contrast = band minus the average of its neighbors. B: band '
        'contrast in each 64 ms moment. L06\'s band stays above the line '
        'despite traffic; U35\'s apparent band is one impact. Sites whose '
        'band is present in fewer than 90% of moments are scored down in '
        'proportion.')
    fig.subplots_adjust(top=0.86, bottom=bottom + 0.1, left=0.08,
                        right=0.98)
    return _save(fig, 'model1_fig2_how_score_works')


# ---------------------------------------------------------------------------
# Figure 3
# ---------------------------------------------------------------------------
def plot_decision_map(scores, probabilities):
    """Fig 3 (INTERPRETABLE). Contrast vs. persistence for every site."""
    main = scores[scores['version'] == 'floor']
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.add_patch(Rectangle((PERSISTENT_BAND_MIN_CONTRAST_DB,
                            PERSISTENCE_CUTOFF), 30, 0.2,
                           color=FRESNO_BAND_BLUE, zorder=0, linewidth=0))
    ax.text(23.5, PERSISTENCE_CUTOFF - 0.035, 'persistent band\n(leak-like)',
            ha='right', va='top', fontsize=7, color=HEALTH_LAB_TEAL,
            fontweight='bold')
    for ranking, marker in PIPE_MARKERS.items():
        for group in ('unlabeled', NO_LEAK_LABEL, LEAK_LABEL):
            subset = main[(main['ranking'] == ranking)
                          & (main['label_group'] == group)]
            color = (PIPE_COLORS[ranking] if group == 'unlabeled'
                     else LABEL_COLORS[group])
            ax.scatter(subset['contrast_db'], subset['persistence'],
                       marker=marker, s=34 if group == 'unlabeled' else 70,
                       color=color, edgecolor='white', linewidth=0.5,
                       zorder=3)
    flagged = set(probabilities.loc[probabilities['prediction'] == 1,
                                    'clip_id'])
    to_label = main[main['label_group'].isin(DUG_LABELS)
                    | main['clip_id'].isin(flagged | {'U35', 'U16', 'U22',
                                                      'U32'})]
    for record in to_label.itertuples():
        is_dug = record.label_group in DUG_LABELS
        default_offset = (5, -3 if record.persistence > 0.9 else 3)
        ax.annotate(record.clip_id, (record.contrast_db, record.persistence),
                    xytext=DECISION_MAP_LABEL_OFFSETS.get(record.clip_id,
                                                          default_offset),
                    textcoords='offset points', fontsize=6.5,
                    fontweight='bold' if is_dug else 'normal',
                    color=(LABEL_COLORS[record.label_group] if is_dug
                           else TEXT_DARK))
    ax.axhline(PERSISTENCE_CUTOFF, color=TEXT_GREY, linestyle='--',
               linewidth=0.7)
    ax.axvline(PERSISTENT_BAND_MIN_CONTRAST_DB, color=TEXT_GREY,
               linestyle='--', linewidth=0.7)
    ax.set_xlim(-9, 24)
    ax.set_ylim(-0.03, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
    ax.set_xlabel('Contrast: how many dB the band stands out above its '
                  'neighbors')
    ax.set_ylabel('Persistence: share of moments\nthe band is present')
    ax.grid(axis='x', color='#EFEFEF')
    ax.legend(handles=[
        Line2D([], [], marker='o', linestyle='', color=HEALTH_LAB_NAVY,
               label='Cast iron, not dug (400-1,300 Hz band)'),
        Line2D([], [], marker='^', linestyle='', color=HEALTH_LAB_LIGHT_TEAL,
               label='PVC, not dug (250-400 Hz band)'),
        Line2D([], [], marker='o', linestyle='', color=LABEL_COLORS[
            LEAK_LABEL], label='Dug: leak'),
        Line2D([], [], marker='o', linestyle='', color=LABEL_COLORS[
            NO_LEAK_LABEL], label='Dug: no leak')],
        loc='center left', fontsize=6.5)
    bottom = add_title_and_note(
        fig, 3, 'The decision map: four cast-iron sites sit with the known '
        'leaks',
        'Each point is one recording (noise floor removed). Cast-iron sites '
        'are measured on the 400-1,300 Hz leak band, PVC sites on the '
        '250-400 Hz band. The shaded corner (band present in at least 90% of '
        'moments and standing out by at least 3 dB) holds all three dug '
        'leaks, no dug non-leak, four undug cast-iron sites and two PVC '
        f'sites. {ANALYSIS_EXCLUSION_REASON}')
    fig.subplots_adjust(top=0.9, bottom=bottom, left=0.12, right=0.98)
    return _save(fig, 'model1_fig3_decision_map')


# ---------------------------------------------------------------------------
# Figure 4
# ---------------------------------------------------------------------------
def build_robustness_grid(scores, probabilities, fold_tables, members):
    """Rows = candidate sites; columns = checks; values 1/0/NaN (n/a)."""
    dig = probabilities.dropna(subset=['dig_priority']).sort_values(
        'dig_priority')['clip_id'].tolist()
    predicted = probabilities.loc[probabilities['prediction'] == 1,
                                  'clip_id'].tolist()
    main_cast_iron = scores[(scores['version'] == 'floor')
                            & (scores['ranking'] == 'cast_iron')
                            & (scores['label_group'] == 'unlabeled')]
    already = set(dig) | set(predicted)
    extras = main_cast_iron[~main_cast_iron['clip_id'].isin(already)
                            ].nlargest(EXTRA_GRID_ROWS, 'score')['clip_id']
    rows = dig + [c for c in predicted if c not in dig] + list(extras)
    columns = {}
    for version, title in (('floor', 'Main version'),
                           ('raw', 'Raw version')):
        subset = scores[scores['version'] == version].set_index('clip_id')
        columns[title] = has_persistent_band(subset)
    for held_out, table in fold_tables.items():
        columns[f'Drop {held_out}'] = has_persistent_band(
            table.set_index('clip_id'))
    grid = pd.DataFrame(index=rows, columns=list(columns), dtype=float)
    for title, flags in columns.items():
        grid[title] = [float(flags[c]) if c in flags.index else np.nan
                       for c in rows]
    for column, title in CLUSTER_COLUMNS.items():
        grid[title] = [float(c in members[column]) for c in rows]
    return grid


def plot_robustness_grid(grid, pipe_by_clip):
    """Fig 4 (ACCURATE). Filled = flagged leak-like under that check."""
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    n_rows, n_columns = grid.shape
    for row, clip_id in enumerate(grid.index):
        for column, title in enumerate(grid.columns):
            value = grid.loc[clip_id, title]
            face = (PANEL_GREY if np.isnan(value)
                    else UCHICAGO_MAROON if value else 'white')
            ax.add_patch(Rectangle((column, n_rows - 1 - row), 0.92, 0.85,
                                   facecolor=face, edgecolor=UNLABELED_GREY,
                                   linewidth=0.5))
            if np.isnan(value):
                ax.text(column + 0.46, n_rows - 1 - row + 0.42, 'n/a',
                        ha='center', va='center', fontsize=5.5,
                        color=TEXT_GREY)
    ax.set_xlim(0, n_columns)
    ax.set_ylim(0, n_rows)
    ax.set_xticks(np.arange(n_columns) + 0.46)
    ax.set_xticklabels(grid.columns, rotation=40, ha='right', fontsize=6.5)
    ax.set_yticks(np.arange(n_rows) + 0.42)
    ax.set_yticklabels([f'{c} ({PIPE_NAMES[pipe_by_clip[c]]})'
                        for c in grid.index[::-1]], fontsize=7)
    for boundary in (2, 8):
        ax.axvline(boundary - 0.04, color=TEXT_DARK, linewidth=0.8)
    for position, text in ((1, 'Model 1 version'),
                           (5, 'Leave one dug site out'),
                           (10, 'Clustering (no labels)')):
        ax.text(position, n_rows + 0.25, text, ha='center', fontsize=7,
                fontweight='bold', color=TEXT_DARK)
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(handles=[Patch(color=UCHICAGO_MAROON, label='Flagged as '
                             'leak-like'),
                       Patch(facecolor='white', edgecolor=UNLABELED_GREY,
                             label='Not flagged'),
                       Patch(color=PANEL_GREY, label='Not applicable')],
              loc='upper center', bbox_to_anchor=(0.5, -0.32), ncol=3,
              fontsize=6.5)
    bottom = add_title_and_note(
        fig, 4, 'Which picks survive every check?',
        '"Flagged" means a persistent band (Model 1 columns) or membership '
        'in the cluster holding the dug leaks (clustering columns). Leave-'
        'one-out rebuilds the cast-iron band without each dug site; it does '
        'not apply to PVC, whose band is fixed in advance. U17 is flagged '
        'under every check; U07 under all but two clusterings. U08 and U36 '
        'fall just under the 3 dB line in the raw version, and U15\'s band '
        'drops below 90% presence there.')
    fig.subplots_adjust(top=0.85, bottom=bottom + 0.12, left=0.16,
                        right=0.98)
    return _save(fig, 'model1_fig4_robustness_grid')


def make_model1_figures():
    """Build all four Model 1 memo figures."""
    apply_urban_labs_style()
    scores = pd.read_csv(MODEL1_SCORES_PATH)
    probabilities = pd.read_csv(MODEL1_PROBABILITY_PATH)
    metadata = load_metadata()
    metadata = metadata[~metadata['clip_id'].isin(ANALYSIS_EXCLUDED_CLIPS)]
    amplitudes = load_all_amplitudes(metadata)
    inputs = build_spectral_inputs(metadata, amplitudes)
    leak_ids = metadata.loc[metadata['label_group'] == LEAK_LABEL,
                            'clip_id'].tolist()
    leak_band_hz = derive_leak_band(inputs['welch_excess'], leak_ids)
    cast_iron_sites = metadata[metadata['pipe_material'] == 'cast_iron']
    _, _, fold_tables = leave_one_out(metadata, inputs, cast_iron_sites)
    grid = build_robustness_grid(scores, probabilities, fold_tables,
                                 leak_cluster_members(metadata))
    pipe_by_clip = scores.drop_duplicates('clip_id').set_index(
        'clip_id')['ranking']
    assert INFORMATIVE_DIG_SITE in grid.index
    return [plot_dig_list(probabilities),
            plot_how_score_works(inputs, leak_band_hz),
            plot_decision_map(scores, probabilities),
            plot_robustness_grid(grid, pipe_by_clip)]


if __name__ == '__main__':
    for path in make_model1_figures():
        print(f'Saved {path}')
