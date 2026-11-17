"""Exploratory figures: summary features and what besides leaks moves them.

Fig 8  key features, dug leak vs. no leak vs. undug
Fig 9  features by clock time (commute confound)
Fig 10 logger and pipe effects
Fig 11 correlations among candidate features
Fig 12 principal-component overview
Fig 13 steadiness vs. pitch (the field note as a map)

Figures are tagged in their docstrings as LAY (readable by a non-technical
audience), INTERPRETABLE (axes with a plain physical meaning) or ACCURATE
(information-dense, needs training to read).

Run from the repo root:  python -m src.plot_eda_features
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.compute_clip_features import (
    FEATURE_COLUMNS,
    FEATURE_LABELS,
    build_clip_feature_table,
)
from src.config import (
    HIGH_PITCH_CUTOFF_HZ,
    LEAK_LABEL,
    MORNING_COMMUTE_START_HOUR,
    NO_LEAK_LABEL,
)
from src.load_recordings import load_all_amplitudes, load_metadata
from src.plot_style import (
    ANNOTATION_FONT_SIZE,
    CRIME_LAB_GOLD,
    FIRST_AXIS_HOUR,
    HEALTH_LAB_NAVY,
    HEALTH_LAB_TEAL,
    LABEL_COLORS,
    LABEL_FONT_SIZE,
    LAST_AXIS_HOUR,
    LIGHT_GRID_GREY,
    LOGGER_MARKERS,
    MEDIUM_FIGURE,
    PIPE_COLORS,
    PIPE_MARKERS,
    POINTS_PER_INCH,
    SMALL_FIGURE,
    SMALL_FONT_SIZE,
    TALL_FIGURE,
    TEXT_DARK,
    TEXT_GREY,
    TINY_FONT_SIZE,
    UCHICAGO_MAROON,
    add_title_and_note,
    apply_urban_labs_style,
    finish_and_save,
    format_overnight_axis,
    label_legend_handles,
    shade_commute_window,
)

JITTER_SEED = 20261117
JITTER_WIDTH = 0.18
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
# Extra bottom margin (figure fraction) for rotated heatmap labels.
CORRELATION_TICK_LABEL_ALLOWANCE = 0.16
PCA_COMPONENTS = 2
# Clips singled out in the checks (outliers, unusual spectra, dropout).
NOTABLE_UNDUG_CLIPS = ('U16', 'U22', 'U32', 'U35', 'U40')
# Features that span orders of magnitude are log-scaled for display/PCA.
LOG_SCALED_FEATURES = ('rms_loudness', 'peak_amplitude', 'crest_factor',
                       'loudness_range_ratio')
# loudness_range_ratio is undefined for the clip with a dropout (U40).
PCA_FEATURES = [feature for feature in FEATURE_COLUMNS
                if feature != 'loudness_range_ratio']
CORRELATION_COLORMAP = LinearSegmentedColormap.from_list(
    'teal_white_maroon', [HEALTH_LAB_TEAL, '#FFFFFF', UCHICAGO_MAROON])


def _jitter(n_points, rng):
    """Uniform vertical jitter so overlapping dots stay visible."""
    return rng.uniform(-JITTER_WIDTH, JITTER_WIDTH, n_points)


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
                    fontsize=SMALL_FONT_SIZE, color=color, zorder=5)


def _annotate_clips(ax, df, x_column, y_column, clip_ids, color=TEXT_DARK):
    """Write clip IDs beside selected scatter points without overlaps."""
    selected = df[df['clip_id'].isin(clip_ids)]
    _place_labels(ax, [(record.clip_id, getattr(record, x_column),
                        getattr(record, y_column), color)
                       for record in selected.itertuples()],
                  SCATTER_LABEL_OFFSETS, 'left')


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
                                fontsize=ANNOTATION_FONT_SIZE,
                                color=LABEL_COLORS[group])
        ax.set_xlabel(_display_label(feature), fontsize=LABEL_FONT_SIZE)
        ax.grid(axis='y', visible=False)
        ax.grid(axis='x', color=LIGHT_GRID_GREY)
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
                     loc='left', fontsize=LABEL_FONT_SIZE, fontweight='normal',
                     color=TEXT_GREY)
        ax.set_ylabel(_display_label(feature), fontsize=LABEL_FONT_SIZE)
        format_overnight_axis(ax, FIRST_AXIS_HOUR, LAST_AXIS_HOUR)
        ax.set_xticks([-2, 0, 3, 6])
        ax.set_xticklabels(['10 pm', '12 am', '3 am', '6 am'])
    legend_handles = label_legend_handles() + [
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
                         fontsize=LABEL_FONT_SIZE, fontweight='normal',
                         color=TEXT_GREY)
            ax.set_yticks([1, 2])
            ax.set_yticklabels([
                f'{row_name} {level.replace("_", " ")} (n={len(g)})'
                for level, g in zip(levels,
                                    grouped)], fontsize=LABEL_FONT_SIZE)
            ax.set_xlabel(_display_label(feature), fontsize=LABEL_FONT_SIZE)
            ax.grid(axis='y', visible=False)
            ax.grid(axis='x', color=LIGHT_GRID_GREY)
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
    ax.set_xticklabels(short_labels, rotation=55, ha='right',
                       fontsize=SMALL_FONT_SIZE)
    ax.set_yticklabels(short_labels, fontsize=SMALL_FONT_SIZE)
    for row_index in range(len(FEATURE_COLUMNS)):
        for column_index in range(len(FEATURE_COLUMNS)):
            value = correlations.iloc[row_index, column_index]
            ax.text(column_index, row_index, f'{value:.1f}', ha='center',
                    va='center', fontsize=TINY_FONT_SIZE,
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
    ax.grid(axis='x', color=LIGHT_GRID_GREY)
    legend_handles = label_legend_handles() + [
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
    ax.grid(axis='x', color=LIGHT_GRID_GREY)
    legend_handles = label_legend_handles() + [
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


def make_plot_eda_features_figures():
    """Load data once and draw every figure in this file."""
    apply_urban_labs_style()
    metadata = load_metadata()
    amplitudes = load_all_amplitudes(metadata)
    clip_features = build_clip_feature_table(metadata, amplitudes)
    return [
        plot_feature_dot_plots(clip_features),
        plot_features_by_clock_time(clip_features),
        plot_device_and_pipe_effects(clip_features),
        plot_feature_correlations(clip_features),
        plot_principal_components(clip_features),
        plot_steadiness_vs_pitch(clip_features),
    ]


if __name__ == '__main__':
    for figure_path in make_plot_eda_features_figures():
        print(f'Saved {figure_path}')
