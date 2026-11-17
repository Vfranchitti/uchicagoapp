"""Exploratory figures: who was recorded, when, and with what.

Fig 1  coverage timeline (recording time x logger x pipe)
Fig 2  dug vs. undug counts by logger, pipe, and timing
Fig 16 descriptive overview (counts by logger, pipe, hour)

Figures are tagged in their docstrings as LAY (readable by a non-technical
audience), INTERPRETABLE (axes with a plain physical meaning) or ACCURATE
(information-dense, needs training to read).

Run from the repo root:  python -m src.plot_eda_coverage
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

from src.config import LEAK_LABEL, MORNING_COMMUTE_START_HOUR, NO_LEAK_LABEL
from src.load_recordings import load_metadata
from src.plot_style import (
    ANNOTATION_FONT_SIZE,
    FIRST_AXIS_HOUR,
    FRESNO_BAND_BLUE,
    HEALTH_LAB_LIGHT_TEAL,
    HEALTH_LAB_NAVY,
    HEALTH_LAB_TEAL,
    LABEL_COLORS,
    LABEL_DISPLAY_NAMES,
    LABEL_FONT_SIZE,
    LAST_AXIS_HOUR,
    LIGHT_GRID_GREY,
    LOGGER_COLORS,
    PANEL_GREY,
    PIPE_COLORS,
    PIPE_DISPLAY_NAMES,
    SMALL_FIGURE,
    SMALL_FONT_SIZE,
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

MIN_LABEL_SEPARATION_HOURS = 0.6    # closer dug-site labels alternate
MIN_SEGMENT_FOR_COUNT_LABEL = 3     # smaller bar segments go unlabeled
# Descriptive overview (Fig 16). Hours are clock hours, evening first.
OVERVIEW_FIGURE = (7.5, 8.0)
OVERNIGHT_HOUR_BINS = (22, 23, 0, 1, 2, 3, 4, 5, 6)
STAT_TILE_VALUE_SIZE = 22
HOUR_BAR_LABEL_GAP = 1.1
HOUR_BAR_HEADROOM = 5
MATRIX_COLOR_HEADROOM = 1.6         # keeps the darkest cell readable
MATRIX_COLORMAP = LinearSegmentedColormap.from_list(
    'pale_teal', ['#FFFFFF', HEALTH_LAB_LIGHT_TEAL])


def _annotate_row_without_overlap(ax, labeled_in_row, row_position):
    """Label dug sites above their dots, dropping below when crowded."""
    previous_hour = None
    for record in labeled_in_row.sort_values('overnight_hour').itertuples():
        is_crowded = (previous_hour is not None and record.overnight_hour
                      - previous_hour < MIN_LABEL_SEPARATION_HOURS)
        ax.annotate(record.clip_id, (record.overnight_hour, row_position),
                    xytext=(0, -15 if is_crowded else 9),
                    textcoords='offset points', ha='center',
                    fontsize=SMALL_FONT_SIZE,
                    color=LABEL_COLORS[record.label_group],
                    fontweight='bold')
        previous_hour = record.overnight_hour


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
    ax.grid(axis='x', color=LIGHT_GRID_GREY)
    ax.grid(axis='y', visible=False)
    format_overnight_axis(ax, FIRST_AXIS_HOUR, LAST_AXIS_HOUR)
    ax.set_xlabel('Clock time the recording was made')
    ax.legend(handles=label_legend_handles(), loc='upper center',
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
                            ha='center', va='center', fontsize=SMALL_FONT_SIZE,
                            color='white' if group != 'unlabeled'
                            else TEXT_DARK, fontweight='bold')
            left += values
        ax.set_yticks(range(len(counts)))
        ax.set_yticklabels([
            f'{level}\n{row[LEAK_LABEL]} leak, {row[NO_LEAK_LABEL]} no leak,'
            f'\n{row["unlabeled"]} not dug'
            for level, row in counts.iterrows()], fontsize=SMALL_FONT_SIZE)
        ax.set_title(factor_title, loc='left')
        ax.grid(axis='y', visible=False)
        ax.grid(axis='x', color=LIGHT_GRID_GREY)
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


def _hour_bin_labels():
    """Clock labels for the overnight hour bins (10 pm ... 6 am)."""
    return [f'{hour % 12 or 12} {"pm" if hour >= 12 else "am"}'
            for hour in OVERNIGHT_HOUR_BINS]


def _counts_by_hour(df, group_column, levels):
    """Recordings per clock hour (rows) and group level (columns)."""
    clock_hour = df['recorded_at'].str[:2].astype(int)
    return (pd.crosstab(clock_hour, df[group_column])
            .reindex(index=OVERNIGHT_HOUR_BINS, columns=levels,
                     fill_value=0))


def _draw_stat_tile(ax, value_text, label_text):
    """Big-number tile in the style of the CARE and HSP reports."""
    ax.set_facecolor(PANEL_GREY)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(0.5, 0.62, value_text, ha='center', va='center',
            fontsize=STAT_TILE_VALUE_SIZE, fontweight='bold',
            color=UCHICAGO_MAROON, transform=ax.transAxes)
    ax.text(0.5, 0.24, label_text, ha='center', va='center',
            fontsize=SMALL_FONT_SIZE,
            color=TEXT_DARK, transform=ax.transAxes, linespacing=1.2)


def _draw_logger_pipe_matrix(ax, metadata):
    """2 x 2 table of recordings (and dug sites) by logger and pipe."""
    loggers, pipes = ('A', 'B'), ('cast_iron', 'PVC')
    totals = pd.crosstab(metadata['logger_model'], metadata['pipe_material']
                         ).reindex(index=loggers, columns=pipes,
                                   fill_value=0)
    dug = metadata[metadata['is_labeled']]
    leaks = pd.crosstab(dug['logger_model'], dug['pipe_material'],
                        values=dug['is_leak'].astype(int), aggfunc='sum'
                        ).reindex(index=loggers, columns=pipes,
                                  fill_value=0).fillna(0)
    n_dug = pd.crosstab(dug['logger_model'], dug['pipe_material']
                        ).reindex(index=loggers, columns=pipes,
                                  fill_value=0)
    ax.imshow(totals.to_numpy(), cmap=MATRIX_COLORMAP, vmin=0,
              vmax=totals.to_numpy().max() * MATRIX_COLOR_HEADROOM)
    for row, logger in enumerate(loggers):
        for column, pipe in enumerate(pipes):
            ax.text(column, row - 0.1, str(totals.loc[logger, pipe]),
                    ha='center', va='center', fontsize=16,
                    fontweight='bold', color=HEALTH_LAB_NAVY)
            n_site_dug = int(n_dug.loc[logger, pipe])
            n_site_leaks = int(leaks.loc[logger, pipe])
            detail = (f'{n_site_dug} dug ({n_site_leaks} leak)'
                      if n_site_dug else 'none dug')
            ax.text(column, row + 0.25, detail, ha='center', va='center',
                    fontsize=SMALL_FONT_SIZE, color=TEXT_GREY if n_site_dug
                    else UCHICAGO_MAROON)
    ax.set_xticks(range(len(pipes)))
    ax.set_xticklabels([f'{PIPE_DISPLAY_NAMES[p]}\n'
                        f'({totals[p].sum()} sites)' for p in pipes])
    ax.set_yticks(range(len(loggers)))
    ax.set_yticklabels([f'Logger {logger}\n({totals.loc[logger].sum()})'
                        for logger in loggers])
    ax.tick_params(length=0)
    ax.xaxis.tick_top()
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title('Recordings by logger model and pipe material', loc='left',
                 pad=28)


def _draw_hour_bars(ax, metadata, group_column, levels, colors, names,
                    title):
    """Stacked bars of recordings per clock hour, commute hour shaded."""
    counts = _counts_by_hour(metadata, group_column, levels)
    positions = np.arange(len(OVERNIGHT_HOUR_BINS))
    commute_position = OVERNIGHT_HOUR_BINS.index(
        int(MORNING_COMMUTE_START_HOUR))
    ax.axvspan(commute_position - 0.5, commute_position + 0.5,
               color=FRESNO_BAND_BLUE, zorder=0, linewidth=0)
    bottoms = np.zeros(len(positions))
    for level in levels:
        values = counts[level].to_numpy()
        ax.bar(positions, values, bottom=bottoms, color=colors[level],
               edgecolor='white', linewidth=0.6, width=0.75,
               label=names[level])
        bottoms += values
    for position, total in zip(positions, bottoms):
        ax.text(position, total + 0.15, f'{total:.0f}', ha='center',
                va='bottom', fontsize=SMALL_FONT_SIZE, color=TEXT_DARK)
    dug = metadata[metadata['is_labeled']].copy()
    dug['clock_hour'] = dug['recorded_at'].str[:2].astype(int)
    for hour, in_hour in dug.groupby('clock_hour'):
        position = OVERNIGHT_HOUR_BINS.index(hour)
        marker_text = '\n'.join(
            f'{record.clip_id}' for record in in_hour.itertuples())
        marker_color = (LABEL_COLORS[in_hour['label_group'].iloc[0]]
                        if in_hour['label_group'].nunique() == 1
                        else TEXT_DARK)
        ax.text(position, bottoms[position] + HOUR_BAR_LABEL_GAP,
                marker_text, ha='center', va='bottom', fontsize=TINY_FONT_SIZE,
                color=marker_color, fontweight='bold', linespacing=1.0)
    ax.set_xticks(positions)
    ax.set_xticklabels(_hour_bin_labels(), fontsize=SMALL_FONT_SIZE)
    ax.set_ylim(0, bottoms.max() + HOUR_BAR_HEADROOM)
    ax.set_ylabel('Recordings')
    ax.set_title(title, loc='left')
    ax.legend(loc='upper left', ncol=len(levels), fontsize=SMALL_FONT_SIZE)
    ax.text(commute_position, ax.get_ylim()[1] * 0.97, 'COMMUTE',
            ha='center', va='top', fontsize=TINY_FONT_SIZE,
            color=HEALTH_LAB_TEAL,
            fontweight='bold')


def plot_descriptive_overview(metadata):
    """Fig 16 (LAY). What data do we have? Counts by logger, pipe, hour."""
    fig = plt.figure(figsize=OVERVIEW_FIGURE)
    grid = fig.add_gridspec(3, 4, height_ratios=[0.45, 1, 1], hspace=0.6,
                            wspace=0.35)
    dug = metadata[metadata['is_labeled']]
    n_commute = int(metadata['is_commute_hours'].sum())
    n_commute_dug = int(dug['is_commute_hours'].sum())
    tiles = [
        (f'{len(metadata)}', 'overnight recordings\n(one per hydrant)'),
        (f'{len(dug)}', f'dug so far:\n{int(dug["is_leak"].sum())} leak, '
                        f'{int((~dug["is_leak"].astype(bool)).sum())} '
                        'no leak'),
        (f'{(metadata["pipe_material"] == "PVC").sum()}',
         'PVC sites,\nnone dug'),
        (f'{n_commute}', f'recorded at/after 6 am\n({n_commute_dug} of '
                         'them dug)'),
    ]
    for column, (value_text, label_text) in enumerate(tiles):
        _draw_stat_tile(fig.add_subplot(grid[0, column]), value_text,
                        label_text)
    _draw_logger_pipe_matrix(fig.add_subplot(grid[1, :2]), metadata)
    status_ax = fig.add_subplot(grid[1, 2:])
    groups = [('A', 'cast_iron'), ('A', 'PVC'), ('B', 'cast_iron'),
              ('B', 'PVC')]
    status_order = (LEAK_LABEL, NO_LEAK_LABEL, 'unlabeled')
    for position, (logger, pipe) in enumerate(groups):
        in_group = metadata[(metadata['logger_model'] == logger)
                            & (metadata['pipe_material'] == pipe)]
        left = 0
        for status in status_order:
            n_status = int((in_group['label_group'] == status).sum())
            status_ax.barh(position, n_status, left=left,
                           color=LABEL_COLORS[status], edgecolor='white',
                           height=0.6)
            if n_status:
                status_ax.text(left + n_status / 2, position, str(n_status),
                               ha='center', va='center',
                               fontsize=SMALL_FONT_SIZE,
                               fontweight='bold',
                               color='white' if status != 'unlabeled'
                               else TEXT_DARK)
            left += n_status
    status_ax.set_yticks(range(len(groups)))
    status_ax.set_yticklabels(
        [f'{logger}, {pipe.replace("_", " ")}' for logger, pipe in groups],
        fontsize=LABEL_FONT_SIZE)
    status_ax.invert_yaxis()
    status_ax.set_xlabel('Recordings')
    status_ax.grid(axis='y', visible=False)
    status_ax.grid(axis='x', color=LIGHT_GRID_GREY)
    status_ax.set_title('Dig status by logger and pipe', loc='left')
    status_ax.legend(handles=[Patch(color=LABEL_COLORS[s],
                                    label=LABEL_DISPLAY_NAMES[s])
                              for s in status_order],
                     loc='lower right', fontsize=ANNOTATION_FONT_SIZE)
    _draw_hour_bars(fig.add_subplot(grid[2, :2]), metadata, 'logger_model',
                    ('A', 'B'), LOGGER_COLORS,
                    {'A': 'Logger A', 'B': 'Logger B'},
                    'Recordings per clock hour, by logger')
    _draw_hour_bars(fig.add_subplot(grid[2, 2:]), metadata, 'pipe_material',
                    ('cast_iron', 'PVC'), PIPE_COLORS,
                    {'cast_iron': 'Cast iron', 'PVC': 'PVC'},
                    'Recordings per clock hour, by pipe')
    bottom = add_title_and_note(
        fig, 16, 'What data are available? Recordings by logger, pipe, and '
        'hour (N=46)',
        'Hours are pooled by clock hour (e.g., "3 am" = 3:00-3:59). Clip '
        'IDs above the bars mark the six dug sites (maroon = leak, teal = '
        'no leak). Logger A outnumbers logger B almost 2 to 1. Only one '
        'dug site (L06) was recorded before 1 am, none of the 12 PVC sites '
        'has been dug, and both dug sites recorded in the 6 am hour '
        '(shaded, when the morning commute is likely under way) had no '
        'leak.')
    return finish_and_save(fig, 'eda_16_descriptive_overview', bottom,
                           top=0.92)


def make_plot_eda_coverage_figures():
    """Load data once and draw every figure in this file."""
    apply_urban_labs_style()
    metadata = load_metadata()
    return [
        plot_coverage_timeline(metadata),
        plot_coverage_bars(metadata),
        plot_descriptive_overview(metadata),
    ]


if __name__ == '__main__':
    for figure_path in make_plot_eda_coverage_figures():
        print(f'Saved {figure_path}')
