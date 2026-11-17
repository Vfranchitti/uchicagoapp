"""Exploratory figure: every recording at once.

Fig 15 spectrograms of all 46 recordings, logger A above B,
       cast iron left of PVC

Figures are tagged in their docstrings as LAY (readable by a non-technical
audience), INTERPRETABLE (axes with a plain physical meaning) or ACCURATE
(information-dense, needs training to read).

Run from the repo root:  python -m src.plot_eda_overview_grid
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from src.config import (
    CLIP_DURATION_SECONDS,
    LEAK_LABEL,
    NO_LEAK_LABEL,
    NYQUIST_HZ,
)
from src.load_recordings import load_all_amplitudes, load_metadata
from src.plot_style import (
    HEADER_FONT_SIZE,
    LABEL_COLORS,
    LOGGER_COLORS,
    MICRO_FONT_SIZE,
    PIPE_COLORS,
    SMALL_FONT_SIZE,
    SPECTROGRAM_COLORMAP,
    SPECTROGRAM_DB_CEILING_PERCENTILE,
    SPECTROGRAM_DB_FLOOR_PERCENTILE,
    TEXT_DARK,
    TINY_FONT_SIZE,
    add_title_and_note,
    apply_urban_labs_style,
    finish_and_save,
    format_clock,
    spectrogram_db,
)

# All-recordings spectrogram grid (Fig 15).
FULL_PAGE_FIGURE = (7.5, 10.5)
ALL_SPECTROGRAM_CAST_COLUMNS = 4
ALL_SPECTROGRAM_PVC_COLUMNS = 2
BLOCK_GAP_RATIO = 0.18              # gap row/column, relative to a panel
BLOCK_DIVIDER_COLOR = '#BDBDBD'
BLOCK_HEADER_HEIGHT = 0.045         # figure fraction above the grid
BLOCK_HEADER_WIDTH = 0.045          # figure fraction left of the grid
ALL_SPECTROGRAM_TOP = 0.885
ALL_SPECTROGRAM_LEFT = 0.10
ALL_SPECTROGRAM_RIGHT = 0.90


def _block_cells(n_rows, n_columns, row_start, column_start):
    """Grid cells of one block, filled left to right, then top to bottom."""
    return [(row_start + row, column_start + column)
            for row in range(n_rows) for column in range(n_columns)]


def _grid_span(fig, grid, rows, columns):
    """Figure-fraction (left, right, bottom, top) covered by grid cells."""
    first = grid[rows[0], columns[0]].get_position(fig)
    last = grid[rows[-1], columns[-1]].get_position(fig)
    return first.x0, last.x1, last.y0, first.y1


def plot_all_spectrograms(metadata, amplitudes):
    """Fig 15 (ACCURATE). Spectrogram of every recording, by logger and pipe.

    Logger A blocks sit above logger B; cast iron sits left of PVC. Within
    each block, recordings run in clock order (earliest first).
    """
    is_cast = metadata['pipe_material'] == 'cast_iron'
    n_cast_rows = {logger: int(np.ceil(
        ((metadata['logger_model'] == logger) & is_cast).sum()
        / ALL_SPECTROGRAM_CAST_COLUMNS)) for logger in ('A', 'B')}
    n_pvc_rows = {logger: int(np.ceil(
        ((metadata['logger_model'] == logger) & ~is_cast).sum()
        / ALL_SPECTROGRAM_PVC_COLUMNS)) for logger in ('A', 'B')}
    block_rows = {logger: max(n_cast_rows[logger], n_pvc_rows[logger])
                  for logger in ('A', 'B')}
    n_columns = ALL_SPECTROGRAM_CAST_COLUMNS + 1 + ALL_SPECTROGRAM_PVC_COLUMNS
    n_rows = block_rows['A'] + 1 + block_rows['B']
    pvc_first_column = ALL_SPECTROGRAM_CAST_COLUMNS + 1
    b_first_row = block_rows['A'] + 1
    fig = plt.figure(figsize=FULL_PAGE_FIGURE)
    grid = fig.add_gridspec(
        n_rows, n_columns, hspace=0.75, wspace=0.12,
        width_ratios=([1] * ALL_SPECTROGRAM_CAST_COLUMNS + [BLOCK_GAP_RATIO]
                      + [1] * ALL_SPECTROGRAM_PVC_COLUMNS),
        height_ratios=([1] * block_rows['A'] + [BLOCK_GAP_RATIO]
                       + [1] * block_rows['B']))
    spectrograms = {clip_id: spectrogram_db(amplitudes[clip_id])
                    for clip_id in metadata['clip_id']}
    all_db = np.concatenate([db.ravel() for _, _, db in
                             spectrograms.values()])
    color_floor, color_ceiling = np.percentile(
        all_db, [SPECTROGRAM_DB_FLOOR_PERCENTILE,
                 SPECTROGRAM_DB_CEILING_PERCENTILE])
    blocks = {
        ('A', 'cast_iron'): _block_cells(block_rows['A'],
                                         ALL_SPECTROGRAM_CAST_COLUMNS, 0, 0),
        ('A', 'PVC'): _block_cells(block_rows['A'],
                                   ALL_SPECTROGRAM_PVC_COLUMNS, 0,
                                   pvc_first_column),
        ('B', 'cast_iron'): _block_cells(block_rows['B'],
                                         ALL_SPECTROGRAM_CAST_COLUMNS,
                                         b_first_row, 0),
        ('B', 'PVC'): _block_cells(block_rows['B'],
                                   ALL_SPECTROGRAM_PVC_COLUMNS, b_first_row,
                                   pvc_first_column),
    }
    for (logger, pipe), cells in blocks.items():
        in_block = metadata[(metadata['logger_model'] == logger)
                            & (metadata['pipe_material'] == pipe)
                            ].sort_values('overnight_hour')
        for (row, column), record in zip(cells, in_block.itertuples()):
            ax = fig.add_subplot(grid[row, column])
            times_seconds, frequencies_hz, power_db = spectrograms[
                record.clip_id]
            image = ax.pcolormesh(times_seconds, frequencies_hz, power_db,
                                  cmap=SPECTROGRAM_COLORMAP,
                                  vmin=color_floor, vmax=color_ceiling,
                                  shading='auto', rasterized=True)
            ax.grid(False)
            ax.set_ylim(0, NYQUIST_HZ)
            ax.set_xlim(0, CLIP_DURATION_SECONDS)
            ax.tick_params(labelsize=MICRO_FONT_SIZE, length=1.5, pad=1)
            ax.set_yticks([0, 1000, 2000])
            ax.set_yticklabels(['0', '1k', '2k'])
            ax.set_xticks([0, 2, 4])
            if column not in (0, pvc_first_column):
                ax.set_yticklabels([])
            if row != n_rows - 1:
                ax.set_xticklabels([])
            title_color = (TEXT_DARK if record.label_group == 'unlabeled'
                           else LABEL_COLORS[record.label_group])
            dug_tag = {LEAK_LABEL: '  LEAK', NO_LEAK_LABEL: '  NO LEAK'
                       }.get(record.label_group, '')
            ax.set_title(
                f'{record.clip_id}{dug_tag}\n'
                f'{logger} | {pipe.replace("_", " ")} | '
                f'{format_clock(record.recorded_at)}',
                loc='left', fontsize=MICRO_FONT_SIZE, color=title_color, pad=2,
                linespacing=1.1)
            for spine in ax.spines.values():
                spine.set_visible(record.label_group != 'unlabeled')
                spine.set_color(title_color)
                spine.set_linewidth(1.2)
    bottom = add_title_and_note(
        fig, 15, 'Spectrograms of all 46 recordings, by logger model and '
        'pipe material',
        'Each panel shows sound power by pitch (vertical, 0-2,000 Hz) and '
        'time (horizontal, 0-5 s) on one shared color scale; darker is '
        'louder. Logger A is on top, logger B below; cast iron is left, '
        'PVC right. Within each block, recordings run in clock order. Dug '
        'sites are outlined and labeled (maroon = leak, teal = no leak). '
        'All six dug sites are cast iron; no PVC site has been dug. Pitch '
        'axis in Hz (1k = 1,000); time axis in seconds.')
    fig.subplots_adjust(top=ALL_SPECTROGRAM_TOP, bottom=bottom,
                        left=ALL_SPECTROGRAM_LEFT, right=ALL_SPECTROGRAM_RIGHT)
    # Light lines through the gap row and column separate the four blocks.
    grid_left, grid_right, grid_bottom, grid_top = _grid_span(
        fig, grid, [0, n_rows - 1], [0, n_columns - 1])
    gap_left, gap_right, _, _ = _grid_span(
        fig, grid, [0], [ALL_SPECTROGRAM_CAST_COLUMNS])
    _, _, gap_bottom, gap_top = _grid_span(
        fig, grid, [block_rows['A']], [0])
    divider_x = (gap_left + gap_right) / 2
    divider_y = (gap_bottom + gap_top) / 2
    fig.add_artist(Line2D([divider_x, divider_x],
                          [grid_bottom, grid_top + BLOCK_HEADER_HEIGHT],
                          color=BLOCK_DIVIDER_COLOR, linewidth=0.9))
    fig.add_artist(Line2D([grid_left - BLOCK_HEADER_WIDTH, grid_right],
                          [divider_y, divider_y],
                          color=BLOCK_DIVIDER_COLOR, linewidth=0.9))
    # Column headers (pipe) and row headers (logger), with counts.
    counts = metadata.groupby(['logger_model', 'pipe_material']).size()
    pipe_spans = {'cast_iron': list(range(ALL_SPECTROGRAM_CAST_COLUMNS)),
                  'PVC': list(range(pvc_first_column, n_columns))}
    for pipe, columns in pipe_spans.items():
        span_left, span_right, _, _ = _grid_span(fig, grid, [0], columns)
        n_pipe = int(counts.xs(pipe, level='pipe_material').sum())
        fig.text((span_left + span_right) / 2,
                 grid_top + BLOCK_HEADER_HEIGHT * 0.55,
                 f'{pipe.replace("_", " ").upper()} PIPE ({n_pipe})',
                 ha='center', va='center', fontsize=HEADER_FONT_SIZE,
                 fontweight='bold',
                 color=PIPE_COLORS[pipe])
    logger_spans = {'A': list(range(block_rows['A'])),
                    'B': list(range(b_first_row, n_rows))}
    for logger, rows in logger_spans.items():
        _, _, span_bottom, span_top = _grid_span(fig, grid, rows, [0])
        n_logger = int(counts.xs(logger, level='logger_model').sum())
        fig.text(grid_left - BLOCK_HEADER_WIDTH * 0.7,
                 (span_bottom + span_top) / 2,
                 f'LOGGER {logger} ({n_logger})', rotation=90, ha='center',
                 va='center', fontsize=HEADER_FONT_SIZE, fontweight='bold',
                 color=LOGGER_COLORS[logger])
    colorbar_axis = fig.add_axes([ALL_SPECTROGRAM_RIGHT + 0.015,
                                  grid_bottom + (grid_top - grid_bottom) / 4,
                                  0.012, (grid_top - grid_bottom) / 2])
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label('Power (dB, darker = louder)', fontsize=SMALL_FONT_SIZE)
    colorbar.ax.tick_params(labelsize=TINY_FONT_SIZE)
    colorbar.outline.set_visible(False)
    return finish_and_save(fig, 'eda_15_all_spectrograms', bottom,
                           top=ALL_SPECTROGRAM_TOP)


def make_plot_eda_overview_grid_figures():
    """Load data once and draw every figure in this file."""
    apply_urban_labs_style()
    metadata = load_metadata()
    amplitudes = load_all_amplitudes(metadata)
    return [
        plot_all_spectrograms(metadata, amplitudes),
    ]


if __name__ == '__main__':
    for figure_path in make_plot_eda_overview_grid_figures():
        print(f'Saved {figure_path}')
