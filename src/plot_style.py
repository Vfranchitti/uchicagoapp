"""UChicago Urban Labs figure style, shared by every plotting script.

Colors and typography follow the Health Lab briefs in briefs/ (Housing
Specialist Pilot report, Safer Futures brief, CARE evaluation) and the
EPIC/E&E Lab Fresno research summary:
- UChicago maroon for the headline series and "Figure N." labels,
- Health Lab teal / navy / light teal for comparison series,
- a pale blue band to mark a time window (as in the Fresno "PILOT" band),
- Arial (the sans-serif used in Health Lab briefs), light gridlines,
  no top/right spines, a bold title above and a grey note below with the
  data source.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

from src.config import (
    DATA_SOURCE_NOTE,
    EDA_FIGURE_DIR,
    FIGURE_DPI,
    LEAK_LABEL,
    NO_LEAK_LABEL,
)

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
UCHICAGO_MAROON = '#800000'
HEALTH_LAB_TEAL = '#00758F'
HEALTH_LAB_NAVY = '#0E2A3B'
HEALTH_LAB_LIGHT_TEAL = '#4CB3C9'
CRIME_LAB_GOLD = '#EAAA00'
FRESNO_BAND_BLUE = '#D6E8F3'
PANEL_GREY = '#F4F4F2'
UNLABELED_GREY = '#B5B5B5'
TEXT_DARK = '#222222'
TEXT_GREY = '#595959'
GRID_GREY = '#E3E3E3'

LABEL_COLORS = {
    LEAK_LABEL: UCHICAGO_MAROON,
    NO_LEAK_LABEL: HEALTH_LAB_TEAL,
    'unlabeled': UNLABELED_GREY,
}
LABEL_DISPLAY_NAMES = {
    LEAK_LABEL: 'Dug: leak found',
    NO_LEAK_LABEL: 'Dug: no leak',
    'unlabeled': 'Not yet dug',
}
# Double encoding (shape as well as color) so groups survive grayscale
# printing and color-vision deficiency (Foster et al., ch. 6).
LOGGER_MARKERS = {'A': 'o', 'B': 's'}
PIPE_MARKERS = {'cast_iron': 'o', 'PVC': '^'}
LOGGER_COLORS = {'A': HEALTH_LAB_NAVY, 'B': CRIME_LAB_GOLD}
PIPE_COLORS = {'cast_iron': HEALTH_LAB_NAVY, 'PVC': HEALTH_LAB_LIGHT_TEAL}

# ---------------------------------------------------------------------------
# Typography and layout
# ---------------------------------------------------------------------------
FONT_FAMILY = 'Arial'
BASE_FONT_SIZE = 9
TITLE_FONT_SIZE = 11
NOTE_FONT_SIZE = 7.5
TITLE_X = 0.01
TITLE_Y = 0.985
TITLE_GAP = 0.012           # space between "Figure N." and the title
NOTE_Y = 0.012
NOTE_WRAP_CHARACTERS = 135
LAYOUT_TOP = 0.88
POINTS_PER_INCH = 72
LINE_SPACING = 1.25         # matplotlib's default line height multiple
# Height (inches) under the lowest axis for tick labels and an x-label.
AXIS_LABEL_ALLOWANCE_INCHES = 0.62


def apply_urban_labs_style():
    """Set matplotlib defaults to the Urban Labs brief look."""
    mpl.rcParams.update({
        'font.family': FONT_FAMILY,
        'font.size': BASE_FONT_SIZE,
        'axes.titlesize': BASE_FONT_SIZE,
        'axes.titleweight': 'bold',
        'axes.titlecolor': TEXT_DARK,
        'axes.labelsize': BASE_FONT_SIZE,
        'axes.labelcolor': TEXT_DARK,
        'axes.edgecolor': TEXT_GREY,
        'axes.linewidth': 0.6,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'axes.grid.axis': 'y',
        'grid.color': GRID_GREY,
        'grid.linewidth': 0.6,
        'axes.axisbelow': True,
        'xtick.color': TEXT_GREY,
        'ytick.color': TEXT_GREY,
        'xtick.labelsize': BASE_FONT_SIZE - 1,
        'ytick.labelsize': BASE_FONT_SIZE - 1,
        'legend.frameon': False,
        'legend.fontsize': BASE_FONT_SIZE - 1,
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
        'figure.dpi': 100,
    })


def _wrap(text, width):
    """Greedy word wrap for figure notes."""
    lines, current = [], ''
    for word in text.split():
        candidate = f'{current} {word}'.strip()
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    lines.append(current)
    return '\n'.join(lines)


def add_title_and_note(fig, figure_number, title, note):
    """Add 'Figure N. Title' above and a grey note + data source below.

    Returns the bottom margin (figure fraction) needed to clear the note.
    """
    figure_label = fig.text(
        TITLE_X, TITLE_Y, f'Figure {figure_number}.', ha='left', va='top',
        fontsize=TITLE_FONT_SIZE, fontweight='bold', color=UCHICAGO_MAROON)
    # Measure the rendered label so the title starts right after it.
    label_extent = figure_label.get_window_extent(
        renderer=fig.canvas.get_renderer())
    label_right = fig.transFigure.inverted().transform(
        (label_extent.x1, label_extent.y1))[0]
    fig.text(label_right + TITLE_GAP, TITLE_Y, title, ha='left', va='top',
             fontsize=TITLE_FONT_SIZE, fontweight='bold', color=TEXT_DARK)
    wrapped_note = _wrap(f'Note: {note} {DATA_SOURCE_NOTE}',
                         NOTE_WRAP_CHARACTERS)
    fig.text(TITLE_X, NOTE_Y, wrapped_note, ha='left', va='bottom',
             fontsize=NOTE_FONT_SIZE, color=TEXT_GREY, style='italic')
    n_note_lines = wrapped_note.count('\n') + 1
    note_height_inches = (n_note_lines * NOTE_FONT_SIZE * LINE_SPACING
                          / POINTS_PER_INCH)
    figure_height_inches = fig.get_figheight()
    return (NOTE_Y + (note_height_inches + AXIS_LABEL_ALLOWANCE_INCHES)
            / figure_height_inches)


def finish_and_save(fig, file_stem, bottom, top=LAYOUT_TOP, left=None):
    """Reserve title/note space, then save PNG to the EDA figure folder."""
    fig.subplots_adjust(top=top, bottom=bottom)
    if left is not None:
        fig.subplots_adjust(left=left)
    EDA_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = EDA_FIGURE_DIR / f'{file_stem}.png'
    fig.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(fig)
    return output_path


def shade_commute_window(ax, start_hour, end_hour, label_y=None):
    """Shade the approximate morning-commute window (Fresno-style band)."""
    ax.axvspan(start_hour, end_hour, color=FRESNO_BAND_BLUE, zorder=0,
               linewidth=0)
    if label_y is not None:
        ax.text(start_hour + 0.05, label_y, 'MORNING\nCOMMUTE',
                fontsize=BASE_FONT_SIZE - 2, color=HEALTH_LAB_TEAL,
                va='bottom', ha='left', fontweight='bold')


def format_overnight_axis(ax, first_hour, last_hour):
    """Label an overnight-hour axis with clock times (10 pm ... 7 am)."""
    tick_hours = range(int(first_hour), int(last_hour) + 1)
    ax.set_xticks(list(tick_hours))
    ax.set_xticklabels([_clock_label(hour) for hour in tick_hours])
    ax.set_xlim(first_hour, last_hour)


def _clock_label(overnight_hour):
    """Turn -2 into '10 pm', 0 into '12 am', 6 into '6 am'."""
    hour_of_day = overnight_hour % 24
    suffix = 'am' if hour_of_day < 12 else 'pm'
    display_hour = hour_of_day % 12 or 12
    return f'{display_hour} {suffix}'
