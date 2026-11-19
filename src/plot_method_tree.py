"""Figure A1: the methodological decision tree.

One row per decision, in pipeline order. The left column is the route
taken; the right column lists the options considered and not taken, each
with its main reason. The content is a record of decisions made during the
analysis (see notes/modeling/model_assumptions_checklist.md), so this
script reads no data.

Output
------
outputs/figures/2_model/method_decision_tree.png
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from src.config import METHOD_TREE_FIGURE_PATH
from src.plot_style import (
    ANNOTATION_FONT_SIZE,
    HEADER_FONT_SIZE,
    PANEL_GREY,
    PANEL_TITLE_FONT_SIZE,
    TEXT_DARK,
    TEXT_GREY,
    TINY_FONT_SIZE,
    UCHICAGO_MAROON,
    UNLABELED_GREY,
    add_title_and_note,
    apply_urban_labs_style,
    save_figure,
)

FIGURE_NUMBER = 'A1'
TITLE = 'How we reached the dig list: each decision and the options weighed'
NOTE = ('Each row is one decision, in pipeline order. Left: the route taken. '
        'Right: options considered and not taken, with the main reason. '
        'LR = likelihood ratio; LOO = leave-one-out; A5 and A8 are '
        'assumptions in the modeling checklist.')

# (decision, choice, detail, alternatives). An empty alternatives tuple
# means the step was fixed by an assumption, not chosen among options.
DECISIONS = (
    ('Model family', 'Rule-based score (B1)',
     'Written from the crew chief\'s description; the six labels only '
     'check it',
     ('Penalized logit as the main model (B2): 3 events per class',
      'Template, naive Bayes, LDA, k-NN: 3 cases per class to estimate',
      'Forests, boosting, SVM, neural nets: would memorize six labels')),
    ('Damaged recording (U40)', 'Exclude; make no prediction',
     '15% of samples clipped and a 1.2 s dropout',
     ('Keep in clustering with a quality flag',
      'Predict no leak at the 12.5% base rate')),
    ('Recording-time confound', 'Never use time as an input',
     'Leaks before 3:30 am, non-leaks after 4:15 am; timing assumed random',
     ()),
    ('15 quiet clips', 'Use as each logger\'s noise baseline',
     'Baseline subtracted per logger before scoring',
     ('Label them as non-leaks: about 15 unverified labels',)),
    ('Device correction', 'One offset per logger',
     'Logger B = A + 5.8 dB with the same shape (r = 0.99); U34 left out',
     ('Pitch-by-pitch correction: not needed, shapes match',
      'Raw spectra only: kept as a robustness check',
      'Keep U34 in the baseline (removing it moved the floor < 0.3 dB)')),
    ('Score form', 'Contrast × persistence gate',
     '400–1,300 Hz vs. the 250–400 and 1,300–2,000 Hz neighbors, in dB',
     ('Contrast × steadiness as a continuous product',
      'Contrast only: a single loud impact (U35) scores high')),
    ('Traffic and presence test', 'Steady level (L90) band test',
     'Level held 90% of the time; bursts drop out, no frames discarded',
     ('Drop burst frames (> 6 dB): busy 6 am clips lose most data',
      'Persistence in 90% of 64 ms frames: traffic masks bands',
      'Wobble (dB spread) gate: a 1.3 dB cutoff rejects leak L06')),
    ('PVC sites (none dug)', 'Search all pitches for steady bands',
     'Distinct from the PVC shared background; found near 320–450 Hz',
     ('Fixed 250–400 Hz slice: set by looking at U34 and U15',
      'Cast-iron rule with a low-confidence flag: plastic damps highs',
      'Exclude PVC from the ranking')),
    ('Probability mapping', 'Two-tier Bayes (option A)',
     'LR from 3 of 3 vs. 0 of 3, Laplace or Jeffreys: 36–50% vs. 2–3%',
     ('Logit as the probability layer (option B): curve set by penalty',
      'Bayesian logit, weak prior (C3): the prior drives the slope')),
    ('PVC likelihood ratio', 'PVC leaks always hiss (A8)',
     'Cast-iron LR applies; PVC without a band about 3%, as cast iron',
     ('LR bounds 1 to 4–7 with a credence weight of 0.5–0.75',
      'Prevalence matching (about 60%): fragile',
      'PVC without a band about 8%: if PVC leaks are harder to hear')),
    ('Confidence within a tier', 'Graded on band quality',
     'steadiness^0.6 × intenseness^0.4; intenseness = band minus noise',
     ('Contrast in dB as intenseness: falls with traffic, not the leak',
      'Tier midpoint or average of settings: no order among band sites',
      'Grade across the full range (logit-like): intenseness dominates')),
    ('PVC discount', 'Signature match w (share of 5 traits)',
     'p = w·p + (1 − w)·12.5%: U34 w = 0.6 gives 35%; U15, U03 w = 0.4',
     ('Average over LR 1 to 4–7 (about 26–35%)',
      'Fit w to a PVC base rate (1.5 expected leaks)',
      'Fixed credence of 0.75')),
    ('Dig list', 'Four cast-iron sites + U34',
     'U34 framed as an informative dig that tests the PVC assumption',
     ('Slot 5: weak cast-iron tie (U35 or U12), about 2%',
      'Slot 4: swap U36 for U15',
      'Fill confidence from the logit or a model average')),
    ('Robustness checks', 'LOO (all six), raw spectra, clustering',
     'Plus label spreading and self-training; literature as a sanity check',
     ('LOO on the three leaks only',
      'Coverage of at least 90% as a gate: would drop U36',
      'Literature inside the model: work within the provided data')),
    ('Alternative model', 'Firth logit on the Model 1 score',
     'Intercept shifted to 1 in 8; 14 refits; PVC discount applied',
     ('Ridge penalty: needs a tuning value six points cannot set',
      'Contrast-only or two-input logit',
      'Profile-likelihood interval instead of the refit range')),
)
FINAL_TITLE = 'Predictions and dig list'
FINAL_DETAIL = 'U17, U07, U08, U36 + U34 (informative dig)'
ASSUMPTION_TEXT = 'Fixed by assumption A5; no alternative was run'

# Layout, in inches from the top-left corner of the figure.
FIGURE_SIZE = (7.5, 10.6)
HEADER_Y = 0.62
FIRST_ROW_Y = 0.78
ROW_PITCH = 0.565
BOX_HEIGHT = 0.45
ROUTE_X, ROUTE_WIDTH = 0.30, 3.20
ALTERNATIVE_X, ALTERNATIVE_WIDTH = 3.85, 3.35
BRACKET_X = 3.68
ALTERNATIVE_HEIGHT = 0.133
ALTERNATIVE_GAP = 0.025
CORNER_RADIUS = 0.04
TEXT_PAD = 0.08
# Text baselines inside a route box, as offsets from its top edge.
DECISION_OFFSET, CHOICE_OFFSET, DETAIL_OFFSET = 0.095, 0.225, 0.355
ARROW_CLEARANCE = 0.012
LINE_WIDTH = 0.7
DASH_PATTERN = (0, (3, 2))


def _box(ax, x, y, width, height, face, edge, dashed=False):
    """Draw a rounded box with its top-left corner at (x, y)."""
    ax.add_patch(FancyBboxPatch(
        (x, y), width, height,
        boxstyle=f'round,pad=0,rounding_size={CORNER_RADIUS}',
        facecolor=face, edgecolor=edge, linewidth=LINE_WIDTH,
        linestyle=DASH_PATTERN if dashed else 'solid'))


def _route_box(ax, top, decision, choice, detail):
    """Chosen option: maroon outline, decision name above the choice."""
    _box(ax, ROUTE_X, top, ROUTE_WIDTH, BOX_HEIGHT, PANEL_GREY,
         UCHICAGO_MAROON)
    left = ROUTE_X + TEXT_PAD
    ax.text(left, top + DECISION_OFFSET, decision.upper(), va='center',
            fontsize=TINY_FONT_SIZE, fontweight='bold',
            color=UCHICAGO_MAROON)
    ax.text(left, top + CHOICE_OFFSET, choice, va='center',
            fontsize=PANEL_TITLE_FONT_SIZE, fontweight='bold',
            color=TEXT_DARK)
    ax.text(left, top + DETAIL_OFFSET, detail, va='center',
            fontsize=TINY_FONT_SIZE, color=TEXT_GREY)


def _alternative_boxes(ax, top, alternatives):
    """Options not taken: dashed grey boxes joined by a dashed bracket."""
    middle = top + BOX_HEIGHT / 2
    if not alternatives:
        _box(ax, ALTERNATIVE_X, middle - ALTERNATIVE_HEIGHT / 2,
             ALTERNATIVE_WIDTH, ALTERNATIVE_HEIGHT, 'white', UNLABELED_GREY)
        ax.plot([ROUTE_X + ROUTE_WIDTH, ALTERNATIVE_X], [middle, middle],
                color=UNLABELED_GREY, linewidth=LINE_WIDTH)
        ax.text(ALTERNATIVE_X + TEXT_PAD, middle, ASSUMPTION_TEXT,
                va='center', fontsize=TINY_FONT_SIZE, style='italic',
                color=TEXT_GREY)
        return
    block = (len(alternatives) * ALTERNATIVE_HEIGHT
             + (len(alternatives) - 1) * ALTERNATIVE_GAP)
    centers = [middle - block / 2 + ALTERNATIVE_HEIGHT / 2
               + index * (ALTERNATIVE_HEIGHT + ALTERNATIVE_GAP)
               for index in range(len(alternatives))]
    bracket = dict(color=UNLABELED_GREY, linewidth=LINE_WIDTH,
                   linestyle=DASH_PATTERN)
    ax.plot([ROUTE_X + ROUTE_WIDTH, BRACKET_X], [middle, middle], **bracket)
    ax.plot([BRACKET_X, BRACKET_X], [centers[0], centers[-1]], **bracket)
    for center, text in zip(centers, alternatives):
        ax.plot([BRACKET_X, ALTERNATIVE_X], [center, center], **bracket)
        _box(ax, ALTERNATIVE_X, center - ALTERNATIVE_HEIGHT / 2,
             ALTERNATIVE_WIDTH, ALTERNATIVE_HEIGHT, 'white', UNLABELED_GREY,
             dashed=True)
        ax.text(ALTERNATIVE_X + TEXT_PAD, center, text, va='center',
                fontsize=TINY_FONT_SIZE, color=TEXT_GREY)


def _route_arrow(ax, from_y, to_y):
    """Maroon arrow down the route, between consecutive boxes."""
    center_x = ROUTE_X + ROUTE_WIDTH / 2
    ax.annotate('', xy=(center_x, to_y - ARROW_CLEARANCE),
                xytext=(center_x, from_y + ARROW_CLEARANCE),
                arrowprops=dict(arrowstyle='-|>', color=UCHICAGO_MAROON,
                                linewidth=LINE_WIDTH, mutation_scale=7))


def _final_box(ax, top):
    """End of the route: solid maroon box with the dig list."""
    _box(ax, ROUTE_X, top, ROUTE_WIDTH, BOX_HEIGHT, UCHICAGO_MAROON,
         UCHICAGO_MAROON)
    left = ROUTE_X + TEXT_PAD
    ax.text(left, top + BOX_HEIGHT * 0.36, FINAL_TITLE, va='center',
            fontsize=PANEL_TITLE_FONT_SIZE, fontweight='bold', color='white')
    ax.text(left, top + BOX_HEIGHT * 0.70, FINAL_DETAIL, va='center',
            fontsize=ANNOTATION_FONT_SIZE, color='white')


def plot_method_tree():
    """Draw the decision tree and save it as a PNG; return the path."""
    apply_urban_labs_style()
    fig = plt.figure(figsize=FIGURE_SIZE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FIGURE_SIZE[0])
    ax.set_ylim(FIGURE_SIZE[1], 0)
    ax.axis('off')
    add_title_and_note(fig, FIGURE_NUMBER, TITLE, NOTE)
    for x, label in ((ROUTE_X, 'ROUTE TAKEN'),
                     (ALTERNATIVE_X, 'CONSIDERED, NOT TAKEN')):
        ax.text(x, HEADER_Y, label, va='center', fontsize=HEADER_FONT_SIZE,
                fontweight='bold', color=UCHICAGO_MAROON)
    for index, (decision, choice, detail, alternatives) in enumerate(
            DECISIONS):
        top = FIRST_ROW_Y + index * ROW_PITCH
        _route_box(ax, top, decision, choice, detail)
        _alternative_boxes(ax, top, alternatives)
        _route_arrow(ax, top + BOX_HEIGHT, top + ROW_PITCH)
    _final_box(ax, FIRST_ROW_Y + len(DECISIONS) * ROW_PITCH)
    return save_figure(fig, METHOD_TREE_FIGURE_PATH.parent,
                       METHOD_TREE_FIGURE_PATH.stem)


if __name__ == '__main__':
    print(plot_method_tree())
