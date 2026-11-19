"""Figures for Model 2 (Firth logistic regression on the Model 1 score).

Fig 5 (INTERPRETABLE)  The fitted S-curve: score -> leak probability.
Fig 6 (LAY)            Model 1 vs. Model 2 for the leading sites.

Reads the Model 1 and Model 2 tables (run src/run_model_logistic.py first).
Outputs: outputs/figures/2_model/model2_fig5_s_curve.png,
         outputs/figures/2_model/model2_fig6_model_comparison.png
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src.config import (
    ANALYSIS_EXCLUDED_CLIPS,
    HISTORICAL_LEAK_RATE,
    LEAK_LABEL,
    MODEL1_PROBABILITY_PATH,
    MODEL1_SCORES_PATH,
    MODEL2_FITS_PATH,
    MODEL2_PROBABILITY_PATH,
    MODEL_FIGURE_DIR,
    NO_LEAK_LABEL,
)
from src.plot_style import (
    ANNOTATION_FONT_SIZE,
    HEALTH_LAB_LIGHT_TEAL,
    HEALTH_LAB_NAVY,
    LABEL_COLORS,
    LIGHT_GRID_GREY,
    RANKING_COLORS,
    RANKING_MARKERS,
    RANKING_NAMES,
    SMALL_FONT_SIZE,
    TEXT_DARK,
    TEXT_GREY,
    UCHICAGO_MAROON,
    UNLABELED_GREY,
    add_title_and_note,
    apply_urban_labs_style,
    save_figure,
)
from src.run_model_logistic import FULL_FIT, MAIN_VERSION, predict

SCORE_GRID = np.linspace(0, 22, 300)
EXTRA_COMPARISON_SITES = ('U32', 'U35')
S_CURVE_LABEL_OFFSETS = {'U32': (0, 12), 'U36': (0, 9), 'U08': (0, -13),
                         'U07': (0, 9), 'U17': (-4, 9), 'U15': (16, -3),
                         'U34': (16, -3)}


def _sample_scale(fit, score_values):
    """Probability on the 6-site sample (no base-rate shift)."""
    return 1 / (1 + np.exp(-(fit['intercept']
                             + fit['slope'] * np.asarray(score_values))))


def plot_s_curve(fits, scores, model2):
    """Fig 5. Fitted curves, dug outcomes, and undug band sites."""
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    main_fit = fits[(fits['version'] == MAIN_VERSION)
                    & (fits['held_out'] == FULL_FIT)].iloc[0]
    for fit in fits.itertuples():
        if fit.version == MAIN_VERSION and fit.held_out == FULL_FIT:
            continue
        ax.plot(SCORE_GRID, predict(fit._asdict(), SCORE_GRID),
                color=UNLABELED_GREY, linewidth=0.6, alpha=0.7, zorder=1)
    ax.plot(SCORE_GRID, _sample_scale(main_fit, SCORE_GRID),
            color=UCHICAGO_MAROON, linewidth=1, linestyle=':', zorder=2)
    ax.plot(SCORE_GRID, predict(main_fit, SCORE_GRID),
            color=UCHICAGO_MAROON, linewidth=2, zorder=3)
    ax.axhline(HISTORICAL_LEAK_RATE, color=TEXT_GREY, linestyle='--',
               linewidth=0.7)
    ax.text(21.8, HISTORICAL_LEAK_RATE + 0.015, 'typical site (1 in 8)',
            ha='right', fontsize=ANNOTATION_FONT_SIZE, color=TEXT_GREY)
    dug = scores[(scores['version'] == MAIN_VERSION)
                 & (scores['ranking'] == 'cast_iron')
                 & scores['label_group'].isin((LEAK_LABEL, NO_LEAK_LABEL))]
    for record in dug.itertuples():
        is_leak = record.label_group == LEAK_LABEL
        ax.scatter(record.score, 1.0 if is_leak else 0.0, s=60,
                   color=LABEL_COLORS[record.label_group], zorder=4,
                   clip_on=False, edgecolor='white')
    ax.text(0.3, 0.965, 'dug leaks (outcome = 1)',
            fontsize=ANNOTATION_FONT_SIZE,
            color=LABEL_COLORS[LEAK_LABEL], va='top')
    ax.text(9.5, 0.012, 'dug non-leaks sit at 0 (left)',
            fontsize=ANNOTATION_FONT_SIZE,
            va='bottom',
            color=LABEL_COLORS[NO_LEAK_LABEL])
    band_sites = model2[model2['score'] > 1]
    for record in band_sites[band_sites['signature_weight'] < 1].itertuples():
        ax.scatter(record.score, record.curve_probability, s=46,
                   marker=RANKING_MARKERS[record.ranking], facecolor='white',
                   edgecolor=RANKING_COLORS[record.ranking], linewidth=1,
                   zorder=5)
        ax.annotate('', xy=(record.score, record.confidence),
                    xytext=(record.score, record.curve_probability),
                    arrowprops=dict(arrowstyle='->', color=TEXT_GREY,
                                    linewidth=0.8), zorder=4)
    for record in band_sites.itertuples():
        ax.scatter(record.score, record.confidence, s=46,
                   marker=RANKING_MARKERS[record.ranking],
                   color=RANKING_COLORS[record.ranking], edgecolor=TEXT_DARK,
                   linewidth=0.5, zorder=5)
        ax.annotate(record.clip_id, (record.score, record.confidence),
                    xytext=S_CURVE_LABEL_OFFSETS.get(record.clip_id, (0, 8)),
                    textcoords='offset points', ha='center',
                    fontsize=ANNOTATION_FONT_SIZE,
                    color=TEXT_DARK)
    ax.set_xlim(0, 22)
    ax.set_ylim(-0.02, 1.02)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
    ax.set_xlabel('Model 1 score: band intenseness (contrast in dB) x '
                  'persistence gate')
    ax.set_ylabel('Estimated chance of a leak')
    ax.grid(axis='x', color=LIGHT_GRID_GREY)
    handles = [
        Line2D([], [], color=UCHICAGO_MAROON, linewidth=2,
               label='Model 2 (base rate 1 in 8)'),
        Line2D([], [], color=UCHICAGO_MAROON, linestyle=':',
               label='Same curve on the 6 dug sites (half leaks)'),
        Line2D([], [], color=UNLABELED_GREY,
               label='Refits: raw version, each dug site left out'),
        Line2D([], [], marker='o', linestyle='', color=HEALTH_LAB_NAVY,
               label='Undug cast iron'),
        Line2D([], [], marker='^', linestyle='', color=HEALTH_LAB_LIGHT_TEAL,
               label='Undug PVC, after signature discount'),
        Line2D([], [], marker='^', linestyle='', markerfacecolor='white',
               color=HEALTH_LAB_LIGHT_TEAL,
               label='Undug PVC, on the curve before discount')]
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.0, 0.93),
              fontsize=ANNOTATION_FONT_SIZE)
    bottom = add_title_and_note(
        fig, 5, 'Model 2: greater band intenseness (contrast in dB) means a '
        'likelier leak',
        'Firth-penalized logistic regression of dig outcome on the Model 1 '
        'score, fitted to the six dug sites and shifted from their 50% leak '
        'share to the 1-in-8 base rate. Grey curves refit the model on the '
        'raw-spectrum scores and with each dug site left out. PVC sites are '
        'read off the curve at their 250-400 Hz score (hollow), then '
        'discounted toward the base rate by the share of leak-signature '
        'properties their band shares (filled; U34 3 of 5, U15 and U03 2 '
        'of 5). U40 excluded (insufficient data quality).')
    fig.subplots_adjust(top=0.9, bottom=bottom, left=0.1, right=0.98)
    return save_figure(fig, MODEL_FIGURE_DIR, 'model2_fig5_s_curve')


def plot_model_comparison(model1, model2):
    """Fig 6. Model 1 and Model 2 estimates and ranges, side by side."""
    merged = model2.merge(model1[['clip_id', 'confidence', 'probability_low',
                                  'probability_high']],
                          on='clip_id', suffixes=('_m2', '_m1'))
    leading = merged[(merged['confidence_m1'] > HISTORICAL_LEAK_RATE)
                     | merged['clip_id'].isin(EXTRA_COMPARISON_SITES)]
    leading = leading.sort_values('score', ascending=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    positions = np.arange(len(leading))
    offset = 0.17
    for position, record in zip(positions, leading.itertuples()):
        for shift, low, high, value, color, marker in (
                (offset, record.probability_low_m1,
                 record.probability_high_m1, record.confidence_m1,
                 HEALTH_LAB_NAVY, 'o'),
                (-offset, record.probability_low_m2,
                 record.probability_high_m2, record.confidence_m2,
                 UCHICAGO_MAROON, 'D')):
            ax.plot([low, high], [position + shift] * 2, color=color,
                    linewidth=3, alpha=0.35, solid_capstyle='round')
            ax.plot(value, position + shift, marker=marker, color=color,
                    markersize=6.5, markeredgecolor='white')
    ax.axvline(HISTORICAL_LEAK_RATE, color=TEXT_GREY, linestyle='--',
               linewidth=0.7)
    ax.set_yticks(positions)
    ax.set_yticklabels([f'{r.clip_id} ({RANKING_NAMES[r.ranking]}, score '
                        f'{r.score:.1f})' for r in leading.itertuples()],
                       fontsize=SMALL_FONT_SIZE)
    ax.set_xlim(0, 0.8)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
    ax.set_xlabel('Estimated chance of a leak')
    ax.grid(axis='y', visible=False)
    ax.grid(axis='x', color=LIGHT_GRID_GREY)
    handles = [
        Line2D([], [], marker='o', color=HEALTH_LAB_NAVY,
               label='Model 1: presence first, then band quality'),
        Line2D([], [], marker='D', color=UCHICAGO_MAROON,
               label='Model 2: chance rises with intenseness (dB contrast)')]
    ax.legend(handles=handles, loc='lower right',
              fontsize=ANNOTATION_FONT_SIZE)
    bottom = add_title_and_note(
        fig, 6, 'The two models flag similar sites but rank them differently',
        'Markers are each model\'s main estimate; bars are its range across '
        'analysis choices; dashed line is the 1-in-8 base rate. Both apply '
        'the same PVC signature discount. Model 1 gives every site with a '
        'steady band 36-50% (before the PVC discount) and grades within '
        'that by steadiness (weighted 0.6) and intenseness net of neighbor '
        'noise (0.4), so the four cast-iron sites lead. Model 2 grades by '
        'contrast in dB alone, which traffic lowers: U34 leads, U17 and U15 '
        'stay above the base rate, U07 sits near it, and U08, U36 and U03 '
        '(whose contrast traffic lowers to under 1 dB) fall below it. Sites '
        'without a band are low under both models.')
    fig.subplots_adjust(top=0.9, bottom=bottom, left=0.25, right=0.98)
    return save_figure(fig, MODEL_FIGURE_DIR, 'model2_fig6_model_comparison')


def make_model2_figures():
    """Load Model 1 and 2 outputs and draw Figures 5 and 6."""
    apply_urban_labs_style()
    fits = pd.read_csv(MODEL2_FITS_PATH)
    scores = pd.read_csv(MODEL1_SCORES_PATH)
    scores = scores[~scores['clip_id'].isin(ANALYSIS_EXCLUDED_CLIPS)]
    model1 = pd.read_csv(MODEL1_PROBABILITY_PATH)
    model2 = pd.read_csv(MODEL2_PROBABILITY_PATH)
    return [plot_s_curve(fits, scores, model2),
            plot_model_comparison(model1, model2)]


if __name__ == '__main__':
    for path in make_model2_figures():
        print(f'Saved {path}')
