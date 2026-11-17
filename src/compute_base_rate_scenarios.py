"""How the 1-in-8 leak rate turns flags into false alarms.

For any screening rule, the share of flagged sites that truly leak (the
positive predictive value, PPV) depends on how common leaks are, not only
on how good the rule is (Foster et al., ch. 10, "rare populations"). This
script tabulates PPV under the historical base rate and under the 50% leak
share of the dug sites, translates it into digs and dollars, and shows how
little six labeled sites can reveal about a rule's accuracy.

Outputs
-------
outputs/tables/base_rate_ppv_scenarios.csv
outputs/tables/base_rate_summary.md
outputs/figures/eda/eda_17_base_rate_false_alarms.png
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle
from scipy import stats

from src.config import (
    BASE_RATE_SUMMARY_PATH,
    BASE_RATE_TABLE_PATH,
    COST_PER_DIG_DOLLARS,
    HISTORICAL_LEAK_RATE,
    MAX_DIGS_THIS_WEEK,
    N_LABELED_EXPECTED,
    N_UNLABELED_EXPECTED,
    TABLE_DIR,
)
from src.load_recordings import load_metadata
from src.plot_style import (
    CRIME_LAB_GOLD,
    HEALTH_LAB_TEAL,
    PANEL_GREY,
    TEXT_DARK,
    UCHICAGO_MAROON,
    UNLABELED_GREY,
    add_title_and_note,
    apply_urban_labs_style,
    finish_and_save,
)

SENSITIVITIES = (0.5, 0.7, 0.8, 0.9)
SPECIFICITIES = (0.7, 0.8, 0.9, 0.95)
CONFIDENCE_LEVEL = 0.95
# Illustration for the lay panel: a rule that catches 4 of the 5 expected
# leaks and wrongly flags 4 of the 35 sound sites (specificity ~0.89).
ILLUSTRATION_TRUE_POSITIVES = 4
ILLUSTRATION_FALSE_POSITIVES = 4
ILLUSTRATION_SENSITIVITY = 0.8
SPECIFICITY_CURVE = np.linspace(0.5, 0.995, 200)
WAFFLE_COLUMNS = 8
FIGURE_SIZE = (7.5, 4.4)


def positive_predictive_value(sensitivity, specificity, prevalence):
    """Share of flagged sites that truly leak (Bayes' rule)."""
    true_flags = sensitivity * prevalence
    false_flags = (1 - specificity) * (1 - prevalence)
    return true_flags / (true_flags + false_flags)


def build_ppv_table(labeled_prevalence):
    """PPV and expected flag counts for a grid of rule accuracies."""
    expected_leaks = HISTORICAL_LEAK_RATE * N_UNLABELED_EXPECTED
    expected_sound = N_UNLABELED_EXPECTED - expected_leaks
    rows = []
    for sensitivity in SENSITIVITIES:
        for specificity in SPECIFICITIES:
            ppv = positive_predictive_value(sensitivity, specificity,
                                            HISTORICAL_LEAK_RATE)
            rows.append({
                'sensitivity': sensitivity,
                'specificity': specificity,
                'ppv_at_1_in_8': ppv,
                'ppv_at_labeled_share': positive_predictive_value(
                    sensitivity, specificity, labeled_prevalence),
                'expected_true_flags_of_40': sensitivity * expected_leaks,
                'expected_false_flags_of_40':
                    (1 - specificity) * expected_sound,
                'dollars_per_leak_found': COST_PER_DIG_DOLLARS / ppv,
            })
    return pd.DataFrame(rows)


def summarize_blind_digging():
    """What five digs find with no information at all."""
    expected_leaks = MAX_DIGS_THIS_WEEK * HISTORICAL_LEAK_RATE
    chance_of_none = (1 - HISTORICAL_LEAK_RATE) ** MAX_DIGS_THIS_WEEK
    return {
        'expected_leaks_in_5_blind_digs': expected_leaks,
        'chance_5_blind_digs_find_nothing': chance_of_none,
        'dollars_per_leak_found_blind':
            COST_PER_DIG_DOLLARS / HISTORICAL_LEAK_RATE,
    }


def summarize_validation_limits():
    """How uncertain is accuracy measured on 3 leaks and 3 non-leaks?"""
    n_per_class = N_LABELED_EXPECTED // 2
    perfect_class = stats.binomtest(n_per_class, n_per_class)
    perfect_all = stats.binomtest(N_LABELED_EXPECTED, N_LABELED_EXPECTED)
    return {
        'lower_bound_sensitivity_if_3_of_3':
            perfect_class.proportion_ci(CONFIDENCE_LEVEL).low,
        'lower_bound_accuracy_if_6_of_6':
            perfect_all.proportion_ci(CONFIDENCE_LEVEL).low,
        'chance_coin_flip_gets_6_of_6': 0.5 ** N_LABELED_EXPECTED,
    }


def write_summary(ppv_table, blind, limits, labeled_prevalence):
    """Plain-language Markdown summary of the base-rate arithmetic."""
    example = ppv_table[(ppv_table['sensitivity'] == 0.8)
                        & (ppv_table['specificity'] == 0.9)].iloc[0]
    lines = [
        '# Base rates and false alarms',
        '',
        f'- Expected leaks among the 40 undug sites at 1 in 8: '
        f'**{HISTORICAL_LEAK_RATE * N_UNLABELED_EXPECTED:.0f}**.',
        f'- Five blind digs find '
        f'**{blind["expected_leaks_in_5_blind_digs"]:.2f}** leaks on '
        f'average; there is a '
        f'**{blind["chance_5_blind_digs_find_nothing"]:.0%}** chance they '
        f'find none. Cost per leak found: '
        f'**${blind["dollars_per_leak_found_blind"]:,.0f}**.',
        f'- A rule that catches 80% of leaks and clears 90% of sound sites '
        f'is right only **{example["ppv_at_1_in_8"]:.0%}** of the time it '
        f'says "leak" at 1 in 8, but would look '
        f'**{example["ppv_at_labeled_share"]:.0%}** right on a sample that is '
        f'{labeled_prevalence:.0%} leaks, like the dug sites. Applied to '
        f'all 40 sites it would flag about '
        f'{example["expected_true_flags_of_40"]:.0f} leaks and '
        f'{example["expected_false_flags_of_40"]:.1f} false alarms.',
        f'- Even a perfect score on the dug sites says little: 3 of 3 leaks '
        f'caught is consistent (95% CI) with true sensitivity as low as '
        f'**{limits["lower_bound_sensitivity_if_3_of_3"]:.0%}**; 6 of 6 '
        f'correct is consistent with accuracy as low as '
        f'**{limits["lower_bound_accuracy_if_6_of_6"]:.0%}**. A coin flip '
        f'gets all 6 right {limits["chance_coin_flip_gets_6_of_6"]:.1%} of '
        f'the time.',
        '',
        'Full grid: `base_rate_ppv_scenarios.csv`.',
    ]
    BASE_RATE_SUMMARY_PATH.write_text('\n'.join(lines) + '\n',
                                      encoding='utf-8')


def _draw_ppv_curves(ax, labeled_prevalence):
    """PPV vs. specificity at the base rate and at the dug-site share."""
    for prevalence, color, label in [
            (labeled_prevalence, HEALTH_LAB_TEAL,
             f'If leaks were {labeled_prevalence:.0%} of sites (as among '
             f'the 6 dug)'),
            (HISTORICAL_LEAK_RATE, UCHICAGO_MAROON,
             'At the real rate, 1 in 8 sites')]:
        ax.plot(SPECIFICITY_CURVE,
                positive_predictive_value(ILLUSTRATION_SENSITIVITY,
                                          SPECIFICITY_CURVE, prevalence),
                color=color, linewidth=2, label=label)
    ax.set_xlabel('Share of sound sites the rule correctly clears '
                  '(specificity)')
    ax.set_ylabel('Share of flagged sites that truly leak')
    ax.set_ylim(0, 1)
    ax.set_xlim(SPECIFICITY_CURVE[0], 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
    ax.legend(loc='lower right', fontsize=7)
    ax.set_title(f'A rule that catches {ILLUSTRATION_SENSITIVITY:.0%} of '
                 'leaks', loc='left')


def _draw_waffle(ax):
    """40 squares: what one realistic rule would flag among undug sites."""
    n_leaks = round(HISTORICAL_LEAK_RATE * N_UNLABELED_EXPECTED)
    categories = (
        [('Leak, flagged', UCHICAGO_MAROON)] * ILLUSTRATION_TRUE_POSITIVES
        + [('Leak, missed', CRIME_LAB_GOLD)]
        * (n_leaks - ILLUSTRATION_TRUE_POSITIVES)
        + [('No leak, flagged (false alarm)', HEALTH_LAB_TEAL)]
        * ILLUSTRATION_FALSE_POSITIVES
        + [('No leak, cleared', PANEL_GREY)]
        * (N_UNLABELED_EXPECTED - n_leaks - ILLUSTRATION_FALSE_POSITIVES))
    n_rows = int(np.ceil(len(categories) / WAFFLE_COLUMNS))
    for index, (_, color) in enumerate(categories):
        row, column = divmod(index, WAFFLE_COLUMNS)
        ax.add_patch(Rectangle((column, n_rows - 1 - row), 0.88, 0.88,
                               facecolor=color, edgecolor=UNLABELED_GREY
                               if color == PANEL_GREY else color,
                               linewidth=0.5))
    ax.set_xlim(-0.1, WAFFLE_COLUMNS)
    ax.set_ylim(-0.1, n_rows)
    ax.set_aspect('equal')
    ax.axis('off')
    legend_order = dict.fromkeys(categories)
    ax.legend(handles=[Patch(facecolor=color, edgecolor=UNLABELED_GREY,
                             label=label) for label, color in legend_order],
              loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=2,
              fontsize=6.5)
    n_flagged = ILLUSTRATION_TRUE_POSITIVES + ILLUSTRATION_FALSE_POSITIVES
    n_false = ILLUSTRATION_FALSE_POSITIVES
    ax.set_title(f'Of 40 undug sites (~{n_leaks} leaks), it flags '
                 f'{n_flagged};\n{n_false} of them are false alarms',
                 loc='left', color=TEXT_DARK)


def plot_base_rate_figure(labeled_prevalence):
    """Fig 17 (LAY). False alarms compound when leaks are rare."""
    apply_urban_labs_style()
    fig, (curve_ax, waffle_ax) = plt.subplots(
        1, 2, figsize=FIGURE_SIZE, gridspec_kw={'width_ratios': [1.15, 1]})
    _draw_ppv_curves(curve_ax, labeled_prevalence)
    _draw_waffle(waffle_ax)
    fig.subplots_adjust(wspace=0.3)
    bottom = add_title_and_note(
        fig, 17, 'Because leaks are rare, even a good rule raises many '
        'false alarms',
        'Left: share of flagged sites that truly leak, for a rule that '
        'catches 80% of leaks, as its false-alarm rate varies. The same '
        'rule looks far better on a half-leak sample like the six dug sites '
        'than it will on the undug sites. Right: an illustrative rule that '
        'catches 4 of 5 leaks and wrongly flags 4 of 35 sound sites '
        '(specificity 89%). Five blind digs would find 0.6 leaks on '
        'average.')
    return finish_and_save(fig, 'eda_17_base_rate_false_alarms', bottom,
                           top=0.84)


def run_base_rate_analysis():
    """Compute tables, write the summary, and draw the figure."""
    metadata = load_metadata()
    labeled_prevalence = metadata.loc[metadata['is_labeled'],
                                      'is_leak'].astype(float).mean()
    ppv_table = build_ppv_table(labeled_prevalence)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    ppv_table.round(3).to_csv(BASE_RATE_TABLE_PATH, index=False)
    blind = summarize_blind_digging()
    limits = summarize_validation_limits()
    write_summary(ppv_table, blind, limits, labeled_prevalence)
    return plot_base_rate_figure(labeled_prevalence)


if __name__ == '__main__':
    print(f'Saved {run_base_rate_analysis()}')
    print(BASE_RATE_SUMMARY_PATH.read_text(encoding='utf-8'))
