"""Robustness check: do semi-supervised methods agree with Model 1?

Two methods use the 6 dug sites plus the 39 scorable undug sites:

1. Label spreading (k-nearest-neighbor graph). Each site is described by
   the three band properties Model 1 uses, each standardized across all
   sites: intenseness (contrast in dB), persistence over time and coverage
   across pitch. Labels flow from the dug sites to their neighbors. Output:
   a 0-1 leak share per site, plus whether it is labeled leak.
2. Self-training. Fit Model 2 (Firth logistic regression on the Model 1
   score) to the dug sites; add undug sites predicted with >= 90% (leak) or
   <= 10% (no leak) on the sample scale as pseudo-labels; refit; repeat
   until nothing new is added.

Both learn from a 50%-leak sample, so their outputs are compared with
Model 1 by ranking and by which sites they call leaks, not by probability
level. Check only; nothing feeds the predictions.

Outputs
-------
outputs/tables/5_checks/check_self_training.csv
outputs/tables/5_checks/check_self_training.md
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelSpreading

from src.config import (
    ANALYSIS_EXCLUDED_CLIPS,
    LABEL_SPREADING_ALPHA,
    LABEL_SPREADING_NEIGHBORS,
    LEAK_LABEL,
    MAIN_VERSION,
    MAX_DIGS_THIS_WEEK,
    MODEL1_PROBABILITY_PATH,
    MODEL1_SCORES_PATH,
    NO_LEAK_LABEL,
    SELF_TRAINING_LEAK_CUTOFF,
    SELF_TRAINING_MAX_ROUNDS,
    SELF_TRAINING_NO_LEAK_CUTOFF,
    SELF_TRAINING_SUMMARY_PATH,
    SELF_TRAINING_TABLE_PATH,
    SUMMARY_TABLE_ROWS,
    TABLE_DECIMALS,
)
from src.measure_band_evenness import compute_band_evenness
from src.run_model_logistic import fit_firth_logistic
from src.utils import make_output_dirs

UNLABELED_CODE = -1
FEATURES = ('contrast_db', 'persistence', 'bin_coverage')


def build_feature_table():
    """Main-version band properties for every scorable site."""
    scores = pd.read_csv(MODEL1_SCORES_PATH)
    scores = scores[(scores['version'] == MAIN_VERSION)
                    & ~scores['clip_id'].isin(ANALYSIS_EXCLUDED_CLIPS)]
    evenness = compute_band_evenness()[['clip_id', 'bin_coverage']]
    return scores.merge(evenness, on='clip_id').reset_index(drop=True)


def run_label_spreading(table):
    """Leak share per site from a kNN label-spreading graph."""
    features = StandardScaler().fit_transform(table[list(FEATURES)])
    labels = np.full(len(table), UNLABELED_CODE)
    labels[table['label_group'] == LEAK_LABEL] = 1
    labels[table['label_group'] == NO_LEAK_LABEL] = 0
    model = LabelSpreading(kernel='knn',
                           n_neighbors=LABEL_SPREADING_NEIGHBORS,
                           alpha=LABEL_SPREADING_ALPHA).fit(features, labels)
    leak_column = list(model.classes_).index(1)
    return (model.label_distributions_[:, leak_column],
            model.transduction_ == 1)


def _sample_probability(coefficients, score_values):
    intercept, slope = coefficients
    return 1 / (1 + np.exp(-(intercept + slope * score_values)))


def run_self_training(table):
    """Iteratively pseudo-label confident undug sites and refit Model 2."""
    is_dug = table['label_group'].isin((LEAK_LABEL, NO_LEAK_LABEL))
    labels = pd.Series(np.nan, index=table.index)
    labels[is_dug] = (table.loc[is_dug, 'label_group']
                      == LEAK_LABEL).astype(float)
    history = []
    for round_number in range(1, SELF_TRAINING_MAX_ROUNDS + 1):
        known = labels.notna()
        coefficients = fit_firth_logistic(
            table.loc[known, 'score'].to_numpy(),
            labels[known].to_numpy())
        probability = _sample_probability(coefficients,
                                          table['score'].to_numpy())
        new_leaks = (~known) & (probability >= SELF_TRAINING_LEAK_CUTOFF)
        new_no_leaks = (~known) & (probability
                                   <= SELF_TRAINING_NO_LEAK_CUTOFF)
        history.append({
            'round': round_number, 'n_labeled': int(known.sum()),
            'slope': coefficients[1],
            'added_leak': table.loc[new_leaks, 'clip_id'].tolist(),
            'added_no_leak_count': int(new_no_leaks.sum())})
        if not (new_leaks.any() or new_no_leaks.any()):
            break
        labels[new_leaks] = 1.0
        labels[new_no_leaks] = 0.0
    return probability, labels, pd.DataFrame(history)


def compare_with_model1(table, spread_share, spread_is_leak,
                        self_probability, self_labels):
    """Undug sites: Model 1 confidence and rank vs. both checks."""
    model1 = pd.read_csv(MODEL1_PROBABILITY_PATH)[
        ['clip_id', 'confidence', 'prediction']]
    result = table[['clip_id', 'ranking', 'label_group', 'score']].copy()
    result['spread_leak_share'] = spread_share
    result['spread_is_leak'] = spread_is_leak.astype(int)
    result['self_training_probability'] = self_probability
    result['self_training_label'] = self_labels
    result = result[result['label_group'] == 'unlabeled'].merge(
        model1, on='clip_id')
    for column in ('confidence', 'spread_leak_share',
                   'self_training_probability'):
        result[f'rank_{column}'] = result[column].rank(ascending=False,
                                                       method='min')
    return result.sort_values('confidence', ascending=False)


def _write_summary(result, history):
    """Plain-language Markdown summary."""
    model1_leaks = set(result.loc[result['prediction'] == 1, 'clip_id'])
    spread_leaks = set(result.loc[result['spread_is_leak'] == 1, 'clip_id'])
    self_leaks = set(result.loc[result['self_training_label'] == 1,
                                'clip_id'])
    top = {name: result.nlargest(MAX_DIGS_THIS_WEEK,
                                 column)['clip_id'].tolist()
           for name, column in (('Model 1', 'confidence'),
                                ('Label spreading', 'spread_leak_share'),
                                ('Self-training',
                                 'self_training_probability'))}
    spearman = result[['confidence', 'spread_leak_share',
                       'self_training_probability']].corr(method='spearman')
    lines = ['# Robustness check: label spreading and self-training', '',
             f'- Features for label spreading: {", ".join(FEATURES)} '
             f'(standardized), k = {LABEL_SPREADING_NEIGHBORS}, alpha = '
             f'{LABEL_SPREADING_ALPHA}.',
             f'- Model 1 predicted leaks: {sorted(model1_leaks)}.',
             f'- Label spreading leaks: {sorted(spread_leaks)} '
             f'(agreement with Model 1: '
             f'{len(model1_leaks & spread_leaks)} shared).',
             f'- Self-training pseudo-labeled leaks: {sorted(self_leaks)}.',
             f'- Top 5 by method: {top}.',
             f'- Rank correlation with Model 1 (Spearman): label spreading '
             f'{spearman.loc["confidence", "spread_leak_share"]:.2f}, '
             f'self-training '
             f'{spearman.loc["confidence", "self_training_probability"]:.2f}.',
             '', '## Self-training rounds', '',
             '| Round | Labeled sites | Slope | Added as leak | Added as no '
             'leak |', '|---|---|---|---|---|']
    lines += [f'| {r.round} | {r.n_labeled} | {r.slope:.2f} | '
              f'{r.added_leak} | {r.added_no_leak_count} |'
              for r in history.itertuples()]
    lines += ['', '## Leading sites', '',
              '| Site | Pipe | Model 1 | Label-spread share | Self-training '
              'probability (sample scale) |', '|---|---|---|---|---|']
    lines += [f'| {r.clip_id} | {r.ranking} | {r.confidence:.1%} | '
              f'{r.spread_leak_share:.2f} | '
              f'{r.self_training_probability:.2f} |'
              for r in result.head(SUMMARY_TABLE_ROWS).itertuples()]
    SELF_TRAINING_SUMMARY_PATH.write_text('\n'.join(lines) + '\n',
                                          encoding='utf-8')


def run_self_training_check():
    """Run both checks and save the comparison."""
    table = build_feature_table()
    spread_share, spread_is_leak = run_label_spreading(table)
    self_probability, self_labels, history = run_self_training(table)
    result = compare_with_model1(table, spread_share, spread_is_leak,
                                 self_probability, self_labels)
    make_output_dirs()
    result.round(TABLE_DECIMALS).to_csv(SELF_TRAINING_TABLE_PATH, index=False)
    _write_summary(result, history)
    return result


if __name__ == '__main__':
    run_self_training_check()
    print(SELF_TRAINING_SUMMARY_PATH.read_text(encoding='utf-8'))
