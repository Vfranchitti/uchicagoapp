"""Model 2: Firth-penalized logistic regression on the Model 1 score.

One input (the Model 1 score, in dB) predicts leak / no leak on the six dug
sites. Their perfect split would make ordinary logistic regression diverge,
so Firth's correction (equivalent to a Jeffreys prior) is used: it has no
tuning setting and keeps the slope finite.

Base rate: each fit learns from its own sample (3 leaks of 6, or 2 of 5 /
3 of 5 when a dug site is left out). The intercept is shifted by
log(prior odds / sample odds) so predictions refer to the 1-in-8 base rate
(prior correction).

Fits: {noise-floor-subtracted, raw} x {all 6 dug sites, each left out} =
14 fits. Point estimate = full noise-floor fit; range = min-max across all
14. PVC sites are placed on the same curve with their 250-400 Hz score
(assumption A8), then discounted toward the base rate by the same
signature-match weight w used in Model 1's probabilities (U34 0.6, U15
0.4): p = w x curve p + (1 - w) x base rate. The undiscounted curve value
is kept as curve_probability.

Outputs
-------
outputs/tables/model2_fits.csv
outputs/tables/model2_probabilities.csv
outputs/tables/model2_summary.md
"""
import numpy as np
import pandas as pd

from src.config import (
    ANALYSIS_EXCLUDED_CLIPS,
    ANALYSIS_EXCLUSION_REASON,
    FIRTH_MAX_ITERATIONS,
    FIRTH_TOLERANCE,
    HISTORICAL_LEAK_RATE,
    LEAK_LABEL,
    MODEL1_PROBABILITY_PATH,
    MODEL1_SCORES_PATH,
    MODEL2_FITS_PATH,
    MODEL2_PROBABILITY_PATH,
    MODEL2_SUMMARY_PATH,
    NO_LEAK_LABEL,
    TABLE_DIR,
)

VERSIONS = ('floor', 'raw')
MAIN_VERSION = 'floor'
FULL_FIT = 'all six'
DUG_LABELS = (LEAK_LABEL, NO_LEAK_LABEL)
PRIOR_LOG_ODDS = np.log(HISTORICAL_LEAK_RATE / (1 - HISTORICAL_LEAK_RATE))
MAX_NEWTON_STEP = 5.0   # caps each update; keeps early iterations stable


def _sigmoid(values):
    return 1 / (1 + np.exp(-values))


def fit_firth_logistic(x, y):
    """Firth-penalized logistic regression with one predictor.

    Maximizes log-likelihood + 0.5 * log det(Fisher information) by Newton
    steps on the modified score U* = X'(y - p + h (0.5 - p)), where h are
    the hat-matrix diagonals. Returns (intercept, slope).
    """
    design = np.column_stack([np.ones_like(x, dtype=float), x])
    coefficients = np.zeros(design.shape[1])
    for _ in range(FIRTH_MAX_ITERATIONS):
        probabilities = _sigmoid(design @ coefficients)
        weights = probabilities * (1 - probabilities)
        information = design.T @ (design * weights[:, None])
        inverse_information = np.linalg.inv(information)
        hat = weights * np.einsum('ij,jk,ik->i', design,
                                  inverse_information, design)
        score = design.T @ (y - probabilities + hat * (0.5 - probabilities))
        step = inverse_information @ score
        largest = np.max(np.abs(step))
        if largest > MAX_NEWTON_STEP:
            step *= MAX_NEWTON_STEP / largest
        coefficients = coefficients + step
        if largest < FIRTH_TOLERANCE:
            break
    return float(coefficients[0]), float(coefficients[1])


def fit_all(scores):
    """The 14 fits: two versions x (all dug sites, each left out)."""
    rows = []
    for version in VERSIONS:
        dug = scores[(scores['version'] == version)
                     & (scores['ranking'] == 'cast_iron')
                     & scores['label_group'].isin(DUG_LABELS)]
        for held_out in [FULL_FIT] + dug['clip_id'].tolist():
            train = dug[dug['clip_id'] != held_out]
            y = (train['label_group'] == LEAK_LABEL).astype(float).to_numpy()
            intercept, slope = fit_firth_logistic(train['score'].to_numpy(),
                                                  y)
            sample_log_odds = np.log(y.sum() / (len(y) - y.sum()))
            rows.append({
                'version': version, 'held_out': held_out,
                'n_leaks': int(y.sum()), 'n_sites': len(y),
                'intercept': intercept, 'slope': slope,
                'base_rate_shift': PRIOR_LOG_ODDS - sample_log_odds,
                'score_at_50pct_sample': -intercept / slope,
            })
    return pd.DataFrame(rows)


def predict(fit, score_values):
    """Base-rate-corrected leak probability for scores under one fit."""
    return _sigmoid(fit['intercept'] + fit['base_rate_shift']
                    + fit['slope'] * np.asarray(score_values))


def site_probabilities(scores, fits):
    """Point estimate, range, and held-out checks per site."""
    undug = scores[scores['label_group'] == 'unlabeled']
    per_fit = []
    for fit in fits.itertuples():
        subset = undug[undug['version'] == fit.version]
        per_fit.append(pd.DataFrame({
            'clip_id': subset['clip_id'].to_numpy(),
            'version': fit.version, 'held_out': fit.held_out,
            'probability': predict(fit._asdict(), subset['score'])}))
    per_fit = pd.concat(per_fit, ignore_index=True)
    full_main = per_fit[(per_fit['version'] == MAIN_VERSION)
                        & (per_fit['held_out'] == FULL_FIT)].set_index(
                            'clip_id')['probability']
    grouped = per_fit.groupby('clip_id')['probability']
    main_scores = undug[undug['version'] == MAIN_VERSION].set_index(
        'clip_id')
    summary = pd.DataFrame({
        'confidence': full_main,
        'probability_low': grouped.min(),
        'probability_high': grouped.max(),
    }).join(main_scores[['ranking', 'logger_model', 'score']])
    return summary.reset_index()


def apply_pvc_discount(summary, model1):
    """Discount PVC sites by Model 1's signature-match weight w."""
    weights = model1.set_index('clip_id')['signature_weight']
    summary = summary.copy()
    summary['signature_weight'] = summary['clip_id'].map(weights).fillna(
        1.0)
    summary['curve_probability'] = summary['confidence']
    for column in ('confidence', 'probability_low', 'probability_high'):
        summary[column] = (summary['signature_weight'] * summary[column]
                           + (1 - summary['signature_weight'])
                           * HISTORICAL_LEAK_RATE)
    return summary


def held_out_check(scores, fits):
    """Each dug site's probability from the fit that left it out."""
    rows = []
    for fit in fits[fits['held_out'] != FULL_FIT].itertuples():
        site = scores[(scores['version'] == fit.version)
                      & (scores['ranking'] == 'cast_iron')
                      & (scores['clip_id'] == fit.held_out)].iloc[0]
        sample_probability = _sigmoid(fit.intercept
                                      + fit.slope * site['score'])
        rows.append({'version': fit.version, 'held_out': fit.held_out,
                     'label': site['label_group'],
                     'score': site['score'],
                     'probability_sample_scale': sample_probability,
                     'probability_base_rate': predict(fit._asdict(),
                                                      [site['score']])[0]})
    return pd.DataFrame(rows)


def write_summary(fits, summary, held_out, model1):
    """Plain-language Markdown summary with a Model 1 comparison."""
    full = fits[(fits['version'] == MAIN_VERSION)
                & (fits['held_out'] == FULL_FIT)].iloc[0]
    expected = HISTORICAL_LEAK_RATE * len(summary)
    lines = ['# Model 2: Firth logistic regression on the Model 1 score', '',
             f'- {ANALYSIS_EXCLUSION_REASON}',
             f'- Main fit: slope {full["slope"]:.2f} per dB; the curve '
             f'crosses 50% (on the 6-site sample) at '
             f'{full["score_at_50pct_sample"]:.1f} dB. Base-rate shift '
             f'{full["base_rate_shift"]:+.2f} on the log-odds scale.',
             f'- Slope across all 14 fits: {fits["slope"].min():.2f}-'
             f'{fits["slope"].max():.2f} per dB.',
             '- PVC band sites discounted by the Model 1 signature-match '
             'weight w (U34 0.6, U15 0.4); undiscounted curve values are in '
             'curve_probability.',
             f'- Calibration: sum of confidence over {len(summary)} '
             f'scorable undug sites = {summary["confidence"].sum():.1f} vs. '
             f'{expected:.1f} expected at 1 in 8.', '',
             '## Held-out dug sites (6-site sample scale; 50% = coin flip)',
             '', '| Version | Held out | Label | Score | P(leak), sample '
             'scale |', '|---|---|---|---|---|']
    lines += [f'| {r.version} | {r.held_out} | {r.label} | {r.score:.1f} | '
              f'{r.probability_sample_scale:.0%} |'
              for r in held_out.itertuples()]
    comparison = summary.merge(
        model1[['clip_id', 'confidence']].rename(
            columns={'confidence': 'model1_confidence'}), on='clip_id')
    comparison = comparison.sort_values('confidence', ascending=False)
    lines += ['', '## Model 2 vs. Model 1 (top 12 by Model 2)', '',
              '| Site | Pipe | Score | Model 2 | Model 2 range | Model 1 |',
              '|---|---|---|---|---|---|']
    lines += [f'| {r.clip_id} | {r.ranking} | {r.score:.1f} | '
              f'{r.confidence:.0%} | {r.probability_low:.0%}-'
              f'{r.probability_high:.0%} | {r.model1_confidence:.0%} |'
              for r in comparison.head(12).itertuples()]
    MODEL2_SUMMARY_PATH.write_text('\n'.join(lines) + '\n',
                                   encoding='utf-8')


def run_model_two():
    """Fit, predict, check, and save Model 2 tables."""
    scores = pd.read_csv(MODEL1_SCORES_PATH)
    scores = scores[~scores['clip_id'].isin(ANALYSIS_EXCLUDED_CLIPS)]
    fits = fit_all(scores)
    summary = apply_pvc_discount(site_probabilities(scores, fits),
                                 pd.read_csv(MODEL1_PROBABILITY_PATH))
    held_out = held_out_check(scores, fits)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    fits.round(4).to_csv(MODEL2_FITS_PATH, index=False)
    summary.round(4).to_csv(MODEL2_PROBABILITY_PATH, index=False)
    write_summary(fits, summary, held_out,
                  pd.read_csv(MODEL1_PROBABILITY_PATH))
    return fits, summary, held_out


if __name__ == '__main__':
    run_model_two()
    print(MODEL2_SUMMARY_PATH.read_text(encoding='utf-8'))
