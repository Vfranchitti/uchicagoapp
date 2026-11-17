"""Turn Model 1 scores into leak probabilities: presence first, then quality.

1. Presence sets the tier. A site either has a persistent band (present in
   at least 90% of 64 ms frames and standing out by at least 3 dB) or not.
   Bayes' rule updates the 1-in-8 base rate with a likelihood ratio from
   the dug sites (3 of 3 leaks have the band, 0 of 3 non-leaks), smoothed
   two ways (Jeffreys +0.5 and Laplace +1 per cell). That gives each tier a
   range: about 36-50% with a band, about 2.0-3.4% without.
2. Quality places a site inside its tier's range, on the log-odds scale:
       quality = steadiness^0.6 x intenseness^0.4
   steadiness  = persistence over time x coverage across pitch, each
                 relative to the dug leaks (capped at 1);
   intenseness = contrast in dB relative to the weakest dug leak (capped
                 at 1, so a band more intense than a confirmed leak earns
                 nothing extra).
   Quality 1 puts a site at the top of its range; quality 0 at the bottom.
3. PVC band sites are discounted by how many leak-signature properties
   their band shares (w, from five properties fixed in advance):
       p = w x graded p + (1 - w) x base rate.
   PVC sites without a band are graded like cast iron (user decision).

Confidence = the graded main-version (noise-floor) probability. Range =
min-max across the earlier analysis settings (two smoothing choices x
noise-floor and raw versions), with the same PVC discount, widened to
include the confidence itself.

Prediction (1/0): 1 if the site has a persistent band in the main version.

Outputs
-------
outputs/tables/model1_probabilities.csv
outputs/tables/model1_probability_summary.md
"""
import numpy as np
import pandas as pd

from src.config import (
    ANALYSIS_EXCLUDED_CLIPS,
    ANALYSIS_EXCLUSION_REASON,
    HISTORICAL_LEAK_RATE,
    INFORMATIVE_DIG_SITE,
    INTENSENESS_WEIGHT,
    LEAK_LABEL,
    MAX_DIGS_THIS_WEEK,
    MODEL1_PROBABILITY_PATH,
    MODEL1_PROBABILITY_SUMMARY_PATH,
    MODEL1_SCORES_PATH,
    NO_LEAK_LABEL,
    PERSISTENCE_CUTOFF,
    PERSISTENT_BAND_MIN_CONTRAST_DB,
    SIGNATURE_GAP_FRACTION,
    SIGNATURE_WIDTH_TOLERANCE_OCTAVES,
    SMOOTHING_PSEUDO_COUNTS,
    STEADINESS_WEIGHT,
    TABLE_DIR,
)
from src.measure_band_evenness import compute_band_evenness
from src.measure_band_geometry import measure_band_geometry

PRIOR_ODDS = HISTORICAL_LEAK_RATE / (1 - HISTORICAL_LEAK_RATE)
MAIN_VERSION = 'floor'
SIGNATURE_PROPERTIES = ('persistent_over_time', 'stands_out',
                        'even_across_band', 'leak_like_width',
                        'gap_above_background')


def has_persistent_band(scores):
    """Boolean Series: band present in >= 90% of frames and >= 3 dB."""
    return ((scores['persistence'] >= PERSISTENCE_CUTOFF)
            & (scores['contrast_db'] >= PERSISTENT_BAND_MIN_CONTRAST_DB))


def likelihood_ratios(dug_scores, pseudo_count):
    """(LR if band present, LR if band absent) from the dug sites."""
    band = has_persistent_band(dug_scores)
    is_leak = dug_scores['label_group'] == LEAK_LABEL
    is_no_leak = dug_scores['label_group'] == NO_LEAK_LABEL
    sensitivity = ((band & is_leak).sum() + pseudo_count) / (
        is_leak.sum() + 2 * pseudo_count)
    false_band_rate = ((band & is_no_leak).sum() + pseudo_count) / (
        is_no_leak.sum() + 2 * pseudo_count)
    return (sensitivity / false_band_rate,
            (1 - sensitivity) / (1 - false_band_rate))


def posterior(likelihood_ratio):
    """Probability after updating the 1-in-8 prior."""
    odds = PRIOR_ODDS * likelihood_ratio
    return odds / (1 + odds)


def _logit(probability):
    return np.log(probability / (1 - probability))


def _expit(log_odds):
    return 1 / (1 + np.exp(-log_odds))


def tier_ranges(lr_table):
    """(low, high) probability for the band (True) and no-band tiers."""
    band = [posterior(present) for present, _ in lr_table.values()]
    no_band = [posterior(absent) for _, absent in lr_table.values()]
    return {True: (min(band), max(band)),
            False: (min(no_band), max(no_band))}


def quality_scores(main, evenness):
    """Steadiness, intenseness and combined quality, relative to leaks."""
    table = main.merge(evenness[['clip_id', 'bin_coverage']], on='clip_id')
    leaks = table[(table['ranking'] == 'cast_iron')
                  & (table['label_group'] == LEAK_LABEL)]
    reference = {'persistence': leaks['persistence'].min(),
                 'coverage': leaks['bin_coverage'].min(),
                 'contrast_db': leaks['contrast_db'].min()}
    persistence_match = (table['persistence']
                         / reference['persistence']).clip(0, 1)
    coverage_match = (table['bin_coverage']
                      / reference['coverage']).clip(0, 1)
    table['steadiness'] = persistence_match * coverage_match
    table['intenseness'] = (table['contrast_db']
                            / reference['contrast_db']).clip(0, 1)
    table['quality'] = (table['steadiness'] ** STEADINESS_WEIGHT
                        * table['intenseness'] ** INTENSENESS_WEIGHT)
    return table, reference


def signature_match(table, geometry):
    """Share of five leak-signature properties each PVC band site shares."""
    leaks = geometry[geometry['group'] == 'cast-iron leak']
    median_width = leaks['band_width_octaves'].median()
    min_gap = leaks['gap_above_shelf_hz'].min()
    geometry = geometry.set_index('clip_id')
    rows = []
    is_pvc_band = (table['ranking'] == 'pvc') & table['has_band_main']
    for record in table[is_pvc_band].itertuples():
        if record.clip_id not in geometry.index:
            raise ValueError(f'No band geometry for {record.clip_id}; add it '
                             'to PVC_BAND_CLIPS in measure_band_geometry.')
        site = geometry.loc[record.clip_id]
        checks = {
            'persistent_over_time': record.persistence >= PERSISTENCE_CUTOFF,
            'stands_out': (record.contrast_db
                           >= PERSISTENT_BAND_MIN_CONTRAST_DB),
            'even_across_band': record.bin_coverage >= PERSISTENCE_CUTOFF,
            'leak_like_width': (abs(site['band_width_octaves']
                                    - median_width)
                                <= SIGNATURE_WIDTH_TOLERANCE_OCTAVES),
            'gap_above_background': (site['gap_above_shelf_hz']
                                     >= SIGNATURE_GAP_FRACTION * min_gap),
        }
        rows.append({'clip_id': record.clip_id, **checks,
                     'signature_weight': float(np.mean(
                         list(checks.values())))})
    return pd.DataFrame(rows, columns=['clip_id', *SIGNATURE_PROPERTIES,
                                       'signature_weight'])


def _discount(probability, weight):
    """Mix toward the base rate by (1 - weight)."""
    return weight * probability + (1 - weight) * HISTORICAL_LEAK_RATE


def setting_probabilities(scores, lr_table):
    """Two-tier probability under every (version, smoothing) setting."""
    rows = []
    for record in scores.itertuples():
        is_band = (record.persistence >= PERSISTENCE_CUTOFF
                   and record.contrast_db >= PERSISTENT_BAND_MIN_CONTRAST_DB)
        for pseudo_count, (lr_present, lr_absent) in lr_table.items():
            rows.append({'clip_id': record.clip_id,
                         'version': record.version,
                         'pseudo_count': pseudo_count,
                         'has_band': is_band,
                         'probability': posterior(
                             lr_present if is_band else lr_absent)})
    return pd.DataFrame(rows)


def build_summary(scores, lr_table, evenness, geometry):
    """One row per undug site: confidence, range, components, prediction."""
    main = scores[scores['version'] == MAIN_VERSION]
    table, reference = quality_scores(main, evenness)
    table['has_band_main'] = has_persistent_band(table)
    ranges = tier_ranges(lr_table)
    table['graded_probability'] = [
        _expit(_logit(ranges[band][0]) + quality
               * (_logit(ranges[band][1]) - _logit(ranges[band][0])))
        for band, quality in zip(table['has_band_main'], table['quality'])]
    weights = signature_match(table, geometry)
    table = table.merge(weights[['clip_id', 'signature_weight']],
                        on='clip_id', how='left')
    table['signature_weight'] = table['signature_weight'].fillna(1.0)
    table['confidence'] = _discount(table['graded_probability'],
                                    table['signature_weight'])
    settings = setting_probabilities(scores, lr_table).merge(
        table[['clip_id', 'signature_weight']], on='clip_id')
    settings['probability'] = _discount(settings['probability'],
                                        settings['signature_weight'])
    grouped = settings.groupby('clip_id')['probability']
    table = table.set_index('clip_id')
    table['probability_low'] = np.minimum(grouped.min(), table['confidence'])
    table['probability_high'] = np.maximum(grouped.max(),
                                           table['confidence'])
    table['has_band_any_version'] = settings.groupby('clip_id')[
        'has_band'].any()
    table['prediction'] = table['has_band_main'].astype(int)
    table = table[table['label_group'] == 'unlabeled'].reset_index()
    return table, reference, weights


def build_dig_list(summary):
    """Top persistent cast-iron sites by confidence, plus informative dig."""
    cast_iron = summary[(summary['ranking'] == 'cast_iron')
                        & summary['has_band_main']]
    picks = cast_iron.nlargest(MAX_DIGS_THIS_WEEK - 1,
                               'confidence')['clip_id'].tolist()
    return picks + [INFORMATIVE_DIG_SITE]


def write_summary(lr_table, summary, dig_list, reference, weights):
    """Plain-language Markdown summary."""
    lines = ['# Model 1 probabilities: presence first, then quality', '',
             f'- {ANALYSIS_EXCLUSION_REASON} No prediction for U40.',
             f'- Prior: 1 in 8 ({HISTORICAL_LEAK_RATE:.1%}).']
    for pseudo_count, (lr_present, lr_absent) in lr_table.items():
        lines.append(
            f'- Smoothing +{pseudo_count:g}: LR(band) = {lr_present:.1f} -> '
            f'{posterior(lr_present):.1%}; LR(no band) = {lr_absent:.2f} -> '
            f'{posterior(lr_absent):.1%}.')
    lines.append(
        f'- Quality = steadiness^{STEADINESS_WEIGHT} x intenseness^'
        f'{INTENSENESS_WEIGHT}. References (weakest dug leak): persistence '
        f'{reference["persistence"]:.3f}, coverage '
        f'{reference["coverage"]:.2f}, contrast '
        f'{reference["contrast_db"]:.1f} dB.')
    for record in weights.itertuples():
        shared = [name for name in SIGNATURE_PROPERTIES
                  if getattr(record, name)]
        lines.append(f'- PVC signature match {record.clip_id}: '
                     f'w = {record.signature_weight:.1f} (shares: '
                     f'{", ".join(shared)}).')
    expected = HISTORICAL_LEAK_RATE * len(summary)
    lines += [
        f'- Calibration: sum of confidence over {len(summary)} scorable '
        f'undug sites = {summary["confidence"].sum():.1f} vs. '
        f'{expected:.1f} expected at 1 in 8.',
        f'- Dig list: {dig_list} (last = informative dig).', '',
        '| Site | Pipe | Steadiness | Intenseness | Quality | w | '
        'Confidence | Range | Prediction |',
        '|---|---|---|---|---|---|---|---|---|']
    ordered = summary.sort_values('confidence', ascending=False)
    lines += [f'| {r.clip_id} | {r.ranking} | {r.steadiness:.2f} | '
              f'{r.intenseness:.2f} | {r.quality:.2f} | '
              f'{r.signature_weight:.1f} | {r.confidence:.1%} | '
              f'{r.probability_low:.1%}-{r.probability_high:.1%} | '
              f'{r.prediction} |' for r in ordered.itertuples()]
    MODEL1_PROBABILITY_SUMMARY_PATH.write_text('\n'.join(lines) + '\n',
                                               encoding='utf-8')


def run_model1_probabilities():
    """Compute confidence, ranges, predictions and dig list; save tables."""
    scores = pd.read_csv(MODEL1_SCORES_PATH)
    scores = scores[~scores['clip_id'].isin(ANALYSIS_EXCLUDED_CLIPS)]
    dug = scores[(scores['ranking'] == 'cast_iron')
                 & (scores['version'] == MAIN_VERSION)
                 & scores['label_group'].isin((LEAK_LABEL, NO_LEAK_LABEL))]
    lr_table = {pseudo_count: likelihood_ratios(dug, pseudo_count)
                for pseudo_count in SMOOTHING_PSEUDO_COUNTS}
    summary, reference, weights = build_summary(
        scores, lr_table, compute_band_evenness(), measure_band_geometry())
    dig_list = build_dig_list(summary)
    summary['dig_priority'] = summary['clip_id'].map(
        {clip_id: rank for rank, clip_id in enumerate(dig_list, start=1)})
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    summary.round(4).to_csv(MODEL1_PROBABILITY_PATH, index=False)
    write_summary(lr_table, summary, dig_list, reference, weights)
    return summary, dig_list


if __name__ == '__main__':
    run_model1_probabilities()
    print(MODEL1_PROBABILITY_SUMMARY_PATH.read_text(encoding='utf-8'))
