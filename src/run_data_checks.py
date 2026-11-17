"""Run the ordered data-quality and inference checks for the pilot data.

Checks follow the total-error framing in Foster et al., "Big Data and
Social Science," ch. 10 (row/coverage, column/specification, and
cell/measurement error) and the sample-bias and confounding concerns of
ch. 11. They run in order of importance: problems early in the list would
invalidate everything after them.

Outputs
-------
outputs/tables/data_checks.csv   one row per check (status, finding)
outputs/tables/data_checks.md    the same, as a readable table
outputs/tables/coverage_labeled_vs_unlabeled.csv
"""
import hashlib
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from src.compute_clip_features import (
    FEATURE_COLUMNS,
    build_clip_feature_table,
    find_longest_constant_run,
)
from src.config import (
    AMPLITUDE_LIMIT,
    CLIPPING_THRESHOLD,
    COVERAGE_TABLE_PATH,
    CSV_WAV_TOLERANCE,
    DATA_CHECKS_CSV_PATH,
    DATA_CHECKS_MARKDOWN_PATH,
    DUPLICATE_CORRELATION_THRESHOLD,
    HISTORICAL_LEAK_RATE,
    LABELED_CLIP_DIR,
    LOGGER_MODELS,
    MAX_ALLOWED_FLATLINE_SAMPLES,
    MORNING_COMMUTE_START_HOUR,
    N_CLIPS_EXPECTED,
    N_LABELED_EXPECTED,
    N_UNLABELED_EXPECTED,
    NEAR_SILENT_RMS,
    OUTLIER_ROBUST_Z,
    PIPE_MATERIALS,
    SAMPLE_RATE_HZ,
    SAMPLES_PER_CLIP,
    TABLE_DIR,
    TIME_GRID_TOLERANCE_SECONDS,
    UNLABELED_CLIP_DIR,
    VALID_LABELS,
)
from src.load_recordings import (
    load_clip_csv,
    load_clip_wav,
    load_metadata,
)

PASS, FLAG, INFO = 'pass', 'FLAG', 'info'
# Scales the median absolute deviation to a normal-equivalent SD.
MAD_TO_SD = 1.4826
REDUNDANT_CORRELATION = 0.9
# Features used for group-difference checks (device, pipe, clock time).
KEY_FEATURES = ('rms_loudness', 'loudness_variability', 'transient_share',
                'spectral_flatness', 'spectral_centroid_hz')


def make_check_result(check, error_type, status, finding):
    """Package one check's outcome."""
    return {'check': check, 'error_type': error_type, 'status': status,
            'finding': finding}


# ---------------------------------------------------------------------------
# 1-4: Does every record exist, link, and mean what we think it means?
# ---------------------------------------------------------------------------
def check_file_inventory(metadata):
    """Every metadata row has a CSV and WAV in the right folder, no extras."""
    expected_files = {
        (clip_dir / f'{clip_id}{suffix}').name
        for clip_id, is_labeled in zip(metadata['clip_id'],
                                       metadata['is_labeled'])
        for clip_dir in [LABELED_CLIP_DIR if is_labeled
                         else UNLABELED_CLIP_DIR]
        for suffix in ('.csv', '.wav')
    }
    found_files = {path.name
                   for clip_dir in (LABELED_CLIP_DIR, UNLABELED_CLIP_DIR)
                   for path in clip_dir.iterdir()}
    missing = sorted(expected_files - found_files)
    extra = sorted(found_files - expected_files)
    is_ok = (not missing and not extra
             and len(metadata) == N_CLIPS_EXPECTED)
    return make_check_result(
        'File inventory and linkage', 'Row (coverage)',
        PASS if is_ok else FLAG,
        f'{len(metadata)} metadata rows; {len(found_files)} clip files; '
        f'missing: {missing or "none"}; unmatched files: {extra or "none"}.')


def check_unique_identifiers(metadata):
    """clip_id and site_id are unique; L/U prefix matches labeled flag."""
    n_dup_clips = metadata['clip_id'].duplicated().sum()
    n_dup_sites = metadata['site_id'].duplicated().sum()
    prefix_matches = (metadata['clip_id'].str[0].eq('L')
                      == metadata['is_labeled']).all()
    is_ok = n_dup_clips == 0 and n_dup_sites == 0 and prefix_matches
    return make_check_result(
        'Unique identifiers (one recording per site)', 'Row (coverage)',
        PASS if is_ok else FLAG,
        f'Duplicate clip_id: {n_dup_clips}; duplicate site_id: '
        f'{n_dup_sites}; L/U prefix consistent with labeled flag: '
        f'{prefix_matches}.')


def check_label_integrity(metadata):
    """Labeled rows carry a valid label; unlabeled rows carry none."""
    labeled = metadata[metadata['is_labeled']]
    unlabeled = metadata[~metadata['is_labeled']]
    has_valid_labels = labeled['label'].isin(VALID_LABELS).all()
    unlabeled_blank = unlabeled['label'].eq('').all()
    counts_ok = (len(labeled) == N_LABELED_EXPECTED
                 and len(unlabeled) == N_UNLABELED_EXPECTED)
    is_ok = has_valid_labels and unlabeled_blank and counts_ok
    return make_check_result(
        'Label integrity', 'Column (specification)',
        PASS if is_ok else FLAG,
        f'{len(labeled)} labeled ({labeled["label"].value_counts().to_dict()})'
        f', {len(unlabeled)} unlabeled; all unlabeled labels blank: '
        f'{unlabeled_blank}.')


def check_metadata_domains(metadata):
    """Categorical fields use the documented values; times parse."""
    bad_logger = ~metadata['logger_model'].isin(LOGGER_MODELS)
    bad_pipe = ~metadata['pipe_material'].isin(PIPE_MATERIALS)
    bad_time = metadata['overnight_hour'].isna()
    earliest = metadata.loc[metadata['overnight_hour'].idxmin(),
                            'recorded_at']
    latest = metadata.loc[metadata['overnight_hour'].idxmax(),
                          'recorded_at']
    is_ok = not (bad_logger.any() or bad_pipe.any() or bad_time.any())
    return make_check_result(
        'Metadata value domains', 'Column (specification)',
        PASS if is_ok else FLAG,
        f'Invalid logger: {bad_logger.sum()}; invalid pipe: '
        f'{bad_pipe.sum()}; unparseable times: {bad_time.sum()}. '
        f'Recordings span {earliest} to {latest} (all overnight).')


# ---------------------------------------------------------------------------
# 5-11: Is each recording a faithful measurement?
# ---------------------------------------------------------------------------
def check_signal_shape(clip_tables):
    """Each CSV has the expected columns, length, and no missing values."""
    problems = [
        clip_id for clip_id, df in clip_tables.items()
        if list(df.columns) != ['time_s', 'amplitude']
        or len(df) != SAMPLES_PER_CLIP
        or not np.isfinite(df.to_numpy()).all()
    ]
    return make_check_result(
        'Signal length, columns, missing values', 'Cell (measurement)',
        PASS if not problems else FLAG,
        f'{len(clip_tables) - len(problems)} of {len(clip_tables)} clips '
        f'have {SAMPLES_PER_CLIP:,} finite rows with columns time_s, '
        f'amplitude. Problems: {problems or "none"}.')


def check_time_grid(clip_tables):
    """time_s starts at 0 and steps by exactly 1 / sample rate."""
    expected_times = np.arange(SAMPLES_PER_CLIP) / SAMPLE_RATE_HZ
    worst_gap = {
        clip_id: np.max(np.abs(df['time_s'].to_numpy() - expected_times))
        for clip_id, df in clip_tables.items()
    }
    bad = [clip_id for clip_id, gap in worst_gap.items()
           if gap > TIME_GRID_TOLERANCE_SECONDS]
    return make_check_result(
        'Uniform time grid at 4,000 Hz', 'Cell (measurement)',
        PASS if not bad else FLAG,
        f'Largest deviation from the ideal grid: '
        f'{max(worst_gap.values()):.2e} s. Clips off-grid: {bad or "none"}.')


def check_amplitude_range(amplitudes):
    """All amplitudes lie within [-1, 1]."""
    out_of_range = [clip_id for clip_id, amplitude in amplitudes.items()
                    if np.max(np.abs(amplitude)) > AMPLITUDE_LIMIT]
    overall_peak = max(np.max(np.abs(a)) for a in amplitudes.values())
    return make_check_result(
        'Amplitude within [-1, 1]', 'Cell (measurement)',
        PASS if not out_of_range else FLAG,
        f'Largest |amplitude| across all clips: {overall_peak:.3f}. '
        f'Out of range: {out_of_range or "none"}.')


def check_csv_matches_wav(metadata, amplitudes):
    """CSV and WAV versions of each clip contain the same signal."""
    worst_difference = {}
    bad_rates = []
    for clip_id, is_labeled in zip(metadata['clip_id'],
                                   metadata['is_labeled']):
        sample_rate_hz, wav_amplitude = load_clip_wav(clip_id, is_labeled)
        if sample_rate_hz != SAMPLE_RATE_HZ:
            bad_rates.append(clip_id)
        worst_difference[clip_id] = np.max(
            np.abs(wav_amplitude - amplitudes[clip_id]))
    mismatched = [clip_id for clip_id, diff in worst_difference.items()
                  if diff > CSV_WAV_TOLERANCE]
    is_ok = not mismatched and not bad_rates
    return make_check_result(
        'CSV and WAV agree', 'Cell (measurement)',
        PASS if is_ok else FLAG,
        f'Largest CSV-WAV difference: {max(worst_difference.values()):.1e}'
        f' (tolerance {CSV_WAV_TOLERANCE:.0e}). WAV sample-rate '
        f'mismatches: {bad_rates or "none"}; mismatched clips: '
        f'{mismatched or "none"}.')


def check_clipping(amplitudes):
    """No samples pinned at full scale (sensor overload)."""
    clipped_counts = {clip_id: int(np.sum(np.abs(a) >= CLIPPING_THRESHOLD))
                      for clip_id, a in amplitudes.items()}
    clipped = {clip_id: f'{count:,} samples ({count / SAMPLES_PER_CLIP:.0%})'
               for clip_id, count in clipped_counts.items() if count > 0}
    return make_check_result(
        'Clipping (sensor overload)', 'Cell (measurement)',
        PASS if not clipped else FLAG,
        f'Clips with samples at |amplitude| >= {CLIPPING_THRESHOLD} (the '
        f'effective sensor ceiling): {clipped or "none"}. Clipped clips '
        f'understate true loudness and distort pitch content.')


def check_dropouts(amplitudes):
    """No flatlined stretches that would indicate a dropped signal."""
    longest_runs = {clip_id: find_longest_constant_run(a)[1]
                    for clip_id, a in amplitudes.items()}
    flagged = {k: v for k, v in longest_runs.items()
               if v > MAX_ALLOWED_FLATLINE_SAMPLES}
    return make_check_result(
        'Dropouts / flatlined segments', 'Cell (measurement)',
        PASS if not flagged else FLAG,
        f'Longest run of identical samples in any clip: '
        f'{max(longest_runs.values())} samples. Clips over '
        f'{MAX_ALLOWED_FLATLINE_SAMPLES}: {flagged or "none"}.')


def check_offset_and_silence(amplitudes, clip_features):
    """No large DC offset and no near-silent (dead) recordings."""
    means = {clip_id: float(np.mean(a))
             for clip_id, a in amplitudes.items()}
    worst_offset_clip = max(means, key=lambda k: abs(means[k]))
    silent = clip_features.loc[
        clip_features['rms_loudness'] < NEAR_SILENT_RMS, 'clip_id'].tolist()
    return make_check_result(
        'DC offset and dead recordings', 'Cell (measurement)',
        PASS if not silent else FLAG,
        f'Largest mean offset: {means[worst_offset_clip]:+.4f} '
        f'({worst_offset_clip}). RMS range '
        f'{clip_features["rms_loudness"].min():.4f}-'
        f'{clip_features["rms_loudness"].max():.4f}; near-silent clips: '
        f'{silent or "none"}.')


def check_duplicate_recordings(amplitudes):
    """No recording is an exact or near copy of another."""
    digests = {clip_id: hashlib.sha256(a.tobytes()).hexdigest()
               for clip_id, a in amplitudes.items()}
    n_exact_duplicates = len(digests) - len(set(digests.values()))
    pair_correlations = {
        (first, second): np.corrcoef(amplitudes[first],
                                     amplitudes[second])[0, 1]
        for first, second in combinations(sorted(amplitudes), 2)
    }
    most_similar_pair = max(pair_correlations,
                            key=lambda pair: abs(pair_correlations[pair]))
    near_duplicates = [pair for pair, r in pair_correlations.items()
                       if abs(r) >= DUPLICATE_CORRELATION_THRESHOLD]
    is_ok = n_exact_duplicates == 0 and not near_duplicates
    return make_check_result(
        'Duplicate or copied recordings', 'Row (coverage)',
        PASS if is_ok else FLAG,
        f'Exact duplicates: {n_exact_duplicates}. Most similar pair '
        f'{most_similar_pair}: r = '
        f'{pair_correlations[most_similar_pair]:+.3f}. Near-duplicates '
        f'(|r| >= {DUPLICATE_CORRELATION_THRESHOLD}): '
        f'{near_duplicates or "none"}.')


# ---------------------------------------------------------------------------
# 12-15: Can the labeled sites stand in for the unlabeled ones?
# ---------------------------------------------------------------------------
def check_label_balance(metadata):
    """Compare the labeled leak share with the historical base rate."""
    labeled = metadata[metadata['is_labeled']]
    leak_share = labeled['is_leak'].mean()
    expected_leaks = HISTORICAL_LEAK_RATE * N_UNLABELED_EXPECTED
    return make_check_result(
        'Label balance vs. historical base rate', 'Row (coverage)', FLAG,
        f'Labeled sample is {leak_share:.0%} leaks ({labeled["is_leak"].sum()}'
        f' of {len(labeled)}) vs. ~{HISTORICAL_LEAK_RATE:.1%} historically. '
        f'Dug sites were not a random sample, so raw model probabilities '
        f'must be re-anchored to the base rate. Expect ~{expected_leaks:.0f}'
        f' leaks among the 40 undug sites.')


def build_coverage_table(metadata):
    """Counts of labeled vs. unlabeled clips by logger, pipe, and timing."""
    timing = np.where(metadata['is_commute_hours'],
                      'after commute start', 'before commute start')
    pieces = []
    for factor_name, factor_values in [
            ('logger_model', metadata['logger_model']),
            ('pipe_material', metadata['pipe_material']),
            ('recording_timing', timing)]:
        counts = pd.crosstab(factor_values, metadata['label_group'])
        counts.index.name = 'level'
        counts.insert(0, 'factor', factor_name)
        pieces.append(counts.reset_index())
    return pd.concat(pieces, ignore_index=True).fillna(0)


def check_coverage(metadata):
    """Every kind of unlabeled site has labeled counterparts."""
    labeled = metadata[metadata['is_labeled']]
    unlabeled = metadata[~metadata['is_labeled']]
    uncovered = [
        f'{column}={level} ({(unlabeled[column] == level).sum()} undug)'
        for column in ('logger_model', 'pipe_material')
        for level in unlabeled[column].unique()
        if level not in set(labeled[column])
    ]
    return make_check_result(
        'Coverage: labeled vs. unlabeled sites', 'Row (coverage)',
        FLAG if uncovered else PASS,
        f'Levels present among undug sites but absent from dug sites: '
        f'{uncovered or "none"}. Predictions for these sites are '
        f'extrapolations (sample bias, ch. 11).')


def check_label_confounding(metadata):
    """Is leak status entangled with clock time or logger model?"""
    labeled = metadata[metadata['is_labeled']].sort_values('overnight_hour')
    leak_hours = labeled.loc[labeled['is_leak'].astype(bool),
                             'recorded_at'].tolist()
    no_leak_hours = labeled.loc[~labeled['is_leak'].astype(bool),
                                'recorded_at'].tolist()
    latest_leak = labeled.loc[labeled['is_leak'].astype(bool),
                              'overnight_hour'].max()
    earliest_no_leak = labeled.loc[~labeled['is_leak'].astype(bool),
                                   'overnight_hour'].min()
    is_time_separated = latest_leak < earliest_no_leak
    logger_table = pd.crosstab(labeled['logger_model'],
                               labeled['label']).to_dict()
    n_commute_unlabeled = metadata.loc[~metadata['is_labeled'],
                                       'is_commute_hours'].sum()
    return make_check_result(
        'Label confounded with recording time or logger',
        'Column (specification)',
        FLAG if is_time_separated else PASS,
        f'Leak clips recorded at {leak_hours}; no-leak clips at '
        f'{no_leak_hours}. Clock time alone perfectly separates the labels: '
        f'{is_time_separated}. Logger x label: {logger_table}. '
        f'{n_commute_unlabeled} undug clips were recorded at/after '
        f'{MORNING_COMMUTE_START_HOUR:.0f}:00.')


# ---------------------------------------------------------------------------
# 16-20: Do features behave sensibly, and what besides leaks moves them?
# ---------------------------------------------------------------------------
def _robust_z(values):
    """Median/MAD z-scores (resistant to the outliers being sought).

    Falls back to the standard deviation when over half the values are
    identical (MAD = 0), e.g., the many clips with no loud bursts at all.
    """
    median = np.nanmedian(values)
    spread = np.nanmedian(np.abs(values - median)) * MAD_TO_SD
    if spread == 0:
        spread = np.nanstd(values)
    return (values - median) / spread


def check_feature_outliers(clip_features):
    """Flag clips that are extreme on any feature."""
    flagged = {}
    for feature in FEATURE_COLUMNS:
        z_scores = _robust_z(clip_features[feature].to_numpy())
        for clip_id, z in zip(clip_features['clip_id'], z_scores):
            if abs(z) > OUTLIER_ROBUST_Z:
                flagged.setdefault(clip_id, []).append(feature)
    return make_check_result(
        'Feature outliers (robust z > 3.5)', 'Cell (measurement)',
        FLAG if flagged else PASS,
        f'{len(flagged)} clips extreme on at least one feature: '
        f'{flagged or "none"}. Listen to these before modeling.')


def check_feature_redundancy(clip_features):
    """Highly correlated feature pairs; features vs. labeled sample size."""
    correlations = clip_features[FEATURE_COLUMNS].corr(method='spearman')
    redundant_pairs = [
        f'{first}~{second} ({correlations.loc[first, second]:+.2f})'
        for first, second in combinations(FEATURE_COLUMNS, 2)
        if abs(correlations.loc[first, second]) >= REDUNDANT_CORRELATION
    ]
    return make_check_result(
        'Feature redundancy and dimensionality', 'Column (specification)',
        INFO,
        f'{len(FEATURE_COLUMNS)} candidate features vs. '
        f'{N_LABELED_EXPECTED} labeled clips: any model must use very few '
        f'features. Pairs with |Spearman rho| >= {REDUNDANT_CORRELATION}: '
        f'{redundant_pairs or "none"}.')


def _group_difference(df, group_column, first_level, second_level):
    """Mann-Whitney p and median ratio for each key feature."""
    summaries = []
    for feature in KEY_FEATURES:
        first = df.loc[df[group_column] == first_level, feature]
        second = df.loc[df[group_column] == second_level, feature]
        p_value = stats.mannwhitneyu(first, second).pvalue
        summaries.append(
            f'{feature}: median {first.median():.3g} vs '
            f'{second.median():.3g} (p={p_value:.2f})')
    return summaries


def check_logger_effect(clip_features):
    """Do logger models A and B record systematically different sound?"""
    unlabeled = clip_features[~clip_features['is_labeled']]
    summaries = _group_difference(unlabeled, 'logger_model', 'A', 'B')
    return make_check_result(
        'Device effect: logger A vs. B (undug sites)',
        'Column (specification)', INFO,
        'A vs B. ' + '; '.join(summaries) + '. Small p suggests the device'
        ' changes the measurement, so features may need per-logger '
        'standardization.')


def check_pipe_effect(clip_features):
    """Does pipe material change the recorded sound?"""
    unlabeled = clip_features[~clip_features['is_labeled']]
    summaries = _group_difference(unlabeled, 'pipe_material',
                                  'cast_iron', 'PVC')
    return make_check_result(
        'Pipe-material effect: cast iron vs. PVC (undug sites)',
        'Column (specification)', INFO,
        'Cast iron vs PVC. ' + '; '.join(summaries) + '.')


def check_clock_time_effect(clip_features):
    """Do traffic-like features rise toward the morning commute?"""
    summaries = []
    for feature in KEY_FEATURES:
        rho, p_value = stats.spearmanr(clip_features['overnight_hour'],
                                       clip_features[feature])
        summaries.append(f'{feature}: rho={rho:+.2f} (p={p_value:.2f})')
    return make_check_result(
        'Clock-time effect on features (all 46 clips)',
        'Column (specification)', INFO,
        'Spearman correlation with recording time. '
        + '; '.join(summaries) + '.')


def write_check_tables(check_results, csv_path=DATA_CHECKS_CSV_PATH,
                       markdown_path=DATA_CHECKS_MARKDOWN_PATH):
    """Save ordered check results as CSV and as a Markdown table.

    Any keys beyond the standard four (e.g., 'section') become extra
    leading columns.
    """
    results = pd.DataFrame(check_results)
    results.insert(0, 'order', range(1, len(results) + 1))
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(csv_path, index=False)
    columns = list(results.columns)
    markdown_lines = [
        '| ' + ' | '.join(column.replace('_', ' ').capitalize()
                          for column in columns) + ' |',
        '|' + '---|' * len(columns)]
    markdown_lines += [
        '| ' + ' | '.join(str(value).replace('|', '/') for value in row)
        + ' |' for row in results.itertuples(index=False)]
    markdown_path.write_text('\n'.join(markdown_lines) + '\n',
                             encoding='utf-8')
    return results


def run_all_checks():
    """Run every check in order of importance and save the results."""
    metadata = load_metadata()
    clip_tables = {
        clip_id: load_clip_csv(clip_id, is_labeled)
        for clip_id, is_labeled in zip(metadata['clip_id'],
                                       metadata['is_labeled'])
    }
    amplitudes = {clip_id: df['amplitude'].to_numpy()
                  for clip_id, df in clip_tables.items()}
    clip_features = build_clip_feature_table(metadata, amplitudes)
    check_results = [
        check_file_inventory(metadata),
        check_unique_identifiers(metadata),
        check_label_integrity(metadata),
        check_metadata_domains(metadata),
        check_signal_shape(clip_tables),
        check_time_grid(clip_tables),
        check_amplitude_range(amplitudes),
        check_csv_matches_wav(metadata, amplitudes),
        check_clipping(amplitudes),
        check_dropouts(amplitudes),
        check_offset_and_silence(amplitudes, clip_features),
        check_duplicate_recordings(amplitudes),
        check_label_balance(metadata),
        check_coverage(metadata),
        check_label_confounding(metadata),
        check_feature_outliers(clip_features),
        check_feature_redundancy(clip_features),
        check_logger_effect(clip_features),
        check_pipe_effect(clip_features),
        check_clock_time_effect(clip_features),
    ]
    build_coverage_table(metadata).to_csv(COVERAGE_TABLE_PATH, index=False)
    return write_check_tables(check_results)


if __name__ == '__main__':
    results = run_all_checks()
    with pd.option_context('display.max_colwidth', None,
                           'display.width', 200):
        print(results[['order', 'check', 'status', 'finding']])
