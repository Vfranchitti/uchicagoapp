"""Model 1: a knowledge-based leak score (a rule we write, not learn).

Cast-iron sites
    score = leak-band contrast x persistence gate
    - leak-band contrast: median power in the leak band minus the average
      of its two neighboring bands (dB). Band edges come from the three dug
      leaks (the only use of labels).
    - persistence: share of 64 ms frames in which the band stands out at
      least PERSISTENCE_CONTRAST_DB above its neighbors ("it doesn't stop").
    - gate: score is multiplied by min(1, persistence / PERSISTENCE_CUTOFF).
PVC sites (separate ranking; unconfirmed signature, low confidence)
    the same construction on the fixed 250-400 Hz band against 0-250 Hz and
    400-1,300 Hz, after removing 120 Hz hum lines.

Two versions: 'floor' (spectra minus each logger's noise floor; main) and
'raw' (spectra as recorded; robustness check). U40 is excluded.

Checks: 9-pair ranking check on the dug sites; leave-one-out over all six
dug sites; agreement with the clustering; floor vs. raw.

Outputs
-------
outputs/tables/model1_scores.csv
outputs/tables/model1_leave_one_out.csv
outputs/tables/model1_summary.md
outputs/figures/model/model1_cast_iron_ranking.png
outputs/figures/model/model1_pvc_ranking.png
"""
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy import signal, stats

from src.analyze_noise_floor import (
    compute_clip_spectra_db,
    compute_excess_spectra,
    compute_logger_baselines,
    identify_floor_clips,
)
from src.compute_clip_features import (
    build_clip_feature_table,
    remove_hum_lines,
)
from src.config import (
    ANALYSIS_EXCLUDED_CLIPS,
    ANALYSIS_EXCLUSION_REASON,
    CLUSTER_TABLE_PATH,
    EXCESS_BANDS_HZ,
    FIGURE_DPI,
    LEAK_LABEL,
    MAX_DIGS_THIS_WEEK,
    MODEL1_LOO_PATH,
    MODEL1_SCORES_PATH,
    MODEL1_SUMMARY_PATH,
    MODEL_FIGURE_DIR,
    NO_LEAK_LABEL,
    NYQUIST_HZ,
    PERSISTENCE_CONTRAST_DB,
    PERSISTENCE_CUTOFF,
    PVC_BAND_KEY,
    PVC_NEIGHBOR_KEYS,
    SAMPLE_RATE_HZ,
    SPECTROGRAM_OVERLAP_SAMPLES,
    SPECTROGRAM_SEGMENT_SAMPLES,
    TABLE_DIR,
)
from src.load_recordings import load_all_amplitudes, load_metadata
from src.measure_band_geometry import find_band
from src.plot_style import (
    HEALTH_LAB_NAVY,
    LABEL_COLORS,
    TEXT_DARK,
    TEXT_GREY,
    add_title_and_note,
    apply_urban_labs_style,
)

POWER_TO_DB = 10
VERSIONS = ('floor', 'raw')
DUG_LABELS = (LEAK_LABEL, NO_LEAK_LABEL)
# Lower neighbor of the cast-iron band starts above the logger's low shelf.
CAST_IRON_LOWER_NEIGHBOR_START_HZ = 250
GROUP_COLORS = {**LABEL_COLORS, 'unlabeled': HEALTH_LAB_NAVY}
CAST_IRON_FIGURE = (7.5, 8.6)
PVC_FIGURE = (7.5, 4.2)


# ---------------------------------------------------------------------------
# Spectra
# ---------------------------------------------------------------------------
def _to_db(power):
    return POWER_TO_DB * np.log10(power + np.finfo(float).tiny)


def compute_spectrogram_db(amplitude):
    """(frequencies_hz, dB array of shape frequency x frame)."""
    frequencies_hz, _, power = signal.spectrogram(
        amplitude, fs=SAMPLE_RATE_HZ, nperseg=SPECTROGRAM_SEGMENT_SAMPLES,
        noverlap=SPECTROGRAM_OVERLAP_SAMPLES)
    return frequencies_hz, _to_db(power)


def build_spectral_inputs(metadata, amplitudes):
    """Whole-clip and frame-level spectra for both versions."""
    clip_features = build_clip_feature_table(metadata, amplitudes)
    welch_frequencies_hz, welch_db = compute_clip_spectra_db(metadata,
                                                             amplitudes)
    floor_clip_ids = identify_floor_clips(clip_features)
    welch_baselines = compute_logger_baselines(metadata, welch_db,
                                               floor_clip_ids)
    welch_excess = compute_excess_spectra(metadata, welch_db,
                                          welch_baselines)
    frames = {clip_id: compute_spectrogram_db(amplitudes[clip_id])
              for clip_id in metadata['clip_id']}
    frame_frequencies_hz = next(iter(frames.values()))[0]
    logger_by_clip = metadata.set_index('clip_id')['logger_model']
    frame_baselines = {
        logger: np.median(np.hstack([
            frames[clip_id][1] for clip_id in floor_clip_ids
            if logger_by_clip[clip_id] == logger]), axis=1)
        for logger in logger_by_clip.unique()}
    frame_db = {
        'raw': {clip_id: values for clip_id, (_, values) in frames.items()},
        'floor': {clip_id: values
                  - frame_baselines[logger_by_clip[clip_id]][:, None]
                  for clip_id, (_, values) in frames.items()},
    }
    clip_db = {'raw': welch_db, 'floor': welch_excess}
    return {'clip_frequencies_hz': welch_frequencies_hz,
            'frame_frequencies_hz': frame_frequencies_hz,
            'clip_db': clip_db, 'frame_db': frame_db,
            'welch_excess': welch_excess}


# ---------------------------------------------------------------------------
# Rule
# ---------------------------------------------------------------------------
def _in_band(frequencies_hz, band_hz):
    low_hz, high_hz = band_hz
    return (frequencies_hz > low_hz) & (frequencies_hz <= high_hz)


def band_contrast(values, frequencies_hz, band_hz, neighbors_hz):
    """Median dB in band minus the mean of the neighbors' medians.

    values may be a 1-D spectrum or a frequency x frame array (returns one
    contrast per frame).
    """
    band = np.median(values[_in_band(frequencies_hz, band_hz)], axis=0)
    neighbors = [np.median(values[_in_band(frequencies_hz, edges)], axis=0)
                 for edges in neighbors_hz]
    return band - np.mean(neighbors, axis=0)


def derive_leak_band(welch_excess, leak_ids):
    """Band edges from the median spectrum of the given dug leaks."""
    median_leak = welch_excess.loc[list(leak_ids)].median(axis=0)
    low_hz, high_hz, _ = find_band(median_leak)
    return low_hz, high_hz


def cast_iron_bands(leak_band_hz):
    """Leak band and its two neighbors for the cast-iron rule."""
    low_hz, high_hz = leak_band_hz
    return leak_band_hz, ((CAST_IRON_LOWER_NEIGHBOR_START_HZ, low_hz),
                          (high_hz, NYQUIST_HZ))


def pvc_bands():
    """Fixed PVC band and neighbors (agreed in advance; no labels used)."""
    return (EXCESS_BANDS_HZ[PVC_BAND_KEY],
            tuple(EXCESS_BANDS_HZ[key] for key in PVC_NEIGHBOR_KEYS))


def score_site(clip_id, inputs, version, band_hz, neighbors_hz,
               is_hum_removed):
    """Contrast, persistence, gate factor and score for one recording."""
    clip_frequencies_hz = inputs['clip_frequencies_hz']
    frame_frequencies_hz = inputs['frame_frequencies_hz']
    spectrum = inputs['clip_db'][version].loc[clip_id].to_numpy()
    frames = inputs['frame_db'][version][clip_id]
    if is_hum_removed:
        spectrum = remove_hum_lines(clip_frequencies_hz, spectrum)
        frames = remove_hum_lines(frame_frequencies_hz, frames)
    contrast = float(band_contrast(spectrum, clip_frequencies_hz, band_hz,
                                   neighbors_hz))
    frame_contrast = band_contrast(frames, frame_frequencies_hz, band_hz,
                                   neighbors_hz)
    persistence = float(np.mean(frame_contrast >= PERSISTENCE_CONTRAST_DB))
    gate = min(1.0, persistence / PERSISTENCE_CUTOFF)
    return {'contrast_db': contrast, 'persistence': persistence,
            'gate_factor': gate, 'score': contrast * gate}


def score_group(metadata, inputs, version, band_hz, neighbors_hz,
                is_hum_removed, ranking):
    """Score every site in one ranking (cast iron or PVC)."""
    rows = [{'clip_id': record.clip_id, 'ranking': ranking,
             'version': version, 'label_group': record.label_group,
             'logger_model': record.logger_model,
             'pipe_material': record.pipe_material,
             **score_site(record.clip_id, inputs, version, band_hz,
                          neighbors_hz, is_hum_removed)}
            for record in metadata.itertuples()]
    scores = pd.DataFrame(rows)
    is_undug = scores['label_group'] == 'unlabeled'
    scores['rank_among_undug'] = np.nan
    scores.loc[is_undug, 'rank_among_undug'] = scores.loc[
        is_undug, 'score'].rank(ascending=False, method='min')
    return scores


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def ranking_check(scores):
    """Share of the 9 leak/non-leak pairs where the leak scores higher."""
    leaks = scores.loc[scores['label_group'] == LEAK_LABEL, 'score']
    non_leaks = scores.loc[scores['label_group'] == NO_LEAK_LABEL, 'score']
    pairs = list(product(leaks, non_leaks))
    return sum(leak > non_leak for leak, non_leak in pairs), len(pairs)


def leave_one_out(metadata, inputs, cast_iron_sites):
    """Cycle through the 6 dug sites, rebuilding the band without each.

    Returns (held-out table, undug scores per fold, full score table per
    fold keyed by the held-out site).
    """
    leak_ids = metadata.loc[metadata['label_group'] == LEAK_LABEL,
                            'clip_id'].tolist()
    dug_ids = metadata.loc[metadata['label_group'].isin(DUG_LABELS),
                           'clip_id'].tolist()
    held_out_rows, fold_scores, fold_tables = [], [], {}
    for held_out in dug_ids:
        leaks_used = [clip_id for clip_id in leak_ids if clip_id != held_out]
        band_hz, neighbors_hz = cast_iron_bands(
            derive_leak_band(inputs['welch_excess'], leaks_used))
        scores = score_group(cast_iron_sites, inputs, 'floor', band_hz,
                             neighbors_hz, False, 'cast_iron')
        fold_tables[held_out] = scores
        undug = scores[scores['label_group'] == 'unlabeled']
        held = scores.set_index('clip_id').loc[held_out]
        held_out_rows.append({
            'held_out': held_out, 'label_group': held['label_group'],
            'band_low_hz': band_hz[0], 'band_high_hz': band_hz[1],
            'held_out_score': held['score'],
            'undug_scoring_higher': int((undug['score']
                                         > held['score']).sum()),
        })
        fold_scores.append(undug.set_index('clip_id')['score']
                           .rename(held_out))
    return (pd.DataFrame(held_out_rows), pd.concat(fold_scores, axis=1),
            fold_tables)


def leak_cluster_members(metadata):
    """Undug clips sharing a cluster with most dug leaks, per setting."""
    assignments = pd.read_csv(CLUSTER_TABLE_PATH)
    assignments = assignments[assignments['run'] == 'all'].merge(
        metadata[['clip_id', 'label_group']], on='clip_id')
    members = {}
    for column in [c for c in assignments.columns
                   if c.startswith(('kmeans_', 'ward_'))]:
        majority = assignments.loc[assignments['label_group']
                                   == LEAK_LABEL, column].mode().iloc[0]
        members[column] = set(assignments.loc[
            (assignments[column] == majority)
            & (assignments['label_group'] == 'unlabeled'), 'clip_id'])
    return members


def top_undug(scores, n_sites=MAX_DIGS_THIS_WEEK):
    """Highest-scoring undug sites, best first."""
    undug = scores[scores['label_group'] == 'unlabeled']
    return undug.nlargest(n_sites, 'score')['clip_id'].tolist()


def compare_versions(scores, ranking):
    """Rank agreement and top-5 overlap between floor and raw versions."""
    subset = scores[(scores['ranking'] == ranking)
                    & (scores['label_group'] == 'unlabeled')]
    wide = subset.pivot(index='clip_id', columns='version', values='score')
    rho = stats.spearmanr(wide['floor'], wide['raw']).statistic
    top = {version: top_undug(subset[subset['version'] == version])
           for version in VERSIONS}
    return rho, top


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_ranking(scores, fold_spread, title, figure_number, note, file_stem,
                 figsize):
    """Horizontal bars: contrast before gate (light) and score (solid)."""
    apply_urban_labs_style()
    ordered = scores.sort_values('score')
    fig, ax = plt.subplots(figsize=figsize)
    positions = np.arange(len(ordered))
    colors = [GROUP_COLORS[group] for group in ordered['label_group']]
    ax.barh(positions, ordered['contrast_db'], color=colors, alpha=0.25,
            height=0.75)
    ax.barh(positions, ordered['score'], color=colors, height=0.75)
    if fold_spread is not None:
        for position, clip_id in zip(positions, ordered['clip_id']):
            if clip_id in fold_spread.index:
                low, high = fold_spread.loc[clip_id, ['min', 'max']]
                ax.plot([low, high], [position, position], color=TEXT_DARK,
                        linewidth=0.9)
    for position, record in zip(positions, ordered.itertuples()):
        if record.persistence < PERSISTENCE_CUTOFF:
            ax.text(max(record.contrast_db, record.score, 0) + 0.3,
                    position, f'present {record.persistence:.0%}',
                    va='center', fontsize=5.5, color=TEXT_GREY)
    ax.set_yticks(positions)
    labels = [f'{r.clip_id} ({r.logger_model})' for r in ordered.itertuples()]
    ax.set_yticklabels(labels, fontsize=6.5)
    for tick, group in zip(ax.get_yticklabels(), ordered['label_group']):
        if group in DUG_LABELS:
            tick.set_color(LABEL_COLORS[group])
            tick.set_fontweight('bold')
    ax.axvline(0, color=TEXT_GREY, linewidth=0.6)
    ax.set_xlabel('Score (dB the band stands out, after persistence gate)')
    ax.grid(axis='y', visible=False)
    ax.grid(axis='x', color='#EFEFEF')
    groups_present = set(ordered['label_group'])
    ax.legend(handles=[Patch(color=GROUP_COLORS[g], label=name) for g, name
                       in (('unlabeled', 'Not yet dug'),
                           (LEAK_LABEL, 'Dug: leak'),
                           (NO_LEAK_LABEL, 'Dug: no leak'))
                       if g in groups_present]
              + [Patch(color=HEALTH_LAB_NAVY, alpha=0.25,
                       label='Before persistence gate')],
              loc='lower right', fontsize=6.5)
    bottom = add_title_and_note(fig, figure_number, title, note)
    fig.subplots_adjust(top=0.93, bottom=bottom, left=0.13)
    MODEL_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = MODEL_FIGURE_DIR / f'{file_stem}.png'
    fig.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def write_summary(scores, leak_band_hz, loo_table, cluster_members,
                  version_comparison, floor_persistence):
    """Plain-language Markdown summary of Model 1 and its checks."""
    lines = ['# Model 1: knowledge-based score', '',
             f'- {ANALYSIS_EXCLUSION_REASON}',
             f'- Cast-iron leak band (from the 3 dug leaks): '
             f'{leak_band_hz[0]:.0f}-{leak_band_hz[1]:.0f} Hz. PVC band '
             f'(fixed): {EXCESS_BANDS_HZ[PVC_BAND_KEY][0]}-'
             f'{EXCESS_BANDS_HZ[PVC_BAND_KEY][1]} Hz, hum removed.',
             f'- Persistence on quiet floor clips (null check, should be ~0):'
             f' median {np.median(floor_persistence):.2f}, max '
             f'{np.max(floor_persistence):.2f}.', '']
    for ranking in ('cast_iron', 'pvc'):
        for version in VERSIONS:
            subset = scores[(scores['ranking'] == ranking)
                            & (scores['version'] == version)]
            lines.append(f'## {ranking} - {version} version')
            if ranking == 'cast_iron':
                wins, pairs = ranking_check(subset)
                lines.append(f'- Ranking check: leaks outscore non-leaks in '
                             f'{wins} of {pairs} pairs.')
            lines.append(f'- Top {MAX_DIGS_THIS_WEEK} undug: '
                         f'{top_undug(subset)}')
            lines.append('')
    lines += ['## Leave-one-out (floor version, cast iron)', '',
              '| Held out | Label | Band (Hz) | Held-out score | Undug '
              'sites scoring higher |', '|---|---|---|---|---|']
    lines += [f'| {r.held_out} | {r.label_group} | {r.band_low_hz:.0f}-'
              f'{r.band_high_hz:.0f} | {r.held_out_score:.1f} | '
              f'{r.undug_scoring_higher} |' for r in loo_table.itertuples()]
    cast_iron_top = top_undug(scores[(scores['ranking'] == 'cast_iron')
                                     & (scores['version'] == 'floor')])
    lines += ['', '## Agreement with clustering (floor top 5 vs. leak '
              'cluster)', '']
    lines += [f'- {setting}: leak cluster {sorted(members)}; overlap with '
              f'top 5: {sorted(members & set(cast_iron_top))}'
              for setting, members in cluster_members.items()]
    lines += ['', '## Floor vs. raw', '']
    for ranking, (rho, top) in version_comparison.items():
        overlap = sorted(set(top['floor']) & set(top['raw']))
        lines.append(f'- {ranking}: Spearman rho = {rho:.2f}; top-5 overlap '
                     f'{len(overlap)} of 5 ({overlap}).')
    MODEL1_SUMMARY_PATH.write_text('\n'.join(lines) + '\n',
                                   encoding='utf-8')


def run_model_one():
    """Score all sites, run the checks, and save tables and figures."""
    metadata = load_metadata()
    metadata = metadata[~metadata['clip_id'].isin(ANALYSIS_EXCLUDED_CLIPS)]
    amplitudes = load_all_amplitudes(metadata)
    inputs = build_spectral_inputs(metadata, amplitudes)
    leak_ids = metadata.loc[metadata['label_group'] == LEAK_LABEL,
                            'clip_id'].tolist()
    leak_band_hz = derive_leak_band(inputs['welch_excess'], leak_ids)
    cast_iron_sites = metadata[metadata['pipe_material'] == 'cast_iron']
    pvc_sites = metadata[metadata['pipe_material'] == 'PVC']
    scores = pd.concat(
        [score_group(cast_iron_sites, inputs, version,
                     *cast_iron_bands(leak_band_hz), False, 'cast_iron')
         for version in VERSIONS]
        + [score_group(pvc_sites, inputs, version, *pvc_bands(), True,
                       'pvc') for version in VERSIONS],
        ignore_index=True)
    loo_table, fold_scores, _ = leave_one_out(metadata, inputs,
                                              cast_iron_sites)
    fold_spread = fold_scores.agg(['min', 'max'], axis=1)
    floor_ids = identify_floor_clips(
        build_clip_feature_table(metadata, amplitudes))
    floor_persistence = scores.loc[
        scores['clip_id'].isin(floor_ids)
        & (scores['ranking'] == 'cast_iron')
        & (scores['version'] == 'floor'), 'persistence']
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    scores.round(3).to_csv(MODEL1_SCORES_PATH, index=False)
    loo_table.round(2).to_csv(MODEL1_LOO_PATH, index=False)
    version_comparison = {ranking: compare_versions(scores, ranking)
                          for ranking in ('cast_iron', 'pvc')}
    write_summary(scores, leak_band_hz, loo_table,
                  leak_cluster_members(metadata), version_comparison,
                  floor_persistence)
    is_floor = scores['version'] == 'floor'
    figure_paths = [
        plot_ranking(
            scores[is_floor & (scores['ranking'] == 'cast_iron')],
            fold_spread, 'Model 1: cast-iron sites ranked by a steady '
            'leak-band score', 'M1',
            f'Score = how many dB the {leak_band_hz[0]:.0f}-'
            f'{leak_band_hz[1]:.0f} Hz band stands out above its neighboring '
            'bands (after subtracting each logger\'s noise floor), scaled '
            'down when the band is present in fewer than 90% of 64 ms frames '
            '(light bars show the score before that scaling). Black lines: '
            'range of each undug site\'s score when each dug site is left '
            'out in turn. Bold labels mark dug sites. '
            f'{ANALYSIS_EXCLUSION_REASON}',
            'model1_cast_iron_ranking', CAST_IRON_FIGURE),
        plot_ranking(
            scores[is_floor & (scores['ranking'] == 'pvc')], None,
            'Model 1: PVC sites ranked on the 250-400 Hz band (unconfirmed)',
            'M2',
            'Score = how many dB the 250-400 Hz band stands out above 0-250 '
            'Hz and 400-1,300 Hz (after subtracting each logger\'s noise '
            'floor and removing 120 Hz hum lines), scaled down when present '
            'in fewer than 90% of frames. No PVC site has been dug, so this '
            'signature is unconfirmed and all PVC scores are low confidence.',
            'model1_pvc_ranking', PVC_FIGURE),
    ]
    return figure_paths


if __name__ == '__main__':
    for path in run_model_one():
        print(f'Saved {path}')
    print(MODEL1_SUMMARY_PATH.read_text(encoding='utf-8'))
