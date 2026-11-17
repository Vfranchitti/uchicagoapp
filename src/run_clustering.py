"""Unsupervised check: do the dug leaks form their own group?

Clusters the recordings on five approved features, one from each block of
near-duplicate features, without using any labels. Then reports where the 3
dug leaks and 3 dug non-leaks land (Foster et al., ch. 7, unsupervised
learning). This is a robustness check, not the prediction model.

Features (each standardized within logger model):
  1. typical loudness: median 50 ms RMS (log10)
  2. loudness variability over time: CV of 50 ms RMS (log10)
  3. leak-band contrast: dB above floor in 400-1,300 Hz minus the mean of
     250-400 Hz and 1,300-2,000 Hz
  4. low-band (PVC) contrast: dB above floor in 250-400 Hz minus the mean
     of 0-250 Hz and 400-1,300 Hz
  5. leak-band steadiness: SD over time of 400-1,300 Hz power (dB)

Runs:
  'all'            every recording except ANALYSIS_EXCLUDED_CLIPS (U40)
  'without_floor'  also drops the quiet noise-floor clips, so silence does
                   not dominate the grouping

Outputs
-------
outputs/tables/cluster_features.csv
outputs/tables/cluster_assignments.csv
outputs/tables/cluster_summary.md
outputs/figures/eda/eda_20_cluster_dendrogram.png
outputs/figures/eda/eda_21_cluster_map.png
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, to_hex
from scipy.cluster.hierarchy import (
    dendrogram,
    fcluster,
    linkage,
    set_link_color_palette,
)
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from src.analyze_noise_floor import (
    compute_clip_spectra_db,
    compute_excess_spectra,
    compute_logger_baselines,
    identify_floor_clips,
    summarize_excess_by_band,
)
from src.compute_clip_features import build_clip_feature_table
from src.config import (
    ANALYSIS_EXCLUDED_CLIPS,
    ANALYSIS_EXCLUSION_REASON,
    CLUSTER_COUNTS,
    CLUSTER_FEATURE_TABLE_PATH,
    CLUSTER_SUMMARY_PATH,
    CLUSTER_TABLE_PATH,
    KMEANS_RANDOM_SEED,
    KMEANS_RESTARTS,
    LEAK_LABEL,
    NO_LEAK_LABEL,
    TABLE_DIR,
)
from src.load_recordings import load_all_amplitudes, load_metadata
from src.plot_style import (
    CRIME_LAB_GOLD,
    HEALTH_LAB_LIGHT_TEAL,
    HEALTH_LAB_NAVY,
    HEALTH_LAB_TEAL,
    LABEL_COLORS,
    PIPE_MARKERS,
    TEXT_DARK,
    TEXT_GREY,
    UCHICAGO_MAROON,
    UNLABELED_GREY,
    add_title_and_note,
    apply_urban_labs_style,
    finish_and_save,
)

CLUSTER_FEATURES = {
    'log_typical_loudness': 'Typical loudness (log)',
    'log_loudness_variability': 'Loudness variability (log)',
    'leak_band_contrast_db': 'Leak-band contrast (dB)',
    'low_band_contrast_db': 'Low-band (PVC) contrast (dB)',
    'leak_band_level_variability_db': 'Leak-band unsteadiness (dB)',
}
RUNS = ('all', 'without_floor')
RUN_TITLES = {'all': 'All recordings (U40 excluded)',
              'without_floor': 'Without quiet noise-floor clips'}
CLUSTER_COLORS = (HEALTH_LAB_NAVY, CRIME_LAB_GOLD, HEALTH_LAB_LIGHT_TEAL,
                  '#8C8C8C')
DUG_LABELS = (LEAK_LABEL, NO_LEAK_LABEL)
HEATMAP_Z_LIMIT = 2.5
Z_SCORE_COLORMAP = LinearSegmentedColormap.from_list(
    'teal_white_maroon', [HEALTH_LAB_TEAL, '#FFFFFF', UCHICAGO_MAROON])
DENDROGRAM_FIGURE = (7.5, 6.2)
MAP_FIGURE = (7.5, 4.4)


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def build_cluster_feature_table(metadata, amplitudes):
    """Raw (unstandardized) clustering features, one row per clip."""
    clip_features = build_clip_feature_table(metadata, amplitudes)
    _, spectra_db = compute_clip_spectra_db(metadata, amplitudes)
    floor_clip_ids = identify_floor_clips(clip_features)
    baselines_db = compute_logger_baselines(metadata, spectra_db,
                                            floor_clip_ids)
    bands = summarize_excess_by_band(
        metadata, compute_excess_spectra(metadata, spectra_db,
                                         baselines_db)).set_index('clip_id')
    table = clip_features.set_index('clip_id')
    table['log_typical_loudness'] = np.log10(table['typical_window_rms'])
    table['log_loudness_variability'] = np.log10(
        table['loudness_variability'])
    table['leak_band_contrast_db'] = bands['leak_band_400_1300'] - (
        bands['edge_250_400'] + bands['high_1300_2000']) / 2
    table['low_band_contrast_db'] = bands['edge_250_400'] - (
        bands['low_0_250'] + bands['leak_band_400_1300']) / 2
    table['is_floor_clip'] = table.index.isin(floor_clip_ids)
    keep_columns = (['label_group', 'logger_model', 'pipe_material',
                     'recorded_at', 'is_floor_clip']
                    + list(CLUSTER_FEATURES))
    return table[keep_columns].reset_index()


def select_run(cluster_table, run):
    """Rows used in one clustering run."""
    is_analyzed = ~cluster_table['clip_id'].isin(ANALYSIS_EXCLUDED_CLIPS)
    if run == 'without_floor':
        is_analyzed &= ~cluster_table['is_floor_clip']
    return cluster_table[is_analyzed].reset_index(drop=True)


def standardize_within_logger(df):
    """z-score each feature within logger model (removes device level)."""
    grouped = df.groupby('logger_model')[list(CLUSTER_FEATURES)]
    return ((df[list(CLUSTER_FEATURES)] - grouped.transform('mean'))
            / grouped.transform('std'))


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
def _order_by_leak_contrast(labels, df):
    """Renumber clusters 1..k from most to least leak-band contrast."""
    means = df.groupby(labels)['leak_band_contrast_db'].mean()
    ranking = {old: new for new, old in
               enumerate(means.sort_values(ascending=False).index, start=1)}
    return np.array([ranking[label] for label in labels])


def cluster_run(df, z_scores):
    """K-means and Ward assignments for each cluster count."""
    ward_tree = linkage(z_scores, method='ward')
    assignments, scores = df[['clip_id']].copy(), []
    for n_clusters in CLUSTER_COUNTS:
        kmeans_labels = KMeans(
            n_clusters=n_clusters, n_init=KMEANS_RESTARTS,
            random_state=KMEANS_RANDOM_SEED).fit_predict(z_scores)
        ward_labels = fcluster(ward_tree, n_clusters, criterion='maxclust')
        kmeans_labels = _order_by_leak_contrast(kmeans_labels, df)
        ward_labels = _order_by_leak_contrast(ward_labels, df)
        assignments[f'kmeans_k{n_clusters}'] = kmeans_labels
        assignments[f'ward_k{n_clusters}'] = ward_labels
        scores.append({
            'n_clusters': n_clusters,
            'kmeans_silhouette': silhouette_score(z_scores, kmeans_labels),
            'ward_silhouette': silhouette_score(z_scores, ward_labels),
            'kmeans_ward_agreement': adjusted_rand_score(kmeans_labels,
                                                         ward_labels),
        })
    return assignments, pd.DataFrame(scores), ward_tree


def best_cluster_count(scores):
    """Cluster count with the highest k-means silhouette."""
    return int(scores.loc[scores['kmeans_silhouette'].idxmax(),
                          'n_clusters'])


def describe_dug_placement(df, assignments, column):
    """Which cluster each dug site falls in, for one assignment column."""
    merged = df[['clip_id', 'label_group']].merge(assignments, on='clip_id')
    return {label: merged.loc[merged['label_group'] == label,
                              ['clip_id', column]].values.tolist()
            for label in DUG_LABELS}


def leak_cluster_members(df, assignments, column):
    """Undug clips sharing a cluster with the majority of dug leaks."""
    merged = df.merge(assignments, on='clip_id')
    leak_clusters = merged.loc[merged['label_group'] == LEAK_LABEL, column]
    majority = leak_clusters.mode().iloc[0]
    members = merged[(merged[column] == majority)
                     & (merged['label_group'] == 'unlabeled')]
    return int(majority), members['clip_id'].tolist()


def write_summary(results):
    """Plain-language summary of both runs."""
    lines = ['# Cluster analysis (unsupervised check)', '',
             f'- {ANALYSIS_EXCLUSION_REASON}',
             '- Features (standardized within logger): '
             + ', '.join(CLUSTER_FEATURES.values()) + '.', '']
    for run, result in results.items():
        df, assignments, scores = (result['df'], result['assignments'],
                                   result['scores'])
        best_k = best_cluster_count(scores)
        column = f'kmeans_k{best_k}'
        placement = describe_dug_placement(df, assignments, column)
        majority, members = leak_cluster_members(df, assignments, column)
        sizes = assignments[column].value_counts().sort_index().to_dict()
        lines += [
            f'## Run: {RUN_TITLES[run]} (n = {len(df)})', '',
            '| k | k-means silhouette | Ward silhouette | Agreement '
            '(adj. Rand) |', '|---|---|---|---|']
        lines += [f'| {row.n_clusters} | {row.kmeans_silhouette:.2f} | '
                  f'{row.ward_silhouette:.2f} | '
                  f'{row.kmeans_ward_agreement:.2f} |'
                  for row in scores.itertuples()]
        lines += [
            '',
            f'- Best k by silhouette: {best_k}. Cluster sizes: {sizes} '
            f'(cluster 1 = highest leak-band contrast).',
            f'- Dug leaks: {placement[LEAK_LABEL]}; dug non-leaks: '
            f'{placement[NO_LEAK_LABEL]}.',
            f'- Undug clips sharing cluster {majority} with most dug leaks: '
            f'{members}.']
        for method in ('kmeans', 'ward'):
            for n_clusters in CLUSTER_COUNTS:
                other_column = f'{method}_k{n_clusters}'
                _, other_members = leak_cluster_members(df, assignments,
                                                        other_column)
                placement = describe_dug_placement(df, assignments,
                                                   other_column)
                leak_clusters = [c for _, c in placement[LEAK_LABEL]]
                no_leak_clusters = [c for _, c in placement[NO_LEAK_LABEL]]
                lines.append(
                    f'  - {method}, k = {n_clusters}: leaks in clusters '
                    f'{leak_clusters}, non-leaks in {no_leak_clusters}; '
                    f'undug in leak cluster ({len(other_members)}): '
                    f'{other_members}')
        lines.append('')
    CLUSTER_SUMMARY_PATH.write_text('\n'.join(lines) + '\n',
                                    encoding='utf-8')


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _color_dug_tick_labels(tick_labels, label_by_clip):
    """Bold maroon/teal tick labels for dug sites."""
    for tick in tick_labels:
        group = label_by_clip.get(tick.get_text(), 'unlabeled')
        if group in DUG_LABELS:
            tick.set_color(LABEL_COLORS[group])
            tick.set_fontweight('bold')


def plot_dendrogram_with_profiles(result):
    """Fig 20 (ACCURATE). Ward tree of all clips plus feature profiles."""
    df, z_scores, ward_tree = (result['df'], result['z_scores'],
                               result['ward_tree'])
    best_k = best_cluster_count(result['scores'])
    fig = plt.figure(figsize=DENDROGRAM_FIGURE)
    grid = fig.add_gridspec(2, 1, height_ratios=[1.3, 1], hspace=0.05)
    tree_ax = fig.add_subplot(grid[0])
    heat_ax = fig.add_subplot(grid[1])
    cut_height = np.sort(ward_tree[:, 2])[-(best_k - 1)]
    set_link_color_palette([to_hex(color) for color in CLUSTER_COLORS])
    tree = dendrogram(ward_tree, ax=tree_ax, labels=df['clip_id'].tolist(),
                      color_threshold=cut_height,
                      above_threshold_color=UNLABELED_GREY, no_labels=True)
    tree_ax.axhline(cut_height, color=TEXT_GREY, linestyle='--',
                    linewidth=0.8)
    tree_ax.text(tree_ax.get_xlim()[1], cut_height, f'  cut: {best_k} groups',
                 va='center', ha='left', fontsize=7, color=TEXT_GREY)
    tree_ax.set_ylabel('Ward linkage distance')
    tree_ax.set_xticks([])
    tree_ax.grid(False)
    leaf_order = tree['leaves']
    heat_ax.imshow(z_scores[leaf_order].T, aspect='auto',
                   cmap=Z_SCORE_COLORMAP,
                   vmin=-HEATMAP_Z_LIMIT, vmax=HEATMAP_Z_LIMIT,
                   interpolation='nearest')
    heat_ax.set_yticks(range(len(CLUSTER_FEATURES)))
    heat_ax.set_yticklabels(list(CLUSTER_FEATURES.values()), fontsize=7)
    heat_ax.set_xticks(range(len(leaf_order)))
    heat_ax.set_xticklabels(df['clip_id'].iloc[leaf_order], rotation=90,
                            fontsize=6)
    _color_dug_tick_labels(heat_ax.get_xticklabels(),
                           df.set_index('clip_id')['label_group'].to_dict())
    heat_ax.grid(False)
    heat_ax.spines[:].set_visible(False)
    bottom = add_title_and_note(
        fig, 20, 'Which recordings group together? Ward clustering on five '
        'features',
        f'Top: hierarchical (Ward) clustering of {len(df)} recordings on '
        'five features, each standardized within logger model; branch '
        f'colors show the {best_k}-group cut chosen by silhouette score. '
        'Bottom: each feature for each recording (maroon = above the '
        'logger\'s average, teal = below; standard deviations, capped at '
        f'±{HEATMAP_Z_LIMIT}). Bold labels mark dug sites (maroon = leak, '
        f'teal = no leak). {ANALYSIS_EXCLUSION_REASON}')
    return finish_and_save(fig, 'eda_20_cluster_dendrogram', bottom,
                           top=0.91, left=0.2)


def _draw_cluster_map(ax, result):
    """Leak-band contrast vs. unsteadiness, colored by k-means cluster."""
    df, assignments = result['df'], result['assignments']
    best_k = best_cluster_count(result['scores'])
    merged = df.merge(assignments, on='clip_id')
    column = f'kmeans_k{best_k}'
    for cluster in range(1, best_k + 1):
        for pipe, marker in PIPE_MARKERS.items():
            subset = merged[(merged[column] == cluster)
                            & (merged['pipe_material'] == pipe)]
            ax.scatter(subset['leak_band_contrast_db'],
                       subset['leak_band_level_variability_db'],
                       color=CLUSTER_COLORS[cluster - 1], marker=marker,
                       s=36, edgecolor='white', linewidth=0.5, zorder=3)
    dug = merged[merged['label_group'].isin(DUG_LABELS)]
    ax.scatter(dug['leak_band_contrast_db'],
               dug['leak_band_level_variability_db'], s=110,
               facecolor='none', linewidth=1.6, zorder=4,
               edgecolor=[LABEL_COLORS[g] for g in dug['label_group']])
    for record in merged[~merged['is_floor_clip']].itertuples():
        is_dug = record.label_group in DUG_LABELS
        ax.annotate(record.clip_id, (record.leak_band_contrast_db,
                                     record.leak_band_level_variability_db),
                    xytext=(4, 3), textcoords='offset points',
                    fontsize=6.5 if is_dug else 5.5,
                    color=(LABEL_COLORS[record.label_group] if is_dug
                           else TEXT_GREY),
                    fontweight='bold' if is_dug else 'normal', zorder=5)
    ax.set_xlabel('Leak-band contrast (dB)  → more leak-like')
    ax.set_ylabel('Leak-band unsteadiness (dB)  ↑ less steady')
    ax.set_title(f'{RUN_TITLES[result["run"]]}: {best_k} groups '
                 f'(n = {len(df)})', loc='left', fontsize=8)
    ax.grid(axis='x', color='#EFEFEF')


def plot_cluster_maps(results):
    """Fig 21 (INTERPRETABLE). Cluster membership on two key features."""
    fig, axes = plt.subplots(1, 2, figsize=MAP_FIGURE, sharey=True)
    for ax, run in zip(axes, RUNS):
        _draw_cluster_map(ax, results[run])
    max_k = max(best_cluster_count(results[run]['scores']) for run in RUNS)
    handles = ([Line2D([], [], marker='o', linestyle='',
                       color=CLUSTER_COLORS[i], label=f'Group {i + 1}')
                for i in range(max_k)]
               + [Line2D([], [], marker=m, linestyle='',
                         markerfacecolor='white', color=TEXT_DARK,
                         label=p.replace('_', ' '))
                  for p, m in PIPE_MARKERS.items()]
               + [Line2D([], [], marker='o', linestyle='', markersize=9,
                         markerfacecolor='none', color=LABEL_COLORS[g],
                         label=name)
                  for g, name in ((LEAK_LABEL, 'Dug: leak'),
                                  (NO_LEAK_LABEL, 'Dug: no leak'))])
    fig.legend(handles=handles, loc='upper center', ncol=len(handles),
               bbox_to_anchor=(0.5, 0.9), fontsize=6.5)
    fig.subplots_adjust(wspace=0.08)
    bottom = add_title_and_note(
        fig, 21, 'Where the dug sites fall among the clusters',
        'K-means clusters on five features standardized within logger '
        '(group 1 = highest average leak-band contrast); the number of '
        'groups in each run is chosen by silhouette score. Only two of the '
        'five features are plotted. Right panel drops the quiet '
        'noise-floor recordings, so silence cannot drive the grouping; in '
        'the left panel they form the unlabeled clump near (0, 0.6). '
        f'{ANALYSIS_EXCLUSION_REASON}')
    return finish_and_save(fig, 'eda_21_cluster_map', bottom, top=0.78)


def run_clustering():
    """Build features, cluster both runs, save tables and figures."""
    apply_urban_labs_style()
    metadata = load_metadata()
    amplitudes = load_all_amplitudes(metadata)
    cluster_table = build_cluster_feature_table(metadata, amplitudes)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    cluster_table.round(4).to_csv(CLUSTER_FEATURE_TABLE_PATH, index=False)
    results, assignment_tables = {}, []
    for run in RUNS:
        df = select_run(cluster_table, run)
        z_scores = standardize_within_logger(df).to_numpy()
        assignments, scores, ward_tree = cluster_run(df, z_scores)
        results[run] = {'run': run, 'df': df, 'z_scores': z_scores,
                        'assignments': assignments, 'scores': scores,
                        'ward_tree': ward_tree}
        assignment_tables.append(assignments.assign(run=run))
    pd.concat(assignment_tables).to_csv(CLUSTER_TABLE_PATH, index=False)
    write_summary(results)
    return [plot_dendrogram_with_profiles(results['all']),
            plot_cluster_maps(results)]


if __name__ == '__main__':
    for figure_path in run_clustering():
        print(f'Saved {figure_path}')
    print(CLUSTER_SUMMARY_PATH.read_text(encoding='utf-8'))
