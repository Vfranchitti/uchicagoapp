# Cluster analysis (unsupervised check)

- U40 excluded (insufficient data quality): signal saturation (15% of samples clipped) and a 1.2 s dropout make its features invalid.
- Features (standardized within logger): Typical loudness (log), Loudness variability (log), Leak-band contrast (dB), Low-band (PVC) contrast (dB), Leak-band unsteadiness (dB).

## Run: All recordings (U40 excluded) (n = 45)

| k | k-means silhouette | Ward silhouette | Agreement (adj. Rand) |
|---|---|---|---|
| 2 | 0.45 | 0.44 | 0.91 |
| 3 | 0.46 | 0.43 | 0.82 |
| 4 | 0.50 | 0.45 | 0.80 |

- Best k by silhouette: 4. Cluster sizes: {1: 4, 2: 10, 3: 12, 4: 19} (cluster 1 = highest leak-band contrast).
- Dug leaks: [['L01', 1], ['L03', 1], ['L06', 1]]; dug non-leaks: [['L02', 2], ['L04', 3], ['L05', 2]].
- Undug clips sharing cluster 1 with most dug leaks: ['U17'].
  - kmeans, k = 2: leaks in clusters [1, 1, 1], non-leaks in [1, 1, 1]; undug in leak cluster (20): ['U02', 'U03', 'U04', 'U06', 'U07', 'U08', 'U09', 'U10', 'U11', 'U12', 'U13', 'U17', 'U24', 'U27', 'U28', 'U29', 'U31', 'U35', 'U36', 'U38']
  - kmeans, k = 3: leaks in clusters [1, 1, 1], non-leaks in [2, 2, 2]; undug in leak cluster (2): ['U08', 'U17']
  - kmeans, k = 4: leaks in clusters [1, 1, 1], non-leaks in [2, 3, 2]; undug in leak cluster (1): ['U17']
  - ward, k = 2: leaks in clusters [1, 1, 1], non-leaks in [1, 1, 1]; undug in leak cluster (21): ['U02', 'U03', 'U04', 'U06', 'U07', 'U08', 'U09', 'U10', 'U11', 'U12', 'U13', 'U17', 'U24', 'U27', 'U28', 'U29', 'U31', 'U32', 'U35', 'U36', 'U38']
  - ward, k = 3: leaks in clusters [1, 1, 1], non-leaks in [2, 2, 2]; undug in leak cluster (4): ['U06', 'U07', 'U08', 'U17']
  - ward, k = 4: leaks in clusters [1, 1, 1], non-leaks in [3, 2, 3]; undug in leak cluster (4): ['U06', 'U07', 'U08', 'U17']

## Run: Without quiet noise-floor clips (n = 31)

| k | k-means silhouette | Ward silhouette | Agreement (adj. Rand) |
|---|---|---|---|
| 2 | 0.42 | 0.42 | 1.00 |
| 3 | 0.36 | 0.36 | 1.00 |
| 4 | 0.44 | 0.40 | 0.82 |

- Best k by silhouette: 4. Cluster sizes: {1: 4, 2: 10, 3: 12, 4: 5} (cluster 1 = highest leak-band contrast).
- Dug leaks: [['L01', 1], ['L03', 1], ['L06', 1]]; dug non-leaks: [['L02', 2], ['L04', 3], ['L05', 2]].
- Undug clips sharing cluster 1 with most dug leaks: ['U17'].
  - kmeans, k = 2: leaks in clusters [1, 1, 1], non-leaks in [1, 1, 1]; undug in leak cluster (20): ['U02', 'U03', 'U04', 'U06', 'U07', 'U08', 'U09', 'U10', 'U11', 'U12', 'U13', 'U17', 'U24', 'U27', 'U28', 'U29', 'U31', 'U35', 'U36', 'U38']
  - kmeans, k = 3: leaks in clusters [1, 1, 1], non-leaks in [2, 2, 2]; undug in leak cluster (4): ['U06', 'U07', 'U08', 'U17']
  - kmeans, k = 4: leaks in clusters [1, 1, 1], non-leaks in [2, 3, 2]; undug in leak cluster (1): ['U17']
  - ward, k = 2: leaks in clusters [1, 1, 1], non-leaks in [1, 1, 1]; undug in leak cluster (20): ['U02', 'U03', 'U04', 'U06', 'U07', 'U08', 'U09', 'U10', 'U11', 'U12', 'U13', 'U17', 'U24', 'U27', 'U28', 'U29', 'U31', 'U35', 'U36', 'U38']
  - ward, k = 3: leaks in clusters [1, 1, 1], non-leaks in [2, 2, 2]; undug in leak cluster (4): ['U06', 'U07', 'U08', 'U17']
  - ward, k = 4: leaks in clusters [1, 1, 1], non-leaks in [3, 2, 3]; undug in leak cluster (4): ['U06', 'U07', 'U08', 'U17']

