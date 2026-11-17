# Model 1: knowledge-based score

- U40 excluded: sensor overloaded (15% of samples clipped) and 1.2 s signal dropout, so its features are unreliable.
- Cast-iron leak band (from the 3 dug leaks): 400-1300 Hz. PVC band (fixed): 250-400 Hz, hum removed.
- Persistence on quiet floor clips (null check, should be ~0): median 0.03, max 0.05.

## cast_iron - floor version
- Ranking check: leaks outscore non-leaks in 9 of 9 pairs.
- Top 5 undug: ['U17', 'U07', 'U08', 'U36', 'U35']

## cast_iron - raw version
- Ranking check: leaks outscore non-leaks in 9 of 9 pairs.
- Top 5 undug: ['U17', 'U07', 'U08', 'U36', 'U12']

## pvc - floor version
- Top 5 undug: ['U34', 'U15', 'U32', 'U03', 'U20']

## pvc - raw version
- Top 5 undug: ['U34', 'U15', 'U18', 'U19', 'U25']

## Leave-one-out (floor version, cast iron)

| Held out | Label | Band (Hz) | Held-out score | Undug sites scoring higher |
|---|---|---|---|---|
| L01 | leak | 396-1304 | 15.3 | 0 |
| L02 | no_leak | 400-1300 | 0.2 | 5 |
| L03 | leak | 396-1304 | 21.2 | 0 |
| L04 | no_leak | 400-1300 | 0.2 | 5 |
| L05 | no_leak | 400-1300 | 0.1 | 10 |
| L06 | leak | 400-1300 | 10.9 | 0 |

## Agreement with clustering (floor top 5 vs. leak cluster)

- kmeans_k2: leak cluster ['U02', 'U03', 'U04', 'U06', 'U07', 'U08', 'U09', 'U10', 'U11', 'U12', 'U13', 'U17', 'U24', 'U27', 'U28', 'U29', 'U31', 'U35', 'U36', 'U38']; overlap with top 5: ['U07', 'U08', 'U17', 'U35', 'U36']
- ward_k2: leak cluster ['U02', 'U03', 'U04', 'U06', 'U07', 'U08', 'U09', 'U10', 'U11', 'U12', 'U13', 'U17', 'U24', 'U27', 'U28', 'U29', 'U31', 'U32', 'U35', 'U36', 'U38']; overlap with top 5: ['U07', 'U08', 'U17', 'U35', 'U36']
- kmeans_k3: leak cluster ['U08', 'U17']; overlap with top 5: ['U08', 'U17']
- ward_k3: leak cluster ['U06', 'U07', 'U08', 'U17']; overlap with top 5: ['U07', 'U08', 'U17']
- kmeans_k4: leak cluster ['U17']; overlap with top 5: ['U17']
- ward_k4: leak cluster ['U06', 'U07', 'U08', 'U17']; overlap with top 5: ['U07', 'U08', 'U17']

## Floor vs. raw

- cast_iron: Spearman rho = 0.65; top-5 overlap 4 of 5 (['U07', 'U08', 'U17', 'U36']).
- pvc: Spearman rho = 0.32; top-5 overlap 2 of 5 (['U15', 'U34']).
