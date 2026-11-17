# Model 1 probabilities: presence first, then quality

- U40 excluded: sensor overloaded (15% of samples clipped) and 1.2 s signal dropout, so its features are unreliable. No prediction for U40.
- Prior: 1 in 8 (12.5%).
- Smoothing +0.5: LR(band) = 7.0 -> 50.0%; LR(no band) = 0.14 -> 2.0%.
- Smoothing +1: LR(band) = 4.0 -> 36.4%; LR(no band) = 0.25 -> 3.4%.
- Quality = steadiness^0.6 x intenseness^0.4. References (weakest dug leak): persistence 0.994, coverage 1.00, contrast 10.9 dB.
- PVC signature match U15: w = 0.4 (shares: persistent_over_time, stands_out).
- PVC signature match U34: w = 0.6 (shares: persistent_over_time, stands_out, even_across_band).
- Calibration: sum of confidence over 39 scorable undug sites = 3.1 vs. 4.9 expected at 1 in 8.
- Dig list: ['U17', 'U07', 'U08', 'U36', 'U34'] (last = informative dig).

| Site | Pipe | Steadiness | Intenseness | Quality | w | Confidence | Range | Prediction |
|---|---|---|---|---|---|---|---|---|
| U17 | cast_iron | 1.00 | 0.82 | 0.92 | 1.0 | 48.9% | 36.4%-50.0% | 1 |
| U07 | cast_iron | 1.00 | 0.66 | 0.84 | 1.0 | 47.8% | 36.4%-50.0% | 1 |
| U08 | cast_iron | 0.94 | 0.47 | 0.71 | 1.0 | 46.0% | 2.0%-50.0% | 1 |
| U36 | cast_iron | 0.62 | 0.30 | 0.47 | 1.0 | 42.6% | 2.0%-50.0% | 1 |
| U34 | pvc | 0.97 | 1.00 | 0.98 | 0.6 | 34.9% | 26.8%-35.0% | 1 |
| U15 | pvc | 0.66 | 1.00 | 0.78 | 0.4 | 26.3% | 8.3%-27.5% | 1 |
| U32 | pvc | 0.30 | 0.36 | 0.32 | 1.0 | 2.4% | 2.0%-3.4% | 0 |
| U35 | cast_iron | 0.04 | 0.62 | 0.11 | 1.0 | 2.1% | 2.0%-3.4% | 0 |
| U27 | cast_iron | 0.02 | 0.23 | 0.05 | 1.0 | 2.1% | 2.0%-3.4% | 0 |
| U11 | cast_iron | 0.02 | 0.14 | 0.04 | 1.0 | 2.0% | 2.0%-3.4% | 0 |
| U04 | cast_iron | 0.01 | 0.19 | 0.04 | 1.0 | 2.0% | 2.0%-3.4% | 0 |
| U06 | cast_iron | 0.01 | 0.21 | 0.04 | 1.0 | 2.0% | 2.0%-3.4% | 0 |
| U12 | cast_iron | 0.01 | 0.14 | 0.03 | 1.0 | 2.0% | 2.0%-3.4% | 0 |
| U01 | cast_iron | 0.00 | 0.00 | 0.00 | 1.0 | 2.0% | 2.0%-3.4% | 0 |
| U02 | cast_iron | 0.00 | 0.00 | 0.00 | 1.0 | 2.0% | 2.0%-3.4% | 0 |
| U05 | cast_iron | 0.00 | 0.00 | 0.00 | 1.0 | 2.0% | 2.0%-3.4% | 0 |
| U14 | cast_iron | 0.00 | 0.00 | 0.00 | 1.0 | 2.0% | 2.0%-3.4% | 0 |
| U21 | cast_iron | 0.00 | 0.00 | 0.00 | 1.0 | 2.0% | 2.0%-3.4% | 0 |
| U22 | cast_iron | 0.00 | 0.02 | 0.00 | 1.0 | 2.0% | 2.0%-3.4% | 0 |
| U16 | cast_iron | 0.00 | 0.03 | 0.00 | 1.0 | 2.0% | 2.0%-3.4% | 0 |
| U13 | cast_iron | 0.00 | 0.00 | 0.00 | 1.0 | 2.0% | 2.0%-3.4% | 0 |
| U09 | cast_iron | 0.00 | 0.00 | 0.00 | 1.0 | 2.0% | 2.0%-3.4% | 0 |
| U10 | cast_iron | 0.00 | 0.02 | 0.00 | 1.0 | 2.0% | 2.0%-3.4% | 0 |
| U30 | cast_iron | 0.00 | 0.00 | 0.00 | 1.0 | 2.0% | 2.0%-3.4% | 0 |
| U26 | cast_iron | 0.00 | 0.00 | 0.00 | 1.0 | 2.0% | 2.0%-3.4% | 0 |
| U23 | cast_iron | 0.00 | 0.03 | 0.00 | 1.0 | 2.0% | 2.0%-3.4% | 0 |
| U24 | cast_iron | 0.00 | 0.00 | 0.00 | 1.0 | 2.0% | 2.0%-3.4% | 0 |
| U39 | cast_iron | 0.00 | 0.00 | 0.00 | 1.0 | 2.0% | 2.0%-3.4% | 0 |
| U03 | pvc | 0.00 | 0.08 | 0.00 | 1.0 | 2.0% | 2.0%-3.4% | 0 |
| U38 | cast_iron | 0.00 | 0.01 | 0.00 | 1.0 | 2.0% | 2.0%-3.4% | 0 |
| U37 | cast_iron | 0.00 | 0.00 | 0.00 | 1.0 | 2.0% | 2.0%-3.4% | 0 |
| U19 | pvc | 0.00 | 0.00 | 0.00 | 1.0 | 2.0% | 2.0%-3.4% | 0 |
| U18 | pvc | 0.00 | 0.00 | 0.00 | 1.0 | 2.0% | 2.0%-3.4% | 0 |
| U20 | pvc | 0.00 | 0.01 | 0.00 | 1.0 | 2.0% | 2.0%-3.4% | 0 |
| U25 | pvc | 0.00 | 0.00 | 0.00 | 1.0 | 2.0% | 2.0%-3.4% | 0 |
| U29 | pvc | 0.01 | 0.00 | 0.00 | 1.0 | 2.0% | 2.0%-3.4% | 0 |
| U28 | pvc | 0.00 | 0.00 | 0.00 | 1.0 | 2.0% | 2.0%-3.4% | 0 |
| U31 | pvc | 0.00 | 0.00 | 0.00 | 1.0 | 2.0% | 2.0%-3.4% | 0 |
| U33 | pvc | 0.00 | 0.00 | 0.00 | 1.0 | 2.0% | 2.0%-3.4% | 0 |
