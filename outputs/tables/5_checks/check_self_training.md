# Robustness check: label spreading and self-training

- Features for label spreading: contrast_db, persistence, bin_coverage (standardized), k = 5, alpha = 0.2.
- Model 1 predicted leaks: ['U03', 'U07', 'U08', 'U15', 'U17', 'U34', 'U36'].
- Label spreading leaks: ['U07', 'U08', 'U15', 'U17', 'U34'] (agreement with Model 1: 5 shared).
- Self-training pseudo-labeled leaks: ['U34'].
- Top 5 by method: {'Model 1': ['U07', 'U17', 'U08', 'U36', 'U34'], 'Label spreading': ['U34', 'U15', 'U07', 'U17', 'U08'], 'Self-training': ['U34', 'U15', 'U17', 'U07', 'U08']}.
- Rank correlation with Model 1 (Spearman): label spreading 0.93, self-training 0.66.

## Self-training rounds

| Round | Labeled sites | Slope | Added as leak | Added as no leak |
|---|---|---|---|---|
| 1 | 6 | 0.25 | ['U34'] | 0 |
| 2 | 7 | 0.27 | [] | 0 |

## Leading sites

| Site | Pipe | Model 1 | Label-spread share | Self-training probability (sample scale) |
|---|---|---|---|---|
| U07 | cast_iron | 50.0% | 1.00 | 0.51 |
| U17 | cast_iron | 50.0% | 1.00 | 0.63 |
| U08 | cast_iron | 49.5% | 0.97 | 0.38 |
| U36 | cast_iron | 44.0% | 0.48 | 0.27 |
| U34 | pvc | 34.7% | 1.00 | 0.93 |
| U15 | pvc | 25.8% | 1.00 | 0.76 |
| U03 | pvc | 22.1% | 0.00 | 0.16 |
| U05 | cast_iron | 2.0% | 0.00 | 0.13 |
| U06 | cast_iron | 2.0% | 0.00 | 0.14 |
| U02 | cast_iron | 2.0% | 0.00 | 0.13 |
| U04 | cast_iron | 2.0% | 0.00 | 0.14 |
| U01 | cast_iron | 2.0% | 0.00 | 0.13 |
