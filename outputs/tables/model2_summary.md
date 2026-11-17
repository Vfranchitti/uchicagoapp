# Model 2: Firth logistic regression on the Model 1 score

- U40 excluded: sensor overloaded (15% of samples clipped) and 1.2 s signal dropout, so its features are unreliable.
- Main fit: slope 0.25 per dB; the curve crosses 50% (on the 6-site sample) at 7.3 dB. Base-rate shift -1.95 on the log-odds scale.
- Slope across all 14 fits: 0.19-0.31 per dB.
- Calibration: sum of confidence over 39 scorable undug sites = 2.1 vs. 4.9 expected at 1 in 8.

## Held-out dug sites (6-site sample scale; 50% = coin flip)

| Version | Held out | Label | Score | P(leak), sample scale |
|---|---|---|---|---|
| floor | L01 | leak | 15.2 | 79% |
| floor | L02 | no_leak | 0.2 | 19% |
| floor | L03 | leak | 21.1 | 98% |
| floor | L04 | no_leak | 0.2 | 19% |
| floor | L05 | no_leak | 0.1 | 19% |
| floor | L06 | leak | 10.9 | 55% |
| raw | L01 | leak | 13.4 | 74% |
| raw | L02 | no_leak | 0.0 | 20% |
| raw | L03 | leak | 20.8 | 99% |
| raw | L04 | no_leak | 0.0 | 20% |
| raw | L05 | no_leak | 0.0 | 20% |
| raw | L06 | leak | 9.5 | 52% |

## Model 2 vs. Model 1 (top 12 by Model 2)

| Site | Pipe | Score | Model 2 | Model 2 range | Model 1 |
|---|---|---|---|---|---|
| U34 | pvc | 16.5 | 59% | 21%-73% | 43% |
| U15 | pvc | 11.3 | 28% | 8%-40% | 43% |
| U17 | cast_iron | 9.0 | 18% | 14%-29% | 43% |
| U07 | cast_iron | 7.2 | 12% | 7%-18% | 43% |
| U08 | cast_iron | 5.1 | 8% | 5%-11% | 43% |
| U36 | cast_iron | 3.3 | 5% | 4%-7% | 43% |
| U32 | pvc | 2.0 | 4% | 2%-6% | 3% |
| U03 | pvc | 0.7 | 3% | 1%-4% | 3% |
| U35 | cast_iron | 0.3 | 2% | 2%-4% | 3% |
| U16 | cast_iron | 0.2 | 2% | 2%-4% | 3% |
| U27 | cast_iron | 0.1 | 2% | 2%-4% | 3% |
| U11 | cast_iron | 0.1 | 2% | 2%-4% | 3% |
