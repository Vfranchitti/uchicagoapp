# Exploratory data analysis: plan and results

Reproduce from the repo root with the `chicago` environment:

```
python -m src.compute_clip_features   # outputs/tables/clip_features.csv
python -m src.run_data_checks         # outputs/tables/data_checks.{csv,md}
python -m src.plot_eda_figures        # outputs/figures/eda/*.png
```

Framework: total error (Foster et al., *Big Data and Social Science*,
ch. 10). Row errors are about coverage (who is in the data), column errors
are about specification (what a variable really measures), and cell errors
are about measurement (whether each value is right). Sample bias and
confounding come from ch. 11, and visualization principles from ch. 6.

## 1. Data checks, in order of importance

Checks earlier in the list can invalidate everything after them. Full
findings are in `outputs/tables/data_checks.md`.

| # | Check | Why it matters | Result |
|---|---|---|---|
| 1 | File inventory and linkage | Every metadata row needs a matching CSV and WAV | Pass: 46 rows, 92 files |
| 2 | Unique identifiers | One recording per site, L/U prefix matches the labeled flag | Pass |
| 3 | Label integrity | Dug sites have valid labels; undug sites have none | Pass: 3 leak, 3 no leak, 40 blank |
| 4 | Metadata value domains | Logger, pipe and time fields hold valid values | Pass: 22:08 to 06:31 |
| 5 | Signal length and missing values | 20,000 finite samples per clip | Pass |
| 6 | Uniform 4,000 Hz time grid | Spectral features assume even spacing | Pass |
| 7 | Amplitude within [-1, 1] | Values match the documented scale | Pass |
| 8 | CSV and WAV agree | The two formats contain the same signal | Pass (max difference 3.5e-5) |
| 9 | Clipping | An overloaded sensor distorts loudness and pitch | **Flag: U40, 15% of samples at the ceiling** |
| 10 | Dropouts | A flatlined stretch means lost signal | **Flag: U40, 1.2 s of exact zeros** |
| 11 | DC offset and dead clips | Bias in the signal, or a dead sensor | Pass |
| 12 | Duplicate recordings | A copied clip would leak labels | Pass (max \|r\| = 0.18) |
| 13 | Label balance vs. base rate | Dug sites were not a random sample | **Flag: 50% leaks vs. ~12.5% historically**; probabilities must be re-anchored to the base rate |
| 14 | Coverage | Unlabeled kinds of sites mean predictions are extrapolations | **Flag: 12 PVC sites, none dug** |
| 15 | Label confounded with time or logger | The model could learn the clock instead of the leak | **Flag: recording time perfectly separates leak from no leak**; logger is balanced (2 A, 1 B in each group) |
| 16 | Feature outliers | Single events dominate whole-clip summaries | Flag: U35 (one impact, kurtosis 436), U40 |
| 17 | Feature redundancy | 15 features vs. 6 labels means overfitting risk | Info: two correlated blocks (loudness/burstiness and pitch) |
| 18 | Device effect (undug sites only) | The measurement may depend on the device | Logger B ~3x louder (p = 0.01) |
| 19 | Pipe effect (undug sites only) | Whether PVC sounds different | No clear difference, but there is no PVC leak to compare against |
| 20 | Clock-time effect (all clips) | Traffic noise rises toward dawn | Loudness rises with time (rho = +0.42); burstiness does not |

Checks 18–20 use the undug sites (or all clips), never the labels. That
gives them 40–46 data points instead of 6, and they cannot leak label
information.

## 2. Figures and what each is for

The tags mark what each figure is good for:
- **Lay:** readable by a non-technical audience.
- **Interpretable:** the axes have a plain physical meaning.
- **Accurate:** information-dense, but needs training to read.

| Fig | File | Data type shown | Tags |
|---|---|---|---|
| 1 | `eda_01_coverage_timeline` | Categorical metadata × clock time | Lay |
| 2 | `eda_02_coverage_bars` | Categorical counts | Lay |
| 3 | `eda_03_labeled_waveforms` | Raw time series | Lay |
| 4 | `eda_04_labeled_loudness_envelopes` | Derived time series (the crew chief's rule) | Lay, interpretable |
| 5 | `eda_05_labeled_spectrograms` | Time × frequency | Accurate |
| 6 | `eda_06_power_spectra` | Frequency profile | Accurate |
| 7 | `eda_07_amplitude_distributions` | Amplitude distribution vs. normal | Accurate |
| 8 | `eda_08_feature_dot_plots` | Per-clip features by group | Interpretable |
| 9 | `eda_09_features_by_clock_time` | Feature × time confound | Interpretable |
| 10 | `eda_10_device_and_pipe_effects` | Feature × device and pipe | Interpretable |
| 11 | `eda_11_feature_correlations` | Feature × feature | Accurate |
| 12 | `eda_12_principal_components` | Low-dimensional summary | Accurate, low interpretability |
| 13 | `eda_13_steadiness_vs_pitch` | Two field-note features | Lay, interpretable |
| 14 | `eda_14_quality_spotlight` | Flagged recordings | Lay |

**Accuracy vs. interpretability.** The most information-dense views are the
spectrograms, spectra and principal components (Figs 5, 6, 12). They also
expose the strongest signal, the 400–1,300 Hz band. But their axes are hard
to read, and principal components have no physical meaning at all. The
interpretable views (Figs 4, 8, 13) use quantities a crew chief would
recognize: loudness, steadiness and pitch. The cost is that whole-clip
summaries blur the evidence, which is why L06 hides among the non-leaks
in Fig 8.

**Best for a lay audience:** Fig 1 shows who was recorded when, and how
recording time is entangled with the labels. Fig 4 shows "steady hiss vs.
comes and goes." Fig 13 puts the field note on a two-axis map. Fig 14
shows what a bad recording looks like.

## 3. Key observations

1. **Recording time is confounded with the label.** All three leaks were
   recorded before 3:30 am and all three non-leaks after 4:15 am. A model
   trained on these six can learn "late = no leak." Six undug clips were
   recorded at or after 6:00 am.
2. **Coverage gap.** None of the dug sites are PVC, but 12 of the 40 undug
   sites are.
3. **Device effect.** Logger B is about 3x louder, so raw loudness partly
   measures the device. Features should be scale-free, or standardized
   within each logger.
4. **The strongest separating pattern is a sustained 400–1,300 Hz band.**
   All three leaks, including L06, have it and no non-leak does (Figs 5
   and 6). In an ad-hoc check, mean power in 400–1,300 Hz relative to
   250–380 Hz was +5.7 to +11.6 dB for the leaks and −1.0 to −1.6 dB for
   the non-leaks. With n = 6, that is suggestive, not proven. This band
   contrast is **not yet** part of `compute_clip_features.py` and should be
   added before modeling.
5. **Many undug clips are near-silent, steady background.** About 15 have
   RMS of roughly 0.004–0.009. Steadiness alone would score these as
   "hiss," so steadiness has to be paired with spectral shape.
6. **Two recordings have quality problems.** U40 is clipped and has a
   dropout. U35 is dominated by a single impact. Use burst-resistant
   features, such as median-window statistics, and flag both clips in the
   predictions.
