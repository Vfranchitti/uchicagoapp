# Exploratory data analysis: plan and results

Reproduce everything from the repo root with `python -m src.run_all` (see
README.md). The exploratory steps are `src.compute_clip_features`,
`src.run_data_checks`, `src.run_quality_audit` and the four
`src.plot_eda_*` scripts; tables land in `outputs/tables/1_quality/` and
`outputs/tables/2_eda/`, figures in `outputs/figures/1_eda/`.

Framework: total error (Foster et al., *Big Data and Social Science*,
ch. 10). Row errors are about coverage (who is in the data), column errors
are about specification (what a variable really measures), and cell errors
are about measurement (whether each value is right). Sample bias and
confounding come from ch. 11, and visualization principles from ch. 6.

## 0. Working assumptions (agreed 2026-11-17)

The task says loggers were "left overnight to record short clips of sound
from the pipe." We assume:

1. **One 5-second clip per site,** probably a cost or device limit. There is
   no repeat measurement to average, so each site's features carry the full
   noise of a single clip.
2. **Clip timing was effectively random.** Recording time should then be
   unrelated to leak status. The perfect time split among the 6 dug sites
   (leaks before 3:30 am, non-leaks after 4:15 am) is chance, or comes from
   how dug sites were chosen. It is not a leak signal, so time must not be
   used as a predictor.
3. **Loggers were allocated in proportion across pipe types and
   assigned randomly within each pipe type** (updated 2026-11-17).
   - Each logger covers about the same share of PVC sites: A on 8 of its
     30 sites (27%), B on 4 of its 16 (25%).
   - "What we had in the shop" suggests there was no further targeting.
   - The ~3x loudness difference between loggers is then a device effect,
     not a site difference. That justifies standardizing features within
     each logger model.

Still unknown: how the 6 dug sites were chosen. Their 50% leak share
suggests they were chosen because they seemed suspicious.

## 1. Data checks, in order of importance

Checks earlier in the list can invalidate everything after them. Full
findings are in `outputs/tables/1_quality/data_checks.md`.

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
| 15 | `eda_15_all_spectrograms` | Time × frequency, all 46 clips, by logger × pipe | Accurate |
| 16 | `eda_16_descriptive_overview` | Counts by logger, pipe, dig status, clock hour | Lay |
| 17 | `eda_17_base_rate_false_alarms` | Positive predictive value at 1 in 8 vs. 50% | Lay |
| 18 | `eda_18_logger_noise_floors` | Logger A vs. B noise floors (U34 excluded) | Accurate |
| 19 | `eda_19_excess_over_noise_floor` | dB above own logger's floor, all clips | Accurate |
| 20 | `eda_20_cluster_dendrogram` | Ward clustering + feature profiles (U40 excluded) | Accurate |
| 21 | `eda_21_cluster_map` | K-means clusters on two key features (U40 excluded) | Interpretable |

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
7. **The recordings look synthesized** (`notes/research/cast_iron_realism_checks.md`).
   - **Realistic:** the leak band's pitch (field studies of iron mains
     average 220–1,400 Hz), steadiness, the recording format, and 120 Hz
     hum at U02, U15, U16 and U22.
   - **Unrealistic:** brick-wall band edges (about 20 dB within one 4 Hz
     step), three leaks of identical loudness, flat bands with no
     resonances, and a few noise blocks reused across sites.
   - **What it means:** state in the memo that performance on real pipes is
     untested. The literature also says the cast-iron band rule should not
     be used on PVC (`notes/research/pvc_attenuation_research.md`).
6. **Two recordings have quality problems.** U40 is clipped and has a
   dropout. U35 is dominated by a single impact. Use burst-resistant
   features, such as median-window statistics, and flag both clips in the
   predictions.
