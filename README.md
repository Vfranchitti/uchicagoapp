# Listening for Leaks: acoustic logger pilot analysis

Analysis for the Harris Social Impact Fellowship 2026–27 technical
assessment. A city water department recorded one 5-second clip of pipe
sound at each of 46 fire hydrants. Crews have dug at 6 sites: 3 had leaks
and 3 did not. This project ranks the other 40 sites by how likely they
are to have a leak, recommends 5 sites to dig this week, and gives a leak
probability for each site. The task description is in
`docs/candidate_instructions.pdf`.

## Results

| Dig | Site | Pipe | Leak probability (range) |
|---|---|---|---|
| 1 | U07 | Cast iron | 50% (36–50%) |
| 2 | U17 | Cast iron | 50% (36–50%) |
| 3 | U08 | Cast iron | 49.5% (2–50%) |
| 4 | U36 | Cast iron | 44% (2–50%) |
| 5 | U34 | PVC | 35% (27–35%), informative dig |

U07 and U17 are tied under the model (both at the top of the band tier,
50%). Band strength (the uncapped corrected band level) is used only as a
one-time tie-breaker for dig order; it does not change any confidence
value. A typical surveyed site has a 12.5% (1 in 8) chance. Six dug sites
can lift a site to about 50% at most, so every site with a steady band is
marked `leak`, meaning worth digging, although none exceeds 0.5.

**Deliverables** (file names start with the applicant reference number):

| File | What it is |
|---|---|
| `memo/552098215_memo.pdf` (and `.docx`) | Two-page write-up with figures; supplemental notes, sources and AI-use statement from page 3 |
| `outputs/deliverables/552098215_predictions.csv` | Predictions in the template's three columns: `clip_id`, `prediction` (leak / no_leak), `confidence` (probability of a leak). U40 is unusable, so it gets the 1-in-8 base rate and `no_leak` |
| `outputs/tables/3_model1/predictions_detail.csv` | Supplement: each site's range across analysis choices and a short note |

## Method in brief

With 6 labeled sites, any model that learns many details will memorize
them. The analysis therefore uses a rule written from the crew chief's
description of a leak ("a steady hiss ... it doesn't stop"), and uses the
6 dug sites to check the rule and to estimate how much evidence it gives.

1. **Clean the inputs.** U40 is excluded: 15% of its samples are clipped
   at the sensor limit and it has a 1.2 s dropout. Each logger's own noise
   floor, measured on its quiet recordings, is subtracted (logger B records
   about 6 dB louder with the same spectral shape). Recording time is never
   used as an input: it splits the dug sites perfectly, which is treated as
   a traffic confound.
2. **Measure the steady sound only.** For each pitch, take the level the
   sound stays above for 90% of the recording (the 10th percentile over
   250 ms frames; the L90 statistic used for background sound in
   environmental acoustics). Passing traffic is brief, so it drops out
   without discarding any data; a leak does not stop, so it stays.
3. **Find steady bands without fixing a range in advance**
   (`src/detect_steady_bands.py`). A band is a stretch of pitches at least
   3 dB above the shared background of recordings of the same pipe
   material, at least 50 Hz wide, and at least 3 dB above the pitches on
   both sides. Validated before use on cast iron: it finds the confirmed
   leak band (about 380–1,320 Hz) in all 3 dug leaks and in none of the 3
   dug non-leaks.
4. **Presence sets the probability tier.** Cast iron: a detected band
   covering at least half of 400–1,300 Hz. PVC: any detected band (all
   three found sit near 320–450 Hz). Bayes' rule updates the 1-in-8 base
   rate with the dug-site evidence (band in 3 of 3 leaks, 0 of 3
   non-leaks), smoothed two standard ways (Jeffreys and Laplace) because
   six sites cannot justify certainty: **36–50%** with a band, **2–3%**
   without. This is the standard positive-predictive-value calculation
   for a screening test.
5. **Band quality places a site within its tier**, on the log-odds scale:
   quality = steadiness^0.6 × intenseness^0.4. Steadiness = persistence
   over time × coverage across the band's pitches; intenseness = the
   band's steady power after subtracting its neighbors' (so traffic does
   not lower it), relative to the weakest confirmed leak and capped at 1.
   The weights are a judgment (steadiness first); the order of the band
   sites is identical under 0.5/0.5, 0.8/0.2 and 1.0/0.0.
6. **PVC is discounted.** No PVC site has been dug, so each PVC band site
   is pulled toward the base rate by the share w of five confirmed-leak
   properties its band shares: p = w·p + (1 − w)·12.5%.
7. **Dig list:** the four cast-iron band sites (ties broken by the
   stronger band), plus U34 as an informative dig: the strongest PVC band,
   whose outcome would give the first labeled PVC example.

## Key assumptions

| | Assumption | Status |
|---|---|---|
| A1 | The six dig outcomes are correct | Cannot be checked |
| A2 | One 5-second clip represents a site; a leak does not stop | Supported by the crew chief |
| A3 | Cast-iron leaks sound alike | Three near-identical examples |
| A4 | After floor subtraction, loggers A and B are comparable | Supported: same shape; L03 (B) and L01/L06 (A) match within 0.7 dB |
| A5 | Recording time is unrelated to leaks | Assumed |
| A6 | About 1 in 8 undug sites leaks | Historical rate |
| A7 | Sites are independent | Unknown (no locations) |
| A8 | A PVC leak is also a steady band, at a lower pitch | **Weakest.** The same 320–450 Hz band also appears at a cast-iron site (U02) on the same logger as U34, and published PVC leak spectra do not predict it; PVC probabilities should be read as upper bounds |

## Robustness checks

Leave-one-out over the six dug sites; raw spectra without floor
subtraction; clustering without labels (k-means and Ward); label spreading
and self-training; an alternative model (Firth-penalized logistic
regression on the band score, re-anchored to 1 in 8, 14 refits); the
quality-weight sensitivity above; and a literature-based check of the PVC
calls, kept out of the model. Results are in `outputs/tables/`. The
cast-iron dig set is stable across checks; U08 and U36 are the least
stable (their bands fall below a cut-off in the raw spectra, hence ranges
down to 2%).

**Known limitations.** The confidence values sum to 3.4 expected leaks
across 39 sites, against about 4.9 expected at 1 in 8, so some leaks are
probably missed (most likely silent, intermittent or distant ones, or on
PVC). Steadiness (persistence over time × coverage across pitch) is
still measured on 64 ms frames and on whole-clip spectra, where traffic
can hide part of a band: U03 passes the steady-band test, but its band is
present in 76% of frames and has no coverage in its whole-clip spectrum,
so it sits at the bottom of the band tier. The recordings look synthesized (brick-wall band edges,
identical leak levels), so results describe this pilot, not field
performance.

## Where to find things

| Folder | What is in it |
|---|---|
| `data/` | The provided recordings and metadata (read-only; never modified) |
| `docs/` | Task instructions and reference briefs used for report style |
| `src/` | All analysis code (one script per step; see the pipeline table) |
| `outputs/deliverables/` | `552098215_predictions.csv` |
| `outputs/tables/` | Every intermediate table, in one subfolder per pipeline stage |
| `outputs/figures/` | Every figure: `1_eda/`, `2_model/`, `3_checks/`, `4_memo/`; `2_model/method_decision_tree.png` shows every decision and the alternatives considered |
| `memo/` | The write-up (`552098215_memo.pdf` / `.docx`) |
| `notes/` | `eda_notes.md`, `modeling/` (model choices and the assumptions checklist), `research/` (literature notes) |

## How to run

The code uses the conda environment in `environment.yml` (Python 3.12 with
numpy, pandas, scipy, scikit-learn, statsmodels, matplotlib and flake8).
From the repo root:

```bash
conda env create -f environment.yml
conda activate chicago
python -m src.run_all
```

This rebuilds everything in `outputs/` from `data/` in a few minutes. Any
single step can be run on its own with `python -m src.<script>`, once the
steps before it have run. On Windows, set `OMP_NUM_THREADS=1` to avoid a
scikit-learn memory warning during clustering.

## Pipeline

| # | Script (`src/`) | What it does | Main outputs |
|---|---|---|---|
| 1 | `compute_clip_features.py` | Summarizes each recording (loudness, steadiness, pitch) | `tables/2_eda/clip_features.csv` |
| 2 | `run_data_checks.py` | 20 data checks: files, formats, clipping, dropouts, confounds | `tables/1_quality/data_checks.md` |
| 3 | `run_quality_audit.py` | Quality audit by error type (rows, columns, cells, labels, coverage) | `tables/1_quality/quality_audit.md` |
| 4 | `compute_base_rate_scenarios.py` | How a 1-in-8 leak rate turns flags into false alarms | `tables/1_quality/base_rate_summary.md` |
| 5 | `plot_eda_coverage.py` | Who was recorded, when, with which logger and pipe | `figures/1_eda/` |
| 6 | `plot_eda_signals.py` | Waveforms, loudness over time, spectrograms of the dug sites | `figures/1_eda/` |
| 7 | `plot_eda_features.py` | Summary features and what besides leaks moves them | `figures/1_eda/` |
| 8 | `plot_eda_overview_grid.py` | Spectrograms of all 46 recordings by logger and pipe | `figures/1_eda/` |
| 9 | `analyze_noise_floor.py` | Each logger's own noise floor; sound above it | `tables/2_eda/noise_floor_*` |
| 10 | `measure_band_geometry.py` | Edges, width and level of each contrasting band | `tables/2_eda/band_geometry.csv` |
| 11 | `run_clustering.py` | Unsupervised check: do the dug leaks group together? | `tables/2_eda/cluster_*` |
| 12 | `run_model_knowledge_score.py` | **Model 1** measures: band contrast and persistence, steady-band presence, corrected band level; leave-one-out and other checks | `tables/3_model1/model1_scores.csv` |
| 13 | `detect_steady_bands.py` | Steady-band detection report: bands found, cast-iron validation, logger cross-check | `tables/3_model1/band_scan_*`, `figures/2_model/band_scan_map.png` |
| 14 | `compute_model1_probabilities.py` | Leak probabilities: presence tier, then quality; PVC discount; weight sensitivity | `tables/3_model1/model1_probabilities.csv` |
| 15 | `measure_band_evenness.py` | Band coverage across pitches (used by step 14) | `tables/2_eda/band_evenness.csv` |
| 16 | `write_predictions.py` | Fills the predictions template; writes ranges and notes separately | `deliverables/552098215_predictions.csv`, `tables/3_model1/predictions_detail.csv` |
| 17 | `plot_model1_results.py` | Model 1 figures: dig list, how the score works, decision map, robustness grid | `figures/2_model/model1_fig1–4` |
| 18 | `run_model_logistic.py` | **Model 2** (alternative model): Firth logistic regression on the Model 1 score | `tables/4_model2/` |
| 19 | `plot_model2_results.py` | Model 2 curve; Model 1 vs. Model 2 | `figures/2_model/model2_fig5–6` |
| 20 | `sanity_predict_pvc_leak.py` | Literature-based picture of a PVC leak (**not** used by any model) | `figures/3_checks/` |
| 21 | `sanity_compare_pvc_calls.py` | Compares the PVC calls with that picture | `tables/5_checks/` |
| 22 | `check_self_training.py` | Label spreading and self-training robustness checks | `tables/5_checks/check_self_training.md` |
| 23 | `plot_method_tree.py` | Decision tree: route taken and options considered at each step | `figures/2_model/method_decision_tree.png` |
| 24 | `plot_memo_figures.py` | The four memo figures (recording timeline, leak signature, chances by site, robustness) | `figures/4_memo/` |

Shared code: `config.py` (every path and constant, defined once, with the
reason for each setting), `load_recordings.py` (reads the data),
`spectral_inputs.py` (spectra shared by Model 1 and its checks),
`utils.py` (small helpers) and `plot_style.py` (figure style). Code style:
PEP 8, checked with `flake8 src/`.

The steady-band rules were fixed before running and revised only in their
mechanics (thresholds unchanged) when a run exposed a defect; each
revision and its reason is recorded in `config.py`, and the first two runs
are kept as `band_scan_run1_*` and `band_scan_run2_*` in
`outputs/tables/3_model1/`.
