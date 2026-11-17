# Listening for Leaks: acoustic logger pilot analysis

Analysis for the Harris Social Impact Fellowship 2026–27 technical
assessment. A city water department recorded one 5-second clip of pipe
sound at each of 46 fire hydrants; crews have dug at 6 sites (3 leaks, 3
no leak). This project ranks the other 40 sites by how likely they are to
have a leak, recommends up to 5 sites to dig this week, and gives a leak
probability for each site. The task description is in
`docs/candidate_instructions.pdf`.

The approach in one sentence: a leak shows up as a steady band of hiss in a
fixed pitch range that never stops, so each recording is scored on whether
it has such a band (presence), how steady it is over time and across
pitches (steadiness), and how strongly it stands out (intenseness), then
converted to a probability anchored to the historical 1-in-8 leak rate.

## Where to find things

| Folder | What is in it |
|---|---|
| `data/` | The provided recordings and metadata (read-only; never modified) |
| `docs/` | Task instructions and Urban Labs reference briefs used for report style |
| `src/` | All analysis code (one script per step; see the pipeline table) |
| `outputs/deliverables/` | `predictions.csv`, the file to submit |
| `outputs/tables/` | Every intermediate table, in one subfolder per pipeline stage |
| `outputs/figures/` | Every figure: `1_eda/`, `2_model/`, `3_checks/` |
| `notes/` | Written notes: `eda_notes.md`, `part2_ideas.md`, `modeling/` (model choices and the assumptions checklist), `research/` (literature notes) |

## Setup

The code uses the conda environment in `environment.yml` (Python 3.12 with
numpy, pandas, scipy, scikit-learn, statsmodels, matplotlib and flake8).

```bash
conda env create -f environment.yml
```

On this machine conda is installed through Anaconda Desktop and is not on
the PATH, so call the environment's Python directly:

```powershell
& "C:\Users\<user>\.anaconda-desktop\micromamba\envs\cpu\envs\chicago\python.exe" -m src.run_all
```

In VS Code, open the repo folder, select the `chicago` interpreter, and
use the **Run full pipeline (src.run_all)** configuration in Run and Debug.

## How to run

From the repo root:

```bash
python -m src.run_all
```

This rebuilds everything in `outputs/` from `data/` in about two minutes.
Any single step can be run on its own with `python -m src.<script>`, as
long as the steps before it have run once. On Windows, set
`OMP_NUM_THREADS=1` to avoid a scikit-learn memory warning during
clustering (the VS Code configuration does this).

## Pipeline

| # | Script (`src/`) | What it does | Main outputs |
|---|---|---|---|
| 1 | `compute_clip_features.py` | Summarizes each recording (loudness, steadiness, pitch) | `tables/2_eda/clip_features.csv` |
| 2 | `run_data_checks.py` | 20 data checks: files, formats, clipping, dropouts, confounds | `tables/1_quality/data_checks.md` |
| 3 | `run_quality_audit.py` | Full quality audit by error type (rows, columns, cells, labels, coverage) | `tables/1_quality/quality_audit.md` |
| 4 | `compute_base_rate_scenarios.py` | How a 1-in-8 leak rate turns flags into false alarms | `tables/1_quality/base_rate_summary.md`, Fig 17 |
| 5 | `plot_eda_coverage.py` | Who was recorded, when, with which logger and pipe | Figs 1, 2, 16 |
| 6 | `plot_eda_signals.py` | Waveforms, loudness over time, spectrograms of the dug sites | Figs 3–7, 14 |
| 7 | `plot_eda_features.py` | Summary features and what besides leaks moves them | Figs 8–13 |
| 8 | `plot_eda_overview_grid.py` | Spectrograms of all 46 recordings by logger and pipe | Fig 15 |
| 9 | `analyze_noise_floor.py` | Each logger's own noise floor; sound above it | `tables/2_eda/noise_floor_*`, Figs 18–19 |
| 10 | `measure_band_geometry.py` | Edges, width and level of each contrasting band | `tables/2_eda/band_geometry.csv` |
| 11 | `run_clustering.py` | Unsupervised check: do the dug leaks group together? | `tables/2_eda/cluster_*`, Figs 20–21 |
| 12 | `run_model_knowledge_score.py` | **Model 1**: steady leak-band score, cast iron and PVC; leave-one-out and other checks | `tables/3_model1/model1_scores.csv`, Figs M1–M2 |
| 13 | `compute_model1_probabilities.py` | Turns Model 1 into leak probabilities (presence, then quality; PVC discount) | `tables/3_model1/model1_probabilities.csv` |
| 14 | `measure_band_evenness.py` | Band evenness across pitches (used by step 13) | `tables/2_eda/band_evenness.csv` |
| 15 | `write_predictions.py` | Fills the predictions template | `deliverables/predictions.csv` |
| 16 | `plot_model1_results.py` | Memo figures: dig list, how the score works, decision map, robustness grid | `figures/2_model/model1_fig1–4` |
| 17 | `run_model_logistic.py` | **Model 2**: Firth logistic regression on the Model 1 score (cross-check) | `tables/4_model2/` |
| 18 | `plot_model2_results.py` | Model 2 curve; Model 1 vs. Model 2 | `figures/2_model/model2_fig5–6` |
| 19 | `sanity_predict_pvc_leak.py` | Literature-based picture of a PVC leak (**not** used by any model) | `figures/3_checks/` |
| 20 | `sanity_compare_pvc_calls.py` | Compares final PVC calls with that picture | `tables/5_checks/` |
| 21 | `check_self_training.py` | Label spreading and self-training robustness checks | `tables/5_checks/check_self_training.md` |

Shared code: `config.py` (every path and constant, defined once),
`load_recordings.py` (reads the data), `utils.py` (small helpers) and
`plot_style.py` (Urban Labs colors, fonts and figure helpers).

## Key decisions (details in `notes/modeling/`)

- **U40 is excluded** for insufficient data quality (signal saturation and a
  1.2 s dropout); it gets no prediction.
- **Recording time is never used as a predictor**, even though it splits
  the six dug sites perfectly; it is treated as a traffic confound.
- **Logger differences are removed** by subtracting each logger's own noise
  floor (logger B records about 5.8 dB louder).
- **Probabilities are anchored to the 1-in-8 base rate**, not to the 50%
  leak share among the six dug sites.
- **PVC sites** are scored on a lower band (250–400 Hz) and discounted by
  how many leak-signature properties their band shares, because no PVC
  site has been dug.
- **Outside literature is used only as a sanity check**, never as a model
  input.

## Before submitting

- Rename `outputs/deliverables/predictions.csv` with the application
  reference number (e.g., `1234567_predictions.csv`).
- Do not include `.vscode/`: its settings contain a local user path.
- Code style: PEP 8, checked with `flake8 src/`.
