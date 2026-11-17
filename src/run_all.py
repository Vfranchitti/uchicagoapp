"""Rebuild every table, figure and the predictions file, in pipeline order.

Run from the repo root:  python -m src.run_all

Each step is an ordinary script in src/ that can also be run on its own
(python -m src.<name>). Later steps read files written by earlier ones, so
the order below matters; README.md describes what each step produces.
"""
import subprocess
import sys
import time

PIPELINE = (
    # 1. Data and quality checks
    'compute_clip_features',
    'run_data_checks',
    'run_quality_audit',
    'compute_base_rate_scenarios',
    # 2. Exploratory analysis and band measures
    'plot_eda_coverage',
    'plot_eda_signals',
    'plot_eda_features',
    'plot_eda_overview_grid',
    'analyze_noise_floor',
    'measure_band_geometry',
    'run_clustering',
    # 3. Model 1 (primary model) and the predictions deliverable
    'run_model_knowledge_score',
    'compute_model1_probabilities',
    'measure_band_evenness',
    'write_predictions',
    'plot_model1_results',
    # 4. Model 2 (logistic regression cross-check)
    'run_model_logistic',
    'plot_model2_results',
    # 5. Robustness and sanity checks
    'sanity_predict_pvc_leak',
    'sanity_compare_pvc_calls',
    'check_self_training',
)


def run_step(module_name):
    """Run one script as `python -m src.<name>` in its own process.

    A separate process per step means every step starts clean, exactly as
    when run by hand; the pipeline stops at the first step that fails.
    Returns the step's run time in seconds.
    """
    started = time.perf_counter()
    subprocess.run([sys.executable, '-m', f'src.{module_name}'], check=True)
    return time.perf_counter() - started


def run_pipeline():
    """Run every step in order, printing a one-line status for each."""
    for step_number, module_name in enumerate(PIPELINE, start=1):
        print(f'\n=== [{step_number}/{len(PIPELINE)}] {module_name} ===')
        seconds = run_step(module_name)
        print(f'--- {module_name} done in {seconds:.1f} s')


if __name__ == '__main__':
    run_pipeline()
