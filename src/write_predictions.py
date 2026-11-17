"""Fill the predictions template from Model 1's probabilities.

Columns
-------
clip_id, prediction (leak / no_leak), confidence    template columns
confidence_low, confidence_high                     sensitivity range
leak_flag (1 / 0)                                   same call as prediction
note                                                why a row is special

confidence is the main-analysis probability (see
compute_model1_probabilities.py). Excluded clips (U40) have no score, so
they get no prediction: 'excluded from analysis', blank confidence, and a note.

Output: outputs/deliverables/predictions.csv
"""
import pandas as pd

from src.config import (
    ANALYSIS_EXCLUDED_CLIPS,
    HISTORICAL_LEAK_RATE,
    INFORMATIVE_DIG_SITE,
    MODEL1_PROBABILITY_PATH,
    PREDICTIONS_OUTPUT_PATH,
    PREDICTIONS_TEMPLATE_PATH,
)

LEAK_WORD, NO_LEAK_WORD = 'leak', 'no_leak'
PROBABILITY_DECIMALS = 3
EXCLUDED_PREDICTION = 'excluded: insufficient data quality'
EXCLUDED_NOTE = ('low-quality recording, not fit for analysis: signal '
                 'saturation (about 15% of samples clipped at sensor '
                 'full scale) and a 1.2 s dropout (constant zero-valued '
                 'samples); amplitude and spectral features are invalid, '
                 'so no prediction is made')


def _note(record):
    """Short explanation for rows a reader may ask about."""
    notes = []
    if record.ranking == 'pvc' and record.prediction == 1:
        notes.append('PVC 250-400 Hz band; assumed leak signature')
    if record.prediction == 1 and record.probability_low < record.confidence:
        if record.probability_low < HISTORICAL_LEAK_RATE:
            notes.append('band falls below a cut-off in raw version')
    if record.clip_id == INFORMATIVE_DIG_SITE:
        notes.append('dig 5 (informative)')
    elif record.dig_priority == record.dig_priority:  # not NaN
        notes.append(f'dig {int(record.dig_priority)}')
    return '; '.join(notes)


def build_predictions():
    """Template rows filled with prediction, confidence, and range."""
    template = pd.read_csv(PREDICTIONS_TEMPLATE_PATH)
    probabilities = pd.read_csv(MODEL1_PROBABILITY_PATH)
    probabilities['note'] = [_note(r) for r in probabilities.itertuples()]
    merged = template[['clip_id']].merge(probabilities, on='clip_id',
                                         how='left', validate='one_to_one')
    is_excluded = merged['clip_id'].isin(ANALYSIS_EXCLUDED_CLIPS)
    merged.loc[is_excluded, 'note'] = EXCLUDED_NOTE
    missing = merged.loc[merged['confidence'].isna() & ~is_excluded,
                         'clip_id'].tolist()
    if missing:
        raise ValueError(f'No probability for: {missing}')
    leak_flag = merged['prediction'].astype('Int64')
    prediction = leak_flag.map({1: LEAK_WORD, 0: NO_LEAK_WORD}).astype(
        object)
    prediction[is_excluded] = EXCLUDED_PREDICTION
    return pd.DataFrame({
        'clip_id': merged['clip_id'],
        'prediction': prediction,
        'confidence': merged['confidence'].round(PROBABILITY_DECIMALS),
        'confidence_low': merged['probability_low'].round(
            PROBABILITY_DECIMALS),
        'confidence_high': merged['probability_high'].round(
            PROBABILITY_DECIMALS),
        'leak_flag': leak_flag,
        'note': merged['note'].fillna(''),
    })


if __name__ == '__main__':
    predictions = build_predictions()
    predictions.to_csv(PREDICTIONS_OUTPUT_PATH, index=False)
    print(f'Wrote {len(predictions)} rows to {PREDICTIONS_OUTPUT_PATH}')
    print(predictions.to_string(index=False))
