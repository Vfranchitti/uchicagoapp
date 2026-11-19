"""Fill the predictions template from Model 1's probabilities.

Deliverable (exactly the template's columns)
--------------------------------------------
clip_id, prediction (leak / no_leak), confidence

confidence is the probability from 0 to 1 that the site has a leak, as the
instructions define it (main-analysis estimate; see
compute_model1_probabilities.py). prediction is 'leak' for every site with a
persistent band, meaning "worth digging", although no site exceeds 0.5 (the
evidence from six dug sites cannot lift a 1-in-8 site above about even).

U40 is unusable (sensor overload and a dropout), so its recording carries no
evidence: it gets the 1-in-8 base rate and 'no_leak' (no band was found).

Supplement (ranges and notes, kept out of the deliverable)
----------------------------------------------------------
confidence_low, confidence_high, leak_flag, note

Outputs
-------
outputs/deliverables/<reference>_predictions.csv
outputs/tables/3_model1/predictions_detail.csv
"""
import pandas as pd

from src.config import (
    ANALYSIS_EXCLUDED_CLIPS,
    HISTORICAL_LEAK_RATE,
    INFORMATIVE_DIG_SITE,
    MODEL1_PROBABILITY_PATH,
    PREDICTIONS_DETAIL_PATH,
    PREDICTIONS_OUTPUT_PATH,
    PREDICTIONS_TEMPLATE_PATH,
)
from src.utils import make_output_dirs

LEAK_WORD, NO_LEAK_WORD = 'leak', 'no_leak'
PROBABILITY_DECIMALS = 3
TEMPLATE_COLUMNS = ['clip_id', 'prediction', 'confidence']
EXCLUDED_NOTE = ('unusable recording (about 15% of samples clipped at the '
                 'sensor limit and a 1.2 s dropout), so it carries no '
                 'evidence: confidence is the 1-in-8 base rate')


def _note(record):
    """Short explanation for rows a reader may ask about."""
    notes = []
    if record.ranking == 'pvc' and record.prediction == 1:
        notes.append('steady PVC band near 320-450 Hz; unconfirmed signature')
    if record.prediction == 1 and record.probability_low < record.confidence:
        if record.probability_low < HISTORICAL_LEAK_RATE:
            notes.append('band falls below a cut-off in raw version')
    if record.clip_id == INFORMATIVE_DIG_SITE:
        notes.append('dig 5 (informative)')
    elif record.dig_priority == record.dig_priority:  # not NaN
        notes.append(f'dig {int(record.dig_priority)}')
    return '; '.join(notes)


def build_predictions():
    """All rows: prediction, confidence, range and note per site."""
    template = pd.read_csv(PREDICTIONS_TEMPLATE_PATH)
    probabilities = pd.read_csv(MODEL1_PROBABILITY_PATH)
    probabilities['note'] = [_note(r) for r in probabilities.itertuples()]
    merged = template[['clip_id']].merge(probabilities, on='clip_id',
                                         how='left', validate='one_to_one')
    is_excluded = merged['clip_id'].isin(ANALYSIS_EXCLUDED_CLIPS)
    missing = merged.loc[merged['confidence'].isna() & ~is_excluded,
                         'clip_id'].tolist()
    if missing:
        raise ValueError(f'No probability for: {missing}')
    for column in ('confidence', 'probability_low', 'probability_high'):
        merged.loc[is_excluded, column] = HISTORICAL_LEAK_RATE
    merged.loc[is_excluded, 'prediction'] = 0
    merged.loc[is_excluded, 'note'] = EXCLUDED_NOTE
    leak_flag = merged['prediction'].astype(int)
    return pd.DataFrame({
        'clip_id': merged['clip_id'],
        'prediction': leak_flag.map({1: LEAK_WORD, 0: NO_LEAK_WORD}),
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
    make_output_dirs()
    predictions[TEMPLATE_COLUMNS].to_csv(PREDICTIONS_OUTPUT_PATH, index=False)
    predictions.to_csv(PREDICTIONS_DETAIL_PATH, index=False)
    print(f'Wrote {len(predictions)} rows to {PREDICTIONS_OUTPUT_PATH}')
    print(predictions.to_string(index=False))
