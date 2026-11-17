"""SANITY CHECK ONLY. Compare the final PVC calls with the literature view.

Reads the model's PVC probabilities
(outputs/tables/3_model1/model1_probabilities.csv) and the literature-based
reading of each PVC site
(outputs/tables/5_checks/sanity_pvc_sites_vs_prediction.csv, from
sanity_predict_pvc_leak.py). Nothing here feeds the model.

For each PVC site, it states whether the model call and the literature
reading agree, and why they may not.

Output: outputs/tables/5_checks/sanity_pvc_vs_model_calls.csv
"""
import pandas as pd

from src.config import (
    HISTORICAL_LEAK_RATE,
    MODEL1_PROBABILITY_PATH,
    SANITY_PVC_SITE_TABLE_PATH,
    SANITY_PVC_VS_MODEL_PATH,
    TABLE_DECIMALS,
)

AT_FLOOR = 'at floor'
BELOW_400 = 'energy only below 400 Hz'
FLAT_OR_RISING = 'flat or rising'


def _verdict(prediction, reading):
    """How the model call relates to the literature-based reading."""
    if reading.startswith(AT_FLOOR):
        if prediction == 0:
            return ('agree on no detectable leak; literature adds that a '
                    'distant PVC leak would also look like this, so "no '
                    'leak" here is weak evidence')
        return 'disagree: model flags a site with nothing above the floor'
    if reading.startswith(BELOW_400):
        if prediction == 1:
            return ('model flags; literature cannot produce this band by '
                    'filtering a cast-iron leak, so the call rests on the '
                    'assumed PVC signature (unconfirmed)')
        return 'agree the band is not a filtered cast-iron leak'
    if reading.startswith(FLAT_OR_RISING):
        if prediction == 0:
            return ('agree: energy up to 1,300 Hz is unlike a PVC leak '
                    '(plastic removes high pitches)')
        return 'disagree: model flags a spectrum unlike a PVC leak'
    return ('literature shape of a near-hydrant leak; model '
            + ('agrees' if prediction == 1 else 'does not flag it'))


def compare_pvc_calls():
    """One row per PVC site: model call, literature reading, verdict."""
    model = pd.read_csv(MODEL1_PROBABILITY_PATH)
    literature = pd.read_csv(SANITY_PVC_SITE_TABLE_PATH)
    merged = literature.merge(
        model[['clip_id', 'prediction', 'confidence', 'probability_low',
               'probability_high']], on='clip_id', how='left')
    merged['verdict'] = [_verdict(prediction, reading) for prediction,
                         reading in zip(merged['prediction'],
                                        merged['reading'])]
    merged['above_base_rate'] = merged['confidence'] > HISTORICAL_LEAK_RATE
    merged = merged.sort_values('confidence', ascending=False)
    merged.round(TABLE_DECIMALS).to_csv(SANITY_PVC_VS_MODEL_PATH, index=False)
    return merged


if __name__ == '__main__':
    with pd.option_context('display.width', 220,
                           'display.max_colwidth', 120):
        print(compare_pvc_calls()[['clip_id', 'confidence', 'prediction',
                                   'reading', 'verdict']]
              .to_string(index=False))
