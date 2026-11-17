# Sanity check: what might a PVC leak look like? (NOT part of the model)

**Status: held aside.** Nothing here feeds the features, clustering or
model. The model works only from the pilot data provided. Use this at the
end of modeling to sanity-check the model's PVC calls.

Code: `src/sanity_predict_pvc_leak.py`.
Figure: `outputs/figures/3_checks/sanity_predicted_pvc_leak.png`.
Tables:
- `outputs/tables/5_checks/sanity_predicted_pvc_leak.csv`
- `outputs/tables/5_checks/sanity_pvc_sites_vs_prediction.csv`

## How the prediction was made

1. Start from the three dug cast-iron leaks. Their median band is about
   +21 dB above the logger floor, from 400 to 1,300 Hz.
2. Apply published plastic-pipe loss. Loss rises in step with pitch:
   dB per metre = 8.686 × β × 2π × f (Almeida et al. 2014), with β from
   1 × 10⁻⁴ to 5 × 10⁻⁴ s/m.
3. PVC loses at least 5 times more than iron (Hunaidi et al. 2000), so the
   extra loss over iron is 4/5 of the PVC loss.
4. Repeat for a leak 2, 5, 10 and 25 m from the hydrant.

## Predicted PVC leak (dB above floor)

A band only counts as detectable if it is at least 3 dB above the floor.

| Leak distance | Least lossy pipe | Most lossy pipe |
|---|---|---|
| 2 m | 404–1,300 Hz, peak +18 dB, fading with pitch | 404–476 Hz, peak +5 dB |
| 5 m | 404–952 Hz, peak +13 dB | nothing above floor |
| 10 m | 404–476 Hz, peak +5 dB | nothing above floor |
| 25 m | nothing above floor | nothing above floor |

**In words:** a PVC leak more than about 5–10 m from the hydrant is
expected to be **invisible** in this pilot's recordings. A close one would
look like the bottom part of the cast-iron band, strongest near 400 Hz and
fading toward 1,300 Hz.

## How the 12 PVC sites compare

| Reading | Sites |
|---|---|
| At the floor: no leak, or a leak too far away to hear | U18, U19, U20, U25, U33 |
| Energy only below 400 Hz: cannot come from filtering the cast-iron band | U34, U15 |
| Flat or rising to 1,300 Hz: unlike a PVC leak, because plastic removes high pitches | U03, U28, U29, U31, U32 |
| Fades with pitch, the near-leak shape | none |

## How to use this at the end of modeling

- **Model ranks a PVC site highly because of 400–1,300 Hz energy** (e.g.,
  U32, U28, U03): the prediction says that energy is unlikely to come from
  a PVC leak. Flag it.
- **Model calls a quiet PVC site "no leak"** (U18–U20, U25, U33): the
  prediction says a distant PVC leak would also look quiet, so "no leak" is
  low confidence, not confirmed.
- **U34 and U15:** their band sits below anything a filtered cast-iron leak
  could produce. If they are leaks, the data generator gives PVC leaks a
  different signature. Literature cannot confirm that, and the sharp edges
  suggest a planted band, so it could also be a decoy. One exploratory dig
  would settle it.

## Caveats

- Filtering can only remove pitches. It cannot create the low-pitch energy
  (below about 150 Hz) where published PVC leaks mostly sit.
- The logger's low-pitch response and the leak-to-hydrant distances are
  unknown.
- The recordings appear synthesized (`notes/research/cast_iron_realism_checks.md`).
