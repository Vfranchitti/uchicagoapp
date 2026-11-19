# Band scan: steady bands found without a fixed range

- U40 excluded (insufficient data quality): signal saturation (15% of samples clipped) and a 1.2 s dropout make its features invalid.
- Rules (fixed before running): steady level = 10th percentile over 64 ms frames; band = at least 3 dB above the shared background of its material, at least 50 Hz wide, and at least 3 dB above the 100 Hz on both sides.

## Validation on cast iron (stated before running)

- Passes: **yes**.
- Dug leaks with a band overlapping 400-1,300 Hz: ['L01', 'L03', 'L06'] (missed: []).
- Dug non-leaks flagged: [].
- Undug cast-iron sites with such a band: ['U02', 'U07', 'U17', 'U36'].

## Cast iron (33 recordings)

- **312-1328 Hz**: 7 of 33 recordings (L01, L03, L06, U02, U07, U17, U36).

| Recording | Logger | Status | Band (Hz) | Width (Hz) | dB above floor | dB above shared | dB above sides (below / above) |
|---|---|---|---|---|---|---|---|
| U02 | A | unlabeled | 312-422 | 125 | 5.1 | 11.5 | 5.0 / 12.2 |
| L01 | A | leak | 375-1328 | 969 | 12.5 | 18.1 | 19.9 / 20.5 |
| L03 | B | leak | 375-1312 | 953 | 12.7 | 18.5 | 19.7 / 18.8 |
| L06 | A | leak | 375-1328 | 969 | 12.3 | 18.1 | 17.2 / 16.9 |
| U07 | B | unlabeled | 375-1328 | 969 | 12.9 | 18.7 | 16.6 / 17.9 |
| U17 | A | unlabeled | 375-1328 | 969 | 12.4 | 18.2 | 18.8 / 20.2 |
| U36 | A | unlabeled | 391-1312 | 938 | 2.6 | 8.2 | 7.6 / 8.1 |

- Runs above the shared background that failed a rule: 99 (too narrow: 47; not distinct from both sides: 99).

## PVC (12 recordings)

- **125-438 Hz**: 1 of 12 recordings (U34).

| Recording | Logger | Status | Band (Hz) | Width (Hz) | dB above floor | dB above shared | dB above sides (below / above) |
|---|---|---|---|---|---|---|---|
| U34 | A | unlabeled | 125-438 | 328 | -1.2 | 6.5 | 6.9 / 7.1 |

- Runs above the shared background that failed a rule: 32 (too narrow: 24; not distinct from both sides: 30).
