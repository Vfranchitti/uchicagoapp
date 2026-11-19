# Band scan: steady bands found without a fixed range

- U40 excluded (insufficient data quality): signal saturation (15% of samples clipped) and a 1.2 s dropout make its features invalid.
- Rules (fixed before running): steady level = 10th percentile over 64 ms frames; band = at least 3 dB above the shared background of its material, at least 50 Hz wide, and at least 3 dB above the 100 Hz on both sides.
- Revision after run 1 (mechanics only, thresholds unchanged): spectra smoothed over 3 bins; runs split at dips of 3 dB or more; noise floor on the same percentile basis. Run 1 is kept in band_scan_run1_summary.md.

## Validation on cast iron (stated before running)

- Passes: **yes**.
- Dug leaks with a band overlapping 400-1,300 Hz: ['L01', 'L03', 'L06'] (missed: []).
- Dug non-leaks flagged: [].
- Undug cast-iron sites with such a band: ['U07', 'U17', 'U36'].

## Cast iron (33 recordings)

- **359-1328 Hz**: 6 of 33 recordings (L01, L03, L06, U07, U17, U36).

| Recording | Logger | Status | Band (Hz) | Width (Hz) | dB above floor | dB above shared | dB above sides (below / above) |
|---|---|---|---|---|---|---|---|
| L03 | B | leak | 359-1328 | 984 | 20.7 | 18.4 | 20.1 / 19.2 |
| L01 | A | leak | 375-1328 | 969 | 20.7 | 18.3 | 19.7 / 20.9 |
| L06 | A | leak | 375-1328 | 969 | 20.6 | 18.2 | 16.8 / 17.2 |
| U07 | B | unlabeled | 375-1328 | 969 | 20.9 | 18.5 | 16.8 / 17.1 |
| U17 | A | unlabeled | 375-1328 | 969 | 20.7 | 18.4 | 19.0 / 19.7 |
| U36 | A | unlabeled | 391-1312 | 938 | 10.8 | 8.6 | 7.6 / 8.1 |

- Runs above the shared background that failed a rule: 74 (too narrow: 27; not distinct from both sides: 74).

## PVC (12 recordings)

- **125-438 Hz**: 1 of 12 recordings (U34).

| Recording | Logger | Status | Band (Hz) | Width (Hz) | dB above floor | dB above shared | dB above sides (below / above) |
|---|---|---|---|---|---|---|---|
| U34 | A | unlabeled | 125-438 | 328 | 7.4 | 7.1 | 7.2 / 7.1 |

- Runs above the shared background that failed a rule: 22 (too narrow: 11; not distinct from both sides: 22).
