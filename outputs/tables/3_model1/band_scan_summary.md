# Steady-band detection (Model 1 presence test)

- U40 excluded (insufficient data quality): signal saturation (15% of samples clipped) and a 1.2 s dropout make its features invalid.
- Rules (fixed before running): steady level = 10th percentile over 64 ms frames; band = at least 3 dB above the shared background of its material, at least 50 Hz wide, and at least 3 dB above the 100 Hz on both sides.
- Presence: cast iron needs a detected band covering at least 50% of 400-1,300 Hz; PVC counts any detected band.
- Revisions (mechanics only, thresholds unchanged): after run 1, spectra smoothed over 48 Hz, runs split at dips, and the noise floor measured on the same percentile basis; after run 2, dips measured from medians, not peaks (peaks split U08); after run 3, steady levels measured on 250 ms frames (4 Hz resolution) so the hum removal clears the hum lines (240 Hz hum hid U15 and U02 at 16 Hz resolution); after run 4, runs split wherever the median level steps or dips by 3 dB or more (steps joined U08 to high-pitch traffic and U34 to a low shelf). Runs 1-2: band_scan_run1_summary.md and band_scan_run2_summary.md.

## Logger cross-check (reported only)

- Logger B minus logger A, steady background of cast-iron recordings without a band ({'A': 17, 'B': 9}): +6.7 dB (median over 250-2,000 Hz); shape correlation 0.97. Quiet-clip estimate: +5.8 dB.

## Validation on cast iron (stated before running)

- Passes: **yes**.
- Dug leaks with a band covering 400-1,300 Hz: ['L01', 'L03', 'L06'] (missed: []).
- Dug non-leaks flagged: [].
- Undug cast-iron sites with such a band: ['U07', 'U08', 'U17', 'U36'].

## Cast iron (33 recordings)

- **316-1324 Hz**: 8 of 33 recordings (L01, L03, L06, U02, U07, U08, U17, U36).

| Recording | Logger | Status | Band (Hz) | Width (Hz) | dB above floor | dB above shared | dB above sides (below / above) |
|---|---|---|---|---|---|---|---|
| U02 | A | unlabeled | 316-420 | 108 | 14.6 | 12.4 | 9.0 / 13.7 |
| L01 | A | leak | 380-1324 | 948 | 20.5 | 18.3 | 20.3 / 19.6 |
| L03 | B | leak | 380-1272 | 896 | 21.2 | 18.8 | 20.5 / 14.0 |
| L06 | A | leak | 380-1324 | 948 | 20.9 | 18.8 | 18.2 / 18.5 |
| U07 | B | unlabeled | 380-1276 | 900 | 21.5 | 19.1 | 17.7 / 15.7 |
| U08 | A | unlabeled | 380-1276 | 900 | 21.5 | 19.2 | 15.9 / 14.8 |
| U17 | A | unlabeled | 380-1324 | 948 | 20.7 | 18.3 | 19.9 / 20.2 |
| U36 | A | unlabeled | 384-1320 | 940 | 10.9 | 8.7 | 8.6 / 8.6 |

- Runs above the shared background that failed a rule: 79 (too narrow: 24; not distinct from both sides: 79).

## PVC (12 recordings)

- **324-448 Hz**: 3 of 12 recordings (U03, U15, U34).

| Recording | Logger | Status | Band (Hz) | Width (Hz) | dB above floor | dB above shared | dB above sides (below / above) |
|---|---|---|---|---|---|---|---|
| U34 | A | unlabeled | 324-444 | 124 | 19.5 | 18.8 | 12.9 / 19.1 |
| U03 | B | unlabeled | 328-448 | 124 | 18.5 | 18.0 | 10.4 / 15.5 |
| U15 | B | unlabeled | 340-420 | 84 | 15.4 | 14.2 | 10.4 / 15.6 |

- Runs above the shared background that failed a rule: 25 (too narrow: 10; not distinct from both sides: 25).
