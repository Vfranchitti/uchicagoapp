# Does our data look like real cast-iron leak recordings?

Purpose: compare the pilot clips with published field and lab observations
of leak noise in iron water mains. Where the clips look unlike the real
world, our results may not carry over to real pipes.

Prepared 2026-11-17. This is a companion to `pvc_attenuation_research.md`,
and the source numbers [1]–[6] refer to that note's citation list. New
sources are [9] and [10] at the end.

**How the checks were run.** We used three throwaway scripts, kept outside
the repo in the session scratchpad. They read the WAV files, `src/config.py`
and `compute_welch_psd` and change nothing. Spectra use the repo's own
settings (Welch method, 1,000-sample segments, so each frequency step is
4 Hz), compared against each logger's noise floor in
`outputs/tables/noise_floor_spectra_by_logger.csv`. The scripts are not part
of the pipeline. If any finding goes into the memo, it should be rebuilt as
a proper `src/` script first.

---

## Bottom line

Some things about the data look realistic:
- **Pitch.** The cast-iron leak band (400–1,300 Hz) sits where field studies
  find leak noise in iron mains.
- **Steadiness.** Leak sound is steady over the clip, as real leak noise is.
- **Recording format.** It resembles a commercial noise logger.
- **Hum.** Some sites carry 120 Hz hum, a common real-world nuisance.

Other things do not look like real recordings:
- **Brick-wall edges.** The leak band switches on and off at exactly 400 and
  1,300 Hz, rising or falling about 20 dB within one 4 Hz step. No pipe,
  hydrant or sensor behaves like that. Physical filtering is gradual.
- **Identical leaks.** All three dug leaks are the same loudness (+21.0,
  +21.0, +21.9 dB) and have the same edges. Real leak loudness varies widely
  with flow, pressure and distance to the hydrant, and each site's band edges
  differ.
- **Reused building blocks.** Almost every non-silent clip is a mix of a few
  flat, sharp-edged noise bands that recur from site to site at the same
  frequencies.

The most likely explanation is that the clips were **synthesized**, built
from filtered noise for the exercise, rather than recorded in the field.
That does not make the exercise less useful. It does mean our model learns
the generator's rules, and how well it would work on real pipes is untested.
The memo should say so in its limitations.

---

## 1. Check-by-check results

"Our calc." marks numbers we computed. Every other number is from the cited
source.

| # | What we checked | Real-world benchmark | Our data | Verdict |
|---|---|---|---|---|
| 1 | **Where the leak band sits** | Iron mains, sensors on hydrants, 20 sites with leaks confirmed by digging: leak band 0–650 Hz at the low end to 850–2,000 Hz at the high end; average 220–1,400 Hz. The one cast-iron site: 250–1,000 Hz [5] | 400–1,300 Hz in all three dug leaks | **Realistic** |
| 2 | **How sharp the band edges are** | Pipes lose energy gradually as pitch rises, a smooth low-pass curve [3, 4]. Field transfer functions are "wavy" but trend smoothly [1]. No source shows a brick-wall edge | L03 goes from −0.5 dB at 392 Hz to +18.6 dB at 400 Hz and +22.6 dB at 408 Hz. At the top it falls from +17.2 dB at 1,300 Hz to +1.2 dB at 1,312 Hz. L01 has the same edges (steps of +17 and −19 dB). These are as sharp as a 4 Hz analysis can show (our calc.) | **Not realistic.** Looks like a digital band-pass filter |
| 3 | **Whether the three leaks match each other** | Leak loudness rises steeply with flow. About 50 times the signal (about 34 dB) going from 3 to 7.5 L/min [1]. Signal RMS tracked flow closely (correlation 0.95) [6]. Bands differ site by site [5] | Middle of the band (500–1,200 Hz): L01 +21.00, L03 +21.00, L06 +21.89 dB above the floor. Same 400 and 1,300 Hz edges in all three (our calc.) | **Not realistic.** Three real leaks would not match this closely |
| 4 | **Shape inside the band** | Real spectra show peaks at fixed frequencies, for example at 100, 250, 300, 400 and 550 Hz at one hydrant [5]. Resonances of the pipe, soil and hydrant add peaks [1] | Flat. Bumpiness has a standard deviation of 0.5–0.9 dB, and the tilt is −1.1 to 0.0 dB per octave. No peak stands more than 3.3 dB above the band. That is about what random estimation noise alone produces (our calc.) | **Not realistic.** Smooth white noise with no resonances |
| 5 | **Steadiness over 5 seconds** | Leak noise is treated as steady random noise in the standard models [3, 4]. The field crew chief says a leak "doesn't stop" | Band loudness in 0.25 s frames varies by 0.6 dB (SD) in L03, against 0.3 dB for ideal steady noise. Values look like the bell curve expected of random noise (excess kurtosis 0.08) (our calc.) | **Realistic** (and very clean) |
| 6 | **Background noise shape** | Background in water mains is dominated by low pitches. Hunaidi's ambient noise was strongest below 5 Hz [1] | Both loggers' floors are about 11 dB louder at 4–100 Hz than at 400–1,300 Hz and fall about 3.8 dB per octave (our calc.) | **Plausible** |
| 7 | **Logger A vs. B** | No published comparison of two logger models found. Commercial loggers have their own frequency responses, for example a hydrophone rated about 20 Hz–5 kHz [9] | B is louder by a median of 5.81 dB. The gap barely changes with frequency (standard deviation 0.55 dB from 62.5 to 1,600 Hz), so B behaves like A with the volume turned up (our calc.) | **Questionable.** Different models usually differ in shape, not just volume |
| 8 | **Electrical hum** | Hunaidi lists electrical noise at the mains frequency and rotating machinery such as fans and pumps as sources of spikes [1] | Strong lines at 120 Hz and its multiples (240, 360 Hz): U15 up to 26 dB above neighbouring frequencies, U02 19 dB, U16 and U22 about 9 dB. 120 Hz is twice the North American 60 Hz power frequency, the usual pitch of transformer and motor hum (general engineering knowledge, not from the sources here) | **Realistic** |
| 9 | **Recording format** | A commercial noise logger used in a 2025 field study recorded 10-second clips at 4,096 samples per second, four readings 15 minutes apart [9]. Field surveys are run around midnight to avoid water use and traffic [10] | 5-second clips at 4,000 per second, one clip per site, recorded 22:08 to 06:31 | **Realistic format**, but thinner than practice (one short clip, and some clips run into the morning commute) |

---

## 2. The data look built from a few reusable pieces

We listed every sharp jump of 8 dB or more within 16 Hz in each clip's
spectrum (our calc.). The same few edges recur across sites, logger models
and pipe types, to within a single 4 Hz step:

| Piece | Edges | Level above floor | Where it appears |
|---|---|---|---|
| "Leak" band | 400 → 1,300 Hz | about +21 dB | L01, L03, L06. Smaller steps at the same edges in U07 and U17 |
| Wide upper band | ~300 → ~1,950 Hz | about +15–16 dB | Alone in U16 and U22 (cast iron) and U32 (PVC). A rise at ~300 Hz also appears in many busier clips |
| Low rumble band | ~24 → ~180 Hz | varies | About 23 clips, including the non-leaks L02, L04 and L05 and the leaks L01 and L06 |
| Narrow "PVC" band | ~300 → ~420 Hz | +19 dB (U34), about +12–13 dB (U15) | U34 and U15 only |
| Hum lines | 120, 240, 360 Hz | up to +26 dB | U15, U02, U16, U22 |

Real recordings do not break down into a few flat blocks with shared edges,
because every site's pipe, soil, distance and hydrant shape the sound
differently [1, 5].

---

## 3. What this means for the analysis

1. **The model will fit the data well, but that does not show it works on
   real pipes.** Because each piece has exact edges, simple band-energy
   features will separate the clips very cleanly. That separation reflects
   how the data were generated, not real-world acoustics. Frame claims as
   "within this pilot data" and flag it under measurement (cell) error and
   external validity.
2. **The wide upper band can look like a leak.** U16, U22 and U32 have about
   +15 dB inside 400–1,300 Hz, so a plain "leak-band energy" feature would
   score them as leak-like. Unlike the dug leaks, they also carry the same
   energy from 300–400 Hz and 1,300–1,950 Hz. Features that measure energy
   just *outside* the leak band, as well as inside it, would tell the two
   apart. This is a suggestion for the modeling step; nothing has been
   changed.
3. **This changes our reading of U34 and U15.** They share an identical
   sharp-edged 300–420 Hz band, and U15 also carries strong 120 Hz hum. A
   real PVC leak would show a smooth roll-off, not a flat block
   (`pvc_attenuation_research.md`). In synthetic data, a band that shows up
   only at two PVC sites was probably put there deliberately. It may stand
   for "a PVC leak", or it may be a decoy. The physics cannot settle which.
   The case for treating PVC predictions as low confidence is now stronger.
4. **It matters for the memo.** One sentence in the limitations, such as
   "the recordings have features not seen in field data, for example perfectly
   sharp frequency edges, so performance on real pipes is untested", is
   honest and shows careful checking.

---

## 4. New sources (in addition to [1]–[8] in `pvc_attenuation_research.md`)

[9] El-Zahab, S., Abdelkader, E. M., Fares, A., & Zayed, T. (2025).
Comparative analysis of machine learning techniques in enhancing acoustic
noise loggers' leak detection. *Water*, 17(16), 2427.
https://doi.org/10.3390/w17162427 — open copy:
https://ira.lib.polyu.edu.hk/bitstream/10397/117621/1/water-17-02427.pdf
(PermaNET+ loggers on valves in Hong Kong; hydrophone about 20 Hz–5 kHz;
10 s clips at 4,096 Hz; four readings 15 min apart.)

[10] Liu, R., Tariq, S., Tijani, I. A., Fan, H., Abdelmageed, S., Fares, A.,
& Zayed, T. (2025). Leak detection in water distribution networks using
micro-electromechanical systems-based accelerometers: a machine learning
approach. *Water Practice & Technology*, 20(9), 1900–.
https://doi.org/10.2166/wpt.2025.109 — open copy:
https://ira.lib.polyu.edu.hk/bitstream/10397/117548/1/wpt2025109.pdf
(Field campaign across 75 Hong Kong sites, October 2020 to August 2021,
recorded at midnight; leak-to-sensor distance 1–16 m; sampled at 3,000 Hz.)

Not found: a published comparison of two logger models' frequency
responses, and a published distribution of leak loudness across many iron
leaks measured the same way. [10] reports feature ranges, but its Table 3
did not come through cleanly in our text copy, so we did not use those
numbers.
