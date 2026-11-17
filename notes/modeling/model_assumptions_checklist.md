# Candidate models: benefits, drawbacks, and assumptions to sign off

How to use this: tick `[x]` next to each assumption you are comfortable
making. A model is only usable if we accept all of its **required**
assumptions. Each assumption has a note on what our data says about it.

Our situation: 6 dug sites (3 leak, 3 no leak, all cast iron), 40 undug
sites (12 PVC), two logger models, and about 1 leak in 8 sites
historically. We need a dig list of 5 and a leak probability for each site.

---

## Part A. Shared assumptions (apply to every model)

These sit underneath every option. If one fails, every model is affected.

- [X] **A1. The 6 labels are correct.** A dug "no leak" really had no leak.
  - *Data:* labels come from digs. A small or intermittent leak could be
    missed, so a "no leak" could be wrong. We cannot check this.
- [X] **A2. One 5-second clip represents the site.** A leak that exists
  makes noise during that clip.
  - *Data:* one clip per site (agreed assumption). The crew chief says a
    leak "doesn't stop," which supports this.
- [X] **A3. A leak sounds roughly the same at every cast-iron site.**
  - *Data:* all 3 dug leaks show the same 400–1,300 Hz band at about +21
    dB above the floor (Fig. 19). This is strong support, but there are
    only 3 examples.
- [X] **A4. After adjusting for the logger, A and B recordings can be
  compared.**
  - *Data:* the noise floors have the same shape (correlation 0.99); B is
    5.8 dB louder. One volume adjustment looks sufficient. B's floor rests
    on only 3 clips.
- [X] **A5. Recording time has nothing to do with leaks.** It only adds
  traffic noise.
  - *Data:* agreed assumption (random clip timing). Among the dug sites,
    time perfectly separates leak from no leak, which we treat as
    coincidence. So no model may use time as an input.
- [X] **A6. About 1 in 8 of the 40 undug sites leak.**
  - *Data:* this is the historical rate for surveyed sites. The 6 dug
    sites (50% leaks) were clearly not typical.
- [X] **A7. Sites are independent.** One site's leak doesn't make a
  neighbor's more likely, or louder.
  - *Data:* unknown. We have no locations. A big leak can be heard at
    nearby hydrants.
- [X] **A8. Findings from cast iron can be stretched to PVC.**
  - *Data:* this is the weakest assumption. No PVC site has been dug. U34
    and U15 (both PVC) show a different, lower 250–400 Hz band. If you
    leave this unchecked, PVC predictions should be reported as
    "low confidence" whatever model we choose. Note: indicate that this is a strong assumption. We will use PVC vs. cast-iron leak noise past literature to create a map from cast-iron leaks to modeled PVC leaks data.
  - **Decision (2026-11-17), a strong assumption:**
    - A PVC leak, like a cast-iron one, always produces a steady band that
      does not stop.
    - On PVC that band is looked for just above the background shelf
      (250–400 Hz), not at the cast-iron 400–1,300 Hz.
    - PVC sites with a persistent band get the cast-iron likelihood ratio
      (36–50%; confidence 43%). Those without one get the cast-iron "no
      band" value (2–3%).
    - **Revised (2026-11-17):**
      - Confidence is graded within each tier by band quality:
        steadiness^0.6 × intenseness^0.4. Steadiness is persistence over
        time × coverage across pitch; intenseness is contrast in dB, capped
        at the weakest dug leak.
      - PVC band sites are discounted by their signature match w, the share
        of 5 leak-signature properties they share: U34 w = 0.6 → 34.9%,
        U15 w = 0.4 → 26.3%. This keeps cast iron ranked above PVC.
    - The literature-based map from cast-iron to PVC leaks
      (`notes/modeling/sanity_check_pvc_prediction.md`) is **held out of the
      model**. It is used only as an end-of-modeling sanity check.
  - *Literature* (`notes/research/pvc_attenuation_research.md`): PVC strips out
    high pitches with distance, and the 400–1,300 Hz cast-iron rule should
    not be applied to PVC.
  - *Realism checks* (`notes/research/cast_iron_realism_checks.md`): the U34/U15
    band has identical sharp edges (like the cast-iron leaks), and U15
    carries 120 Hz hum (removed before scoring).
- [ ] **A9. Results hold only within this pilot data.** The recordings look
  synthesized:
  - Brick-wall band edges.
  - The three leaks are identical in loudness (+21.0, +21.0 and +21.9 dB).
  - The same few noise blocks are reused across sites.

  Ticking this means the memo states that limitation.

---

## Part B. Recommended models

### B1. Knowledge-based score (a rule we write, not learn)

**What it does:** we write a scoring rule from the crew chief's description
and the spectra, e.g., "how much the 400–1,300 Hz band stands out, times
how steady it is." Then we rank all 40 sites by that score. The 6 labels
are used only to check the rule, not to build it.

| Benefits | Drawbacks |
|---|---|
| Cannot overfit the 6 labels, because nothing is learned from them | The rule's weights are our judgment; another analyst might weigh things differently |
| Easiest to explain to the Commissioner ("we listened for a steady hiss in this pitch range") | Gives a ranking, not a true probability; turning it into a probability needs a separate step (B2 or a simple mapping) |
| Uses what the crew chief knows, which is worth more than 6 examples | If our idea of what a leak sounds like is wrong (e.g., for PVC), the rule is wrong everywhere |
| Easy to test for stability: change a weight and see if the dig list moves | Choosing the rule *after* looking at the 6 labels could quietly bake in their quirks |

**Assumptions:**
- [X] **B1a (required).** The crew chief's description ("steady hiss that
  doesn't stop") captures what makes a leak different.
  - *Data:* consistent with all 3 dug leaks.
- [ ] **B1b (required).** The features in the rule measure that
  description well.
  - *Data:* the band-contrast measure separates the 3 leaks (+11 to +21
    dB) from the 3 non-leaks (+2 to +4 dB).
- [ ] **B1c (required).** We fix the rule before using it to rank the 40
  undug sites, and don't keep tuning it to get a dig list we like.

**Verdict:** strong first choice for ranking; pair with B2 for probabilities.

---

### B2. Penalized logistic regression (one or two inputs)

**What it does:** it learns how a score (or one or two features) relates to
the chance of a leak, and outputs a probability between 0 and 1. The
"penalty" stops it from becoming absurdly certain when the 6 points split
perfectly, which they do.

| Benefits | Drawbacks |
|---|---|
| Produces the probabilities `predictions.csv` asks for | With 3 leaks, the fitted relationship is very uncertain; the slope could plausibly be much steeper or flatter |
| Simple, standard, and easy to explain ("the higher the score, the higher the chance") | Without the penalty it breaks down: perfect separation gives infinite coefficients and 0%/100% predictions |
| Easy to re-anchor to the 1-in-8 base rate (shift the intercept) | The penalty strength is a judgment call that changes the probabilities |
| Can be refit in seconds once this week's 5 digs come back | A rule of thumb asks for about 10 leaks per input; we have 3, so at most one input |

**Assumptions:**
- [X] **B2a (required).** The chance of a leak rises smoothly and steadily
  with the score (an S-shaped curve). No sudden jumps or reversals.
  - *Data:* untestable with 6 points; it is plausible.
- [X] **B2b (required).** One input (or at most two) is enough.
  - *Data:* forced by sample size.
- [X] **B2c (required).** We accept that the penalty pulls probabilities
  toward the middle. That is deliberate caution, not a fact about the
  pipes.
- [X] **B2d (required).** The probabilities are re-anchored from the dug
  sites' 50% to the 1-in-8 base rate. This needs A6.

**Verdict:** recommended as the "probability layer" on top of B1's score.

---

### B3. Template matching / nearest centroid

**What it does:** it averages the spectra of the 3 leaks into a "leak
template" and the 3 non-leaks into a "no-leak template." Each undug site is
scored by which template its spectrum is closer to.

| Benefits | Drawbacks |
|---|---|
| Uses the whole pitch pattern, not one number, so it picks up the band's sharp edges | Each template is an average of only 3 clips; one odd clip (L06's traffic) shifts it |
| Very few things to learn (two averages), so overfitting risk is low | Closeness is measured over every pitch, so loud traffic can swamp the leak band unless we weight or limit the pitches compared |
| Intuitive picture: "this site's sound looks like the known leaks" | Assumes "no leak" is one kind of sound; really it is many kinds (silence, traffic, hum) |
| Works naturally with the noise-floor adjustment | Distance is not a probability; it needs a conversion step |

**Assumptions:**
- [ ] **B3a (required).** Leaks share one typical spectrum (supports A3). Note: yes, within pipe type
- [ ] **B3b (required).** Non-leaks also share one typical spectrum.
  - *Data:* weak. Undug sites range from silent to heavy traffic, so one
    "no-leak" average is a poor summary.
- [ ] **B3c (required).** Every pitch matters equally when measuring
  closeness, unless we restrict the comparison to chosen pitch ranges.
  - *Data:* traffic sits below 300 Hz; restricting to 250–2,000 Hz would
    help.
- [ ] **B3d (optional).** It is OK to compare to the leak template only
  ("how leak-like is this?") and ignore the no-leak template. This avoids
  B3b.

**Verdict:** good as a cross-check on B1, especially in its "leak template
only" form.

---

### B4. Gaussian naive Bayes

**What it does:** for each class (leak, no leak) it describes the typical
value and spread of each feature, then asks which class a new site's
values fit better.

| Benefits | Drawbacks |
|---|---|
| Learns fast from few examples (known to beat logistic regression when data are scarce) | Needs a mean *and* a spread per feature per class, each estimated from 3 clips, which is very rough |
| Gives probabilities directly | Assumes features are unrelated to each other given the class; ours are often strongly related (Fig. 11), which makes it overconfident |
| Prior can be set to 1 in 8 directly | Assumes bell-shaped spreads; several of our features are skewed (loudness, spikiness) |
| Transparent: you can see each feature's contribution | With 3 points a spread can come out tiny by chance, giving extreme probabilities |

**Assumptions:**
- [ ] **B4a (required).** Within each class, features are unrelated to one
  another.
  - *Data:* fails for most pairs (Spearman up to 0.96). It is acceptable
    only with 2–3 features from different blocks.
- [ ] **B4b (required).** Each feature has a roughly bell-shaped spread
  within each class.
  - *Data:* untestable with 3 per class. Log scales help.
- [ ] **B4c (required).** Three examples per class give usable estimates
  of a typical value and its spread.
  - *Data:* doubtful.

**Verdict:** usable only as a robustness check with 2 features; not a main
model.

---

### B5. Shrinkage linear discriminant analysis (LDA)

**What it does:** it finds the single best line (a weighted combination of
features) that separates leaks from non-leaks. "Shrinkage" pulls its
estimates toward a simple default, so it doesn't overreact to 6 points.

| Benefits | Drawbacks |
|---|---|
| Handles related features better than naive Bayes | Still estimates more than our 6 points can support unless shrinkage is heavy, and then it is close to B3 |
| Gives probabilities; prior can be set to 1 in 8 | Assumes both classes have the same spread; "no leak" is a mix of very different sounds |
| Well-understood and stable with shrinkage | Harder to explain than B1 or B2 |

**Assumptions:**
- [X] **B5a (required).** Leak and no-leak sites have bell-shaped feature
  spreads.
- [ ] **B5b (required).** The two classes vary by the same amount in the
  same directions (equal spread).
  - *Data:* unlikely; non-leaks range from silence to traffic.
- [ ] **B5c (required).** The shrinkage setting is chosen in advance, not
  tuned on the 6 labels.

**Verdict:** adds little beyond B2 and B3 here; optional robustness check.

---

### B6. k-nearest neighbors (k = 1 to 3)

**What it does:** for each undug site it finds the 1–3 most similar dug
sites and copies their label.

| Benefits | Drawbacks |
|---|---|
| Simplest idea there is; covered in the textbook (ch. 7) | With only 6 neighbors to choose from, every undug site is "nearest" to one of a handful of clips; results are lumpy |
| No assumptions about curve shapes or spreads | Probabilities come out as 0, 1/3, 2/3, or 1, which is too coarse for a dig ranking |
| Easy to explain ("this site sounds most like L03") | Very sensitive to which features and distance we choose |
| | Almost nothing is near a PVC site, yet k-NN will still answer confidently |

**Assumptions:**
- [X] **B6a (required).** Sites that sound alike on our features have the
  same leak status.
- [ ] **B6b (required).** The 6 dug sites cover the range of sounds at
  undug sites.
  - *Data:* fails for PVC, and for the 15 silent clips (no dug site is
    silent).

**Verdict:** useful as a plain-language explanation ("most similar to
L01"), not as the model.

---

## Part C. Supporting methods (use the 40 undug clips)

### C1. Clustering (approved; features pending your sign-off)

**What it does:** it groups all 46 clips by similarity without using labels,
then checks where the 3 leaks and 3 non-leaks land.

| Benefits | Drawbacks |
|---|---|
| Uses all 46 clips, not just 6 | Groups by *whatever* differs most (often loudness or traffic), not necessarily leak vs. no leak |
| Tests whether "leak-like" sound is a real group or an artifact of 6 labels | Number of groups and feature choice change the answer |
| Can reveal patterns we haven't named (e.g., the PVC 250–400 Hz band) | Doesn't produce probabilities by itself |

**Assumptions:**
- [ ] **C1a (required).** Leaks form a distinct group in the chosen
  features.
- [ ] **C1b (required).** The chosen features are not dominated by
  traffic, silence or the device (hence logger standardization, and a run
  without floor clips).
- [ ] **C1c (required).** We read the clusters as a check, not as the
  final prediction.

### C2. Self-training / label propagation

**What it does:** it predicts the undug sites, treats the most confident
predictions as new labels, and refits.

| Benefits | Drawbacks |
|---|---|
| Stretches 6 labels further | Early mistakes get copied and amplified |
| | Can make the model look more certain without making it more right |

**Assumptions:**
- [X] **C2a (required).** The first model's most confident calls are
  almost always right.
  - *Data:* we cannot check this.
- [X] **C2b (required).** Undug sites near each other in feature space
  share labels.

**Verdict:** robustness check only, if at all.

### C3. Bayesian logistic regression with a 1-in-8 prior

**What it does:** the same as B2, but it starts from the belief "about 1 in
8 sites leak" and updates that belief using the 6 labels. It returns a
range of plausible probabilities for each site, not one number.

| Benefits | Drawbacks |
|---|---|
| Builds the base rate in directly | Requires choosing priors for the slope, a judgment call that matters a lot with 6 points |
| Gives an honest range (e.g., "20–60%") per site, ideal for the confidence column and Part 2 Q4 | More complex to explain and to run (simulation) |
| Naturally cautious | Same S-curve assumption as B2 |

**Assumptions:**
- [ ] **C3a (required).** Same as B2a and B2b.
- [ ] **C3b (required).** We are comfortable stating a prior belief about
  how strongly the score predicts leaks (e.g., "probably positive,
  probably not huge").
- [ ] **C3c (required).** A6 (1-in-8 base rate) holds.

---

## Part D. Not suitable here (no checklist needed)

| Model | Why not, in plain terms |
|---|---|
| Random forest, gradient boosting | Build hundreds of decision trees that need many examples; with 6 they memorize the 6 |
| Support vector machine (flexible kernel) | Can draw any boundary around 6 points; its probabilities need extra labeled data to calibrate (textbook ch. 7) |
| Neural networks / CNNs on spectrograms | Thousands to millions of settings learned from 6 examples; guaranteed to memorize |
| Any model given 500 features | The Part 2 Q1 trap: with that many measurements, some will split 6 points perfectly by chance |

---

## Suggested combination (once you've ticked the boxes)

1. **B1** knowledge-based score to rank sites.
2. **B2** penalized logistic regression on that score for probabilities,
   re-anchored to 1 in 8.
3. **B3** (leak-template only) and **C1** clustering as cross-checks: does
   the top-5 list survive?
4. Report PVC sites separately as low confidence, unless you tick **A8**.
