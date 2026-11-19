# Listening for Leaks: Where to Dig This Week

*Technical assessment write-up. Draft for review, 2026-11-17. Text target: 2 pages; figures: 2 pages.*

---

## Recommended digs (in priority order)

| # | Site | Pipe | Chance of a leak* | Why it is on the list |
|---|---|---|---|---|
| 1 | **U17** | Cast iron | **49%** (36–50%) | Steady leak-pitch hiss in 99% of the recording; closest match to the confirmed leaks |
| 2 | **U07** | Cast iron | **48%** (36–50%) | Same steady hiss (99%); slightly fainter than U17 |
| 3 | **U08** | Cast iron | **46%** (2–50%) | Same steady hiss (98%), but faint enough to drop out under one analysis choice |
| 4 | **U36** | Cast iron | **43%** (2–50%) | Steady hiss (98%), but the faintest and patchiest of the four |
| 5 | **U34** | PVC | **35%** (27–35%) | *Informative dig:* the strongest steady hiss on plastic pipe, at a lower pitch than any confirmed leak; digging it tests whether that pattern means a leak |

\*Main estimate, with the range across reasonable analysis choices. A typical surveyed site has a **12.5%** (1 in 8) chance.
Together, these five digs are expected to find about **2.2 leaks**, compared with about **0.6** from five digs chosen at random. That cuts the expected cost per leak found from about **$64,000** to about **$18,000**.

---

## Introduction and data

**The decision.** The water department can dig at no more than five of 40 unchecked sites this week, at about $8,000 per dig. A pilot clipped cheap acoustic loggers to fire hydrants overnight, producing one 5-second recording per site. Crews have already dug at six of the 46 sites: three had leaks and three did not. Our task was to rank the 40 undug sites by their chance of a leak.

**How we framed it.** With only six known outcomes, any model that learns many details from the data will simply memorize those six. We therefore treated this as a *ranking* problem built on what crews already know a leak sounds like: *"a steady hiss… and it doesn't stop,"* in the crew chief's words. We used the six dug sites mainly to *check* that rule, not to build it.

**Lessons from related UChicago work.** The Energy & Environment Lab has used data to target scarce inspections before, and three lessons from that work shaped our approach:

- **Measure success against current practice.** In Colorado, a model trained on past inspections raised the share of methane inspections that found a problem from 5.6% to 17%. We likewise judge our list by how many more leaks five digs should find than five chosen at random.
- **Compare like with like.** The Colorado team judged each oil and gas facility's risk against others run by the same operator. We similarly removed each logger model's own background noise before comparing sites.
- **Don't trust results from a hand-picked sample until they are tested in the field.** An EPA hazardous-waste model built on the roughly 2% of sites that had been inspected looked 46% better than standard targeting on past data, and 79% better in a national field test. The two numbers differed because the past data came from a hand-picked sample. Our six dug sites are hand-picked in the same way, so we use them to check our rule rather than to fit a complex model, and we treat this week's digs as the real test.

**What we noticed in the data** (Figures 1–2). Five observations shaped the analysis:

1. **The confirmed leaks share one clear signature.** All three dug leaks carry a block of sound between about 400 and 1,300 Hz that stands out about 21 dB above the logger's own background noise and never drops out. None of the three non-leaks has it.
2. **Recording time is entangled with the outcome.** All three leaks were recorded before 3:30 a.m. and all three non-leaks after 4:15 a.m., as morning traffic began. We assumed clip timing was otherwise random, so we never used time as a predictor; a model could otherwise "learn the clock."
3. **The two logger models differ only in volume.** Logger B records about 5.8 dB louder than logger A (roughly twice the signal amplitude), with an identical pattern across pitches. We removed each logger's own background noise before comparing sites.
4. **The dug sites are not typical, and they include no plastic pipe.** Half of the dug sites leaked, against 1 in 8 historically, so crews likely dug where they already suspected a leak. None of the 12 PVC (plastic) sites has been dug.
5. **One recording is unusable.** U40 is overloaded (15% of its samples clipped at the sensor limit) and goes silent for 1.2 seconds. We excluded it and made no prediction for it.

---

## Methodology

**Primary analysis: a rule written from the crew chief's description.** For each recording we measured two things in the leak pitch range (400–1,300 Hz for cast iron):

- **Contrast:** how far the band stands out above the pitches just below and above it, in decibels (dB).
- **Persistence:** the share of 64-millisecond moments in which the band is present. A leak "doesn't stop," so we required the band in at least 90% of moments. U35, for example, has a strong-looking band that comes from a single loud impact; it is present 5% of the time and scores near zero.

We turned these measurements into probabilities in two steps:

1. **Presence sets the range.** A site either has a persistent band or it does not. Starting from the 1-in-8 base rate, we used Bayes' rule with how much more common the band was among dug leaks than non-leaks (3 of 3 vs. 0 of 3). We smoothed those counts two standard ways so that six sites could not imply certainty. That gives **36–50%** for a site with the band and **2–3%** for one without.
2. **Band quality places a site within its range.** We scored steadiness (persistence over time and coverage across the pitch range) more heavily than intenseness (the contrast), weighting them 0.6 and 0.4, because the crew chief's description stresses steadiness. Each was measured relative to the weakest confirmed leak, and capped so that a band louder than a known leak earns no extra credit.

**Plastic pipe.** We assumed a leak on PVC also produces a steady band, but plastic dampens high pitches, so we searched for it lower, at 250–400 Hz, just above the background. Because no PVC leak has ever been confirmed, we discounted each PVC site by how many of five properties of the confirmed leak signature its band shares. For example, U34 shares three of five (it is steady and stands out, but it is narrower and sits directly on the background).

**Robustness checks.** We tested whether the ranking holds up under different choices:

- **Leave one out:** we rebuilt the rule six times, each time without one dug site.
- **Raw spectra:** we repeated everything without removing each logger's background noise.
- **Clustering:** we grouped all recordings by similarity *without using any outcomes*, and checked where the confirmed leaks fell.
- **Learning from the undug sites:** we spread the six outcomes to similar undug sites (label spreading and self-training).
- **An alternative model:** a statistical model (penalized logistic regression) that lets the chance of a leak rise with contrast alone.
- **Outside literature (kept out of the model):** published research on how plastic pipe changes leak sound, used only to sanity-check the PVC calls.

---

## Results

**Four cast-iron sites sound like the confirmed leaks** (Figure 3). U17, U07, U08 and U36 are the only undug cast-iron sites with a band that stands out and persists through the recording, as all three dug leaks do. Every other cast-iron site has the band at most 8% of the time, like the three dug non-leaks. Two PVC sites, U34 and U15, carry a steady band at the lower pitch.

**The ranking survives every check, with two weaker picks** (Figure 4).

- Each confirmed leak, when left out, still scored above every undug site; no non-leak did.
- **U17** is flagged under every check, including every clustering done without outcomes. **U07** is flagged under all but two clusterings.
- **U08 and U36** are the most fragile. In the raw spectra their bands fall just below the 3 dB line, and the alternative model, which weights contrast alone, places both below the base rate. This is why their ranges stretch down to 2%.
- **PVC** depends on assumptions. Models that treat the PVC band exactly like the cast-iron band rank U34 *first*. The published research suggests a leak on plastic pipe would look different again, which is why we discount PVC and recommend U34 as a test rather than a sure bet.

---

## Discussion and conclusion

**How much should the department trust this ranking?** We trust the *set* of candidates more than the exact percentages. The probabilities rest on six dug sites, so each could plausibly be several points higher or lower. As a result, the expected-leak totals are rough: our estimates add up to about 3 leaks among the 39 scorable sites, against about 5 expected at 1 in 8. **We are probably missing one or two leaks,** most likely on PVC pipe or at sites where a leak was too faint or too far from the hydrant to register.

**The assumptions that matter most:**

1. Every leak produces a steady band (if some leaks are silent, the no-band sites are riskier than 2–3%).
2. Recording time is unrelated to whether a site leaks.
3. A PVC leak keeps the same steady character at a lower pitch.
4. The six dug outcomes are correct.

The recordings also have features we did not expect in field data, such as perfectly sharp pitch edges and three leaks of identical loudness. The rule therefore fits *this* pilot well, but its performance on real pipes is untested.

**What would make the estimate better:**

- **Use this week's digs.** Five new outcomes, including the first PVC site, would roughly double our labeled data and test the PVC assumption directly.
- **Record differently.** Several clips per site, spread across the quietest hours (1–4 a.m.), with one logger model throughout. This would remove the clock and device confounds and catch intermittent leaks.
- **Dig some sites at random.** Sites chosen because they looked suspicious overstate how often leaks occur. A few randomly chosen digs would show how many leaks the rule misses.

**Recommendation.** Dig U17, U07, U08 and U36, the four cast-iron sites that sound like the confirmed leaks. Use the fifth dig at U34 to learn whether plastic-pipe leaks can be heard this way. If U34 proves to be a leak, U15 is the next candidate.

*Use of AI.* Claude Opus 5.5 was used to assist in drafting code for this analysis.

---

## Figure pages (up to 2)

**Page 3 — Data at a glance**

- **Figure 1. What data are available** (descriptive overview: recordings by logger, pipe, dig status and hour; source: `eda_16_descriptive_overview`).
- **Figure 2. What a leak sounds like** (spectrograms of a dug leak, a dug non-leak and the top undug site, with the leak band marked; derived from `model1_fig2_how_score_works`, panel A, or `eda_05_labeled_spectrograms`).

**Page 4 — Results**

- **Figure 3. The decision map** (contrast vs. persistence for every site; source: `model1_fig3_decision_map`).
- **Figure 4. Which picks survive every check** (robustness grid; source: `model1_fig4_robustness_grid`). Optional replacement: Model 1 vs. alternative-model comparison (`model2_fig6_model_comparison`).
- The dig list with probability ranges (`model1_fig1_dig_list`) could replace the table at the top of page 1 if space is tight.
