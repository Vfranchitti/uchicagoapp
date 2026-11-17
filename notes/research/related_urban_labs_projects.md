# Related Energy & Environment Lab projects (reviewed 2026-11-17)

Sources: Urban Labs project pages, the E&E Lab site (eelab.epic.uchicago.edu),
J-PAL project pages, AEA RCT Registry, michaelgreenstone.com/research, and
the GitHub API (name searches).

## Availability at a glance

| Project | Web description | Paper | Short memo / brief | Public data / code |
|---|---|---|---|---|
| Predicting Methane Leaks (CO, CDPHE) | Yes | Working paper (Oct 2024), no public PDF found | None found | None (registry: "no public data") |
| EPA Inspection Targeting (RCRA) | Yes | Working paper (Aug 2025, R&R at *AER*), no public PDF found | None found (2019 ECOS slide deck; EPA FY2020 evaluation report) | None found |
| Smart Water Conservation Enforcement (Fresno) | Yes | Published, *REStat* 107(4), 2025; BFI WP 2022-158 | **Yes: 4-page research brief** (`docs/reference_briefs/Fresno.Water_.Report.112822.pdf`) | **Yes: Harvard Dataverse, doi:10.7910/DVN/PSO21N** |
| Heavy-Duty Truck Emissions (CARB) | Yes | None found | None found | None (AEA registry only) |

## Predicting Methane Leaks: the closest analog to this task

- **Partner:** the Colorado Department of Public Health and Environment
  (CDPHE). Work began in 2018.
- **Team:** Thomas Covert, Michael Greenstone, Ludovica Gazze and Olga
  Rostapshova.
- **Problem:** regulators can inspect only a small share of sites, so they
  need to decide where to look. That is the same decision as our five digs.
- **Model:** supervised machine learning trained on historical inspection
  and permitting records, predicting the risk of a leak or violation at
  each facility. Colorado has used it for annual inspection targeting
  since inspection year 2022.
- **Evaluation:**
  - Inspection hit rate rose from **5.6% in 2021 to 17%** once inspections
    followed the model, roughly triple.
  - Facilities flagged as high-risk also had higher hit rates as measured
    by the team's vehicle-mounted gas analyzer in Weld County.
- **Randomized trials** (AEARCTR-0009071, among others):
  1. Notify operators about a random subset of high-risk facilities.
  2. Notify low-risk facilities when emissions are detected.
  3. A planned third trial on increased scrutiny with continuous monitoring.
- **Trial results:** notifications cut fugitive emissions by **about 50%**.
  Preliminary results suggest firms respond to the risk of enforcement but
  not to information alone.
- **Papers and talks:**
  - "How Large are Methane Emissions from the Upstream Oil and Gas Industry
    and Why are They There? Evidence from Colorado" (working paper, October
    2024).
  - Seminar: "A View from the Fenceline" (Oxford, October 2022).

### Methods in detail (from trial registrations and talk abstracts)

What is documented:

1. **Risk model.** A supervised machine-learning model trained on CDPHE's
   historical inspection outcomes and permitting records gives each
   facility a leak or violation risk score. Satellite measurements were
   proposed as a further input (UChicago Data Science Institute, 2019). The
   algorithm, features and validation are **not published**.
2. **Targeting.** CDPHE plans each inspection year's inspections from the
   scores (inspection year 2022 onward). The hit rate went from 5.6% to 17%.
   That is a **before/after** comparison, not a randomized one.
3. **Risk tiers set within each operator.** In AEARCTR-0009071, a facility
   is "low-risk" if its score is below **its own operator's median score**.
   The comparison is within each company, not across the whole state.
4. **Randomized subsets.** Only a **random subset of high-risk facilities**
   appeared on the notification lists (Oxford abstract, 2022). That lets the
   team estimate the effect of being flagged separately from the effect of
   being risky.
5. **Trial 1: high-risk notifications.** Information is bundled with a
   higher perceived chance of enforcement.
6. **Trial 2 (AEARCTR-0009071).** Low-risk facilities where the team
   detected emissions were randomized half to a notification and half to no
   notification. The sample was small, at least 60 facilities.
7. **Trial 3 (AEARCTR-0010800).** 571 facilities from the inspection year
   2023 list that were within 75 m of a public road and within 2 hours of
   Denver. It used **balanced re-randomization** at the facility level, and
   the plan was 2,500–4,000 visits. The notification signals greater
   regulatory scrutiny.
8. **Outcomes measured independently of the regulator.**
   - The team's own Mobile Air Monitoring Lab, a vehicle-mounted gas
     analyzer, recorded whether emissions were present and how large they
     were.
   - Emissions were attributed to a facility using **meteorological
     conditions** (wind).
   - CDPHE's infrared-camera inspections provided a second, separate
     measure.
9. **Results.** Flagged high-risk facilities had higher hit rates on both
   measures. Notifications cut fugitive emissions by about 50%. Firms
   respond to the risk of enforcement but not to information alone
   (described as preliminary).
10. **Pre-registration.** All trials were registered with the AEA, with
    University of Chicago ethics-board approval. Analysis plans are
    restricted and no public data are available.

### What carries over to the acoustic-logger task

| Methane project | Leak-logger analog |
|---|---|
| Rank facilities by predicted risk; judge success by **hit rate vs. status quo** (5.6% → 17%) | Rank the 40 sites; report expected leaks found by 5 digs vs. ~0.6 (5 × 1/8) with blind digging |
| Learn from historical inspection outcomes | The 6 dug sites are our "inspections," but 6 labels cannot support a many-feature model, so use a few domain-driven features |
| Tiers set relative to **each operator's median** | Standardize features **within logger model** (A vs. B), since the device shifts loudness about 3x |
| **Random subset** of high-risk sites flagged, to learn causal effects | Spend part of the budget learning: e.g., 4 digs on top-ranked sites + 1 on an informative or uncertain site (e.g., a top PVC candidate) |
| **Independent outcome measures** (mobile lab + infrared camera) | Confirm before digging with an independent signal: re-record at 1–4 am with the same logger model, or a second listening method |
| **Attribution via meteorology** to separate the facility from background | Separate leak hiss from background: control for recording hour (commute traffic) and the sensor's noise floor |
| Explicit eligibility frame (≤75 m from road, ≤2 h drive) | Define the eligible set: flag U40 (clipped, dropout) and U35 (impact) as unreliable recordings |
| Pre-registration; field "horse race" vs. status quo (EPA) | Write down the ranking rule before digging; treat this week's 5 digs as an out-of-sample test; propose a model-vs.-crew-chief comparison next round |

What does not carry over: large-scale supervised learning, satellite and
remote-sensing inputs, and the firm-behavior experiments, since the utility
fixes its own pipes.

## EPA Inspection Targeting

- **Partner:** U.S. EPA. The team has worked on site at EPA headquarters
  since 2017.
- **Model:** about 15–20 years of RCRA administrative data with roughly
  10,000 variables, predicting serious hazardous-waste violations.
- **Results:**
  - The **estimate from historical data** was a 46% higher hit rate.
  - A **national field test** against EPA's own targeting found a 79% gain;
    lab materials report 82%.
  - The cost of inspections did not change.
- **Lesson for our task:** the training data covered only about 2% of
  sites, a highly selected sample. The authors treated the
  historical-data estimate as uncertain and ran a field test. Our six dug
  sites are a selected sample in the same way.
- **Outcome:** EPA scaled the model nationwide and set up a National
  Targeting Center. Clean Air Act and Clean Water Act models are in
  development.
- **Paper:** "Can Machine Learning Improve the Environment? Evidence from a
  National Field Test Targeting Hazardous Waste Inspections" (Greenstone,
  Buchsbaum, Meckel), revisions requested at *AER*.

## Smart Water Conservation Enforcement (Fresno)

- **Design:** randomized trial with more than 80,000 households, using
  smart meters. It compared automated and in-person enforcement and varied
  fine levels.
- **Results:**
  - Water use fell 3% and violations fell 17%, saving about 64 million
    gallons in one summer.
  - Complaints rose sharply, and the city ended automated fines.
- **Materials:**
  - Paper: Browne, Gazze, Greenstone and Rostapshova, *REStat*, 2025.
  - Brief: the 4-page BFI/EPIC research brief.
  - Replication files: on Harvard Dataverse.
- **This is the only one of the four projects with a paper, a short brief
  and public replication files.**

## Heavy-Duty Truck Emissions (CARB)

- **Design:** roadside remote sensing flags trucks that are likely high
  emitters. Each flagged truck is randomly assigned either a compliance
  letter or a fleet-level audit.
- **Registry entry:** AEARCTR-0006812, with a planned sample of 6,198
  trucks and 612 fleets. The outcome is black carbon emissions.
- **Status:** no results, paper or brief found.

## GitHub search

- **No UChicago Urban Labs or E&E Lab organization exists on GitHub.** The
  `urbanlabs` organization is unrelated: it hosts travel-demand modeling
  tools.
- **No repository for any of the four projects was found.** Searches for
  methane, RCRA, EPA inspection targeting and Fresno smart meters all
  returned 0 results.
- **Personal accounts that match E&E Lab names:**
  - `tcovert` (Thomas Covert, @uchicago): a personal website and a Julia
    logit package.
  - `lgazze` (Ludovica Gazze): a profile README only.
  - `fburlig` (Fiona Burlig): replication files for unrelated papers.
  - `cjwichman` (Casey Wichman): teaching repositories and replication
    files for an unrelated paper.
  - None contain code for these projects.
- **No accounts found** for Michael Greenstone, Olga Rostapshova, Rajat
  Kochhar, Joe Amick, Federico Domínguez-Molina or Santiago
  Cardenas-Garcia. "Vishal Joseph" matches only unrelated people. This was
  a name search, so accounts under other handles would not appear.
