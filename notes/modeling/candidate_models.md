# Candidate models for the leak classifier

The constraint: 6 labeled clips (3 leak, 3 no leak), 40 unlabeled, a 1-in-8
base rate, and a top-5 dig decision.

Primary reference: Foster et al., *Big Data and Social Science*, ch. 7
(Machine Learning). Its guidance for this case: with little data, put the
complexity in the features and keep the model simple. Never judge a model by
how well it fits its own training data. Rank cases and evaluate precision
in the top k. Choose the threshold from costs rather than defaulting to
0.5. Probabilities from some models (e.g., SVMs) need extra labeled data to
calibrate.

## Recommended (can be fit on 6 labels)

| Model | What it does | Why it suits this data | Main risk |
|---|---|---|---|
| **Knowledge-based score** (no fitted parameters) | Combines 1–2 features chosen from the crew chief's description, e.g., sustained 400–1,300 Hz band energy × steadiness, then ranks sites | Nothing is learned from the 6 labels, so nothing can overfit them; easy to explain to a non-technical reader | Feature weights are a judgment call; the 6 labels serve only as a sanity check |
| **Penalized logistic regression** (1–2 features; ridge or Firth) | Turns a score into a probability | Gives probabilities for `predictions.csv`. The penalty prevents the infinite coefficients that arise when 6 points separate perfectly (they do) | Still 3 events per class; treat coefficients as illustrative |
| **Nearest centroid / template matching** | Measures how close each clip's (normalized) spectrum is to the average leak vs. the average non-leak spectrum | Uses the whole spectral shape with very few parameters (two averages); a k-NN relative from the textbook | Each average comes from only 3 clips; sensitive to device (logger A/B) and pipe differences |
| **Gaussian naive Bayes / shrinkage LDA** | Generative classifier: models each class's feature distribution | Generative models reach their best accuracy with fewer examples than discriminative ones (Ng & Jordan, 2002) | Independence and normality assumptions; still needs few features |

All four need the same two adjustments:
- **Standardize features within each logger model** (the device effect). This
  is the methane project's within-operator logic.
- **Re-anchor probabilities to the base rate.** A model trained on a 50%-leak
  sample overstates risk, so multiply its odds by
  (1/8 ÷ 7/8) ÷ (1/2 ÷ 1/2) = 1/7.

## Supporting methods (use the 40 unlabeled clips)

- **Clustering** (k-means or hierarchical; textbook ch. 7, unsupervised).
  Group all 46 clips by feature similarity, then see which group the 3
  leaks fall into. This checks whether the leak pattern is a real cluster
  or an artifact of the 6 labels.
- **Self-training / label propagation.** Let confident predictions inform
  nearby unlabeled clips. This is risky here because errors spread, so use
  it only as a robustness check.
- **Bayesian logistic regression with an informative prior.** Fix the
  intercept at the 1-in-8 base rate and put weak priors on the slopes. This
  is a formal version of the re-anchoring step.

## Not appropriate here

- **Random forests, gradient boosting, SVMs with flexible kernels, neural
  networks or CNNs on spectrograms.** Each has hundreds to millions of
  parameters. With 6 labels they will fit perfectly and mean nothing; this
  is exactly the Part 2 Q1 trap of 500 features and 6 recordings. SVM
  probabilities also need separate calibration data (Foster et al., ch. 7).
- **Pretrained audio embeddings** (transfer learning). In principle these
  are richer features, but they still feed a classifier trained on 6
  labels, and they need new packages. Worth trying once there are about 300
  labels.

## Urban Labs precedent

- **What was used:**
  - The EPA hazardous-waste model drew on ~15–20 years of data and ~10,000
    variables.
  - The Colorado methane model drew on thousands of past inspections.
  - Both used supervised machine learning with unpublished algorithms.
- **What carries over:**
  - The algorithms do not, because both projects had thousands of labels.
  - The framework does: rank by risk, measure hit rate against the status
    quo, normalize within operator (here, within logger), and validate in
    the field rather than trusting results on historical data.

## Validation options for 6 labels

- **Leave-one-out cross-validation** (6 folds) is the only feasible split,
  and it is coarse: one mistake moves accuracy by 17 points.
  - A perfect 3 of 3 on leaks is consistent with sensitivity as low as 29%.
  - A perfect 6 of 6 overall is consistent with accuracy as low as 54% (95%
    CI).
- **Better checks:**
  - **Stability:** does the top-5 list change when the features, the model,
    or which clip is left out change?
  - **Plausibility:** do the top sites look like leaks on the spectrograms?
  - **Field validation:** the 5 digs this week are the real test set.
