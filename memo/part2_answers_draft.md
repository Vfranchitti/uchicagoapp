# Part 2: Four short questions (draft)

## 1. Testing a model (up to 250 words)

**(a)** Six recordings and 500 measurements give the classifier far more
freedom than data. With that many measurements, some combination will
separate any six labels perfectly, even random ones: a single random
measurement splits three leaks from three non-leaks by chance 10% of the time
(2 of the 20 possible splits), so about 50 of the 500 could do it alone. A
perfect score on its own training data is overfitting, not skill: in-sample
fit says nothing about new sites (Foster et al., ch. 7). The classifier may
also be learning the clock, since recording time alone separates these six.

**(b)** The test should mimic real use: predicting at sites the model has
never heard.

1. Split by site, not by recording. Loggers moved, so a site may have several
   recordings, and all of them must fall on the same side.
2. Set aside about a fifth of the sites as a final test set, balanced by pipe
   material and logger, and use it only once.
3. On the remaining sites, use grouped cross-validation, refitting every
   step (noise adjustment, feature choice, thresholds) inside each training
   fold to prevent leakage. Also hold out each logger in turn, and train on
   earlier recordings and test on later ones, so the model is not rewarded
   for learning a device or a season.
4. Report what matters for digging: leaks found among the top five sites,
   compared with five random digs, at the 1-in-8 rate, with confidence
   intervals, separately for cast iron and PVC.
