# Common A/B Testing Mistakes

- **Too small a sample.** Testing with a handful of users per group won't reliably detect anything but very large effects. Run a sample-size calculation first.
- **Testing too many things at once.** If five things changed between A and B and B won, you don't know which change helped. Test one change at a time, or use a factorial design if you need to test interactions deliberately.
- **Stopping too early / peeking.** Don't stop the moment treatment looks good after an hour — early results are noisy. Commit to a pre-computed sample size or duration.
- **Ignoring "no difference" results.** A null result is still valuable information — it tells you the change doesn't matter, so you can stop investing in it and move to the next idea, rather than treating it as a failed experiment.
- **Multiple comparisons without correction.** Running many simultaneous tests (or checking many metrics/segments) increases the chance that *something* looks significant purely by chance. Correct for this (e.g., Bonferroni or false-discovery-rate control) when you're running or reporting many comparisons at once.
- **Ignoring segment heterogeneity.** An overall null or positive result can hide the fact that the treatment helps one segment and hurts another. Uplift modeling (see the uplift-modeling lesson) is designed specifically to surface this.
