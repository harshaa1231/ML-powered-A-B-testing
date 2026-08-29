# Sample Size and Statistical Power

Statistical power is the probability that your test will detect a real effect, if one exists, at your chosen significance level. Underpowered tests are one of the most common A/B testing mistakes — testing with 20 people per group will not give reliable results.

Three things determine how much data you need:
1. **Baseline rate**: metrics close to 0% or 100% need more samples to detect a given absolute change than metrics near 50%.
2. **Minimum detectable effect (MDE)**: the smallest lift you actually care about detecting. Wanting to detect a tiny 0.5% lift requires far more data than wanting to detect a 20% lift.
3. **Desired power and significance level**: conventionally 80% power and a 5% significance level (p < 0.05).

Rule of thumb: you generally need hundreds to thousands of samples per group, depending on the baseline rate and the size of the effect you're trying to detect. A test with only tens of users per group is very unlikely to reach significance even if a real effect exists.

Run a sample-size calculation *before* launching a test, not after — deciding how long to run the test in advance also protects you from the "peeking problem" (see the lesson on stopping tests early).
