# What Does a P-Value Actually Mean?

A p-value is the probability of seeing a difference at least as large as the one you observed, *if there were actually no real difference between control and treatment* (i.e., if the null hypothesis were true).

- **Low p-value (below 0.05)**: it would be unlikely to see this result by chance alone — the result is probably real.
- **High p-value (above 0.05)**: this result is quite plausible even if the two versions are truly identical — it could easily be luck.

Common misreadings to avoid:
- A p-value is **not** "the probability the treatment doesn't work." It says nothing about the probability that your hypothesis is true — it only describes how surprising the data would be under the assumption of no effect.
- A p-value of 0.049 is not meaningfully different from 0.051 — treat the 0.05 threshold as a convention, not a magic line.
- A non-significant result (p > 0.05) does not prove there's no effect — it may just mean you don't have enough data yet to detect it.
