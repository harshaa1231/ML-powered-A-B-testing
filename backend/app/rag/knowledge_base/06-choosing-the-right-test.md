# Choosing the Right Statistical Test

The right test depends on what kind of metric you're measuring:

- **Two-proportion z-test / chi-square test**: use for binary outcomes — did the user convert or not, click or not, churn or not. Compares conversion rates between two groups.
- **Welch's t-test**: use for continuous, roughly normal metrics — revenue per user, session length, cart value. Welch's variant (rather than the classic Student's t-test) does not assume the two groups have equal variance, which is the safer default for real-world data.
- **Mann-Whitney U test**: use for continuous metrics that are heavily skewed or have outliers (e.g., revenue, which often has a long right tail) — it compares distributions using ranks instead of means, so it's robust to outliers and doesn't assume normality.

A practical platform can auto-recommend a test: categorical/binary outcome → chi-square; continuous outcome → t-test by default, or Mann-Whitney if the data is very skewed or small-sample. When in doubt, run both a t-test and a Mann-Whitney test — if they agree, you can be more confident in the conclusion.
