# A/B Testing Metrics Glossary

**Conversion rate**: the percentage of people who did what you wanted. Example: 50 signups out of 1,000 visitors = 5% conversion rate.

**Lift / uplift**: how much better (or worse) treatment is compared to control, usually shown as a percentage. Example: control converts at 5%, treatment at 6% — that's a 20% relative lift, because 6 is 20% more than 5. Be careful to distinguish *relative* lift (20%) from *absolute* lift (1 percentage point) — both are correct, but they read very differently.

**Confidence level**: how sure you are the result is real; roughly the inverse of the p-value. 95%+ confidence is the common bar for acting on a result; below that, you typically need more data.

**Effect size**: a standardized measure of how large the difference between groups is (e.g., a t-statistic or Cohen's d), independent of sample size — useful for judging practical importance, not just statistical detectability.

**Confidence interval (CI)**: a range of plausible values for the true effect, e.g., "we're 95% confident the true lift is between 2% and 9%." A CI that includes zero means the test is not statistically significant at that confidence level. CIs are more informative than a single p-value because they also convey the precision of your estimate.

**Guardrail metric**: a secondary metric you monitor to make sure the treatment doesn't cause unacceptable harm elsewhere, even if the primary metric improves (e.g., checking that page load time or churn doesn't regress while conversion goes up).
