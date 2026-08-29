export interface GlossaryTerm {
  term: string;
  definition: string;
  category: "core" | "stats" | "advanced";
}

export const GLOSSARY_TERMS: GlossaryTerm[] = [
  {
    term: "A/B Test",
    category: "core",
    definition:
      "An experiment that splits users into two (or more) groups, shows each a different version of something, and measures which performs better on a chosen metric.",
  },
  {
    term: "Control Group",
    category: "core",
    definition: "The group that experiences the current, unchanged version — the baseline everything else is compared against.",
  },
  {
    term: "Treatment Group",
    category: "core",
    definition: "The group that experiences the new version being tested.",
  },
  {
    term: "Hypothesis",
    category: "core",
    definition:
      "A specific, falsifiable prediction stated before running a test — e.g. \"we believe X will cause Y because Z.\" Writing it down first prevents rationalizing whatever the data happens to show.",
  },
  {
    term: "Conversion Rate",
    category: "core",
    definition: "The percentage of users who completed a desired action (e.g. 50 signups out of 1,000 visitors = 5%).",
  },
  {
    term: "Uplift / Lift",
    category: "core",
    definition: "How much better (or worse) the treatment performed vs. control, usually as a relative percentage.",
  },
  {
    term: "P-value",
    category: "stats",
    definition:
      "The probability of seeing a difference at least this large if there were actually no real effect. Low (typically < 0.05) suggests the result probably isn't just noise.",
  },
  {
    term: "Statistical Significance",
    category: "stats",
    definition:
      "A result is \"significant\" when you're confident (conventionally 95%+) the observed difference is real, not random chance. Says nothing about whether the effect is big enough to matter — that's practical significance.",
  },
  {
    term: "Confidence Interval",
    category: "stats",
    definition:
      "A range of plausible values for the true effect (e.g. \"we're 95% confident the real lift is between 2% and 9%\"). More informative than a single p-value since it also conveys precision.",
  },
  {
    term: "Effect Size",
    category: "stats",
    definition: "A standardized measure of how large the difference between groups is, independent of sample size.",
  },
  {
    term: "Sample Size",
    category: "stats",
    definition: "The number of users/observations in each group. Too small, and even a real effect won't reach significance.",
  },
  {
    term: "Statistical Power",
    category: "stats",
    definition:
      "The probability your test will detect a real effect, if one exists, at your chosen significance level. Conventionally targeted at 80%.",
  },
  {
    term: "Minimum Detectable Effect (MDE)",
    category: "stats",
    definition: "The smallest lift you actually care about detecting — wanting to catch a tiny 0.5% change requires far more data than a 20% one.",
  },
  {
    term: "Type I Error",
    category: "stats",
    definition: "A false positive — concluding there's a real effect when there actually isn't one.",
  },
  {
    term: "Type II Error",
    category: "stats",
    definition: "A false negative — failing to detect a real effect that actually exists, usually from an underpowered test.",
  },
  {
    term: "Welch's t-test",
    category: "stats",
    definition: "Compares means of a continuous metric (revenue, session length) between two groups, without assuming equal variance.",
  },
  {
    term: "Mann-Whitney U Test",
    category: "stats",
    definition: "A non-parametric test for continuous metrics that are skewed or have outliers — compares distributions by rank instead of mean.",
  },
  {
    term: "Chi-Square Test",
    category: "stats",
    definition: "Compares proportions/rates between groups for binary or categorical outcomes (converted vs. not).",
  },
  {
    term: "Guardrail Metric",
    category: "advanced",
    definition:
      "A secondary metric you monitor to make sure the treatment doesn't cause unacceptable harm elsewhere, even if the primary metric improves.",
  },
  {
    term: "Sample Ratio Mismatch (SRM)",
    category: "advanced",
    definition:
      "A health check comparing the observed control/treatment split against the expected one (e.g. 50/50). A significant mismatch usually means broken randomization — the results can't be trusted until it's fixed.",
  },
  {
    term: "The Peeking Problem",
    category: "advanced",
    definition:
      "Checking results repeatedly and stopping the moment they look good inflates your false-positive rate well past 5%. Commit to a sample size or duration before starting.",
  },
  {
    term: "Novelty Effect",
    category: "advanced",
    definition: "A temporary lift caused purely by a change being new and getting extra attention, which fades over time — can be mistaken for a durable improvement.",
  },
  {
    term: "Uplift Modeling",
    category: "advanced",
    definition:
      "Predicting each individual's personal treatment effect (not just the group average) to find who actually benefits from a change — useful for targeting a rollout instead of shipping to everyone.",
  },
];
