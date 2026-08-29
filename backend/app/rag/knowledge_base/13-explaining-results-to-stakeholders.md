# Explaining A/B Test Results to a Non-Technical Audience

When presenting results to a manager or stakeholder, lead with the business-relevant numbers, not the statistics jargon:

1. **The headline**: "Treatment increased conversion by X% (from A% to B%), and we're Y% confident this is a real effect, not luck."
2. **Sample size and duration**: how many users were in each group and how long the test ran — this establishes credibility.
3. **Practical significance**: translate the lift into business impact — extra revenue per month, additional signups per week — not just the percentage.
4. **Caveats**: mention any guardrail metrics that moved in a concerning direction, and whether the effect might include a temporary novelty bump that could fade.
5. **Recommendation**: a clear "ship it," "don't ship it," or "we need more data" — don't leave the decision hanging on a p-value with no verdict attached.

Avoid unexplained jargon (p-value, confidence interval, effect size) unless you immediately translate it — "there's less than a 5% chance this result is a fluke" reads much better to a general audience than "p < 0.05."
