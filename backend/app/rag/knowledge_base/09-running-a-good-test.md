# How to Run a Good A/B Test

1. **Test one thing at a time.** If you change the button color, the headline, and the price simultaneously, you won't know which change drove the result. Isolate one variable per test (or use a proper multivariate/factorial design if you need to test combinations).
2. **Split your audience randomly.** Don't show version A to returning customers and version B to new visitors — that confounds the treatment effect with the group difference. Random assignment is what makes the two groups comparable.
3. **Decide your primary success metric before you start.** Clicks? Purchases? Revenue? Time on page? Pick one primary metric and commit to it — picking the best-looking metric after the fact ("metric shopping") is a form of p-hacking.
4. **Run the test long enough.** Compute the required sample size up front and don't stop early (see the peeking-problem lesson). Also run for at least one full business cycle (typically 1-2 weeks) to average out day-of-week effects.
5. **Watch guardrail metrics**, not just the primary metric — a change that boosts conversion but tanks retention or page performance may not be a net win.
