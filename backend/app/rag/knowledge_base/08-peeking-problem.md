# The Peeking Problem: Don't Stop Tests Early

If you check your results every hour and stop the test the moment treatment looks better, you dramatically increase your chance of being fooled by a random streak. This is called "peeking," and it inflates your false-positive rate far above the 5% you think you're protecting with a p < 0.05 threshold — sometimes to 20-30% or higher if you peek repeatedly.

Why this happens: with enough repeated looks at noisy data, the p-value will cross below 0.05 by pure chance at some point during the test, even when there's truly no effect. Stopping right when that happens ("significance chasing") means you're selectively picking the lucky moment, not making an unbiased decision.

How to avoid it:
- Decide your sample size (or test duration) **before** starting the test, based on a power calculation, and commit to running until you hit it.
- If you want to monitor progress along the way, use a sequential testing method designed for repeated looks (e.g., alpha-spending, sequential probability ratio tests) rather than a fixed-sample test checked repeatedly.
- Also watch out for the **novelty effect**: a new feature can show an initial lift purely because it's new and users are curious, which fades over 1-2 weeks. Running tests too short risks mistaking novelty for a durable improvement.
