# Statistical Significance

"Statistically significant" means you've done the math and are at least 95% confident the observed difference between control and treatment is real, not just random chance. This 95% confidence threshold (equivalently, a p-value below 0.05) is the conventional bar in most industries, though some high-stakes fields (medicine) use stricter thresholds.

Significance is about **confidence that an effect exists**, not about **how big or important that effect is** — that's a separate concept called practical significance. A test can be statistically significant with a tiny, business-irrelevant effect size if the sample is large enough, and it can fail to reach significance with a large effect if the sample is too small.

Rule of thumb: always look at both the p-value (is this real?) and the effect size / uplift percentage (does this matter?) before deciding to ship a change.
