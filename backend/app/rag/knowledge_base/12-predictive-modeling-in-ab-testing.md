# Using Predictive ML Models Alongside A/B Tests

Beyond the pure statistical test, training a predictive model (e.g., gradient boosting or random forest) on your experiment data can answer a different question: **which user or session features are associated with the outcome, and how well can we predict it?**

Typical workflow:
1. Auto-detect the group column (control/treatment), the outcome/target column, and candidate feature columns from the uploaded dataset.
2. Train several models (commonly Gradient Boosting and Random Forest, for both classification and regression tasks) and pick the best by cross-validated score (AUC-ROC for classification, R² for regression).
3. Inspect feature importance to understand what's actually driving the outcome — this can surface segments or conditions the raw A/B test result doesn't reveal.
4. Use the trained model to score new/incoming data without re-running the whole experiment, or as the base learners for uplift modeling (see the uplift-modeling lesson).

This is a complement to hypothesis testing, not a replacement — the A/B test still tells you whether the treatment effect is statistically real; the ML model tells you more about the "why" and "for whom."
