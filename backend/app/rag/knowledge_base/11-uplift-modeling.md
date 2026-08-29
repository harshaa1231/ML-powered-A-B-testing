# Uplift Modeling: Who Actually Benefits From Treatment?

A standard A/B test tells you the *average* effect of treatment across everyone in the experiment. But the average can hide a lot: treatment might strongly help one segment of users and slightly hurt another, netting out to a small or even null average effect.

**Uplift modeling** (also called heterogeneous treatment effect estimation) tries to predict, for each individual, how much *more* likely they are to convert under treatment versus control — i.e., their personal treatment effect, not just whether they'll convert.

A common approach is the **T-learner**: train two separate models — one on the control group, one on the treatment group — each predicting the outcome from user features. For any given user, the uplift estimate is the treatment model's prediction minus the control model's prediction. If that's positive and large, treatment is predicted to meaningfully help this user; if negative, treatment may hurt them.

What you get from a good uplift model:
- **Average uplift**: the overall estimated effect (should roughly match the standard A/B test result — a useful sanity check).
- **Positive-uplift percentage**: what fraction of users are predicted to benefit from treatment at all.
- **Feature importance**: which user characteristics most strongly predict who benefits — useful for targeting the rollout (e.g., "only ship this to mobile users" or "exclude power users, who see negative uplift").

Uplift modeling is most valuable when you plan to *personalize* a rollout rather than ship the same version to everyone — for example, targeting a promotion only at users predicted to have high positive uplift.
