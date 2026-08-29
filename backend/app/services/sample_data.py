"""Six pre-built demo datasets across industries, ported from `sample_data.py`."""

from __future__ import annotations

import numpy as np
import pandas as pd


class SampleDataGenerator:
    @staticmethod
    def _make_binary_conversion_sample(n: int, base: float, uplift: float, seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        group = rng.choice(["control", "treatment"], size=n)
        p = np.where(group == "treatment", base + uplift, base)
        converted = (rng.random(n) < p).astype(int)
        return pd.DataFrame({"group": group, "converted": converted})

    @staticmethod
    def _make_continuous_metric_sample(n: int, mean: float, uplift: float, std: float, seed: int = 7) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        group = rng.choice(["control", "treatment"], size=n)
        mu = np.where(group == "treatment", mean + uplift, mean)
        metric = rng.normal(loc=mu, scale=std, size=n)
        return pd.DataFrame({"group": group, "metric": metric})

    @staticmethod
    def get_all_samples() -> dict[str, dict]:
        tech = SampleDataGenerator._make_binary_conversion_sample(n=5000, base=0.10, uplift=0.02, seed=1)

        ecommerce = SampleDataGenerator._make_continuous_metric_sample(
            n=4000, mean=75.0, uplift=8.0, std=30.0, seed=2
        ).rename(columns={"metric": "cart_value"})

        marketing = SampleDataGenerator._make_binary_conversion_sample(
            n=6000, base=0.05, uplift=0.01, seed=3
        ).rename(columns={"converted": "clicked"})

        gaming = SampleDataGenerator._make_continuous_metric_sample(
            n=5000, mean=18.0, uplift=2.0, std=7.0, seed=4
        ).rename(columns={"metric": "session_minutes"})

        finance = SampleDataGenerator._make_binary_conversion_sample(
            n=4500, base=0.18, uplift=0.04, seed=5
        ).rename(columns={"converted": "account_opened"})

        rng = np.random.default_rng(6)
        group = rng.choice(["control", "treatment"], size=5000)
        adherence = rng.beta(a=6, b=2, size=5000)
        adherence = np.where(group == "treatment", np.clip(adherence + 0.03, 0, 1), adherence)
        healthcare = pd.DataFrame({"group": group, "appointment_adherence": adherence})

        return {
            "tech": {
                "name": "Tech - Conversion Rate",
                "description": "Binary conversions with a small positive treatment effect.",
                "df": tech,
                "group_col": "group",
                "metric_col": "converted",
            },
            "ecommerce": {
                "name": "E-commerce - Cart Value",
                "description": "Continuous metric (cart_value) with higher mean under treatment.",
                "df": ecommerce,
                "group_col": "group",
                "metric_col": "cart_value",
            },
            "marketing": {
                "name": "Marketing - Click Rate",
                "description": "Binary clicks with modest uplift in treatment.",
                "df": marketing,
                "group_col": "group",
                "metric_col": "clicked",
            },
            "gaming": {
                "name": "Gaming - Session Minutes",
                "description": "Continuous metric (session length) with small uplift in treatment.",
                "df": gaming,
                "group_col": "group",
                "metric_col": "session_minutes",
            },
            "finance": {
                "name": "Finance - Account Opened",
                "description": "Binary account_opened with treatment uplift.",
                "df": finance,
                "group_col": "group",
                "metric_col": "account_opened",
            },
            "healthcare": {
                "name": "Healthcare - Appointment Adherence",
                "description": "Continuous 0-1 adherence metric with slight treatment lift.",
                "df": healthcare,
                "group_col": "group",
                "metric_col": "appointment_adherence",
            },
        }
