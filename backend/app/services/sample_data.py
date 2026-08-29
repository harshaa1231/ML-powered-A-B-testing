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
    def _make_cookie_cats_sample(seed: int = 8) -> pd.DataFrame:
        """Synthesized to match the real, published Cookie Cats mobile-game A/B test
        (moving the first paywall gate from level 30 to level 40): 90,189 total
        players, gate_30 (control) -> 44.8%/19.0% 1-day/7-day retention, gate_40
        (treatment) -> 44.2%/18.2%. Not a redistribution of the original per-user
        CSV (avoids licensing ambiguity) — generated to match the real published
        aggregate rates, same approach as every other sample dataset here.
        """
        rng = np.random.default_rng(seed)
        n_total = 90_189
        group = rng.choice(["control", "treatment"], size=n_total)
        retention_1_rate = np.where(group == "treatment", 0.442, 0.448)
        retention_7_rate = np.where(group == "treatment", 0.182, 0.190)
        retention_1 = (rng.random(n_total) < retention_1_rate).astype(int)
        retention_7 = (rng.random(n_total) < retention_7_rate).astype(int)
        return pd.DataFrame({"group": group, "retention_1": retention_1, "retention_7": retention_7})

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

        cookie_cats = SampleDataGenerator._make_cookie_cats_sample()

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
            "cookie_cats": {
                "name": "Cookie Cats - Gate Placement (real published experiment)",
                "description": (
                    "Based on a real, published mobile game A/B test: moving the first paywall gate from "
                    "level 30 to 40. 1-day retention isn't significantly different, but 7-day retention is "
                    "a genuine, counterintuitive result — delaying the gate further hurts long-term retention."
                ),
                "df": cookie_cats,
                "group_col": "group",
                "metric_col": "retention_7",
            },
        }
