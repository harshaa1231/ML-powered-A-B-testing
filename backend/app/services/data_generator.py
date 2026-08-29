"""Large-scale synthetic data generator with realistic per-domain patterns.

Ported from `enhanced_data_generator.py`. The original's file-dump
`save_training_data`/`__main__` path is dropped — this runs as an
in-process API service now, callers get a DataFrame directly.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DOMAINS = ["tech", "ecommerce", "marketing", "gaming", "finance", "healthcare"]


class EnhancedDataGenerator:
    @staticmethod
    def generate_multi_domain_training_data(n_samples: int = 100000, seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        samples_per_domain = n_samples // len(DOMAINS)
        all_data = [EnhancedDataGenerator._generate_domain_data(domain, samples_per_domain, rng) for domain in DOMAINS]
        return pd.concat(all_data, ignore_index=True)

    @staticmethod
    def generate_domain(domain: str, n_samples: int, seed: int = 42) -> pd.DataFrame:
        if domain not in DOMAINS:
            raise ValueError(f"Unknown domain '{domain}'. Choose from {DOMAINS}.")
        rng = np.random.default_rng(seed)
        return EnhancedDataGenerator._generate_domain_data(domain, n_samples, rng)

    @staticmethod
    def _generate_domain_data(domain: str, n: int, rng: np.random.Generator) -> pd.DataFrame:
        return {
            "tech": EnhancedDataGenerator._tech_data,
            "ecommerce": EnhancedDataGenerator._ecommerce_data,
            "marketing": EnhancedDataGenerator._marketing_data,
            "gaming": EnhancedDataGenerator._gaming_data,
            "finance": EnhancedDataGenerator._finance_data,
            "healthcare": EnhancedDataGenerator._healthcare_data,
        }[domain](n, rng)

    @staticmethod
    def _tech_data(n: int, rng: np.random.Generator) -> pd.DataFrame:
        segments = rng.choice(["power_user", "casual", "new"], n, p=[0.2, 0.5, 0.3])
        devices = rng.choice(["mobile", "desktop", "tablet"], n, p=[0.6, 0.3, 0.1])
        hour_of_day = rng.integers(0, 24, n)
        day_of_week = rng.integers(0, 7, n)
        treatment = rng.choice([0, 1], n)

        base_rates = {"power_user": 0.25, "casual": 0.15, "new": 0.08}
        load_time = np.where(treatment == 1, rng.gamma(2, 0.8, n), rng.gamma(2, 1.5, n))

        conversion_prob = np.array([base_rates[s] for s in segments])
        conversion_prob += treatment * 0.05
        conversion_prob -= (load_time - 2) * 0.02
        conversion_prob += ((hour_of_day >= 9) & (hour_of_day <= 17)).astype(float) * 0.03
        conversion_prob += (devices == "desktop") * 0.02
        conversion_prob = np.clip(conversion_prob, 0, 1)
        converted = (rng.random(n) < conversion_prob).astype(int)

        session_duration = np.where(converted == 1, rng.exponential(500, n), rng.exponential(200, n))
        pages_viewed = np.where(converted == 1, rng.poisson(8, n), rng.poisson(3, n))

        return pd.DataFrame(
            {
                "domain": "tech",
                "user_segment": segments,
                "device": devices,
                "hour_of_day": hour_of_day,
                "day_of_week": day_of_week,
                "treatment": treatment,
                "page_load_time": load_time,
                "session_duration": session_duration,
                "pages_viewed": pages_viewed,
                "converted": converted,
                "previous_visits": rng.poisson(5, n),
                "account_age_days": rng.exponential(180, n),
            }
        )

    @staticmethod
    def _ecommerce_data(n: int, rng: np.random.Generator) -> pd.DataFrame:
        segments = rng.choice(["vip", "regular", "first_time"], n, p=[0.15, 0.55, 0.3])
        regions = rng.choice(["US", "EU", "ASIA", "OTHER"], n, p=[0.4, 0.3, 0.2, 0.1])
        treatment = rng.choice([0, 1], n)

        cart_value = np.where(treatment == 1, rng.lognormal(4.7, 0.8, n), rng.lognormal(4.5, 0.8, n))
        items_in_cart = rng.poisson(3, n) + treatment

        base_rates = {"vip": 0.65, "regular": 0.40, "first_time": 0.25}
        checkout_prob = np.array([base_rates[s] for s in segments])
        checkout_prob += treatment * 0.12
        checkout_prob += (cart_value < 100) * 0.05
        checkout_prob -= (items_in_cart > 5) * 0.08
        checkout_prob = np.clip(checkout_prob, 0, 1)
        checkout_completed = (rng.random(n) < checkout_prob).astype(int)

        time_on_site = rng.exponential(400, n) + treatment * 50

        return pd.DataFrame(
            {
                "domain": "ecommerce",
                "customer_segment": segments,
                "region": regions,
                "treatment": treatment,
                "cart_value": cart_value,
                "items_in_cart": items_in_cart,
                "time_on_site": time_on_site,
                "checkout_completed": checkout_completed,
                "has_coupon": rng.choice([0, 1], n, p=[0.7, 0.3]),
                "is_mobile": rng.choice([0, 1], n, p=[0.4, 0.6]),
                "shipping_cost": rng.choice([0, 5, 10, 15], n),
            }
        )

    @staticmethod
    def _marketing_data(n: int, rng: np.random.Generator) -> pd.DataFrame:
        subscriber_age_days = rng.exponential(365, n)
        previous_opens = rng.poisson(3, n)
        previous_clicks = rng.poisson(1, n)
        treatment = rng.choice([0, 1], n)

        open_prob = 0.25 + treatment * 0.08 + (previous_opens / 20)
        open_prob = np.clip(open_prob, 0, 1)
        email_opened = (rng.random(n) < open_prob).astype(int)

        click_prob = email_opened * (0.12 + treatment * 0.06) + (previous_clicks / 10)
        click_prob = np.clip(click_prob, 0, 1)
        link_clicked = (rng.random(n) < click_prob).astype(int)

        time_to_click = np.where(link_clicked == 1, rng.exponential(2400 - treatment * 600, n), 0)

        return pd.DataFrame(
            {
                "domain": "marketing",
                "treatment": treatment,
                "subscriber_age_days": subscriber_age_days,
                "previous_opens": previous_opens,
                "previous_clicks": previous_clicks,
                "email_opened": email_opened,
                "link_clicked": link_clicked,
                "time_to_click": time_to_click,
                "device": rng.choice(["mobile", "desktop"], n),
                "sent_hour": rng.integers(6, 22, n),
                "is_weekend": rng.choice([0, 1], n, p=[0.7, 0.3]),
            }
        )

    @staticmethod
    def _gaming_data(n: int, rng: np.random.Generator) -> pd.DataFrame:
        player_level = rng.poisson(15, n)
        game_mode = rng.choice(["casual", "competitive", "social"], n, p=[0.5, 0.3, 0.2])
        treatment = rng.choice([0, 1], n)

        num_sessions = rng.poisson(5 + treatment * 2, n)

        retention_prob = 0.40 + treatment * 0.18 + (player_level / 50) * 0.1 + (num_sessions / 15) * 0.1
        retention_prob = np.clip(retention_prob, 0, 1)
        retained_7day = (rng.random(n) < retention_prob).astype(int)

        avg_session_length = np.where(treatment == 1, rng.exponential(900, n), rng.exponential(600, n))

        return pd.DataFrame(
            {
                "domain": "gaming",
                "treatment": treatment,
                "player_level": player_level,
                "game_mode": game_mode,
                "num_sessions": num_sessions,
                "avg_session_length": avg_session_length,
                "retained_7day": retained_7day,
                "achievements_unlocked": rng.poisson(8, n) + treatment * 2,
                "friends_count": rng.poisson(10, n),
                "in_app_purchases": rng.poisson(2, n),
                "platform": rng.choice(["ios", "android", "pc"], n),
            }
        )

    @staticmethod
    def _finance_data(n: int, rng: np.random.Generator) -> pd.DataFrame:
        credit_score = np.clip(rng.normal(680, 80, n), 300, 850)
        income_bracket = rng.choice(["low", "medium", "high"], n, p=[0.3, 0.5, 0.2])
        treatment = rng.choice([0, 1], n)

        open_prob = 0.22 + treatment * 0.13 + (credit_score - 650) / 2000 + (income_bracket == "high") * 0.1
        open_prob = np.clip(open_prob, 0, 1)
        account_opened = (rng.random(n) < open_prob).astype(int)

        initial_deposit = np.where(account_opened == 1, rng.lognormal(8.3 + treatment * 0.2, 1.2, n), 0)

        return pd.DataFrame(
            {
                "domain": "finance",
                "treatment": treatment,
                "credit_score": credit_score,
                "income_bracket": income_bracket,
                "account_opened": account_opened,
                "initial_deposit": initial_deposit,
                "age": rng.normal(42, 15, n),
                "employment_years": rng.exponential(8, n),
                "existing_accounts": rng.poisson(2, n),
                "referral": rng.choice([0, 1], n, p=[0.85, 0.15]),
            }
        )

    @staticmethod
    def _healthcare_data(n: int, rng: np.random.Generator) -> pd.DataFrame:
        age = np.clip(rng.normal(55, 15, n), 18, 95)
        chronic_conditions = rng.poisson(1.5, n)
        treatment = rng.choice([0, 1], n)

        visits_completed = rng.poisson(4 + treatment * 1.5, n)

        retention_prob = 0.65 + treatment * 0.13 - (age - 55) / 300 + (visits_completed / 15) * 0.1
        retention_prob = np.clip(retention_prob, 0, 1)
        retained_12month = (rng.random(n) < retention_prob).astype(int)

        appointment_adherence = np.clip(rng.uniform(0.5, 1.0, n) + treatment * 0.08, 0, 1)

        return pd.DataFrame(
            {
                "domain": "healthcare",
                "treatment": treatment,
                "age": age,
                "chronic_conditions": chronic_conditions,
                "visits_completed": visits_completed,
                "appointment_adherence": appointment_adherence,
                "retained_12month": retained_12month,
                "insurance_type": rng.choice(["private", "medicare", "medicaid"], n),
                "distance_to_clinic_miles": rng.exponential(15, n),
                "has_caregiver": rng.choice([0, 1], n, p=[0.7, 0.3]),
            }
        )
