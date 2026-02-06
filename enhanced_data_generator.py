# Enhanced Training Data Generator - High-Volume Realistic Datasets
# Author: Harsha
# Generates large-scale synthetic data for robust model training
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class EnhancedDataGenerator:
    """Generates large-scale, realistic datasets with complex patterns"""
    
    @staticmethod
    def generate_multi_domain_training_data(n_samples: int = 100000) -> pd.DataFrame:
        """
        Generate comprehensive training dataset across all domains
        
        Args:
            n_samples: Total number of samples (default 100k)
            
        Returns:
            Large DataFrame with realistic features and outcomes
        """
        np.random.seed(42)
        
        domains = ['tech', 'ecommerce', 'marketing', 'gaming', 'finance', 'healthcare']
        samples_per_domain = n_samples // len(domains)
        
        all_data = []
        
        for domain in domains:
            domain_data = EnhancedDataGenerator._generate_domain_data(domain, samples_per_domain)
            all_data.append(domain_data)
        
        return pd.concat(all_data, ignore_index=True)
    
    @staticmethod
    def _generate_domain_data(domain: str, n: int) -> pd.DataFrame:
        """Generate realistic data for specific domain"""
        
        if domain == 'tech':
            return EnhancedDataGenerator._tech_data(n)
        elif domain == 'ecommerce':
            return EnhancedDataGenerator._ecommerce_data(n)
        elif domain == 'marketing':
            return EnhancedDataGenerator._marketing_data(n)
        elif domain == 'gaming':
            return EnhancedDataGenerator._gaming_data(n)
        elif domain == 'finance':
            return EnhancedDataGenerator._finance_data(n)
        elif domain == 'healthcare':
            return EnhancedDataGenerator._healthcare_data(n)
    
    @staticmethod
    def _tech_data(n: int) -> pd.DataFrame:
        """Tech domain with complex user behavior patterns"""
        # User segments
        segments = np.random.choice(['power_user', 'casual', 'new'], n, p=[0.2, 0.5, 0.3])
        
        # Device types
        devices = np.random.choice(['mobile', 'desktop', 'tablet'], n, p=[0.6, 0.3, 0.1])
        
        # Time-based features
        hour_of_day = np.random.randint(0, 24, n)
        day_of_week = np.random.randint(0, 7, n)
        
        # Treatment assignment (50/50 split)
        treatment = np.random.choice([0, 1], n)
        
        # Base conversion rates by segment
        base_rates = {
            'power_user': 0.25,
            'casual': 0.15,
            'new': 0.08
        }
        
        # Page load time (treatment reduces load time)
        load_time = np.where(
            treatment == 1,
            np.random.gamma(2, 0.8, n),  # Faster
            np.random.gamma(2, 1.5, n)   # Slower
        )
        
        # Complex conversion logic
        conversion_prob = np.array([base_rates[s] for s in segments])
        
        # Uplift effects
        conversion_prob += treatment * 0.05  # Base treatment effect
        conversion_prob -= (load_time - 2) * 0.02  # Load time effect
        conversion_prob += (hour_of_day >= 9) & (hour_of_day <= 17) * 0.03  # Business hours
        conversion_prob += (devices == 'desktop') * 0.02  # Desktop bonus
        
        conversion_prob = np.clip(conversion_prob, 0, 1)
        converted = (np.random.random(n) < conversion_prob).astype(int)
        
        # Session metrics
        session_duration = np.where(
            converted == 1,
            np.random.exponential(500, n),
            np.random.exponential(200, n)
        )
        
        pages_viewed = np.where(
            converted == 1,
            np.random.poisson(8, n),
            np.random.poisson(3, n)
        )
        
        return pd.DataFrame({
            'domain': 'tech',
            'user_segment': segments,
            'device': devices,
            'hour_of_day': hour_of_day,
            'day_of_week': day_of_week,
            'treatment': treatment,
            'page_load_time': load_time,
            'session_duration': session_duration,
            'pages_viewed': pages_viewed,
            'converted': converted,
            'previous_visits': np.random.poisson(5, n),
            'account_age_days': np.random.exponential(180, n)
        })
    
    @staticmethod
    def _ecommerce_data(n: int) -> pd.DataFrame:
        """E-commerce with purchase behavior"""
        # Customer segments
        segments = np.random.choice(['vip', 'regular', 'first_time'], n, p=[0.15, 0.55, 0.3])
        
        # Geographic region
        regions = np.random.choice(['US', 'EU', 'ASIA', 'OTHER'], n, p=[0.4, 0.3, 0.2, 0.1])
        
        treatment = np.random.choice([0, 1], n)
        
        # Cart value (treatment increases value)
        cart_value = np.where(
            treatment == 1,
            np.random.lognormal(4.7, 0.8, n),
            np.random.lognormal(4.5, 0.8, n)
        )
        
        # Number of items
        items_in_cart = np.random.poisson(3, n) + treatment
        
        # Base checkout rates
        base_rates = {'vip': 0.65, 'regular': 0.40, 'first_time': 0.25}
        checkout_prob = np.array([base_rates[s] for s in segments])
        
        # Treatment effect (simplified checkout)
        checkout_prob += treatment * 0.12
        checkout_prob += (cart_value < 100) * 0.05  # Lower cart = higher conversion
        checkout_prob -= (items_in_cart > 5) * 0.08  # Too many items = lower conversion
        
        checkout_prob = np.clip(checkout_prob, 0, 1)
        checkout_completed = (np.random.random(n) < checkout_prob).astype(int)
        
        # Time on site
        time_on_site = np.random.exponential(400, n) + treatment * 50
        
        return pd.DataFrame({
            'domain': 'ecommerce',
            'customer_segment': segments,
            'region': regions,
            'treatment': treatment,
            'cart_value': cart_value,
            'items_in_cart': items_in_cart,
            'time_on_site': time_on_site,
            'checkout_completed': checkout_completed,
            'has_coupon': np.random.choice([0, 1], n, p=[0.7, 0.3]),
            'is_mobile': np.random.choice([0, 1], n, p=[0.4, 0.6]),
            'shipping_cost': np.random.choice([0, 5, 10, 15], n)
        })
    
    @staticmethod
    def _marketing_data(n: int) -> pd.DataFrame:
        """Marketing campaigns with email engagement"""
        # Subscriber age
        subscriber_age_days = np.random.exponential(365, n)
        
        # Engagement history
        previous_opens = np.random.poisson(3, n)
        previous_clicks = np.random.poisson(1, n)
        
        treatment = np.random.choice([0, 1], n)
        
        # Email open (personalized = higher open rate)
        open_prob = 0.25 + treatment * 0.08
        open_prob += (previous_opens / 20)  # History matters
        open_prob = np.clip(open_prob, 0, 1)
        email_opened = (np.random.random(n) < open_prob).astype(int)
        
        # Click-through (conditional on open)
        click_prob = email_opened * (0.12 + treatment * 0.06)
        click_prob += (previous_clicks / 10)
        click_prob = np.clip(click_prob, 0, 1)
        link_clicked = (np.random.random(n) < click_prob).astype(int)
        
        # Time to click (only for clickers)
        time_to_click = np.where(
            link_clicked == 1,
            np.random.exponential(2400 - treatment * 600, n),
            0
        )
        
        return pd.DataFrame({
            'domain': 'marketing',
            'treatment': treatment,
            'subscriber_age_days': subscriber_age_days,
            'previous_opens': previous_opens,
            'previous_clicks': previous_clicks,
            'email_opened': email_opened,
            'link_clicked': link_clicked,
            'time_to_click': time_to_click,
            'device': np.random.choice(['mobile', 'desktop'], n),
            'sent_hour': np.random.randint(6, 22, n),
            'is_weekend': np.random.choice([0, 1], n, p=[0.7, 0.3])
        })
    
    @staticmethod
    def _gaming_data(n: int) -> pd.DataFrame:
        """Gaming with player retention patterns"""
        # Player level
        player_level = np.random.poisson(15, n)
        
        # Game mode preference
        game_mode = np.random.choice(['casual', 'competitive', 'social'], n, p=[0.5, 0.3, 0.2])
        
        treatment = np.random.choice([0, 1], n)
        
        # Session count (treatment increases engagement)
        num_sessions = np.random.poisson(5 + treatment * 2, n)
        
        # Retention logic
        retention_prob = 0.40 + treatment * 0.18
        retention_prob += (player_level / 50) * 0.1  # Higher level = more retention
        retention_prob += (num_sessions / 15) * 0.1
        retention_prob = np.clip(retention_prob, 0, 1)
        retained_7day = (np.random.random(n) < retention_prob).astype(int)
        
        # Session length
        avg_session_length = np.where(
            treatment == 1,
            np.random.exponential(900, n),
            np.random.exponential(600, n)
        )
        
        return pd.DataFrame({
            'domain': 'gaming',
            'treatment': treatment,
            'player_level': player_level,
            'game_mode': game_mode,
            'num_sessions': num_sessions,
            'avg_session_length': avg_session_length,
            'retained_7day': retained_7day,
            'achievements_unlocked': np.random.poisson(8, n) + treatment * 2,
            'friends_count': np.random.poisson(10, n),
            'in_app_purchases': np.random.poisson(2, n),
            'platform': np.random.choice(['ios', 'android', 'pc'], n)
        })
    
    @staticmethod
    def _finance_data(n: int) -> pd.DataFrame:
        """Financial services with account opening"""
        # Credit score
        credit_score = np.random.normal(680, 80, n)
        credit_score = np.clip(credit_score, 300, 850)
        
        # Income bracket
        income_bracket = np.random.choice(['low', 'medium', 'high'], n, p=[0.3, 0.5, 0.2])
        
        treatment = np.random.choice([0, 1], n)
        
        # Account opening (promotional pricing helps)
        open_prob = 0.22 + treatment * 0.13
        open_prob += (credit_score - 650) / 2000  # Credit score effect
        open_prob += (income_bracket == 'high') * 0.1
        open_prob = np.clip(open_prob, 0, 1)
        account_opened = (np.random.random(n) < open_prob).astype(int)
        
        # Initial deposit
        initial_deposit = np.where(
            account_opened == 1,
            np.random.lognormal(8.3 + treatment * 0.2, 1.2, n),
            0
        )
        
        return pd.DataFrame({
            'domain': 'finance',
            'treatment': treatment,
            'credit_score': credit_score,
            'income_bracket': income_bracket,
            'account_opened': account_opened,
            'initial_deposit': initial_deposit,
            'age': np.random.normal(42, 15, n),
            'employment_years': np.random.exponential(8, n),
            'existing_accounts': np.random.poisson(2, n),
            'referral': np.random.choice([0, 1], n, p=[0.85, 0.15])
        })
    
    @staticmethod
    def _healthcare_data(n: int) -> pd.DataFrame:
        """Healthcare patient retention"""
        # Patient age
        age = np.random.normal(55, 15, n)
        age = np.clip(age, 18, 95)
        
        # Chronic conditions
        chronic_conditions = np.random.poisson(1.5, n)
        
        treatment = np.random.choice([0, 1], n)
        
        # Visits completed
        visits_completed = np.random.poisson(4 + treatment * 1.5, n)
        
        # Retention
        retention_prob = 0.65 + treatment * 0.13
        retention_prob -= (age - 55) / 300  # Age effect
        retention_prob += (visits_completed / 15) * 0.1
        retention_prob = np.clip(retention_prob, 0, 1)
        retained_12month = (np.random.random(n) < retention_prob).astype(int)
        
        # Appointment adherence
        appointment_adherence = np.random.uniform(0.5, 1.0, n) + treatment * 0.08
        appointment_adherence = np.clip(appointment_adherence, 0, 1)
        
        return pd.DataFrame({
            'domain': 'healthcare',
            'treatment': treatment,
            'age': age,
            'chronic_conditions': chronic_conditions,
            'visits_completed': visits_completed,
            'appointment_adherence': appointment_adherence,
            'retained_12month': retained_12month,
            'insurance_type': np.random.choice(['private', 'medicare', 'medicaid'], n),
            'distance_to_clinic_miles': np.random.exponential(15, n),
            'has_caregiver': np.random.choice([0, 1], n, p=[0.7, 0.3])
        })
    
    @staticmethod
    def save_training_data(filepath: str = 'training_data.parquet', n_samples: int = 100000):
        """Generate and save large training dataset"""
        print(f"Generating {n_samples:,} samples across all domains...")
        df = EnhancedDataGenerator.generate_multi_domain_training_data(n_samples)
        
        # Save as parquet for efficiency
        df.to_parquet(filepath, index=False, compression='snappy')
        print(f"✓ Saved to {filepath}")
        print(f"  Shape: {df.shape}")
        print(f"  Size: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
        return df


if __name__ == "__main__":
    # Generate training data
    df = EnhancedDataGenerator.save_training_data(
        filepath='/home/claude/training_data.parquet',
        n_samples=100000
    )
    
    print("\n📊 Dataset Summary:")
    print(df.groupby('domain').size())
    print(f"\n✓ Training data ready for model training!")
