# Statistical Testing Module for AB Testing Pro
# Author: Harsha

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
from scipy import stats


@dataclass
class TestResult:
    test_name: str
    p_value: float
    statistic: float
    extra: Dict[str, Any]


class StatisticalTester:
    """
    Basic statistical test helper for A/B testing.

    Supports:
      - two-proportion z test (binary conversions)
      - Welch's t-test (continuous metric, unequal variances)
      - Mann-Whitney U (nonparametric continuous)
      - Chi-square test (2x2 contingency)

    Also includes adapter methods used by your Streamlit apps:
      - independent_ttest
      - chi_square_test
      - mann_whitney_u_test
      - recommend_test
    """

    # ---------------------------
    # Core statistical primitives
    # ---------------------------

    @staticmethod
    def two_proportion_z_test(
        control_success: int,
        control_total: int,
        treatment_success: int,
        treatment_total: int,
    ) -> TestResult:
        if control_total <= 0 or treatment_total <= 0:
            raise ValueError("Totals must be > 0.")
        if not (0 <= control_success <= control_total) or not (0 <= treatment_success <= treatment_total):
            raise ValueError("Success counts must be within [0, total].")

        p1 = control_success / control_total
        p2 = treatment_success / treatment_total
        p_pool = (control_success + treatment_success) / (control_total + treatment_total)

        se = math.sqrt(p_pool * (1 - p_pool) * (1 / control_total + 1 / treatment_total))
        if se == 0:
            return TestResult(
                test_name="Two-Proportion Z-Test",
                p_value=1.0,
                statistic=0.0,
                extra={"p_control": p1, "p_treatment": p2, "note": "No variance (pooled proportion 0 or 1)."},
            )

        z = (p2 - p1) / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

        return TestResult(
            test_name="Two-Proportion Z-Test",
            p_value=float(p_value),
            statistic=float(z),
            extra={"p_control": p1, "p_treatment": p2, "uplift": (p2 - p1)},
        )

    @staticmethod
    def welch_t_test(control_values: np.ndarray, treatment_values: np.ndarray) -> TestResult:
        control_values = np.asarray(control_values, dtype=float)
        treatment_values = np.asarray(treatment_values, dtype=float)

        if control_values.size < 2 or treatment_values.size < 2:
            raise ValueError("Need at least 2 samples per group for t-test.")

        t_stat, p_value = stats.ttest_ind(control_values, treatment_values, equal_var=False, nan_policy="omit")
        return TestResult(
            test_name="Welch's t-test",
            p_value=float(p_value),
            statistic=float(t_stat),
            extra={
                "mean_control": float(np.nanmean(control_values)),
                "mean_treatment": float(np.nanmean(treatment_values)),
                "diff": float(np.nanmean(treatment_values) - np.nanmean(control_values)),
            },
        )

    @staticmethod
    def mann_whitney_u(control_values: np.ndarray, treatment_values: np.ndarray) -> TestResult:
        control_values = np.asarray(control_values, dtype=float)
        treatment_values = np.asarray(treatment_values, dtype=float)

        if control_values.size == 0 or treatment_values.size == 0:
            raise ValueError("Empty group.")

        u_stat, p_value = stats.mannwhitneyu(control_values, treatment_values, alternative="two-sided")
        return TestResult(
            test_name="Mann–Whitney U",
            p_value=float(p_value),
            statistic=float(u_stat),
            extra={},
        )

    @staticmethod
    def chi_square_2x2(a: int, b: int, c: int, d: int) -> TestResult:
        table = np.array([[a, b], [c, d]], dtype=float)
        chi2, p_value, dof, expected = stats.chi2_contingency(table, correction=False)
        return TestResult(
            test_name="Chi-square (2x2)",
            p_value=float(p_value),
            statistic=float(chi2),
            extra={"dof": int(dof), "expected": expected.tolist()},
        )

    # -----------------------------------
    # Adapter methods your app expects
    # -----------------------------------

    def independent_ttest(self, control, treatment) -> Dict[str, Any]:
        """
        Used by app_nextgen.py when test_type == 'ttest'
        Returns a dict with keys the app expects.
        """
        res = self.welch_t_test(control, treatment)

        mean_c = float(np.nanmean(np.asarray(control, dtype=float)))
        mean_t = float(np.nanmean(np.asarray(treatment, dtype=float)))

        uplift_pct = 0.0
        if mean_c != 0 and not np.isnan(mean_c) and not np.isnan(mean_t):
            uplift_pct = ((mean_t - mean_c) / abs(mean_c)) * 100.0

        return {
            "test_name": res.test_name,
            "p_value": float(res.p_value),
            "effect_size": float(res.statistic),  # your UI labels this as "Effect Size"
            "uplift_percentage": float(uplift_pct),
            "is_significant": bool(res.p_value < 0.05),
            "mean_control": mean_c,
            "mean_treatment": mean_t,
        }

    def mann_whitney_u_test(self, control, treatment) -> Dict[str, Any]:
        """
        Used by app_nextgen.py when test_type == 'mann_whitney'
        """
        res = self.mann_whitney_u(control, treatment)

        mean_c = float(np.nanmean(np.asarray(control, dtype=float)))
        mean_t = float(np.nanmean(np.asarray(treatment, dtype=float)))

        uplift_pct = 0.0
        if mean_c != 0 and not np.isnan(mean_c) and not np.isnan(mean_t):
            uplift_pct = ((mean_t - mean_c) / abs(mean_c)) * 100.0

        return {
            "test_name": res.test_name,
            "p_value": float(res.p_value),
            "effect_size": float(res.statistic),
            "uplift_percentage": float(uplift_pct),
            "is_significant": bool(res.p_value < 0.05),
            "mean_control": mean_c,
            "mean_treatment": mean_t,
        }

    def chi_square_test(self, control, treatment) -> Dict[str, Any]:
        """
        Used by app_nextgen.py when test_type == 'chi_square'

        Expects control/treatment arrays to be binary-like (0/1).
        If your metric is already counts, you should call chi_square_2x2 directly.
        """
        control = np.asarray(control, dtype=float)
        treatment = np.asarray(treatment, dtype=float)

        # Treat values as 0/1 conversions
        c_succ = int(np.nansum(control))
        c_tot = int(np.sum(~np.isnan(control)))
        t_succ = int(np.nansum(treatment))
        t_tot = int(np.sum(~np.isnan(treatment)))

        if c_tot <= 0 or t_tot <= 0:
            raise ValueError("Empty group(s) after NaN removal.")

        res = self.chi_square_2x2(
            c_succ,
            c_tot - c_succ,
            t_succ,
            t_tot - t_succ,
        )

        p_control = c_succ / c_tot
        p_treatment = t_succ / t_tot
        uplift_pct = (p_treatment - p_control) * 100.0

        return {
            "test_name": res.test_name,
            "p_value": float(res.p_value),
            "effect_size": float(res.statistic),
            "uplift_percentage": float(uplift_pct),
            "is_significant": bool(res.p_value < 0.05),
            "p_control": float(p_control),
            "p_treatment": float(p_treatment),
        }

    def recommend_test(self, control, treatment, metric_type: str = "continuous") -> str:
        """
        Used by app_nextgen.py when test_type == 'auto'
        """
        metric_type = (metric_type or "").lower().strip()
        if metric_type == "categorical":
            return "chi_square"
        return "ttest"
