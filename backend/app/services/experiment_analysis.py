"""Shared control/treatment split + test dispatch logic for the advanced
(upload-based) analysis flow, ported from the Streamlit app's
`run_ab_analysis` helper.
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd

from app.services.stats_engine import StatisticalTester

CONTROL_LABELS = {"control", "0", "a", "baseline", "unexposed"}
TREATMENT_LABELS = {"treatment", "1", "b", "variant", "exposed", "test"}


def split_control_treatment(df: pd.DataFrame, group_col: str, metric_col: str) -> tuple[pd.Series, pd.Series]:
    group_vals = df[group_col].astype(str).str.strip().str.lower()
    control_data = df[group_vals.isin(CONTROL_LABELS)][metric_col]
    treatment_data = df[group_vals.isin(TREATMENT_LABELS)][metric_col]

    if len(control_data) == 0 or len(treatment_data) == 0:
        unique_groups = group_vals.unique()
        if len(unique_groups) < 2:
            raise ValueError("Could not identify control and treatment groups from the group column.")
        control_data = df[group_vals == unique_groups[0]][metric_col]
        treatment_data = df[group_vals == unique_groups[1]][metric_col]

    if len(control_data) == 0 or len(treatment_data) == 0:
        raise ValueError("Could not identify control and treatment groups from the group column.")

    return control_data, treatment_data


def run_ab_analysis(
    df: pd.DataFrame,
    group_col: str,
    metric_col: str,
    test_type: Literal["auto", "ttest", "chi_square", "mann_whitney"],
    domain: str = "general",
) -> dict[str, Any]:
    control_data, treatment_data = split_control_treatment(df, group_col, metric_col)
    tester = StatisticalTester()

    if test_type == "ttest":
        results = tester.independent_ttest(control_data, treatment_data)
    elif test_type == "chi_square":
        results = tester.chi_square_test(control_data, treatment_data)
    elif test_type == "mann_whitney":
        results = tester.mann_whitney_u_test(control_data, treatment_data)
    else:
        metric_type = "categorical" if df[metric_col].nunique() <= 2 else "continuous"
        recommended = tester.recommend_test(control_data, treatment_data, metric_type)
        if recommended == "chi_square":
            results = tester.chi_square_test(control_data, treatment_data)
        elif recommended == "mann_whitney":
            results = tester.mann_whitney_u_test(control_data, treatment_data)
        else:
            results = tester.independent_ttest(control_data, treatment_data)

    results["domain"] = domain
    results["metric"] = metric_col
    results["test_type"] = test_type
    results["n_control"] = len(control_data)
    results["n_treatment"] = len(treatment_data)

    return results
