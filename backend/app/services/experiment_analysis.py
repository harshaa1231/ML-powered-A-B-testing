"""Shared control/treatment split + test dispatch logic for the advanced
(upload-based) analysis flow, ported from the Streamlit app's
`run_ab_analysis` helper.

Also runs two Statsig-style additions on every advanced analysis: a
Sample Ratio Mismatch (SRM) health check (a real chi-square
goodness-of-fit test flagging broken randomization) and, if guardrail
columns are supplied, the same per-column test dispatch used for the
primary metric, producing a "scorecard" of secondary results.
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd

from app.services.stats_engine import StatisticalTester

CONTROL_LABELS = {"control", "0", "a", "baseline", "unexposed"}
TREATMENT_LABELS = {"treatment", "1", "b", "variant", "exposed", "test"}

TestType = Literal["auto", "ttest", "chi_square", "mann_whitney"]


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


def _run_dispatch(
    tester: StatisticalTester,
    control_data: pd.Series,
    treatment_data: pd.Series,
    test_type: TestType,
    metric_col: str,
) -> dict[str, Any]:
    if test_type == "ttest":
        results = tester.independent_ttest(control_data, treatment_data)
    elif test_type == "chi_square":
        results = tester.chi_square_test(control_data, treatment_data)
    elif test_type == "mann_whitney":
        results = tester.mann_whitney_u_test(control_data, treatment_data)
    else:
        metric_type = "categorical" if control_data.nunique() <= 2 and treatment_data.nunique() <= 2 else "continuous"
        recommended = tester.recommend_test(control_data, treatment_data, metric_type)
        if recommended == "chi_square":
            results = tester.chi_square_test(control_data, treatment_data)
        elif recommended == "mann_whitney":
            results = tester.mann_whitney_u_test(control_data, treatment_data)
        else:
            results = tester.independent_ttest(control_data, treatment_data)

    results["metric"] = metric_col
    return results


def run_ab_analysis(
    df: pd.DataFrame,
    group_col: str,
    metric_col: str,
    test_type: TestType,
    domain: str = "general",
    guardrail_cols: list[str] | None = None,
) -> dict[str, Any]:
    control_data, treatment_data = split_control_treatment(df, group_col, metric_col)
    tester = StatisticalTester()

    results = _run_dispatch(tester, control_data, treatment_data, test_type, metric_col)

    results["domain"] = domain
    results["test_type"] = test_type
    results["n_control"] = len(control_data)
    results["n_treatment"] = len(treatment_data)

    results["health_checks"] = {
        "sample_ratio_mismatch": StatisticalTester.sample_ratio_mismatch(len(control_data), len(treatment_data))
    }

    if guardrail_cols:
        guardrails = []
        for col in guardrail_cols:
            if col not in df.columns or col in (group_col, metric_col):
                continue
            try:
                g_control, g_treatment = split_control_treatment(df, group_col, col)
                guardrails.append(_run_dispatch(tester, g_control, g_treatment, "auto", col))
            except (ValueError, KeyError):
                continue
        results["guardrails"] = guardrails

    return results
