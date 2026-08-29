import numpy as np
import pandas as pd
import pytest

from app.services.experiment_analysis import run_ab_analysis, split_control_treatment


def test_split_control_treatment_recognizes_standard_labels() -> None:
    df = pd.DataFrame({"group": ["control", "treatment", "control", "treatment"], "metric": [1, 2, 3, 4]})
    control, treatment = split_control_treatment(df, "group", "metric")
    assert list(control) == [1, 3]
    assert list(treatment) == [2, 4]


def test_split_control_treatment_falls_back_to_first_two_unique_values() -> None:
    df = pd.DataFrame({"group": ["variant_1", "variant_2", "variant_1", "variant_2"], "metric": [10, 20, 30, 40]})
    control, treatment = split_control_treatment(df, "group", "metric")
    assert len(control) == 2
    assert len(treatment) == 2


def test_split_control_treatment_raises_on_single_group() -> None:
    df = pd.DataFrame({"group": ["control", "control"], "metric": [1, 2]})
    with pytest.raises(ValueError):
        split_control_treatment(df, "group", "metric")


def test_run_ab_analysis_auto_picks_ttest_for_continuous_metric() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "group": ["control"] * 200 + ["treatment"] * 200,
            "revenue": np.concatenate([rng.normal(50, 10, 200), rng.normal(60, 10, 200)]),
        }
    )
    results = run_ab_analysis(df, "group", "revenue", "auto", domain="ecommerce")
    assert results["test_name"] == "Welch's t-test"
    assert results["domain"] == "ecommerce"
    assert results["n_control"] == 200
    assert results["n_treatment"] == 200


def test_run_ab_analysis_auto_picks_chi_square_for_binary_metric() -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "group": ["control"] * 500 + ["treatment"] * 500,
            "converted": np.concatenate(
                [(rng.random(500) < 0.1).astype(int), (rng.random(500) < 0.2).astype(int)]
            ),
        }
    )
    results = run_ab_analysis(df, "group", "converted", "auto")
    assert results["test_name"] == "Chi-square (2x2)"
