import numpy as np
import pytest

from app.services.stats_engine import StatisticalTester


@pytest.fixture
def tester() -> StatisticalTester:
    return StatisticalTester()


def test_two_proportion_z_test_detects_clear_effect(tester: StatisticalTester) -> None:
    result = tester.two_proportion_test(control_success=100, control_total=1000, treatment_success=200, treatment_total=1000)
    assert result["is_significant"] is True
    assert result["p_value"] < 0.05
    assert result["uplift_percentage"] == pytest.approx(100.0, rel=0.01)


def test_two_proportion_z_test_no_effect(tester: StatisticalTester) -> None:
    result = tester.two_proportion_test(control_success=100, control_total=1000, treatment_success=102, treatment_total=1000)
    assert result["is_significant"] is False


def test_two_proportion_z_test_rejects_invalid_totals(tester: StatisticalTester) -> None:
    with pytest.raises(ValueError):
        tester.two_proportion_test(0, 0, 0, 100)


def test_welch_ttest_detects_mean_shift(tester: StatisticalTester) -> None:
    rng = np.random.default_rng(0)
    control = rng.normal(50, 5, 500)
    treatment = rng.normal(56, 5, 500)

    result = tester.independent_ttest(control, treatment)
    assert result["is_significant"] is True
    assert result["mean_treatment"] > result["mean_control"]
    assert result["n_control"] == 500
    assert result["n_treatment"] == 500


def test_welch_ttest_requires_min_samples(tester: StatisticalTester) -> None:
    with pytest.raises(ValueError):
        tester.welch_t_test([1.0], [2.0])


def test_mann_whitney_detects_shift_in_skewed_data(tester: StatisticalTester) -> None:
    rng = np.random.default_rng(1)
    control = rng.exponential(1.0, 300)
    treatment = rng.exponential(1.6, 300)

    result = tester.mann_whitney_u_test(control, treatment)
    assert result["p_value"] < 0.05
    assert result["is_significant"] is True


def test_chi_square_test_on_binary_outcomes(tester: StatisticalTester) -> None:
    control = np.array([1] * 100 + [0] * 900)
    treatment = np.array([1] * 200 + [0] * 800)

    result = tester.chi_square_test(control, treatment)
    assert result["p_control"] == pytest.approx(0.10)
    assert result["p_treatment"] == pytest.approx(0.20)
    assert result["is_significant"] is True


def test_recommend_test_uses_metric_type(tester: StatisticalTester) -> None:
    assert tester.recommend_test([], [], metric_type="categorical") == "chi_square"
    assert tester.recommend_test([], [], metric_type="continuous") == "ttest"
