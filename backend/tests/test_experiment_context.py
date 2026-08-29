"""Fast, no-network unit tests for build_experiment_context — the function that
decides what real numbers get handed to the LLM before it explains a result.

Regression coverage for a real bug found during manual verification: the model
was inventing specific-looking retention percentages it was never given,
because build_experiment_context only forwarded p_value/uplift/n and dropped
the actual per-group rates the stats engine already computes. These tests
pin down that every numeric field the engine can produce actually reaches the
prompt, and that the anti-hallucination instruction is always present.
"""

from app.rag.retriever import build_experiment_context


def test_no_results_returns_empty_context() -> None:
    assert build_experiment_context(None) == ""
    assert build_experiment_context({}) == ""


def test_chi_square_group_rates_are_forwarded() -> None:
    results = {
        "test_name": "Chi-square (2x2)",
        "metric": "retention_7",
        "p_value": 0.006,
        "is_significant": True,
        "uplift_percentage": -0.71,
        "p_control": 0.1873,
        "p_treatment": 0.1802,
        "n_control": 45000,
        "n_treatment": 45000,
    }
    context = build_experiment_context(results)
    assert "18.73%" in context
    assert "18.02%" in context
    assert "retention_7" in context
    assert "45,000" in context


def test_continuous_metric_means_are_forwarded() -> None:
    results = {
        "test_name": "Welch's t-test",
        "metric": "cart_value",
        "p_value": 0.02,
        "is_significant": True,
        "uplift_percentage": 8.5,
        "mean_control": 75.12,
        "mean_treatment": 81.53,
        "n_control": 2000,
        "n_treatment": 2000,
    }
    context = build_experiment_context(results)
    assert "75.1200" in context
    assert "81.5300" in context


def test_guardrail_results_are_summarized() -> None:
    results = {
        "test_name": "Chi-square (2x2)",
        "p_value": 0.03,
        "is_significant": True,
        "guardrails": [
            {"metric": "latency_ms", "p_value": 0.9, "is_significant": False},
            {"metric": "error_rate", "p_value": 0.01, "is_significant": True},
        ],
    }
    context = build_experiment_context(results)
    assert "latency_ms" in context
    assert "error_rate" in context
    assert "not significant" in context


def test_context_always_forbids_inventing_extra_numbers() -> None:
    context = build_experiment_context({"p_value": 0.5})
    assert "never invent additional metrics" in context.lower()


def test_context_omits_fields_not_present() -> None:
    # A ttest result has no p_control/p_treatment — the prompt shouldn't claim it does.
    context = build_experiment_context({"p_value": 0.5, "mean_control": 1.0, "mean_treatment": 1.2})
    assert "Control group rate" not in context
    assert "Control group mean" in context
