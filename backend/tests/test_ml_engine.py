import numpy as np
import pandas as pd
import pytest

from app.services.ml_engine import UniversalMLEngine


def _make_classification_df(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    group = rng.choice(["control", "treatment"], n)
    feature_1 = rng.normal(0, 1, n)
    feature_2 = rng.choice(["a", "b", "c"], n)
    prob = 0.3 + (group == "treatment") * 0.2 + feature_1 * 0.05
    converted = (rng.random(n) < np.clip(prob, 0, 1)).astype(int)
    return pd.DataFrame({"group": group, "feature_1": feature_1, "feature_2": feature_2, "converted": converted})


def test_auto_detect_columns_finds_group_and_target() -> None:
    df = _make_classification_df()
    engine = UniversalMLEngine()
    detection = engine.auto_detect_columns(df)

    assert "group" in detection["potential_group_cols"]
    assert "converted" in detection["potential_target_cols"]
    assert "feature_1" in detection["numeric_cols"]


def test_auto_detect_columns_flags_guardrail_candidates() -> None:
    df = _make_classification_df()
    df["page_load_time"] = np.random.default_rng(1).normal(2.0, 0.5, len(df))
    df["error_rate"] = np.random.default_rng(2).normal(0.02, 0.01, len(df))

    engine = UniversalMLEngine()
    detection = engine.auto_detect_columns(df)

    assert "page_load_time" in detection["potential_guardrail_cols"]
    assert "error_rate" in detection["potential_guardrail_cols"]
    assert "feature_1" not in detection["potential_guardrail_cols"]
    assert "group" not in detection["potential_guardrail_cols"]


def test_train_model_classification_end_to_end() -> None:
    df = _make_classification_df(n=500)
    engine = UniversalMLEngine()
    results = engine.train_model(df, target_col="converted", group_col="group")

    assert results["task_type"] == "classification"
    assert engine.is_trained is True
    assert results["best_model"] in ("gradient_boosting", "random_forest")
    assert 0.0 <= results["best_score"] <= 1.0
    assert results["n_features"] > 0


def test_train_model_rejects_tiny_dataset() -> None:
    df = _make_classification_df(n=5)
    engine = UniversalMLEngine()
    with pytest.raises(ValueError):
        engine.train_model(df, target_col="converted")


def test_train_model_rejects_constant_target() -> None:
    df = _make_classification_df(n=50)
    df["converted"] = 1
    engine = UniversalMLEngine()
    with pytest.raises(ValueError):
        engine.train_model(df, target_col="converted")


def test_train_model_rejects_dataset_with_no_feature_columns_left() -> None:
    """Regression test: a dataset with only a group column and a target column (the
    exact shape of several of this app's own sample datasets) used to reach
    StandardScaler with zero columns and crash with a cryptic sklearn internal error
    ('at least one array or dtype is required') instead of an actionable message."""
    rng = np.random.default_rng(1)
    n = 200
    df = pd.DataFrame({
        "group": rng.choice(["control", "treatment"], n),
        "converted": (rng.random(n) < 0.3).astype(int),
    })
    engine = UniversalMLEngine()
    with pytest.raises(ValueError, match="No usable feature columns remain"):
        engine.train_model(df, target_col="converted", group_col="group")


def test_predict_after_training_returns_expected_shape() -> None:
    df = _make_classification_df(n=500)
    engine = UniversalMLEngine()
    engine.train_model(df, target_col="converted", group_col="group")

    new_data = _make_classification_df(n=20, seed=99).drop(columns=["converted"])
    predictions = engine.predict(new_data)
    assert len(predictions) == 20


def test_uplift_model_end_to_end() -> None:
    df = _make_classification_df(n=500)
    engine = UniversalMLEngine()
    results = engine.train_uplift_model(df, target_col="converted", treatment_col="group")

    assert "avg_uplift" in results
    assert results["n_control"] > 0
    assert results["n_treatment"] > 0


def test_roundtrip_serialization_preserves_predictions() -> None:
    df = _make_classification_df(n=500)
    engine = UniversalMLEngine()
    engine.train_model(df, target_col="converted", group_col="group")

    new_data = _make_classification_df(n=20, seed=7).drop(columns=["converted"])
    original_predictions = engine.predict(new_data)

    restored = UniversalMLEngine.from_bytes(engine.to_bytes())
    restored_predictions = restored.predict(new_data)

    np.testing.assert_allclose(original_predictions, restored_predictions)
