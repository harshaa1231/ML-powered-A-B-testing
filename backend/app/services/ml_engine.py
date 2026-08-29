"""Universal ML engine: auto column detection, multi-model training, and
T-learner uplift modeling.

Ported from the original `ml_engine.py` prototype — same algorithms
(GradientBoosting/RandomForest comparison, T-learner uplift), now typed
and covered by unit tests.
"""

from __future__ import annotations

import io
import warnings
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge  # noqa: F401 (kept for parity/extension)
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")


class UniversalMLEngine:
    def __init__(self) -> None:
        self.models: dict[str, Any] = {}
        self.scalers: dict[str, Any] = {}
        self.encoders: dict[str, LabelEncoder] = {}
        self.feature_names: list[str] = []
        self.target_col: str | None = None
        self.treatment_col: str | None = None
        self.task_type: str | None = None
        self.is_trained = False
        self.training_results: dict[str, Any] = {}
        self.feature_importance: dict[str, dict[str, float]] = {}

    def auto_detect_columns(self, df: pd.DataFrame) -> dict[str, Any]:
        detection: dict[str, list[str]] = {
            "potential_group_cols": [],
            "potential_target_cols": [],
            "potential_guardrail_cols": [],
            "numeric_cols": [],
            "categorical_cols": [],
            "binary_cols": [],
        }

        for col in df.columns:
            unique_vals = df[col].dropna().unique()
            n_unique = len(unique_vals)

            if df[col].dtype == "object" or n_unique <= 10:
                detection["categorical_cols"].append(col)

                if n_unique == 2:
                    detection["binary_cols"].append(col)
                    str_vals = {str(v).lower().strip() for v in unique_vals}
                    group_indicators = {"control", "treatment", "test", "variant", "a", "b", "exposed", "unexposed"}
                    if str_vals & group_indicators:
                        detection["potential_group_cols"].append(col)

                if col.lower() in [
                    "group", "variant", "treatment", "test_group", "ab_group",
                    "experiment_group", "condition", "arm", "segment",
                ]:
                    if col not in detection["potential_group_cols"]:
                        detection["potential_group_cols"].append(col)

            if np.issubdtype(df[col].dtype, np.number):
                detection["numeric_cols"].append(col)
                if n_unique == 2 and set(unique_vals) <= {0, 1}:
                    detection["binary_cols"].append(col)

        target_keywords = [
            "convert", "click", "purchase", "revenue", "sale", "open",
            "retain", "churn", "bounce", "signup", "subscribe", "engaged",
            "completed", "success", "outcome", "target", "response", "value",
            "score", "rate", "metric", "kpi", "result",
        ]
        for col in df.columns:
            col_lower = col.lower()
            for kw in target_keywords:
                if kw in col_lower and col not in detection["potential_group_cols"]:
                    detection["potential_target_cols"].append(col)
                    break

        # Guardrail metrics: numeric columns that look like something you'd want to make
        # sure DIDN'T get worse, even while the primary metric improves. Heuristic only —
        # no model call, deliberately, since pattern-matching does the job for free.
        guardrail_keywords = ["time", "latency", "cost", "error", "churn", "bounce", "complaint"]
        for col in detection["numeric_cols"]:
            col_lower = col.lower()
            if col in detection["potential_group_cols"]:
                continue
            if any(kw in col_lower for kw in guardrail_keywords):
                detection["potential_guardrail_cols"].append(col)

        return detection

    def prepare_features(
        self, df: pd.DataFrame, target_col: str, exclude_cols: list[str] | None = None, fit: bool = True
    ) -> tuple[pd.DataFrame, np.ndarray | None]:
        df = df.copy()
        exclude = set(exclude_cols or [])
        exclude.add(target_col)

        y = None
        if target_col in df.columns:
            y = df[target_col].values
            df = df.drop(columns=[target_col], errors="ignore")

        df = df.drop(columns=[c for c in exclude if c in df.columns], errors="ignore")

        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        for col in cat_cols:
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.encoders[col] = le
            else:
                if col in self.encoders:
                    le = self.encoders[col]
                    df[col] = df[col].astype(str).map(lambda x, _le=le: x if x in _le.classes_ else "unknown")
                    if "unknown" not in le.classes_:
                        le.classes_ = np.append(le.classes_, "unknown")
                    df[col] = le.transform(df[col])
                else:
                    df = df.drop(columns=[col])

        df = df.fillna(df.median(numeric_only=True))
        df = df.select_dtypes(include=[np.number])

        if fit:
            scaler = StandardScaler()
            scaled = scaler.fit_transform(df)
            self.scalers["main"] = scaler
        elif "main" in self.scalers:
            common_cols = [c for c in df.columns if c in self.feature_names]
            missing_cols = [c for c in self.feature_names if c not in df.columns]
            df = df[common_cols]
            for mc in missing_cols:
                df[mc] = 0
            df = df[self.feature_names]
            scaled = self.scalers["main"].transform(df)
        else:
            scaled = df.values

        result = pd.DataFrame(scaled, columns=df.columns, index=df.index)

        if fit:
            self.feature_names = result.columns.tolist()

        return result, y

    def train_model(
        self,
        df: pd.DataFrame,
        target_col: str,
        group_col: str | None = None,
        model_type: str = "auto",
        test_size: float = 0.2,
    ) -> dict[str, Any]:
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset.")
        if df[target_col].dropna().empty:
            raise ValueError(f"Target column '{target_col}' has no valid values.")
        if len(df) < 10:
            raise ValueError("Dataset too small. Need at least 10 rows to train a model.")

        self.target_col = target_col
        self.treatment_col = group_col

        exclude_cols = [group_col] if group_col else []
        X, y = self.prepare_features(df, target_col, exclude_cols=exclude_cols, fit=True)

        if X.shape[1] == 0:
            raise ValueError("No usable features found after preprocessing. Check your data.")

        y_clean = y[~np.isnan(y)] if np.issubdtype(y.dtype, np.number) else y
        n_unique = len(np.unique(y_clean)) if len(y_clean) > 0 else 0

        if n_unique < 2:
            raise ValueError("Target column has only one unique value. Cannot train a model on constant data.")

        if model_type == "auto":
            self.task_type = "classification" if n_unique <= 10 else "regression"
        else:
            self.task_type = model_type

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        results: dict[str, Any] = {}

        if self.task_type == "classification":
            models_to_train = {
                "gradient_boosting": GradientBoostingClassifier(
                    n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, subsample=0.8
                ),
                "random_forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
            }
        else:
            models_to_train = {
                "gradient_boosting": GradientBoostingRegressor(
                    n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, subsample=0.8
                ),
                "random_forest": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
            }

        best_score = -np.inf
        best_name = next(iter(models_to_train))

        for name, model in models_to_train.items():
            model.fit(X_train, y_train)
            self.models[name] = model

            if self.task_type == "classification":
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                y_pred = model.predict(X_test)
                try:
                    auc = roc_auc_score(y_test, y_pred_proba)
                except ValueError:
                    auc = 0.5
                acc = accuracy_score(y_test, y_pred)
                score = auc
                results[name] = {"accuracy": float(acc), "auc_roc": float(auc), "score": float(score)}
            else:
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                score = r2 if not np.isnan(r2) else 0.0
                results[name] = {"rmse": float(rmse), "mae": float(mae), "r2_score": float(score), "score": float(score)}

            if not np.isnan(score) and score > best_score:
                best_score = score
                best_name = name

            if hasattr(model, "feature_importances_"):
                fi = dict(zip(self.feature_names, model.feature_importances_, strict=False))
                self.feature_importance[name] = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))

        if best_score == -np.inf:
            best_score = 0.0

        self.models["best"] = self.models[best_name]
        self.is_trained = True

        self.training_results = {
            "task_type": self.task_type,
            "best_model": best_name,
            "best_score": float(best_score),
            "n_features": len(self.feature_names),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "model_results": results,
            "feature_names": self.feature_names,
            "feature_importance": self.feature_importance.get(best_name, {}),
        }

        return self.training_results

    def train_uplift_model(
        self, df: pd.DataFrame, target_col: str, treatment_col: str, learner_type: str = "t_learner"
    ) -> dict[str, Any]:
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset.")
        if treatment_col not in df.columns:
            raise ValueError(f"Treatment column '{treatment_col}' not found in dataset.")
        if len(df) < 20:
            raise ValueError("Dataset too small for uplift modeling. Need at least 20 rows.")

        self.target_col = target_col
        self.treatment_col = treatment_col

        treatment_vals = df[treatment_col].astype(str).str.strip().str.lower()
        unique_vals = treatment_vals.unique()

        if len(unique_vals) < 2:
            raise ValueError(f"Treatment column '{treatment_col}' needs at least 2 distinct values (e.g., control/treatment).")

        treatment_labels = {"treatment", "1", "true", "yes", "exposed", "b", "variant"}

        treatment_binary = np.zeros(len(df), dtype=int)
        for val in unique_vals:
            if val in treatment_labels:
                treatment_binary[treatment_vals == val] = 1

        if treatment_binary.sum() == 0:
            treatment_binary = (treatment_vals != treatment_vals.iloc[0]).astype(int)

        n_ctrl = int((treatment_binary == 0).sum())
        n_treat = int((treatment_binary == 1).sum())
        if n_ctrl < 5 or n_treat < 5:
            raise ValueError(f"Not enough samples in each group. Control: {n_ctrl}, Treatment: {n_treat}. Need at least 5 in each.")

        X, y = self.prepare_features(df, target_col, exclude_cols=[treatment_col], fit=True)

        if X.shape[1] == 0:
            raise ValueError("No usable features found after preprocessing.")

        y_clean = y[~np.isnan(y)] if np.issubdtype(y.dtype, np.number) else y
        n_unique = len(np.unique(y_clean)) if len(y_clean) > 0 else 0
        if n_unique < 2:
            raise ValueError("Target column has only one unique value. Cannot train uplift model.")

        self.task_type = "classification" if n_unique <= 10 else "regression"

        X_ctrl = X[treatment_binary == 0]
        X_treat = X[treatment_binary == 1]
        y_ctrl = y[treatment_binary == 0]
        y_treat = y[treatment_binary == 1]

        if self.task_type == "classification":
            model_ctrl = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, subsample=0.8)
            model_treat = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, subsample=0.8)
        else:
            model_ctrl = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, subsample=0.8)
            model_treat = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, subsample=0.8)

        model_ctrl.fit(X_ctrl, y_ctrl)
        model_treat.fit(X_treat, y_treat)

        self.models["uplift_control"] = model_ctrl
        self.models["uplift_treatment"] = model_treat

        if self.task_type == "classification":
            pred_ctrl = model_ctrl.predict_proba(X)[:, 1]
            pred_treat = model_treat.predict_proba(X)[:, 1]
        else:
            pred_ctrl = model_ctrl.predict(X)
            pred_treat = model_treat.predict(X)

        uplift = pred_treat - pred_ctrl

        X_train_ctrl, X_test_ctrl, y_train_ctrl, y_test_ctrl = train_test_split(X_ctrl, y_ctrl, test_size=0.2, random_state=42)
        X_train_treat, X_test_treat, y_train_treat, y_test_treat = train_test_split(X_treat, y_treat, test_size=0.2, random_state=42)

        if self.task_type == "classification":
            try:
                auc_ctrl = roc_auc_score(y_test_ctrl, model_ctrl.predict_proba(X_test_ctrl)[:, 1])
            except ValueError:
                auc_ctrl = 0.5
            try:
                auc_treat = roc_auc_score(y_test_treat, model_treat.predict_proba(X_test_treat)[:, 1])
            except ValueError:
                auc_treat = 0.5
        else:
            auc_ctrl = r2_score(y_test_ctrl, model_ctrl.predict(X_test_ctrl))
            auc_ctrl = 0.0 if np.isnan(auc_ctrl) else auc_ctrl
            auc_treat = r2_score(y_test_treat, model_treat.predict(X_test_treat))
            auc_treat = 0.0 if np.isnan(auc_treat) else auc_treat

        if hasattr(model_treat, "feature_importances_"):
            fi = dict(zip(self.feature_names, model_treat.feature_importances_, strict=False))
            self.feature_importance["uplift"] = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))

        self.is_trained = True

        self.training_results = {
            "task_type": self.task_type,
            "learner_type": learner_type,
            "avg_uplift": float(np.mean(uplift)),
            "median_uplift": float(np.median(uplift)),
            "uplift_std": float(np.std(uplift)),
            "positive_uplift_pct": float(np.mean(uplift > 0) * 100),
            "score_control": float(auc_ctrl),
            "score_treatment": float(auc_treat),
            "n_control": int(len(X_ctrl)),
            "n_treatment": int(len(X_treat)),
            "n_features": len(self.feature_names),
            "feature_names": self.feature_names,
            "feature_importance": self.feature_importance.get("uplift", {}),
        }

        return self.training_results

    def predict(self, df: pd.DataFrame, model_name: str = "best") -> np.ndarray:
        if not self.is_trained:
            raise ValueError("No trained model available. Train a model first.")

        X, _ = self.prepare_features(
            df, self.target_col or "__dummy__", exclude_cols=[self.treatment_col] if self.treatment_col else [], fit=False
        )

        model = self.models.get(model_name, self.models.get("best"))
        if model is None:
            raise ValueError(f"Model '{model_name}' not found.")

        if self.task_type == "classification" and hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        return model.predict(X)

    def predict_uplift(self, df: pd.DataFrame) -> np.ndarray:
        if "uplift_control" not in self.models or "uplift_treatment" not in self.models:
            raise ValueError("No uplift model trained. Train an uplift model first.")

        X, _ = self.prepare_features(
            df, self.target_col or "__dummy__", exclude_cols=[self.treatment_col] if self.treatment_col else [], fit=False
        )

        model_ctrl = self.models["uplift_control"]
        model_treat = self.models["uplift_treatment"]

        if self.task_type == "classification":
            pred_ctrl = model_ctrl.predict_proba(X)[:, 1]
            pred_treat = model_treat.predict_proba(X)[:, 1]
        else:
            pred_ctrl = model_ctrl.predict(X)
            pred_treat = model_treat.predict(X)

        return pred_treat - pred_ctrl

    def _state_dict(self) -> dict[str, Any]:
        return {
            "models": self.models,
            "scalers": self.scalers,
            "encoders": self.encoders,
            "feature_names": self.feature_names,
            "target_col": self.target_col,
            "treatment_col": self.treatment_col,
            "task_type": self.task_type,
            "is_trained": self.is_trained,
            "training_results": self.training_results,
            "feature_importance": self.feature_importance,
        }

    def _load_state_dict(self, data: dict[str, Any]) -> None:
        self.models = data["models"]
        self.scalers = data["scalers"]
        self.encoders = data["encoders"]
        self.feature_names = data["feature_names"]
        self.target_col = data.get("target_col")
        self.treatment_col = data.get("treatment_col")
        self.task_type = data.get("task_type")
        self.is_trained = data["is_trained"]
        self.training_results = data.get("training_results", {})
        self.feature_importance = data.get("feature_importance", {})

    def save(self, filepath: str) -> None:
        joblib.dump(self._state_dict(), filepath)

    def load(self, filepath: str) -> None:
        self._load_state_dict(joblib.load(filepath))

    def to_bytes(self) -> bytes:
        buffer = io.BytesIO()
        joblib.dump(self._state_dict(), buffer)
        return buffer.getvalue()

    @classmethod
    def from_bytes(cls, data: bytes) -> UniversalMLEngine:
        engine = cls()
        engine._load_state_dict(joblib.load(io.BytesIO(data)))
        return engine
