# ============================================================
# RECOVERAI MODEL TRAINING
# ============================================================

import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    confusion_matrix
)

from sklearn.calibration import calibration_curve

from feature_engineering import (
    engineer_features,
    MODEL_FEATURES
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("recoverai_transactions.csv")

print("\n========================================")
print("        RECOVERAI MODEL TRAINING")
print("========================================")

print(f"\nDataset shape: {df.shape}")


# ============================================================
# FEATURE ENGINEERING
# ============================================================

df = engineer_features(df)

print("\nEngineered features:")
print(
    df[
        ["customer_success_rate"]
    ].head(10)
)


# ============================================================
# FAILED TRANSACTIONS ONLY
# ============================================================

failed_df = df[
    df["transaction_status"] == "FAILED"
].copy()

print(
    f"\nTotal transactions: {len(df):,}"
)

print(
    f"Failed transactions: {len(failed_df):,}"
)


# ============================================================
# FEATURES + TARGET
# ============================================================

X = failed_df[MODEL_FEATURES].copy()

y = failed_df["recovered"]


print("\nModel features:")
for feature in MODEL_FEATURES:
    print(f"  - {feature}")


print("\nTarget distribution:")
print(y.value_counts())


# ============================================================
# FEATURE TYPES
# ============================================================

categorical_features = [
    "failure_category",
    "failure_reason"
]

numeric_features = [
    "retry_count",
    "previous_successful_payments",
    "previous_failed_payments",
    "subscription_active",
    "customer_success_rate"
]


# ============================================================
# PREPROCESSOR
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        ),
        (
            "numeric",
            SimpleImputer(
                strategy="median"
            ),
            numeric_features
        )
    ]
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# RANDOM FOREST
# ============================================================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42,
    class_weight="balanced"
)


# ============================================================
# COMPLETE PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        )
    ]
)


# ============================================================
# TRAIN
# ============================================================

pipeline.fit(
    X_train,
    y_train
)

print("\nModel training complete!")


# ============================================================
# PREDICTIONS
# ============================================================

y_pred = pipeline.predict(X_test)

y_probability = (
    pipeline.predict_proba(X_test)[:, 1]
)


# ============================================================
# EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


print("\n========================================")
print("           MODEL EVALUATION")
print("========================================")

print(
    f"\nAccuracy: {accuracy:.4f}"
)

print(
    f"ROC-AUC: {roc_auc:.4f}"
)


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# CALIBRATION
# ============================================================

prob_true, prob_pred = calibration_curve(
    y_test,
    y_probability,
    n_bins=10
)

print("\nProbability Calibration:")

for predicted, actual in zip(
    prob_pred,
    prob_true
):
    print(
        f"Predicted: {predicted:.2f} | "
        f"Actual recovery rate: {actual:.2f}"
    )


# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

test_results = X_test.copy()

test_results["actual_recovered"] = (
    y_test.values
)

test_results["model_recovery_probability"] = (
    y_probability
)

print("\nSample Recovery Predictions:")

print(
    test_results[
        [
            "model_recovery_probability",
            "actual_recovered"
        ]
    ].head(10)
)


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    pipeline,
    "recoverai_model.pkl"
)


print("\n========================================")
print("Model saved as recoverai_model.pkl")
print("========================================")