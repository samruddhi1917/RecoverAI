import pandas as pd

df = pd.read_csv("recoverai_transactions.csv")

df["customer_success_rate"] = (
    df["previous_successful_payments"]
    / (
        df["previous_successful_payments"]
        + df["previous_failed_payments"]
    ).replace(0, 1)
)

df["has_been_retried"] = (
    df["retry_count"] > 0
).astype(int)

df["amount_to_lifetime_value_ratio"] = (
    df["amount"] / df["customer_lifetime_value"]
)

df["has_retry_timestamp"] = (
    df["last_retry_hours_ago"].notna()
).astype(int)

print("\nNew engineered features:")

print(
    df[
        [
            "customer_success_rate",
            "has_been_retried",
            "amount_to_lifetime_value_ratio",
            "has_retry_timestamp"
        ]
    ].head(10)
)

print("Dataset shape:", df.shape)
print(df.head())
print(df.columns.tolist())

failed_df = df[df["transaction_status"] == "FAILED"].copy()

print("Total transactions:", len(df))
print("Failed transactions:", len(failed_df))


X = failed_df.drop(columns=[
    "transaction_id",
    "customer_id",
    "recovered",
    "recovery_probability",
    "expected_recovery",
    "transaction_status"
])

y = failed_df["recovered"]

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget distribution:")
print(y.value_counts())

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

categorical_features = [
    "payment_method",
    "merchant_category",
    "failure_category",
    "failure_reason",
    "device_type",
    "region"
]

numeric_features = [
    "amount",
    "retry_count",
    "last_retry_hours_ago",
    "transaction_hour",
    "subscription_active",
    "previous_successful_payments",
    "previous_failed_payments",
    "customer_lifetime_value",
    "customer_success_rate",
    "has_been_retried",
    "amount_to_lifetime_value_ratio",
    "has_retry_timestamp"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numeric",
            SimpleImputer(strategy="median"),
            numeric_features
        )
    ]
)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42,
    class_weight="balanced"
)

from sklearn.pipeline import Pipeline

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

pipeline.fit(X_train, y_train)

print("Model training complete!")

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score
)

y_pred = pipeline.predict(X_test)

y_probability = pipeline.predict_proba(X_test)[:, 1]

print("\nSample Recovery Probabilities:")

for i in range(10):
    print(
        f"Actual: {y_test.iloc[i]} | "
        f"Recovery Probability: {y_probability[i]:.2f}"
    )

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nROC-AUC:")
print(roc_auc_score(y_test, y_probability))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

import pandas as pd

feature_importance = pipeline.named_steps["model"].feature_importances_

feature_names = pipeline.named_steps[
    "preprocessor"
].get_feature_names_out()

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": feature_importance
})

importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
)

print("\nTop 20 Important Features:")
print(importance_df.head(20).to_string(index=False))

from sklearn.metrics import confusion_matrix

y_pred = pipeline.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

from sklearn.calibration import calibration_curve

y_probability = pipeline.predict_proba(X_test)[:, 1]

test_results = X_test.copy()

test_results["actual_recovered"] = y_test.values
test_results["model_recovery_probability"] = y_probability

test_results["model_expected_recovery"] = (
    test_results["amount"] *
    test_results["model_recovery_probability"]
)

print("\nSample Revenue Recovery Predictions:")

print(
    test_results[
        [
            "amount",
            "model_recovery_probability",
            "model_expected_recovery",
            "actual_recovered"
        ]
    ].head(10)
)

prob_true, prob_pred = calibration_curve(
    y_test,
    y_probability,
    n_bins=10
)

print("\nProbability Calibration:")

for predicted, actual in zip(prob_pred, prob_true):
    print(
        f"Predicted: {predicted:.2f} | "
        f"Actual recovery rate: {actual:.2f}"
    )

import joblib

joblib.dump(pipeline, "recoverai_model.pkl")

print("\nModel saved successfully!")