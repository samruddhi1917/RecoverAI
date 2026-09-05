import pandas as pd
import joblib

from feature_engineering import (
    engineer_features,
    MODEL_FEATURES
)

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


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("recoverai_transactions.csv")

print("Dataset shape:", df.shape)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

df = engineer_features(df)

print("\nModel features:")
print(MODEL_FEATURES)


# ============================================================
# FAILED TRANSACTIONS ONLY
# ============================================================

failed_df = df[
    df["transaction_status"] == "FAILED"
].copy()

print("\nTotal transactions:", len(df))
print("Failed transactions:", len(failed_df))


# ============================================================
# FEATURES AND TARGET
# ============================================================

X = failed_df[MODEL_FEATURES].copy()

y = failed_df["recovered"]


print("\nFeatures used for training:")
print(X.columns.tolist())

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
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================================
# MODEL
# ============================================================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42,
    class_weight="balanced"
)


# ============================================================
# PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
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
# EVALUATION
# ============================================================

y_pred = pipeline.predict(X_test)

y_probability = pipeline.predict_proba(
    X_test
)[:, 1]


print("\nAccuracy:")
print(
    accuracy_score(
        y_test,
        y_pred
    )
)


print("\nROC-AUC:")
print(
    roc_auc_score(
        y_test,
        y_probability
    )
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
# FEATURE IMPORTANCE
# ============================================================

feature_importance = (
    pipeline
    .named_steps["model"]
    .feature_importances_
)

feature_names = (
    pipeline
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": feature_importance
})

importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
)

print("\nTop Important Features:")
print(
    importance_df
    .head(20)
    .to_string(index=False)
)


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
# SAVE MODEL
# ============================================================

joblib.dump(
    pipeline,
    "recoverai_model.pkl"
)

print("\n========================================")
print("Model saved as recoverai_model.pkl")
print("========================================")

