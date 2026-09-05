# ============================================================
# RECOVERAI SHARED FEATURE ENGINEERING
# ============================================================

import pandas as pd


# Lean feature set validated through permutation importance.
# These are the features that actually carry meaningful
# recovery signal in the current dataset.
MODEL_FEATURES = [
    "failure_category",
    "failure_reason",
    "retry_count",
    "previous_successful_payments",
    "previous_failed_payments",
    "subscription_active",
    "customer_success_rate",
]


def engineer_features(df):
    """
    Add all engineered features required by RecoverAI.

    Works for both:
    - a full transaction DataFrame
    - a single-row prediction DataFrame
    """

    df = df.copy()

    # --------------------------------------------------------
    # CUSTOMER SUCCESS RATE
    # --------------------------------------------------------

    total_previous_payments = (
        df["previous_successful_payments"]
        + df["previous_failed_payments"]
    )

    df["customer_success_rate"] = (
        df["previous_successful_payments"]
        / total_previous_payments.replace(0, 1)
    )

    # --------------------------------------------------------
    # SAFETY
    # --------------------------------------------------------

    df["customer_success_rate"] = (
        df["customer_success_rate"]
        .fillna(0)
        .clip(0, 1)
    )

    return df