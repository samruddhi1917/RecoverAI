import pandas as pd
import joblib

# Load trained model
model = joblib.load("recoverai_model.pkl")

print("RecoverAI model loaded successfully!")


# Function to make prediction
def predict_recovery(case_name, transaction):

    probability = model.predict_proba(transaction)[0][1]

    amount = transaction["amount"].iloc[0]

    expected_recovery = amount * probability

    if probability >= 0.70:
        recommendation = "HIGH PRIORITY RETRY"
    elif probability >= 0.40:
        recommendation = "RETRY WITH CAUTION"
    else:
        recommendation = "LOW PRIORITY RETRY"

    print("\n========================================")
    print(case_name)
    print("========================================")

    print(f"Transaction Amount: Rs. {amount:.2f}")
    print(f"Recovery Probability: {probability:.2%}")
    print(f"Expected Recovery: Rs. {expected_recovery:.2f}")
    print(f"Recommendation: {recommendation}")


# ==================================================
# CASE A - HIGH POTENTIAL
# ==================================================

case_a = pd.DataFrame([{
    "amount": 5000,
    "payment_method": "UPI",
    "merchant_category": "ELECTRONICS",
    "failure_category": "TEMPORARY",
    "failure_reason": "NETWORK_TIMEOUT",
    "retry_count": 0,
    "last_retry_hours_ago": None,
    "device_type": "PHONE",
    "region": "MAHARASHTRA",
    "transaction_hour": 10,
    "subscription_active": 1,
    "previous_successful_payments": 18,
    "previous_failed_payments": 1,
    "customer_lifetime_value": 50000,
    "customer_success_rate": 18 / 19,
    "has_been_retried": 0,
    "amount_to_lifetime_value_ratio": 5000 / 50000,
    "has_retry_timestamp": 0
}])

predict_recovery("CASE A - HIGH POTENTIAL", case_a)


# ==================================================
# CASE B - LOW POTENTIAL
# ==================================================

case_b = pd.DataFrame([{
    "amount": 5000,
    "payment_method": "CARD",
    "merchant_category": "ELECTRONICS",
    "failure_category": "CUSTOMER_ACTION",
    "failure_reason": "INSUFFICIENT_FUNDS",
    "retry_count": 5,
    "last_retry_hours_ago": 1,
    "device_type": "PHONE",
    "region": "MAHARASHTRA",
    "transaction_hour": 10,
    "subscription_active": 1,
    "previous_successful_payments": 2,
    "previous_failed_payments": 7,
    "customer_lifetime_value": 30000,
    "customer_success_rate": 2 / 9,
    "has_been_retried": 1,
    "amount_to_lifetime_value_ratio": 5000 / 30000,
    "has_retry_timestamp": 1
}])

predict_recovery("CASE B - LOW POTENTIAL", case_b)


# ==================================================
# CASE C - FIRST RETRY
# ==================================================

case_c = pd.DataFrame([{
    "amount": 2000,
    "payment_method": "UPI",
    "merchant_category": "CLOTHING",
    "failure_category": "TEMPORARY",
    "failure_reason": "TEMPORARY_SERVICE_ERROR",
    "retry_count": 0,
    "last_retry_hours_ago": None,
    "device_type": "PHONE",
    "region": "MAHARASHTRA",
    "transaction_hour": 14,
    "subscription_active": 1,
    "previous_successful_payments": 10,
    "previous_failed_payments": 0,
    "customer_lifetime_value": 25000,
    "customer_success_rate": 1.0,
    "has_been_retried": 0,
    "amount_to_lifetime_value_ratio": 2000 / 25000,
    "has_retry_timestamp": 0
}])

predict_recovery("CASE C - FIRST RETRY", case_c)


# ==================================================
# CASE D - NETWORK FAILURE
# ==================================================

case_d = pd.DataFrame([{
    "amount": 3000,
    "payment_method": "NETBANKING",
    "merchant_category": "HOME_APPLIANCES",
    "failure_category": "TEMPORARY",
    "failure_reason": "NETWORK_TIMEOUT",
    "retry_count": 1,
    "last_retry_hours_ago": 2,
    "device_type": "DESKTOP",
    "region": "MAHARASHTRA",
    "transaction_hour": 18,
    "subscription_active": 1,
    "previous_successful_payments": 12,
    "previous_failed_payments": 2,
    "customer_lifetime_value": 40000,
    "customer_success_rate": 12 / 14,
    "has_been_retried": 1,
    "amount_to_lifetime_value_ratio": 3000 / 40000,
    "has_retry_timestamp": 1
}])

predict_recovery("CASE D - NETWORK FAILURE", case_d)