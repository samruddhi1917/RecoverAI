import pandas as pd
import joblib

# Load trained model
model = joblib.load("recoverai_model.pkl")

print("RecoverAI model loaded successfully!")

# Example failed transaction
transaction = pd.DataFrame([{
    "amount": 590,
    "payment_method": "UPI",
    "merchant_category": "ELECTRONICS",
    "failure_category": "TEMPORARY",
    "failure_reason": "NETWORK_TIMEOUT",
    "retry_count": 1,
    "last_retry_hours_ago": 2,
    "device_type": "PHONE",
    "region": "MAHARASHTRA",
    "transaction_hour": 10,
    "subscription_active": 1,
    "previous_successful_payments": 8,
    "previous_failed_payments": 1,
    "customer_lifetime_value": 25000,
    "customer_success_rate": 8 / 9,
    "has_been_retried": 1,
    "amount_to_lifetime_value_ratio": 590 / 25000,
    "has_retry_timestamp": 1
}])

# Predict recovery probability
probability = model.predict_proba(transaction)[0][1]

# Calculate expected recovery
expected_recovery = 590 * probability

print("\n========== RECOVERAI PREDICTION ==========")

print(f"Transaction Amount: Rs. 590")
print(f"Recovery Probability: {probability:.2%}")
print(f"Expected Recovery: Rs. {expected_recovery:.2f}")

if probability >= 0.70:
    recommendation = "HIGH PRIORITY RETRY"
elif probability >= 0.40:
    recommendation = "RETRY WITH CAUTION"
else:
    recommendation = "LOW PRIORITY RETRY"

print(f"Recommendation: {recommendation}")