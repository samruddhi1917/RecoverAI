import pandas as pd
import joblib

# ==============================
# LOAD TRAINED MODEL
# ==============================

model = joblib.load("recoverai_model.pkl")

print("RecoverAI model loaded successfully!")

print("\n========== RECOVERAI PREDICTION ==========")


# ==============================
# GET TRANSACTION DETAILS
# ==============================

amount = float(input("Transaction amount (Rs.): "))

payment_method = input(
    "Payment method (UPI/CARD/NETBANKING/WALLET): "
).upper()

merchant_category = input(
    "Merchant category "
    "(ELECTRONICS/CLOTHING/GROCERIES/FOOD/TRAVEL/HOME_APPLIANCES): "
).upper()

failure_category = input(
    "Failure category "
    "(TEMPORARY/CUSTOMER_ACTION/PAYMENT_METHOD/SYSTEM): "
).upper()

failure_reason = input(
    "Failure reason "
    "(NETWORK_TIMEOUT/GATEWAY_TIMEOUT/INSUFFICIENT_FUNDS/"
    "AUTHENTICATION_REQUIRED/EXPIRED_CARD/INVALID_CARD/"
    "PROCESSING_ERROR/TEMPORARY_SERVICE_ERROR): "
).upper()

retry_count = int(input("Retry count: "))

last_retry_hours_ago = float(
    input("Hours since last retry (enter 0 if never retried): ")
)

device_type = input(
    "Device type (PHONE/DESKTOP/TABLET): "
).upper()

region = input("Region/state: ").upper()

transaction_hour = int(
    input("Transaction hour (0-23): ")
)

subscription_active = int(
    input("Is subscription active? (1 = Yes, 0 = No): ")
)

previous_successful_payments = int(
    input("Previous successful payments: ")
)

previous_failed_payments = int(
    input("Previous failed payments: ")
)

customer_lifetime_value = float(
    input("Customer lifetime value (Rs.): ")
)

# ============================================================
# AUDIT TRAIL
# ============================================================

st.divider()

st.header("🔐 Decision Audit Trail")

st.write(
    "RecoverAI records the key inputs and decision generated "
    "for each prediction to support transparency and auditing."
)

if st.button("📝 Generate Audit Record", width="stretch"):

    import datetime
    import uuid

    audit_id = "AUD-" + str(uuid.uuid4())[:8].upper()

    audit_record = pd.DataFrame({
        "Field": [
            "Audit ID",
            "Timestamp",
            "Transaction Amount",
            "Payment Method",
            "Failure Category",
            "Failure Reason",
            "Retry Count",
            "Recovery Probability",
            "Expected Recovery",
            "Retry Priority",
            "Recommended Action"
        ],
        "Value": [
            audit_id,
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            f"₹{amount:,.2f}",
            payment_method,
            failure_category,
            failure_reason,
            str(retry_count),
            f"{recovery_probability:.2%}",
            f"₹{expected_recovery:,.2f}",
            priority,
            smart_action
        ]
    })

    st.success("✅ Audit record generated successfully.")

    st.dataframe(
        audit_record,
        width="stretch",
        hide_index=True
    )

    st.download_button(
        "⬇️ Download Audit Record",
        audit_record.to_csv(index=False),
        file_name=f"{audit_id}.csv",
        mime="text/csv",
        width="stretch"
    )


# ==============================
# ENGINEERED FEATURES
# ==============================

total_previous_payments = (
    previous_successful_payments
    + previous_failed_payments
)

customer_success_rate = (
    previous_successful_payments
    / total_previous_payments
    if total_previous_payments > 0
    else 0
)

has_been_retried = int(retry_count > 0)

amount_to_lifetime_value_ratio = (
    amount / customer_lifetime_value
    if customer_lifetime_value > 0
    else 0
)

has_retry_timestamp = int(retry_count > 0)

if retry_count == 0:
    last_retry_hours_ago = None


# ==============================
# CREATE TRANSACTION
# ==============================

transaction = pd.DataFrame([{

    "amount": amount,

    "payment_method": payment_method,

    "merchant_category": merchant_category,

    "failure_category": failure_category,

    "failure_reason": failure_reason,

    "retry_count": retry_count,

    "last_retry_hours_ago": last_retry_hours_ago,

    "device_type": device_type,

    "region": region,

    "transaction_hour": transaction_hour,

    "subscription_active": subscription_active,

    "previous_successful_payments":
        previous_successful_payments,

    "previous_failed_payments":
        previous_failed_payments,

    "customer_lifetime_value":
        customer_lifetime_value,

    "customer_success_rate":
        customer_success_rate,

    "has_been_retried":
        has_been_retried,

    "amount_to_lifetime_value_ratio":
        amount_to_lifetime_value_ratio,

    "has_retry_timestamp":
        has_retry_timestamp

}])


# ==============================
# MODEL PREDICTION
# ==============================

recovery_probability = model.predict_proba(
    transaction
)[0][1]

expected_recovery = (
    amount * recovery_probability
)


# ==============================
# RECOVERAI DECISION ENGINE
# ==============================

if recovery_probability >= 0.75:

    recommendation = "HIGH PRIORITY RETRY"
    priority = "HIGH"

elif recovery_probability >= 0.50:

    recommendation = "RETRY WITH CAUTION"
    priority = "MEDIUM"

elif recovery_probability >= 0.25:

    recommendation = "LOW PRIORITY RETRY"
    priority = "LOW"

else:

    recommendation = "DO NOT RETRY"
    priority = "VERY LOW"


# ==============================
# FINAL RESULT
# ==============================

print("\n========================================")
print("        RECOVERAI DECISION")
print("========================================")

print(
    f"Transaction Amount: Rs. {amount:.2f}"
)

print(
    f"Recovery Probability: "
    f"{recovery_probability:.2%}"
)

print(
    f"Expected Recovery: "
    f"Rs. {expected_recovery:.2f}"
)

print(
    f"Retry Priority: {priority}"
)

print(
    f"Recommendation: {recommendation}"
)

print("========================================")