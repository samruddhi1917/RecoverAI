import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RecoverAI",
    page_icon="💳",
    layout="wide"
)


# ============================================================
# CUSTOM UI STYLING
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 0px;
}

.subtitle {
    font-size: 18px;
    opacity: 0.75;
    margin-top: -8px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("recoverai_model.pkl")


model = load_model()

# ============================================================
# LOAD TRANSACTION DATASET
# ============================================================

@st.cache_data
def load_transactions():
    return pd.read_csv("recoverai_transactions.csv")


transactions = load_transactions()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">💳 RecoverAI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-Powered Payment Recovery Intelligence</div>',
    unsafe_allow_html=True
)

st.write(
    "RecoverAI predicts the probability of recovering a failed payment "
    "and estimates the expected revenue that can be recovered."
)

st.divider()


# ============================================================
# TRANSACTION ANALYTICS
# ============================================================

st.header("📈 Transaction Analytics")

total_transactions = len(transactions)

failed_transactions = (
    transactions["failure_category"]
    .notna()
    .sum()
)

total_transaction_value = transactions["amount"].sum()

st.caption(
    f"Analyzing {total_transactions:,} transactions from RecoverAI's dataset."
)

metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric(
        "Total Transactions",
        f"{total_transactions:,}"
    )

with metric2:
    st.metric(
        "Failed Transactions",
        f"{failed_transactions:,}"
    )

with metric3:
    st.metric(
        "Transaction Value",
        f"₹{total_transaction_value:,.0f}"
    )

st.divider()

# ============================================================
# TRANSACTION DATA
# ============================================================

st.subheader("📋 Transaction Dataset")

st.dataframe(
    transactions.head(10),
    width="stretch",
    hide_index=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Transaction Details")

amount = st.sidebar.number_input(
    "Transaction Amount (₹)",
    min_value=500.01,
    max_value=99999.99,
    value=5000.00,
    step=100.00
)

payment_method = st.sidebar.selectbox(
    "Payment Method",
    ["UPI", "CARD", "NETBANKING", "WALLET"]
)

merchant_category = st.sidebar.selectbox(
    "Merchant Category",
    [
        "ELECTRONICS",
        "CLOTHING",
        "GROCERIES",
        "FOOD",
        "TRAVEL",
        "HOME_APPLIANCES"
    ]
)

failure_category = st.sidebar.selectbox(
    "Failure Category",
    [
        "TEMPORARY",
        "CUSTOMER_ACTION",
        "PAYMENT_METHOD",
        "SYSTEM"
    ]
)

failure_reason = st.sidebar.selectbox(
    "Failure Reason",
    [
        "NETWORK_TIMEOUT",
        "GATEWAY_TIMEOUT",
        "INSUFFICIENT_FUNDS",
        "AUTHENTICATION_REQUIRED",
        "EXPIRED_CARD",
        "INVALID_CARD",
        "PROCESSING_ERROR",
        "TEMPORARY_SERVICE_ERROR"
    ]
)

retry_count = st.sidebar.number_input(
    "Retry Count",
    min_value=0,
    max_value=10,
    value=1,
    step=1
)

if retry_count > 0:

    last_retry_hours_ago = st.sidebar.number_input(
        "Hours Since Last Retry",
        min_value=0.1,
        max_value=72.0,
        value=2.0,
        step=0.1
    )

else:

    last_retry_hours_ago = None


device_type = st.sidebar.selectbox(
    "Device Type",
    ["PHONE", "DESKTOP", "TABLET"]
)

region = st.sidebar.selectbox(
    "Region",
    [
        "MAHARASHTRA",
        "KARNATAKA",
        "DELHI",
        "TELANGANA",
        "TAMIL_NADU",
        "GUJARAT",
        "WEST_BENGAL",
        "UTTAR_PRADESH"
    ]
)

transaction_hour = st.sidebar.slider(
    "Transaction Hour",
    min_value=0,
    max_value=23,
    value=12
)

subscription_active = st.sidebar.selectbox(
    "Subscription Active?",
    ["Yes", "No"]
)

previous_successful_payments = st.sidebar.number_input(
    "Previous Successful Payments",
    min_value=0,
    max_value=100,
    value=8
)

previous_failed_payments = st.sidebar.number_input(
    "Previous Failed Payments",
    min_value=0,
    max_value=100,
    value=1
)

customer_lifetime_value = st.sidebar.number_input(
    "Customer Lifetime Value (₹)",
    min_value=500.01,
    max_value=100000.00,
    value=25000.00,
    step=500.00
)


# ============================================================
# ENGINEERED FEATURES
# ============================================================

total_previous_payments = (
    previous_successful_payments
    + previous_failed_payments
)

if total_previous_payments > 0:

    customer_success_rate = (
        previous_successful_payments
        / total_previous_payments
    )

else:

    customer_success_rate = 0


has_been_retried = int(retry_count > 0)


if customer_lifetime_value > 0:

    amount_to_lifetime_value_ratio = (
        amount / customer_lifetime_value
    )

else:

    amount_to_lifetime_value_ratio = 0


has_retry_timestamp = int(retry_count > 0)


# ============================================================
# MAIN DASHBOARD
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Transaction Amount",
        f"₹{amount:,.2f}"
    )

with col2:

    st.metric(
        "Customer Success Rate",
        f"{customer_success_rate:.1%}"
    )

with col3:

    st.metric(
        "Previous Failed Payments",
        previous_failed_payments
    )


st.divider()


# ============================================================
# PREDICTION BUTTON
# ============================================================

if st.button(
    "🔮 Predict Recovery",
    width="stretch"
):

    # --------------------------------------------------------
    # SUBSCRIPTION CONVERSION
    # --------------------------------------------------------

    subscription_value = (
        1 if subscription_active == "Yes" else 0
    )


    # --------------------------------------------------------
    # CREATE INPUT DATAFRAME
    # --------------------------------------------------------

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

        "subscription_active": subscription_value,

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


    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    recovery_probability = model.predict_proba(
        transaction
    )[0][1]


    expected_recovery = (
        amount * recovery_probability
    )

    # ==============================
    # EXPLAINABLE AI
    # ==============================

    reasons = []

    # Customer history
    if customer_success_rate >= 0.75:
        reasons.append(
            "Strong customer payment history"
        )
    elif customer_success_rate <= 0.40:
        reasons.append(
            "Low customer payment success rate"
        )

    # Failure type
    if failure_category == "TEMPORARY":
        reasons.append(
            "Temporary failure is often recoverable"
        )
    elif failure_category == "SYSTEM":
        reasons.append(
            "System-related failure may resolve on retry"
        )
    elif failure_category == "PAYMENT_METHOD":
        reasons.append(
            "Payment-method issue increases retry risk"
        )

    # Failure reason
    if failure_reason in [
        "NETWORK_TIMEOUT",
        "GATEWAY_TIMEOUT",
        "TEMPORARY_SERVICE_ERROR"
    ]:
        reasons.append(
            "Temporary connectivity/service issue detected"
        )

    elif failure_reason == "INSUFFICIENT_FUNDS":
        reasons.append(
            "Insufficient funds may reduce recovery probability"
        )

    # Retry history
    if retry_count == 0:
        reasons.append(
            "No previous retry attempt"
        )
    elif retry_count == 1:
        reasons.append(
            "Only one retry attempt has been made"
        )
    else:
        reasons.append(
            "Multiple retry attempts detected"
        )

    # Subscription
    if subscription_active == "Yes":
        reasons.append(
            "Active subscription indicates an ongoing customer relationship"
        )

    # Customer value
    if customer_lifetime_value >= 50000:
        reasons.append(
            "High customer lifetime value"
        )

    # Amount relative to customer value
    if amount_to_lifetime_value_ratio <= 0.10:
        reasons.append(
            "Transaction amount is relatively small compared with customer lifetime value"
        )

    # Fallback
    if len(reasons) == 0:
        reasons.append(
            "Prediction is based on transaction, customer and failure characteristics"
        )

    # ========================================================
    # DECISION ENGINE
    # ========================================================

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


    # ========================================================
    # RESULTS
    # ========================================================

    st.divider()

    st.header("📊 RecoverAI Prediction")


    result1, result2, result3 = st.columns(3)

    with result1:

        st.metric(
            "Recovery Probability",
            f"{recovery_probability:.2%}"
        )

    with result2:

        st.metric(
            "Expected Recovery",
            f"₹{expected_recovery:,.2f}"
        )

    with result3:

        st.metric(
            "Retry Priority",
            priority
        )

    st.markdown("---")

    st.subheader("🤖 Why RecoverAI Made This Decision")

    st.write(
        "RecoverAI analyzed the transaction, failure characteristics, "
        "retry history and customer payment behavior."
    )

    for reason in reasons[:5]:
        st.markdown(f"✅ {reason}")



    # --------------------------------------------------------
    # PROBABILITY BAR
    # --------------------------------------------------------

    st.subheader("Recovery Potential")

    st.progress(
        float(recovery_probability)
    )

    # --------------------------------------------------------
    # EXPECTED RECOVERY
    # --------------------------------------------------------

    st.subheader("Expected Revenue Recovery")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Transaction Value",
            f"₹{amount:,.2f}"
        )

    with col2:
        st.metric(
            "Recovery Probability",
            f"{recovery_probability:.1%}"
        )

    with col3:
        st.metric(
            "Expected Recovery",
            f"₹{expected_recovery:,.2f}"
        )

    # ========================================================
    # BUSINESS IMPACT
    # ========================================================

    st.subheader("💰 Business Impact")

    revenue_at_risk = amount - expected_recovery

    st.info(
        f"**₹{amount:,.2f} was at risk from this failed payment.** "
        f"RecoverAI estimates that **₹{expected_recovery:,.2f} "
        f"could potentially be recovered**, leaving approximately "
        f"**₹{revenue_at_risk:,.2f} still at risk.**"
    )

    st.caption(
        f"Estimated recoverable share: "
        f"{recovery_probability:.1%} of the transaction value"
    )

    # --------------------------------------------------------
    # RECOMMENDATION
    # --------------------------------------------------------

    if priority == "HIGH":

        st.success(
            f"🚀 {recommendation}"
        )

    elif priority == "MEDIUM":

        st.warning(
            f"⚠️ {recommendation}"
        )

    elif priority == "LOW":

        st.info(
            f"ℹ️ {recommendation}"
        )

    else:

        st.error(
            f"🛑 {recommendation}"
        )

    # ========================================================
    # SMART RETRY STRATEGY
    # ========================================================

    st.subheader("🎯 Smart Retry Strategy")

    if recovery_probability >= 0.75:

        if failure_category == "TEMPORARY" or failure_reason in [
            "NETWORK_TIMEOUT",
            "GATEWAY_TIMEOUT",
            "TEMPORARY_SERVICE_ERROR"
        ]:

            smart_action = "Retry payment now"
            smart_reason = (
                "High recovery potential combined with a "
                "potentially temporary failure."
            )

        elif failure_category == "PAYMENT_METHOD":

            smart_action = "Retry with an alternative payment method"
            smart_reason = (
                "Recovery potential is high, but the current "
                "payment method may be contributing to the failure."
            )

        else:

            smart_action = "Retry payment"
            smart_reason = (
                "The model predicts a high probability of recovery."
            )

    elif recovery_probability >= 0.50:

        if retry_count >= 3:

            smart_action = "Avoid another immediate retry"
            smart_reason = (
                "Moderate recovery potential, but multiple "
                "previous retry attempts have already occurred."
            )

        elif failure_reason in [
            "NETWORK_TIMEOUT",
            "GATEWAY_TIMEOUT",
            "TEMPORARY_SERVICE_ERROR"
        ]:

            smart_action = "Retry after a short delay"
            smart_reason = (
                "The failure appears temporary and the model "
                "still identifies meaningful recovery potential."
            )

        else:

            smart_action = "Retry with caution"
            smart_reason = (
                "The model predicts moderate recovery potential."
            )

    elif recovery_probability >= 0.25:

        if failure_reason == "INSUFFICIENT_FUNDS":

            smart_action = "Avoid immediate retry"
            smart_reason = (
                "Insufficient funds may cause another failed "
                "payment attempt."
            )

        elif retry_count >= 3:

            smart_action = "Stop repeated retries"
            smart_reason = (
                "Several retry attempts have already been made "
                "with limited recovery potential."
            )

        elif failure_category == "PAYMENT_METHOD":

            smart_action = "Suggest an alternative payment method"
            smart_reason = (
                "The current payment method may continue to fail."
            )

        else:

            smart_action = "Low-priority retry"
            smart_reason = (
                "Recovery potential exists, but the probability "
                "is relatively low."
            )

    else:

        smart_action = "Do not retry"
        smart_reason = (
            "The model predicts very low recovery potential, "
            "so another immediate attempt may not be worthwhile."
        )

    st.info(
        f"**Recommended Action:** {smart_action}"
    )

    st.write(
        f"**Why:** {smart_reason}"
    )

    # ========================================================
    # EXPLANATION
    # ========================================================

    st.subheader("🧠 Recovery Analysis")

    analysis = []

    if failure_category == "TEMPORARY":
        analysis.append(
            "Temporary failure detected — retrying may be viable."
        )

    if failure_reason in [
        "NETWORK_TIMEOUT",
        "GATEWAY_TIMEOUT",
        "TEMPORARY_SERVICE_ERROR"
    ]:
        analysis.append(
            "The failure reason indicates a potentially temporary issue."
        )

    if customer_success_rate >= 0.70:
        analysis.append(
            "Customer has a strong historical payment success rate."
        )

    elif customer_success_rate < 0.40:
        analysis.append(
            "Customer has a relatively low historical payment success rate."
        )

    if previous_failed_payments >= 5:
        analysis.append(
            "High previous failure count reduces recovery confidence."
        )

    if retry_count >= 3:
        analysis.append(
            "Multiple retries have already occurred."
        )

    if subscription_value == 1:
        analysis.append(
            "Active subscription is a positive customer signal."
        )

    if not analysis:
        analysis.append(
            "Recovery decision is primarily based on the trained ML model."
        )

    for item in analysis:

        st.write("•", item)


    # ========================================================
    # TRANSACTION SUMMARY
    # ========================================================

    with st.expander("Transaction Details"):

        display_transaction = pd.DataFrame({
            "Field": [
                "Transaction Amount",
                "Payment Method",
                "Merchant Category",
                "Failure Category",
                "Failure Reason",
                "Retry Count",
                "Device Type",
                "Region",
                "Transaction Hour",
                "Subscription Active",
                "Previous Successful Payments",
                "Previous Failed Payments",
                "Customer Lifetime Value"
            ],
            "Value": [
                f"Rs. {amount:,.2f}",
                str(payment_method),
                str(merchant_category),
                str(failure_category),
                str(failure_reason),
                str(retry_count),
                str(device_type),
                str(region),
                str(transaction_hour),
                subscription_active,
                str(previous_successful_payments),
                str(previous_failed_payments),
                f"Rs. {customer_lifetime_value:,.2f}"
            ]
        })

        st.dataframe(
            display_transaction,
            width="stretch",
            hide_index=True
        )

    