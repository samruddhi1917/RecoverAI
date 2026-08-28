import streamlit as st
import pandas as pd
import joblib


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="RecoverAI",
    page_icon="💰",
    layout="wide"
)


# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():
    return joblib.load("recoverai_model.pkl")


model = load_model()


# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.main {
    background-color: #f8fafc;
}

.title {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 0px;
}

.subtitle {
    font-size: 18px;
    color: #64748b;
    margin-bottom: 30px;
}

.result-card {
    padding: 25px;
    border-radius: 15px;
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    text-align: center;
}

.result-number {
    font-size: 36px;
    font-weight: 800;
}

.result-label {
    font-size: 15px;
    color: #64748b;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# HEADER
# ==========================================

st.markdown(
    '<div class="title">💰 RecoverAI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered payment failure recovery and revenue optimization'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ==========================================
# INPUT SECTION
# ==========================================

st.subheader("📋 Transaction Details")

col1, col2, col3 = st.columns(3)


with col1:

    amount = st.number_input(
        "Transaction Amount (₹)",
        min_value=0.01,
        max_value=99999.99,
        value=590.00,
        step=10.0
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "UPI",
            "CARD",
            "NETBANKING",
            "WALLET"
        ]
    )

    merchant_category = st.selectbox(
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

    failure_category = st.selectbox(
        "Failure Category",
        [
            "TEMPORARY",
            "CUSTOMER_ACTION",
            "PAYMENT_METHOD",
            "SYSTEM"
        ]
    )


with col2:

    failure_reason = st.selectbox(
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

    retry_count = st.number_input(
        "Retry Count",
        min_value=0,
        max_value=20,
        value=1,
        step=1
    )

    last_retry_hours_ago = st.number_input(
        "Hours Since Last Retry",
        min_value=0.0,
        max_value=720.0,
        value=2.0,
        step=1.0
    )

    device_type = st.selectbox(
        "Device Type",
        [
            "PHONE",
            "DESKTOP",
            "TABLET"
        ]
    )


with col3:

    region = st.text_input(
        "Region / State",
        value="MAHARASHTRA"
    ).upper()

    transaction_hour = st.slider(
        "Transaction Hour",
        min_value=0,
        max_value=23,
        value=10
    )

    subscription_active = st.selectbox(
        "Subscription Active?",
        [
            1,
            0
        ],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    customer_lifetime_value = st.number_input(
        "Customer Lifetime Value (₹)",
        min_value=0.01,
        max_value=100000.00,
        value=25000.00,
        step=100.0
    )


st.subheader("👤 Customer Payment History")

col4, col5 = st.columns(2)

with col4:

    previous_successful_payments = st.number_input(
        "Previous Successful Payments",
        min_value=0,
        max_value=1000,
        value=8,
        step=1
    )

with col5:

    previous_failed_payments = st.number_input(
        "Previous Failed Payments",
        min_value=0,
        max_value=1000,
        value=1,
        step=1
    )


st.write("")


# ==========================================
# PREDICTION BUTTON
# ==========================================

predict_button = st.button(
    "🚀 Predict Recovery",
    use_container_width=True
)


# ==========================================
# PREDICTION
# ==========================================

if predict_button:

    # --------------------------------------
    # ENGINEERED FEATURES
    # --------------------------------------

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


    has_been_retried = int(
        retry_count > 0
    )


    if customer_lifetime_value > 0:

        amount_to_lifetime_value_ratio = (
            amount
            / customer_lifetime_value
        )

    else:

        amount_to_lifetime_value_ratio = 0


    has_retry_timestamp = int(
        retry_count > 0
    )


    if retry_count == 0:

        last_retry_hours_ago = None


    # --------------------------------------
    # CREATE TRANSACTION DATAFRAME
    # --------------------------------------

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


    # --------------------------------------
    # MODEL PREDICTION
    # --------------------------------------

    recovery_probability = model.predict_proba(
        transaction
    )[0][1]


    expected_recovery = (
        amount * recovery_probability
    )


    # ======================================
    # DECISION ENGINE
    # ======================================

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


    # ======================================
    # RESULTS
    # ======================================

    st.divider()

    st.subheader("🤖 RecoverAI Prediction")

    r1, r2, r3 = st.columns(3)


    with r1:

        st.metric(
            "Recovery Probability",
            f"{recovery_probability:.2%}"
        )


    with r2:

        st.metric(
            "Expected Recovery",
            f"₹{expected_recovery:,.2f}"
        )


    with r3:

        st.metric(
            "Retry Priority",
            priority
        )


    st.write("")


    # ======================================
    # RECOMMENDATION
    # ======================================

    if priority == "HIGH":

        st.success(
            f"🚀 **{recommendation}**"
        )

    elif priority == "MEDIUM":

        st.warning(
            f"⚠️ **{recommendation}**"
        )

    elif priority == "LOW":

        st.info(
            f"ℹ️ **{recommendation}**"
        )

    else:

        st.error(
            f"🛑 **{recommendation}**"
        )


    # ======================================
    # REVENUE INSIGHT
    # ======================================

    st.subheader("💡 Revenue Insight")

    col6, col7 = st.columns(2)

    with col6:

        st.write(
            f"**Transaction Value:** "
            f"₹{amount:,.2f}"
        )

        st.write(
            f"**Potential Revenue Recovered:** "
            f"₹{expected_recovery:,.2f}"
        )


    with col7:

        unrecovered_amount = (
            amount - expected_recovery
        )

        st.write(
            f"**Potential Unrecovered Amount:** "
            f"₹{unrecovered_amount:,.2f}"
        )

        st.write(
            f"**Customer Success Rate:** "
            f"{customer_success_rate:.2%}"
        )


    # ======================================
    # MODEL INPUT PREVIEW
    # ======================================

    with st.expander("🔍 View Model Input"):

        st.dataframe(
            transaction,
            use_container_width=True
        )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "RecoverAI • AI-powered revenue recovery prototype"
)