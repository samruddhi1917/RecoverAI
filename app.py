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
# LOAD TRANSACTION DATA
# ============================================================

@st.cache_data
def load_transactions():
    return pd.read_csv("recoverai_transactions.csv")


transactions = load_transactions()

# ============================================================
# HERO SECTION
# ============================================================

st.markdown(
    '<div class="main-title">💳 RecoverAI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Powered Payment Recovery Intelligence'
    '</div>',
    unsafe_allow_html=True
)

st.info(
    "RecoverAI uses machine learning to identify failed payments "
    "with recovery potential, estimate recoverable revenue, and "
    "recommend the smartest retry action."
)

st.divider()


# ============================================================
# TRANSACTION ANALYTICS
# ============================================================

st.header("📈 Transaction Analytics")

total_transactions = len(transactions)

failed_transactions = (
    transactions["transaction_status"] == "FAILED"
).sum()

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
# TRANSACTION EXPLORER
# ============================================================

st.divider()

st.header("📋 Transaction Explorer")

st.write(
    "Explore historical payment transactions and identify "
    "revenue recovery opportunities."
)

# ------------------------------------------------------------
# FILTERS
# ------------------------------------------------------------

filter_col1, filter_col2, filter_col3 = st.columns(3)

with filter_col1:

    status_filter = st.selectbox(
        "Transaction Status",
        ["ALL", "SUCCESS", "FAILED"]
    )

with filter_col2:

    category_filter = st.selectbox(
        "Failure Category",
        ["ALL"] +
        sorted(
            transactions["failure_category"]
            .dropna()
            .unique()
            .tolist()
        )
    )

with filter_col3:

    payment_filter = st.selectbox(
        "Payment Method",
        ["ALL"] +
        sorted(
            transactions["payment_method"]
            .dropna()
            .unique()
            .tolist()
        )
    )


# ------------------------------------------------------------
# APPLY FILTERS
# ------------------------------------------------------------

filtered_transactions = transactions.copy()

if status_filter != "ALL":

    filtered_transactions = filtered_transactions[
        filtered_transactions["transaction_status"]
        == status_filter
    ]

if category_filter != "ALL":

    filtered_transactions = filtered_transactions[
        filtered_transactions["failure_category"]
        == category_filter
    ]

if payment_filter != "ALL":

    filtered_transactions = filtered_transactions[
        filtered_transactions["payment_method"]
        == payment_filter
    ]


# ------------------------------------------------------------
# BUSINESS METRICS
# ------------------------------------------------------------

total_transactions = len(filtered_transactions)

total_value = filtered_transactions["amount"].sum()

total_expected_recovery = (
    filtered_transactions["expected_recovery"].sum()
)

revenue_at_risk = (
    total_value - total_expected_recovery
)


metric1, metric2, metric3, metric4 = st.columns(4)

with metric1:

    st.metric(
        "Transactions",
        f"{total_transactions:,}"
    )

with metric2:

    st.metric(
        "Transaction Value",
        f"₹{total_value:,.0f}"
    )

with metric3:

    st.metric(
        "Expected Recovery",
        f"₹{total_expected_recovery:,.0f}"
    )

with metric4:

    st.metric(
        "Revenue Still at Risk",
        f"₹{revenue_at_risk:,.0f}"
    )


# ------------------------------------------------------------
# TRANSACTION TABLE
# ------------------------------------------------------------

st.subheader("Filtered Transactions")

display_columns = [
    "transaction_id",
    "customer_id",
    "amount",
    "payment_method",
    "transaction_status",
    "failure_category",
    "failure_reason",
    "retry_count",
    "recovery_probability",
    "recovered",
    "expected_recovery"
]

st.dataframe(
    filtered_transactions[display_columns],
    width="stretch",
    hide_index=True
)

# ============================================================
# RECOVERY OPPORTUNITY DASHBOARD
# ============================================================

st.divider()

st.header("💰 Recovery Opportunity")

st.write(
    "RecoverAI identifies failed transactions with potential "
    "revenue recovery opportunities."
)

# ------------------------------------------------------------
# FAILED TRANSACTIONS
# ------------------------------------------------------------

failed_data = transactions[
    transactions["transaction_status"] == "FAILED"
].copy()


# ------------------------------------------------------------
# RECOVERY METRICS
# ------------------------------------------------------------

failed_value = failed_data["amount"].sum()

potential_recovery = failed_data["expected_recovery"].sum()

revenue_at_risk = failed_value - potential_recovery

if failed_value > 0:

    recovery_rate = potential_recovery / failed_value

else:

    recovery_rate = 0


# ------------------------------------------------------------
# DISPLAY METRICS
# ------------------------------------------------------------

op1, op2, op3, op4 = st.columns(4)

with op1:

    st.metric(
        "Failed Payment Value",
        f"₹{failed_value:,.0f}"
    )

with op2:

    st.metric(
        "Potential Recovery",
        f"₹{potential_recovery:,.0f}"
    )

with op3:

    st.metric(
        "Revenue Still at Risk",
        f"₹{revenue_at_risk:,.0f}"
    )

with op4:

    st.metric(
        "Recovery Rate",
        f"{recovery_rate:.1%}"
    )


# ------------------------------------------------------------
# RECOVERY PROGRESS
# ------------------------------------------------------------

st.subheader("Recovery Potential")

st.progress(
    float(recovery_rate)
)

st.caption(
    f"RecoverAI estimates that approximately "
    f"{recovery_rate:.1%} of failed-payment value "
    f"could potentially be recovered."
)

# ============================================================
# RECOVERY OPPORTUNITY BREAKDOWN
# ============================================================

st.subheader("📊 Recovery Opportunity Breakdown")

chart_data = pd.DataFrame({
    "Category": [
        "Failed Payment Value",
        "Potential Recovery",
        "Revenue Still at Risk"
    ],
    "Amount": [
        failed_value,
        potential_recovery,
        revenue_at_risk    ]
})

st.bar_chart(
    chart_data.set_index("Category")
)

st.caption(
    "This breakdown shows how RecoverAI estimates the recoverable "
    "portion of failed-payment value."
)

# ============================================================
# RECOVERY PRIORITY DISTRIBUTION
# ============================================================

st.subheader("🎯 Recovery Priority Distribution")

failed_priority_data = failed_data.copy()

def get_priority(probability):

    if probability >= 0.75:
        return "HIGH"

    elif probability >= 0.50:
        return "MEDIUM"

    elif probability >= 0.25:
        return "LOW"

    else:
        return "VERY LOW"


failed_priority_data["priority"] = (
    failed_priority_data["recovery_probability"]
    .apply(get_priority)
)

priority_counts = (
    failed_priority_data["priority"]
    .value_counts()
    .reindex(
        ["HIGH", "MEDIUM", "LOW", "VERY LOW"],
        fill_value=0
    )
)

st.bar_chart(priority_counts)

st.caption(
    "Transactions are prioritized based on their predicted "
    "recovery probability."
)

# ------------------------------------------------------------
# RECOVERY BREAKDOWN
# ------------------------------------------------------------

st.subheader("📊 Recovery Breakdown")

recovery_chart = pd.DataFrame({
    "Category": [
        "Potential Recovery",
        "Revenue Still at Risk"
    ],
    "Amount": [
        potential_recovery,
        revenue_at_risk
    ]
})

st.bar_chart(
    recovery_chart.set_index("Category")
)

st.caption(
    "This breakdown shows how RecoverAI estimates the failed-payment "
    "value is divided between potential recovery and remaining risk."
)

# ------------------------------------------------------------
# TOP RECOVERY OPPORTUNITIES
# ------------------------------------------------------------

st.subheader("🎯 Top Recovery Opportunities")

if not failed_data.empty:

    opportunity_data = failed_data.copy()

    opportunity_data["recovery_priority"] = (
        opportunity_data["recovery_probability"]
        .apply(
            lambda x:
                "HIGH" if x >= 0.75
                else "MEDIUM" if x >= 0.50
                else "LOW"
        )
    )

    opportunity_data = opportunity_data.sort_values(
        by="expected_recovery",
        ascending=False
    )

    opportunity_columns = [
        "transaction_id",
        "customer_id",
        "amount",
        "failure_category",
        "failure_reason",
        "recovery_probability",
        "expected_recovery",
        "recovery_priority"
    ]

    st.dataframe(
        opportunity_data[opportunity_columns].head(10),
        width="stretch",
        hide_index=True
    )

else:

    st.info("No failed transactions available for recovery analysis.")

# ============================================================
# STORE PREDICTION RESULTS
# ============================================================

if "prediction_generated" not in st.session_state:
    st.session_state.prediction_generated = False

if "recovery_probability" not in st.session_state:
    st.session_state.recovery_probability = 0.0

if "expected_recovery" not in st.session_state:
    st.session_state.expected_recovery = 0.0

# ============================================================
# PREDICTION BUTTON
# ============================================================

if st.button(
    "🔮 Predict Recovery",
    width="stretch",
    type="primary"
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

    # Store results so other dashboard sections can use them
    st.session_state.recovery_probability = recovery_probability
    st.session_state.expected_recovery = expected_recovery
    st.session_state.prediction_generated = True

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
    # SAVE PREDICTION FOR AUDIT TRAIL
    # ========================================================

    st.session_state.last_prediction = {
        "recovery_probability": recovery_probability,
        "expected_recovery": expected_recovery,
        "priority": priority,
        "recommendation": recommendation,
        "smart_action": ""
    }


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

        # ============================================================
    # TOP RECOVERY OPPORTUNITIES
    # ============================================================

    st.subheader("🔥 Top Recovery Opportunities")

    st.write(
        "Failed transactions ranked by estimated recoverable revenue."
    )

    # Work only with failed payments
    opportunity_data = transactions[
        transactions["transaction_status"] == "FAILED"
    ].copy()

    # Rank by expected recovery
    opportunity_data = opportunity_data.sort_values(
        by="expected_recovery",
        ascending=False
    )

    # Show top 10 opportunities
    top_opportunities = opportunity_data.head(10).copy()

    # Select useful columns
    opportunity_columns = [
        "transaction_id",
        "customer_id",
        "amount",
        "failure_category",
        "failure_reason",
        "retry_count",
        "recovery_probability",
        "expected_recovery"
    ]

    st.dataframe(
        top_opportunities[opportunity_columns],
        width="stretch",
        hide_index=True
    )

    # Total value represented by top opportunities
    top_opportunity_value = top_opportunities[
        "expected_recovery"
    ].sum()

    st.caption(
        f"💡 Top 10 opportunities represent approximately "
        f"₹{top_opportunity_value:,.0f} in potential recovery."
    )

    # ------------------------------------------------------------
    # RECOVERY OPPORTUNITY
    # ------------------------------------------------------------

    st.subheader("💡 Recovery Opportunity")

    opportunity_amount = expected_recovery

    if recovery_probability >= 0.75:

        opportunity_message = (
            f"High recovery opportunity — approximately "
            f"₹{opportunity_amount:,.2f} may be recoverable."
        )

    elif recovery_probability >= 0.50:

        opportunity_message = (
            f"Moderate recovery opportunity — approximately "
            f"₹{opportunity_amount:,.2f} may be recoverable."
        )

    elif recovery_probability >= 0.25:

        opportunity_message = (
            f"Limited recovery opportunity — approximately "
            f"₹{opportunity_amount:,.2f} may be recoverable."
        )

    else:

        opportunity_message = (
            f"Low recovery opportunity — only approximately "
            f"₹{opportunity_amount:,.2f} may be recoverable."
        )

    st.info(opportunity_message)

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

    st.session_state.last_prediction["smart_action"] = smart_action

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

# ============================================================
# AUDIT TRAIL
# ============================================================

st.divider()

st.header("🧾 Audit Trail")

st.write(
    "Generate a transparent record of RecoverAI's recovery decision."
)

# Initialize audit record
if "audit_record" not in st.session_state:
    st.session_state.audit_record = None


# ------------------------------------------------------------
# GENERATE AUDIT TRAIL
# ------------------------------------------------------------

if st.button(
    "📋 Generate Audit Trail Record",
    width="stretch",
    type="secondary"
):

    # Check whether prediction exists
    if "last_prediction" not in st.session_state:

        st.warning(
            "⚠️ Please click '🔮 Predict Recovery' first."
        )

    else:

        prediction = st.session_state.last_prediction

        audit_record = {
            "Transaction Amount": f"₹{amount:,.2f}",
            "Payment Method": payment_method,
            "Merchant Category": merchant_category,
            "Failure Category": failure_category,
            "Failure Reason": failure_reason,
            "Retry Count": retry_count,
            "Customer Success Rate": f"{customer_success_rate:.1%}",
            "Customer Lifetime Value": f"₹{customer_lifetime_value:,.2f}",

            "Recovery Probability":
                f"{prediction['recovery_probability']:.2%}",

            "Expected Recovery":
                f"₹{prediction['expected_recovery']:,.2f}",

            "Priority":
                prediction["priority"],

            "Recommendation":
                prediction["recommendation"],

            "Smart Action":
                prediction["smart_action"]
        }

        st.session_state.audit_record = audit_record

        st.success(
            "✅ Audit trail record generated successfully."
        )


# ------------------------------------------------------------
# DISPLAY AUDIT RECORD
# ------------------------------------------------------------

if st.session_state.audit_record is not None:

    st.subheader("📄 Audit Record")

    audit_df = pd.DataFrame(
        list(st.session_state.audit_record.items()),
        columns=["Field", "Value"]
    )

    st.dataframe(
        audit_df,
        width="stretch",
        hide_index=True
    )

    st.caption(
        "This record captures the inputs, ML prediction, "
        "recovery estimate and recommended recovery action."
    )

    # ========================================================
    # DOWNLOAD AUDIT TRAIL
    # ========================================================

    audit_csv = audit_df.to_csv(index=False)

    st.download_button(
        label="📥 Download Audit Trail",
        data=audit_csv,
        file_name="recoverai_audit_trail.csv",
        mime="text/csv",
        width="stretch"
    )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "💳 RecoverAI • AI-powered revenue recovery intelligence"
)

st.caption(
    "Predict → Explain → Prioritize → Recover"
)