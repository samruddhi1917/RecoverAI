import pandas as pd
import numpy as np

NUM_CUSTOMERS = 1000

customer_ids = [
    f"USER{i:04d}"
    for i in range(1, NUM_CUSTOMERS + 1)
]

print(customer_ids[:10])

previous_successful = np.random.randint(0, 20, NUM_CUSTOMERS)

print(previous_successful[:10])

previous_failed = np.random.randint(0, 8, NUM_CUSTOMERS)

print(previous_failed[:10])

lifetime_value = np.round(
    np.random.uniform(500, 100000, NUM_CUSTOMERS),
    2
)

print(lifetime_value[:10])

customers = pd.DataFrame({
    "customer_id": customer_ids,
    "previous_successful_payments": previous_successful,
    "previous_failed_payments": previous_failed,
    "customer_lifetime_value": lifetime_value
})

print(customers.head())

NUM_TRANSACTIONS = 10000

transaction_ids = [
    f"TXN{i:06d}"
    for i in range(1, NUM_TRANSACTIONS + 1)
]

transaction_customer_ids = np.random.choice(
    customer_ids,
    NUM_TRANSACTIONS
)

print(transaction_ids[:10])
print(transaction_customer_ids[:10])

transaction_amounts = np.round(
    np.random.uniform(500, 100000, NUM_TRANSACTIONS),
    2
)

print(transaction_amounts[:10])

payment_methods = np.random.choice(
    ["UPI", "CARD", "NETBANKING", "WALLET"],
    NUM_TRANSACTIONS
)

print(payment_methods[:10])

merchant_categories = np.random.choice(
    [
        "ELECTRONICS",
        "CLOTHING",
        "HOME_APPLIANCES",
        "GROCERIES",
        "TRAVEL",
        "FOOD"
    ],
    NUM_TRANSACTIONS
)

print(merchant_categories[:10])

transaction_status = np.random.choice(
    ["SUCCESS", "FAILED"],
    NUM_TRANSACTIONS,
    p=[0.75, 0.25]
)

print(transaction_status[:10])

print(
    pd.Series(transaction_status).value_counts()
)

failure_reasons = {
    "TEMPORARY": [
        "NETWORK_TIMEOUT",
        "GATEWAY_TIMEOUT",
        "TEMPORARY_SERVICE_ERROR"
    ],

    "CUSTOMER_ACTION": [
        "INSUFFICIENT_FUNDS",
        "AUTHENTICATION_REQUIRED"
    ],

    "PAYMENT_METHOD": [
        "EXPIRED_CARD",
        "INVALID_CARD"
    ],

    "SYSTEM": [
        "PROCESSING_ERROR"
    ]
}

failure_categories = np.full(NUM_TRANSACTIONS, None, dtype=object)

failed_mask = transaction_status == "FAILED"

failure_categories[failed_mask] = np.random.choice(
    ["TEMPORARY", "CUSTOMER_ACTION", "PAYMENT_METHOD", "SYSTEM"],
    failed_mask.sum(),
    p=[0.40, 0.30, 0.20, 0.10]
)

print(failure_categories[:20])

failure_reason = np.full(NUM_TRANSACTIONS, None, dtype=object)

failed_indices = np.where(failed_mask)[0]

for i in failed_indices:
    category = failure_categories[i]
    failure_reason[i] = np.random.choice(
        failure_reasons[category]
    )

print(failure_reason[:20])

retry_counts = np.zeros(NUM_TRANSACTIONS, dtype=int)

retry_counts[failed_mask] = np.random.choice(
    [0, 1, 2, 3, 4, 5],
    failed_mask.sum(),
    p=[0.40, 0.30, 0.15, 0.08, 0.05, 0.02]
)

print(retry_counts[:20])

last_retry_hours_ago = np.full(
    NUM_TRANSACTIONS,
    np.nan
)

retry_mask = failed_mask & (retry_counts > 0)

last_retry_hours_ago[retry_mask] = np.round(
    np.random.uniform(0.1, 72, retry_mask.sum()),
    2
)

print(last_retry_hours_ago[:20])

device_types = np.random.choice(
    ["PHONE", "DESKTOP", "TABLET"],
    NUM_TRANSACTIONS,
    p=[0.65, 0.25, 0.10]
)

print(device_types[:10])

regions = np.random.choice(
    [
        "MAHARASHTRA",
        "KARNATAKA",
        "DELHI",
        "TELANGANA",
        "TAMIL_NADU",
        "GUJARAT",
        "WEST_BENGAL",
        "UTTAR_PRADESH"
    ],
    NUM_TRANSACTIONS
)

print(regions[:10])

transaction_hour = np.random.randint(
    0,
    24,
    NUM_TRANSACTIONS
)

print(transaction_hour[:10])

subscription_active = np.random.choice(
    [0, 1],
    NUM_TRANSACTIONS,
    p=[0.30, 0.70]
)

print(subscription_active[:10])

# ==========================================
# RECOVERY SCORE
# ==========================================

recovery_score = np.zeros(NUM_TRANSACTIONS)

failed_indices = np.where(failed_mask)[0]

for i in failed_indices:

    # Start with a neutral recovery probability
    score = 0.50

    # --------------------------------------
    # 1. Failure reason
    # --------------------------------------

    if failure_reason[i] == "NETWORK_TIMEOUT":
        score += 0.20

    elif failure_reason[i] == "GATEWAY_TIMEOUT":
        score += 0.15

    elif failure_reason[i] == "TEMPORARY_SERVICE_ERROR":
        score += 0.12

    elif failure_reason[i] == "INSUFFICIENT_FUNDS":
        score -= 0.15

    elif failure_reason[i] == "AUTHENTICATION_REQUIRED":
        score -= 0.10

    elif failure_reason[i] == "EXPIRED_CARD":
        score -= 0.20

    elif failure_reason[i] == "INVALID_CARD":
        score -= 0.20

    elif failure_reason[i] == "PROCESSING_ERROR":
        score -= 0.05

    # --------------------------------------
    # 2. Customer history
    # --------------------------------------

    customer_index = customer_ids.index(
        transaction_customer_ids[i]
    )

    successful_history = previous_successful[customer_index]
    failed_history = previous_failed[customer_index]

    if successful_history >= 10:
        score += 0.10

    elif successful_history <= 2:
        score -= 0.05

    if failed_history >= 5:
        score -= 0.10

    # --------------------------------------
    # 3. Retry count
    # --------------------------------------

    score -= retry_counts[i] * 0.08

    # --------------------------------------
    # 4. Subscription status
    # --------------------------------------

    if subscription_active[i] == 1:
        score += 0.05

    else:
        score -= 0.05

    # --------------------------------------
    # Keep probability between 0 and 1
    # --------------------------------------

    recovery_score[i] = np.clip(score, 0, 1)


print("\nRecovery probabilities:")
print(recovery_score[:20])


# ==========================================
# RECOVERY OUTCOME
# ==========================================

recovered = np.zeros(
    NUM_TRANSACTIONS,
    dtype=int
)

for i in failed_indices:

    recovered[i] = (
        np.random.random() < recovery_score[i]
    )


print("\nRecovery outcomes:")
print(recovered[:20])


# ==========================================
# EXPECTED RECOVERY
# ==========================================

expected_recovery = np.round(
    transaction_amounts * recovery_score,
    2
)

# Successful transactions have no money to recover
expected_recovery[~failed_mask] = 0


print("\nExpected recovery:")
print(expected_recovery[:20])


# ==========================================
# RECOVERY STATISTICS
# ==========================================

print("\nRecovery statistics:")

print(
    "Failed transactions:",
    failed_mask.sum()
)

print(
    "Recovered transactions:",
    recovered.sum()
)

print(
    "Recovery rate among failed:",
    round(
        recovered[failed_mask].mean(),
        4
    )
)

print(
    "Total expected recovery:",
    round(
        expected_recovery.sum(),
        2
    )
)


# ==========================================
# CREATE FINAL TRANSACTION DATAFRAME
# ==========================================

transactions = pd.DataFrame({

    "transaction_id": transaction_ids,

    "customer_id": transaction_customer_ids,

    "amount": transaction_amounts,

    "payment_method": payment_methods,

    "merchant_category": merchant_categories,

    "transaction_status": transaction_status,

    "failure_category": failure_categories,

    "failure_reason": failure_reason,

    "retry_count": retry_counts,

    "last_retry_hours_ago": last_retry_hours_ago,

    "device_type": device_types,

    "region": regions,

    "transaction_hour": transaction_hour,

    "subscription_active": subscription_active,

    "recovery_probability": recovery_score,

    "recovered": recovered,

    "expected_recovery": expected_recovery
})


# ==========================================
# MERGE CUSTOMER INFORMATION
# ==========================================

transactions = transactions.merge(
    customers,
    on="customer_id",
    how="left"
)


# ==========================================
# SAVE DATASET
# ==========================================

transactions.to_csv(
    "recoverai_transactions.csv",
    index=False
)


print("\nDataset saved successfully!")

print(
    "Dataset shape:",
    transactions.shape
)

print("\nFirst 5 rows:")

print(
    transactions.head()
)

# ==========================================
# DAY 2 DATA QUALITY CHECK
# ==========================================

print("\n========== DATA QUALITY CHECK ==========")

# 1. Check number of rows
print("Number of transactions:", len(transactions))

# 2. Check duplicate transaction IDs
print(
    "Duplicate transaction IDs:",
    transactions["transaction_id"].duplicated().sum()
)

# 3. Check missing values
print("\nMissing values:")
print(transactions.isnull().sum())

# 4. Check invalid probabilities
print(
    "\nInvalid recovery probabilities:",
    (
        (transactions["recovery_probability"] < 0) |
        (transactions["recovery_probability"] > 1)
    ).sum()
)

# 5. Expected recovery should never exceed transaction amount
print(
    "Expected recovery > transaction amount:",
    (
        transactions["expected_recovery"] >
        transactions["amount"]
    ).sum()
)

# 6. Successful transactions should not have failure reasons
print(
    "Successful transactions with failure reason:",
    (
        (transactions["transaction_status"] == "SUCCESS") &
        (transactions["failure_reason"].notna())
    ).sum()
)

# 7. Failed transactions should have failure reasons
print(
    "Failed transactions without failure reason:",
    (
        (transactions["transaction_status"] == "FAILED") &
        (transactions["failure_reason"].isna())
    ).sum()
)

# 8. Successful transactions should not be marked recovered
print(
    "Successful transactions marked recovered:",
    (
        (transactions["transaction_status"] == "SUCCESS") &
        (transactions["recovered"] == 1)
    ).sum()
)

# 9. Retry count should not be negative
print(
    "Negative retry counts:",
    (transactions["retry_count"] < 0).sum()
)

print("\n========== CHECK COMPLETE ==========")