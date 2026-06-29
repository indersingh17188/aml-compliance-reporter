import pandas as pd
import numpy as np


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [col.strip().replace(" ", "_") for col in df.columns]
    return df


def reduce_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")

    return df


def find_target_column(df: pd.DataFrame) -> str:
    possible_targets = [
        "Is_Laundering",
        "Is Laundering",
        "Laundering",
        "laundering",
        "is_laundering",
        "target",
        "label",
    ]

    for col in possible_targets:
        if col in df.columns:
            return col

    raise ValueError(f"Target column not found. Columns are: {list(df.columns)}")


def engineer_aml_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    AML-specific feature engineering for IBM AML dataset.

    Expected columns:
    Timestamp, From_Bank, Account, To_Bank, Account.1,
    Amount_Received, Receiving_Currency,
    Amount_Paid, Payment_Currency, Payment_Format, Is_Laundering
    """

    df = df.copy()

    # -------------------------
    # Timestamp features
    # -------------------------
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

        df["transaction_hour"] = df["Timestamp"].dt.hour
        df["transaction_day"] = df["Timestamp"].dt.day
        df["transaction_month"] = df["Timestamp"].dt.month
        df["transaction_dayofweek"] = df["Timestamp"].dt.dayofweek

        df["is_weekend"] = df["transaction_dayofweek"].isin([5, 6]).astype(int)
        df["is_night_transaction"] = df["transaction_hour"].between(0, 5).astype(int)

    # -------------------------
    # Amount-based features
    # -------------------------
    if "Amount_Paid" in df.columns and "Amount_Received" in df.columns:
        df["amount_difference"] = df["Amount_Paid"] - df["Amount_Received"]
        df["absolute_amount_difference"] = df["amount_difference"].abs()

        df["amount_ratio_paid_to_received"] = (
            df["Amount_Paid"] / (df["Amount_Received"] + 1e-6)
        )

        df["log_amount_paid"] = np.log1p(df["Amount_Paid"])
        df["log_amount_received"] = np.log1p(df["Amount_Received"])

    # -------------------------
    # Currency features
    # -------------------------
    if "Payment_Currency" in df.columns and "Receiving_Currency" in df.columns:
        df["same_currency"] = (
            df["Payment_Currency"] == df["Receiving_Currency"]
        ).astype(int)

        df["cross_currency"] = 1 - df["same_currency"]

    # -------------------------
    # Bank movement features
    # -------------------------
    if "From_Bank" in df.columns and "To_Bank" in df.columns:
        df["same_bank_transfer"] = (df["From_Bank"] == df["To_Bank"]).astype(int)
        df["cross_bank_transfer"] = 1 - df["same_bank_transfer"]

    return df


def select_model_features(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Select clean and explainable features for AML model.

    We intentionally remove account identifiers and raw timestamps to avoid leakage/memorization.
    """

    df = df.copy()

    selected_features = [
        # Original useful fields
        "From_Bank",
        "To_Bank",
        "Amount_Received",
        "Receiving_Currency",
        "Amount_Paid",
        "Payment_Currency",
        "Payment_Format",

        # Engineered time features
        "transaction_hour",
        "transaction_day",
        "transaction_month",
        "transaction_dayofweek",
        "is_weekend",
        "is_night_transaction",

        # Engineered amount features
        "amount_difference",
        "absolute_amount_difference",
        "amount_ratio_paid_to_received",
        "log_amount_paid",
        "log_amount_received",

        # Engineered currency/bank features
        "same_currency",
        "cross_currency",
        "same_bank_transfer",
        "cross_bank_transfer",
    ]

    available_features = [col for col in selected_features if col in df.columns]

    print("\nSelected model features:")
    for col in available_features:
        print(f" - {col}")

    X = df[available_features]
    y = df[target_col].astype(int)

    return X, y