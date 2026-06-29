import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

def get_transformed_feature_names(pipeline):
    """
    Get feature names after preprocessing.
    Works with ColumnTransformer containing numeric + OneHotEncoder categorical features.
    """
    preprocessor = pipeline.named_steps["preprocessor"]

    feature_names = []

    for name, transformer, columns in preprocessor.transformers_:
        if name == "num":
            feature_names.extend(columns)

        elif name == "cat":
            encoder = transformer
            encoded_names = encoder.get_feature_names_out(columns)
            feature_names.extend(encoded_names)

    return feature_names


def explain_single_transaction(pipeline, input_df: pd.DataFrame, top_n: int = 5):
    """
    Generate SHAP explanation for one transaction.

    Parameters
    ----------
    pipeline:
        Trained sklearn pipeline containing preprocessor + XGBoost model.

    input_df:
        Single-row dataframe with same feature columns used during training.

    top_n:
        Number of top contributing features to return.

    Returns
    -------
    explanation_df:
        DataFrame containing top SHAP contributors.

    expected_value:
        Base SHAP value.

    risk_score:
        Model probability for laundering class.
    """

    if len(input_df) != 1:
        raise ValueError("input_df must contain exactly one transaction row.")

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    X_transformed = preprocessor.transform(input_df)

    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    feature_names = get_transformed_feature_names(pipeline)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_values_single = shap_values[0]

    risk_score = pipeline.predict_proba(input_df)[0, 1]

    explanation_df = pd.DataFrame(
        {
            "feature": feature_names,
            "shap_value": shap_values_single,
            "absolute_impact": np.abs(shap_values_single),
        }
    )

    explanation_df = explanation_df.sort_values(
        by="absolute_impact",
        ascending=False,
    ).head(top_n)

    explanation_df["direction"] = explanation_df["shap_value"].apply(
        lambda x: "increases risk" if x > 0 else "decreases risk"
    )

    return explanation_df, explainer.expected_value, risk_score


def make_display_feature_name(feature_name: str) -> str:
    readable_map = {
        "Amount_Paid": "Amount paid",
        "Amount_Received": "Amount received",
        "absolute_amount_difference": "Amount difference",
        "amount_difference": "Amount difference",
        "amount_ratio_paid_to_received": "Paid/received amount ratio",
        "log_amount_paid": "Large paid amount pattern",
        "log_amount_received": "Large received amount pattern",
        "same_currency": "Same currency transaction",
        "cross_currency": "Cross-currency transaction",
        "same_bank_transfer": "Same-bank transfer",
        "cross_bank_transfer": "Cross-bank transfer",
        "transaction_hour": "Transaction hour",
        "transaction_day": "Transaction day",
        "transaction_month": "Transaction month",
        "transaction_dayofweek": "Day of week",
        "is_weekend": "Weekend transaction",
        "is_night_transaction": "Night-time transaction",
        "From_Bank": "Originating bank",
        "To_Bank": "Receiving bank",
    }

    for key, readable in readable_map.items():
        if feature_name == key or feature_name.startswith(key + "_"):
            return readable

    if feature_name.startswith("Payment_Format_"):
        return feature_name.replace("Payment_Format_", "Payment format: ")

    if feature_name.startswith("Payment_Currency_"):
        return feature_name.replace("Payment_Currency_", "Payment currency: ")

    if feature_name.startswith("Receiving_Currency_"):
        return feature_name.replace("Receiving_Currency_", "Receiving currency: ")

    return feature_name.replace("_", " ")


def add_human_readable_reasons(explanation_df: pd.DataFrame) -> pd.DataFrame:
    explanation_df = explanation_df.copy()

    explanation_df["display_feature"] = explanation_df["feature"].apply(
        make_display_feature_name
    )

    explanation_df["reason"] = explanation_df.apply(
        lambda row: f"{row['display_feature']} {row['direction']}.",
        axis=1,
    )

    return explanation_df


def plot_shap_bar(explanation_df):
    """
    Create a simple SHAP bar chart for top features.
    """

    df_plot = explanation_df.copy()
    df_plot = df_plot.sort_values("shap_value")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.barh(
        df_plot["display_feature"],
        df_plot["shap_value"],
    )

    ax.axvline(0, linestyle="--", linewidth=1)

    ax.set_xlabel("SHAP value impact")
    ax.set_ylabel("Feature")
    ax.set_title("Top SHAP Risk Drivers")

    plt.tight_layout()

    return fig