import joblib
import pandas as pd
import streamlit as st

from utils.explainability import (
    explain_single_transaction,
    add_human_readable_reasons,
    plot_shap_bar,
)


MODEL_PATH = "models/xgb_aml_pipeline_v2.pkl"
SAMPLE_INPUTS_PATH = "data/sample_app_inputs_v2.csv"


def get_risk_level(score):
    if score >= 0.98:
        return "High Risk", "Escalate for AML analyst review"
    elif score >= 0.70:
        return "Medium Risk", "Review if customer profile or history is unusual"
    else:
        return "Low Risk", "No immediate escalation suggested"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_sample_inputs():
    return pd.read_csv(SAMPLE_INPUTS_PATH)


st.set_page_config(
    page_title="AML Risk Report Generator",
    page_icon="🏦",
    layout="wide",
)

st.title("🏦 Explainable AML Risk Report Generator")

st.markdown(
    """
This demo uses **XGBoost** for AML risk classification and **SHAP** for explainability.
The goal is to support suspicious transaction review with a transparent risk score and analyst-friendly explanations.
"""
)

pipeline = load_model()
sample_df = load_sample_inputs()

st.sidebar.header("Transaction Selection")

sample_index = st.sidebar.selectbox(
    "Choose a sample transaction",
    options=list(range(len(sample_df))),
)

input_df = sample_df.iloc[[sample_index]].copy()

st.subheader("Selected Transaction")
st.dataframe(input_df, use_container_width=True)

if st.button("Predict AML Risk"):
    risk_score = pipeline.predict_proba(input_df)[0, 1]
    risk_level, recommendation = get_risk_level(risk_score)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("AML Risk Score", f"{risk_score * 100:.2f} / 100")

    with col2:
        st.metric("Risk Level", risk_level)

    with col3:
        st.metric("Model", "XGBoost + SHAP")

    if risk_level == "High Risk":
        st.error(f"Recommendation: {recommendation}")
    elif risk_level == "Medium Risk":
        st.warning(f"Recommendation: {recommendation}")
    else:
        st.success(f"Recommendation: {recommendation}")

    st.subheader("Top Risk Drivers")

    explanation_df, expected_value, risk_score = explain_single_transaction(
        pipeline=pipeline,
        input_df=input_df,
        top_n=7,
    )

    explanation_df = add_human_readable_reasons(explanation_df)

    st.dataframe(
        explanation_df[
            ["display_feature", "shap_value", "direction", "reason"]
        ],
        use_container_width=True,
    )

    st.subheader("SHAP Visualization")

    fig = plot_shap_bar(explanation_df)
    st.pyplot(fig)

    st.subheader("Draft Analyst Summary")

    reasons = explanation_df["reason"].tolist()

    report = f"""
AML Analyst Summary

Risk Level: {risk_level}
Risk Score: {risk_score * 100:.2f} / 100

The transaction was assessed using an XGBoost-based AML risk model. 
The most important factors influencing the model decision were:

{chr(10).join(["- " + r for r in reasons])}

Recommendation:
{recommendation}

Note:
This is a prototype demo for explainable AML risk scoring. It is not a production compliance system.
"""

    st.text_area("Generated Draft Report", report, height=300)

    st.download_button(
        label="Download Report",
        data=report,
        file_name="aml_risk_report.txt",
        mime="text/plain",
    )