import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import joblib
import pandas as pd
import streamlit as st

from rag.retriever import (
    AMLKnowledgeRetriever,
    build_query_from_prediction,
)

from rag.report_generator import generate_rag_enriched_report

from utils.explainability import (
    explain_single_transaction,
    add_human_readable_reasons,
    plot_shap_bar,
)


MODEL_PATH = "models/xgb_aml_pipeline_v2.pkl"
SAMPLE_FILES = {
    "Original random test samples": "data/sample_app_inputs_v2.csv",
    "Positive laundering - high risk": "data/testing_samples/positive_high_risk_with_metadata.csv",
    "Positive laundering - random": "data/testing_samples/positive_random_with_metadata.csv",
    "Negative normal - low risk": "data/testing_samples/negative_low_risk_with_metadata.csv",
    "Negative normal - high-score false positive candidates": "data/testing_samples/negative_high_score_false_positive_candidates_with_metadata.csv",
    "Negative normal - random": "data/testing_samples/negative_random_with_metadata.csv",
    "Mixed testing samples": "data/testing_samples/mixed_testing_samples_with_metadata.csv",
}

def prepare_model_input(row_df):
    """
    Remove testing metadata columns before sending input to the model.
    """

    metadata_cols = [
        "true_label",
        "model_risk_score",
    ]

    model_input = row_df.drop(
        columns=[col for col in metadata_cols if col in row_df.columns],
        errors="ignore",
    )

    return model_input
    
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
def load_sample_inputs(file_path):
    return pd.read_csv(file_path)

@st.cache_resource
def load_retriever():
    return AMLKnowledgeRetriever()

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
# sample_df = load_sample_inputs()
retriever = load_retriever()

st.sidebar.header("Transaction Selection")

sample_group = st.sidebar.selectbox(
    "Choose sample group",
    options=list(SAMPLE_FILES.keys()),
)

sample_file_path = SAMPLE_FILES[sample_group]

sample_df = load_sample_inputs(sample_file_path)

sample_index = st.sidebar.selectbox(
    "Choose a sample transaction",
    options=list(range(len(sample_df))),
)

selected_row_df = sample_df.iloc[[sample_index]].copy()

input_df = prepare_model_input(selected_row_df)

st.subheader("Selected Transaction")

st.write(f"Sample group: **{sample_group}**")

st.dataframe(selected_row_df, use_container_width=True)

if "true_label" in selected_row_df.columns:
    true_label = int(selected_row_df["true_label"].iloc[0])

    if true_label == 1:
        st.info("Ground truth label: Laundering transaction")
    else:
        st.info("Ground truth label: Normal transaction")

if "model_risk_score" in selected_row_df.columns:
    stored_score = float(selected_row_df["model_risk_score"].iloc[0])
    st.write(f"Stored model score from sample creation: **{stored_score:.4f}**")

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

    retrieval_query = build_query_from_prediction(
        risk_level=risk_level,
        shap_reasons=reasons,
        transaction_row=input_df.iloc[0],
    )

    retrieved_guidance = retriever.retrieve(
        query=retrieval_query,
        top_k=4,
    )

    st.subheader("Retrieved AML Guidance")

    for i, doc in enumerate(retrieved_guidance, start=1):
        with st.expander(
            f"{i}. {doc['source']} | similarity score: {doc['score']:.3f}"
        ):
            st.write(doc["text"])

    report = generate_rag_enriched_report(
        risk_level=risk_level,
        risk_score=risk_score,
        shap_reasons=reasons,
        retrieved_docs=retrieved_guidance,
        recommendation=recommendation,
    )

    st.subheader("RAG-Enriched Analyst Report")

    st.text_area(
        "Generated Draft Report",
        report,
        height=400,
    )

    st.download_button(
        label="Download RAG-Enriched Report",
        data=report,
        file_name="aml_rag_enriched_report.txt",
        mime="text/plain",
    )