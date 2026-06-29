import joblib
import pandas as pd

from utils.explainability import (
    explain_single_transaction,
    add_human_readable_reasons,
)

pipeline = joblib.load("models/xgb_aml_pipeline_v2.pkl")

sample_df = pd.read_csv("data/sample_app_inputs_v2.csv")

single_transaction = sample_df.head(1)

explanation_df, expected_value, risk_score = explain_single_transaction(
    pipeline=pipeline,
    input_df=single_transaction,
    top_n=5,
)

explanation_df = add_human_readable_reasons(explanation_df)

print("\nRisk score:")
print(risk_score)

print("\nTop SHAP explanations:")
print(explanation_df[["display_feature", "shap_value", "direction", "reason"]])