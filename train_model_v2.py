import os
import argparse
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from utils.feature_engineering import (
    clean_column_names,
    reduce_memory_usage,
    find_target_column,
    engineer_aml_features,
    select_model_features,
)


def evaluate_at_thresholds(y_true, y_proba, thresholds):
    print("\nThreshold analysis:")
    print("-" * 70)
    print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Flagged':<12}")
    print("-" * 70)

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)

        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()

        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        flagged = y_pred.sum()

        print(f"{threshold:<12.2f} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {flagged:<12}")


def main(data_path, sample_size=None):
    print(f"Loading dataset from: {data_path}")

    df = pd.read_csv(data_path)
    print(f"Original shape: {df.shape}")

    df = clean_column_names(df)
    df = reduce_memory_usage(df)

    print("\nColumns:")
    print(list(df.columns))

    target_col = find_target_column(df)
    print(f"\nDetected target column: {target_col}")

    df = df.dropna(subset=[target_col])
    df[target_col] = df[target_col].astype(int)

    if sample_size is not None and sample_size < len(df):
        print(f"\nUsing sample of {sample_size} rows for faster experiment.")
        df = df.sample(n=sample_size, random_state=42)

    print("\nEngineering AML features...")
    df = engineer_aml_features(df)

    X, y = select_model_features(df, target_col)

    print("\nClass distribution:")
    print(y.value_counts())

    print("\nClass distribution percentage:")
    print(y.value_counts(normalize=True) * 100)

    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    print("\nNumeric features:")
    print(numeric_cols)

    print("\nCategorical features:")
    print(categorical_cols)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    negative_count = (y == 0).sum()
    positive_count = (y == 1).sum()
    scale_pos_weight = negative_count / positive_count

    print(f"\nscale_pos_weight: {scale_pos_weight:.2f}")

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
        scale_pos_weight=scale_pos_weight,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    print("\nTraining XGBoost AML model...")
    pipeline.fit(X_train, y_train)

    print("\nEvaluating model at default threshold 0.50...")
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred_default = (y_proba >= 0.50).astype(int)

    print("\nClassification report:")
    print(classification_report(y_test, y_pred_default))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred_default))

    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    print(f"\nROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")

    evaluate_at_thresholds(
        y_true=y_test.values,
        y_proba=y_proba,
        thresholds=[0.50, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99],
    )

    os.makedirs("models", exist_ok=True)

    model_path = "models/xgb_aml_pipeline_v2.pkl"
    joblib.dump(pipeline, model_path)
    print(f"\nSaved model pipeline to: {model_path}")

    sample_output_path = "data/sample_app_inputs_v2.csv"
    X_test.head(50).to_csv(sample_output_path, index=False)
    print(f"Saved sample app inputs to: {sample_output_path}")

    feature_info = {
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "target_column": target_col,
    }

    joblib.dump(feature_info, "models/feature_info_v2.pkl")
    print("Saved feature info to: models/feature_info_v2.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to AML CSV dataset",
    )

    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Optional row sample for quick testing",
    )

    args = parser.parse_args()

    main(
        data_path=args.data_path,
        sample_size=args.sample_size,
    )