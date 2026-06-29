import os
import argparse
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    average_precision_score,
)
from typing import Optional
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


def find_target_column(df: pd.DataFrame) -> str:
    """
    Try to automatically find the AML label column.
    """
    possible_targets = [
        "Is Laundering",
        "Laundering",
        "is_laundering",
        "laundering",
        "target",
        "label",
        "Label",
        "class",
        "Class",
    ]

    for col in possible_targets:
        if col in df.columns:
            return col

    raise ValueError(
        f"Could not automatically find target column. Available columns are:\n{list(df.columns)}"
    )


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names slightly.
    Example: 'Is Laundering' -> 'Is_Laundering'
    """
    df = df.copy()
    df.columns = [col.strip().replace(" ", "_") for col in df.columns]
    return df


def find_target_column_after_cleaning(df: pd.DataFrame) -> str:
    possible_targets = [
        "Is_Laundering",
        "Laundering",
        "is_laundering",
        "laundering",
        "target",
        "label",
        "Label",
        "class",
        "Class",
    ]

    for col in possible_targets:
        if col in df.columns:
            return col

    raise ValueError(
        f"Could not automatically find target column. Available columns are:\n{list(df.columns)}"
    )


def basic_feature_cleanup(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Drop columns that are probably not useful for a first simple baseline.
    We can improve later with time/account-based feature engineering.
    """
    df = df.copy()

    drop_keywords = [
        "id",
        "account",
        "name",
        "customer",
        "timestamp",
        "date",
        "time",
    ]

    cols_to_drop = []

    for col in df.columns:
        if col == target_col:
            continue

        col_lower = col.lower()

        # Drop very ID-like columns for first baseline
        if any(keyword in col_lower for keyword in drop_keywords):
            cols_to_drop.append(col)

    if cols_to_drop:
        print("\nDropping possible ID/time columns for baseline:")
        for col in cols_to_drop:
            print(f" - {col}")
        df = df.drop(columns=cols_to_drop)

    return df


def reduce_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Helpful if the dataset is large.
    Downcasts numeric columns to reduce memory.
    """
    df = df.copy()

    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")

    return df


def main(data_path: str, sample_size: Optional[int] = None):
    print(f"Loading dataset from: {data_path}")

    df = pd.read_csv(data_path)
    print(f"Original shape: {df.shape}")

    df = clean_column_names(df)
    print("\nColumns:")
    print(list(df.columns))

    df = reduce_memory_usage(df)

    target_col = find_target_column_after_cleaning(df)
    print(f"\nDetected target column: {target_col}")

    # Drop rows with missing target
    df = df.dropna(subset=[target_col])

    # Optional sampling for fast first run
    if sample_size is not None and sample_size < len(df):
        print(f"\nUsing sample of {sample_size} rows for fast training.")
        df = df.sample(n=sample_size, random_state=42)

    # Ensure target is numeric 0/1
    y = df[target_col]

    if y.dtype == "object":
        y = y.astype(str).str.lower().map(
            {
                "true": 1,
                "false": 0,
                "yes": 1,
                "no": 0,
                "1": 1,
                "0": 0,
            }
        )

    y = y.astype(int)

    df_features = df.drop(columns=[target_col])
    df_features[target_col] = y

    df_features = basic_feature_cleanup(df_features, target_col)

    X = df_features.drop(columns=[target_col])
    y = df_features[target_col]

    print("\nClass distribution:")
    print(y.value_counts())
    print("\nClass distribution percentage:")
    print(y.value_counts(normalize=True) * 100)

    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    print("\nNumeric columns:")
    print(numeric_cols)

    print("\nCategorical columns:")
    print(categorical_cols)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    # Important for imbalanced AML data
    negative_count = (y == 0).sum()
    positive_count = (y == 1).sum()

    if positive_count > 0:
        scale_pos_weight = negative_count / positive_count
    else:
        scale_pos_weight = 1

    print(f"\nscale_pos_weight: {scale_pos_weight:.2f}")

    model = XGBClassifier(
        n_estimators=150,
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

    print("\nTraining model...")
    pipeline.fit(X_train, y_train)

    print("\nEvaluating model...")
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    print(f"\nROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")

    os.makedirs("models", exist_ok=True)

    model_path = "models/xgb_aml_pipeline.pkl"
    joblib.dump(pipeline, model_path)

    print(f"\nSaved model pipeline to: {model_path}")

    # Save a few test samples for the future Streamlit app
    sample_output_path = "data/sample_app_inputs.csv"
    X_test.head(20).to_csv(sample_output_path, index=False)

    print(f"Saved sample app inputs to: {sample_output_path}")


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
        help="Optional number of rows for quick training test",
    )

    args = parser.parse_args()

    main(
        data_path=args.data_path,
        sample_size=args.sample_size,
    )