import os
import argparse
import joblib
import pandas as pd
import numpy as np

from utils.feature_engineering import (
    clean_column_names,
    reduce_memory_usage,
    find_target_column,
    engineer_aml_features,
    select_model_features,
)


def score_in_batches(pipeline, X, batch_size=100_000):
    """
    Predict probabilities in batches to avoid memory issues.
    """
    scores = []

    for start in range(0, len(X), batch_size):
        end = start + batch_size
        batch = X.iloc[start:end]
        batch_scores = pipeline.predict_proba(batch)[:, 1]
        scores.extend(batch_scores)

    return np.array(scores)


def save_app_ready_and_metadata(df, feature_cols, output_dir, base_name):
    """
    Saves two versions:
    1. app_ready: only model feature columns, safe to load into Streamlit.
    2. with_metadata: includes true_label and model_risk_score for inspection.
    """

    app_ready_path = os.path.join(output_dir, f"{base_name}_app_ready.csv")
    metadata_path = os.path.join(output_dir, f"{base_name}_with_metadata.csv")

    df[feature_cols].to_csv(app_ready_path, index=False)
    df.to_csv(metadata_path, index=False)

    print(f"Saved app-ready file: {app_ready_path}")
    print(f"Saved metadata file : {metadata_path}")


def main(
    data_path,
    model_path,
    output_dir,
    n_samples,
    negative_candidate_size,
):
    os.makedirs(output_dir, exist_ok=True)

    print("Loading dataset...")
    df = pd.read_csv(data_path)

    df = clean_column_names(df)
    df = reduce_memory_usage(df)

    target_col = find_target_column(df)
    print(f"Target column: {target_col}")

    df[target_col] = df[target_col].astype(int)

    print("\nClass distribution:")
    print(df[target_col].value_counts())

    print("\nEngineering AML features...")
    df_engineered = engineer_aml_features(df)

    X_all, y_all = select_model_features(df_engineered, target_col)
    feature_cols = X_all.columns.tolist()

    print("\nLoading trained model...")
    pipeline = joblib.load(model_path)

    # ------------------------------------------------------------------
    # Positive laundering examples
    # ------------------------------------------------------------------
    print("\nPreparing positive laundering examples...")

    positive_indices = y_all[y_all == 1].index
    X_positive = X_all.loc[positive_indices].copy()
    y_positive = y_all.loc[positive_indices].copy()

    positive_scores = score_in_batches(pipeline, X_positive)

    positive_df = X_positive.copy()
    positive_df["true_label"] = y_positive.values
    positive_df["model_risk_score"] = positive_scores

    positive_high_risk = positive_df.sort_values(
        by="model_risk_score",
        ascending=False,
    ).head(n_samples)

    positive_random = positive_df.sample(
        n=min(n_samples, len(positive_df)),
        random_state=42,
    )

    save_app_ready_and_metadata(
        positive_high_risk,
        feature_cols,
        output_dir,
        "positive_high_risk",
    )

    save_app_ready_and_metadata(
        positive_random,
        feature_cols,
        output_dir,
        "positive_random",
    )

    # ------------------------------------------------------------------
    # Negative normal examples
    # ------------------------------------------------------------------
    print("\nPreparing negative normal examples...")

    negative_indices = y_all[y_all == 0].index

    if len(negative_indices) > negative_candidate_size:
        negative_indices = pd.Series(negative_indices).sample(
            n=negative_candidate_size,
            random_state=42,
        ).values

    X_negative = X_all.loc[negative_indices].copy()
    y_negative = y_all.loc[negative_indices].copy()

    negative_scores = score_in_batches(pipeline, X_negative)

    negative_df = X_negative.copy()
    negative_df["true_label"] = y_negative.values
    negative_df["model_risk_score"] = negative_scores

    negative_low_risk = negative_df.sort_values(
        by="model_risk_score",
        ascending=True,
    ).head(n_samples)

    negative_high_score = negative_df.sort_values(
        by="model_risk_score",
        ascending=False,
    ).head(n_samples)

    negative_random = negative_df.sample(
        n=min(n_samples, len(negative_df)),
        random_state=42,
    )

    save_app_ready_and_metadata(
        negative_low_risk,
        feature_cols,
        output_dir,
        "negative_low_risk",
    )

    save_app_ready_and_metadata(
        negative_high_score,
        feature_cols,
        output_dir,
        "negative_high_score_false_positive_candidates",
    )

    save_app_ready_and_metadata(
        negative_random,
        feature_cols,
        output_dir,
        "negative_random",
    )

    # ------------------------------------------------------------------
    # Mixed test set
    # ------------------------------------------------------------------
    print("\nCreating mixed test sample...")

    mixed_df = pd.concat(
        [
            positive_high_risk,
            positive_random,
            negative_low_risk,
            negative_high_score,
            negative_random,
        ],
        axis=0,
    ).sample(frac=1, random_state=42)

    mixed_app_ready_path = os.path.join(output_dir, "mixed_testing_samples_app_ready.csv")
    mixed_metadata_path = os.path.join(output_dir, "mixed_testing_samples_with_metadata.csv")

    mixed_df[feature_cols].to_csv(mixed_app_ready_path, index=False)
    mixed_df.to_csv(mixed_metadata_path, index=False)

    print(f"Saved mixed app-ready file: {mixed_app_ready_path}")
    print(f"Saved mixed metadata file : {mixed_metadata_path}")

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print("\nTesting sample creation complete.")

    print("\nPositive high-risk examples:")
    print(
        positive_high_risk[
            [
                "Amount_Paid",
                "Amount_Received",
                "Payment_Currency",
                "Receiving_Currency",
                "Payment_Format",
                "cross_currency",
                "cross_bank_transfer",
                "is_night_transaction",
                "true_label",
                "model_risk_score",
            ]
        ].head()
    )

    print("\nNegative low-risk examples:")
    print(
        negative_low_risk[
            [
                "Amount_Paid",
                "Amount_Received",
                "Payment_Currency",
                "Receiving_Currency",
                "Payment_Format",
                "cross_currency",
                "cross_bank_transfer",
                "is_night_transaction",
                "true_label",
                "model_risk_score",
            ]
        ].head()
    )

    print("\nFiles saved in:")
    print(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=str,
        default="data/HI-Small_Trans.csv",
        help="Path to raw AML dataset CSV",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="models/xgb_aml_pipeline_v2.pkl",
        help="Path to trained model pipeline",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/testing_samples",
        help="Directory where testing samples will be saved",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=25,
        help="Number of samples to save per category",
    )

    parser.add_argument(
        "--negative_candidate_size",
        type=int,
        default=100000,
        help="Number of negative examples to sample and score",
    )

    args = parser.parse_args()

    main(
        data_path=args.data_path,
        model_path=args.model_path,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        negative_candidate_size=args.negative_candidate_size,
    )