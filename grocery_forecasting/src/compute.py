import os
import numpy as np
import pandas as pd
import lightgbm as lgb

from data_pipeline import GroceryDataPipeline
from features import FeatureEngineer
from validation import TimeAwareValidation


def calculate_log_rmsle(y_true, pred_log):
    """
    RMSLE in log space.
    Model is trained on log1p(sales).
    """
    y_true_log = np.log1p(np.clip(y_true, 0, None))

    return float(
        np.sqrt(
            np.mean(
                (pred_log - y_true_log) ** 2
            )
        )
    )


def run_submission_pipeline():

    data_dir = "data"

    print("=" * 70)
    print("GROCERY SALES FORECASTING PIPELINE")
    print("=" * 70)

    # ---------------------------------------------------------
    # LOAD DATA
    # ---------------------------------------------------------

    print("\n[-] Loading and merging source tables...")

    pipeline = GroceryDataPipeline(data_dir=data_dir)

    train_complete, test_complete = pipeline.run()

    num_train_rows = len(train_complete)

    print(
        f"[+] Train Rows: {num_train_rows:,}"
    )

    print(
        f"[+] Test Rows : {len(test_complete):,}"
    )

    # ---------------------------------------------------------
    # COMBINE TRAIN + TEST
    # ---------------------------------------------------------

    print(
        "\n[-] Combining train and test for "
        "time-series feature generation..."
    )

    combined_df = pd.concat(
        [train_complete, test_complete],
        ignore_index=True,
        sort=False
    )

    # ---------------------------------------------------------
    # FEATURE ENGINEERING
    # ---------------------------------------------------------

    print("\n[-] Engineering features...")

    engineer = FeatureEngineer()

    # Do NOT fill test row sales with 0.
    # Test rows have NaN sales — leave them as NaN so that lag features
    # looking back into training rows pick up real historical sales.
    # features.py handles start-of-series NaNs internally via fillna(0).
    combined_df["sales"] = combined_df["sales"]

    combined_feats = engineer.fit_transform(
        combined_df.copy(),
        is_train=True,
        num_train_rows=num_train_rows
    )

    # ---------------------------------------------------------
    # SPLIT BACK
    # ---------------------------------------------------------

    train_feats = (
        combined_feats
        .iloc[:num_train_rows]
        .copy()
    )

    test_feats = (
        combined_feats
        .iloc[num_train_rows:]
        .copy()
    )

    print(
        f"[+] Feature Matrix Train: "
        f"{train_feats.shape}"
    )

    print(
        f"[+] Feature Matrix Test : "
        f"{test_feats.shape}"
    )

    # ---------------------------------------------------------
    # VALIDATION SPLIT
    # ---------------------------------------------------------

    validator = TimeAwareValidation(
        train_feats,
        val_days=32
    )

    train_split, val_split = (
        validator.train_test_split_by_date()
    )

    # ---------------------------------------------------------
    # FEATURE SELECTION
    # ---------------------------------------------------------

    drop_cols = [
        "id",
        "date",
        "sales"
    ]

    feature_cols = [
        col
        for col in train_feats.columns
        if col not in drop_cols
    ]

    print(
        f"\n[+] Total Features Used: "
        f"{len(feature_cols)}"
    )

    # ---------------------------------------------------------
    # BUILD MATRICES
    # ---------------------------------------------------------

    X_train = train_split[feature_cols]

    y_train = train_split["sales"].values

    X_val = val_split[feature_cols]

    y_val = val_split["sales"].values

    X_full = train_feats[feature_cols]

    y_full = train_feats["sales"].values

    X_test = test_feats[feature_cols]

    # ---------------------------------------------------------
    # LIGHTGBM DATASETS
    # ---------------------------------------------------------

    train_dataset = lgb.Dataset(
        X_train,
        label=np.log1p(y_train)
    )

    valid_dataset = lgb.Dataset(
        X_val,
        label=np.log1p(y_val),
        reference=train_dataset
    )

    # ---------------------------------------------------------
    # MODEL PARAMETERS
    # ---------------------------------------------------------

    params = {
        "objective": "regression",
        "metric": "rmse",

        "boosting_type": "gbdt",

        "learning_rate": 0.02,

        "num_leaves": 255,

        "feature_fraction": 0.80,

        "bagging_fraction": 0.80,

        "bagging_freq": 1,

        "min_data_in_leaf": 50,

        "lambda_l1": 0.10,

        "lambda_l2": 0.10,

        "seed": 42,

        "verbose": -1
    }

    # ---------------------------------------------------------
    # VALIDATION MODEL
    # ---------------------------------------------------------

    print(
        "\n[-] Training validation model "
        "(progress logged every 1000 rounds)..."
    )

    val_model = lgb.train(
        params,
        train_dataset,
        num_boost_round=7000,
        valid_sets=[valid_dataset],
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=200,
                verbose=True
            ),
            lgb.log_evaluation(period=1000)
        ]
    )

    # ---------------------------------------------------------
    # VALIDATION SCORE
    # ---------------------------------------------------------

    val_preds_log = val_model.predict(
        X_val,
        num_iteration=val_model.best_iteration
    )

    validation_rmsle = calculate_log_rmsle(
        y_val,
        val_preds_log
    )

    print("\n" + "=" * 70)

    print(
        f"[+] Validation RMSLE: "
        f"{validation_rmsle:.6f}"
    )

    print(
        f"[+] Best Iteration: "
        f"{val_model.best_iteration}"
    )

    print("=" * 70)

    # ---------------------------------------------------------
    # FINAL MODEL
    # ---------------------------------------------------------

    print(
        "\n[-] Training final production model "
        "(progress logged every 1000 rounds)..."
    )

    full_dataset = lgb.Dataset(
        X_full,
        label=np.log1p(y_full)
    )

    final_model = lgb.train(
        params,
        full_dataset,
        num_boost_round=val_model.best_iteration,
        callbacks=[
            lgb.log_evaluation(period=1000)
        ]
    )

    # ---------------------------------------------------------
    # FEATURE IMPORTANCE
    # ---------------------------------------------------------

    print(
        "\n[-] Exporting feature importance..."
    )

    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": final_model.feature_importance(
            importance_type="gain"
        )
    })

    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False
        )
    )

    importance_path = os.path.join(
        data_dir,
        "feature_importance.csv"
    )

    importance_df.to_csv(
        importance_path,
        index=False
    )

    print(
        f"[+] Saved: {importance_path}"
    )

    # ---------------------------------------------------------
    # INFERENCE
    # ---------------------------------------------------------

    print(
        "\n[-] Generating test predictions..."
    )

    test_preds_log = final_model.predict(
        X_test
    )

    test_preds = np.expm1(
        test_preds_log
    )

    test_preds = np.clip(
        test_preds,
        0,
        None
    )

    # ---------------------------------------------------------
    # SUBMISSION
    # ---------------------------------------------------------

    submission = pd.DataFrame({
        "id": test_complete["id"].astype(int),
        "sales": test_preds
    })

    output_path = os.path.join(
        data_dir,
        "submission.csv"
    )

    submission.to_csv(
        output_path,
        index=False
    )

    print("\n" + "=" * 70)

    print(
        "[+] Submission file generated."
    )

    print(
        f"[+] Saved To: "
        f"{os.path.abspath(output_path)}"
    )

    print(
        f"[+] Predictions: "
        f"{len(submission):,}"
    )

    print("=" * 70)


if __name__ == "__main__":
    run_submission_pipeline()