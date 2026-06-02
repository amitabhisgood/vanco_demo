import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from data_pipeline import GroceryDataPipeline
from validation import TimeAwareValidation
from features import FeatureEngineer

def run_submission_pipeline():
    data_dir = "data"
    
    # 1. Ingest and Merge Relational Tables
    pipeline = GroceryDataPipeline(data_dir=data_dir)
    train_complete, test_complete = pipeline.run()
    
    # Track where the training data ends and test data begins
    num_train_rows = len(train_complete)
    
    # Combine datasets temporarily to ensure perfect, unified categorical encoding matches
    print("[-] Unifying train and test distributions for categorical consistency...")
    combined_df = pd.concat([train_complete, test_complete], ignore_index=True)
    
    # 2. Feature Engineering Sequence (State Isolated to Prevent Leakage)
    engineer = FeatureEngineer()
    
    # Run feature engineering with precise training vs test contextual masking
    # This prevents uninitialized test window targets from polluting historical moving averages
    combined_feats_train = engineer.fit_transform(combined_df.copy(), is_train=True)
    combined_feats_test = engineer.fit_transform(combined_df.copy(), is_train=False)
    
    # Separate back into clean train and test datasets using row indices
    train_feats = combined_feats_train.iloc[:num_train_rows].copy()
    test_feats = combined_feats_test.iloc[num_train_rows:].copy()
    
    # 3. Define Training Feature Columns
    drop_cols = ["id", "date", "sales", "locale", "locale_name"]
    feature_cols = [col for col in train_feats.columns if col not in drop_cols]
    
    # 4. Generate Internal Out-of-Time Validation Split for Direct Score Verification
    print("[-] Designing time-aware validation split for direct score verification...")
    # Emulate the 16-day competitive window using the tail-end of training data
    val_split_date = "2017-07-30"
    
    val_mask = train_feats["date"] > val_split_date
    train_split = train_feats[~val_mask]
    val_split = train_feats[val_mask]
    
    X_train_split = train_split[feature_cols]
    y_train_split = train_split["sales"]
    X_val_split = val_split[feature_cols]
    y_val_split = val_split["sales"]
    
    # Train validation model using log space labels
    val_dataset = lgb.Dataset(X_train_split, label=np.log1p(y_train_split))
    val_monitor = lgb.Dataset(X_val_split, label=np.log1p(y_val_split), reference=val_dataset)
    
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "seed": 42,
        "verbose": -1
    }
    
    print("[-] Tuning and identifying optimal iterations on out-of-time validation slice...")
    val_model = lgb.train(
        params,
        val_dataset,
        num_boost_round=1000,
        valid_sets=[val_dataset, val_monitor],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
    )
    
    # Predict on validation segment (Returns values strictly in log-space)
    val_preds_log = val_model.predict(X_val_split)
    
    # Secure the validation ground truth directly in matching log-space
    y_val_log = np.log1p(np.clip(y_val_split, 0, None))
    
    # Calculate RMSLE directly (RMSE over log-space labels is mathematically identical to RMSLE)
    internal_rmsle = float(np.sqrt(np.mean((val_preds_log - y_val_log) ** 2)))
    print(f"\n[+] Clean Aligned Out-of-Time Validation RMSLE Score: {internal_rmsle:.5f}")
    
    # 5. Train Final Production Model on 100% of Historical Data Matrix
    X_train_full, y_train_full = train_feats[feature_cols], train_feats["sales"]
    X_test_full = test_feats[feature_cols]
    
    print(f"\n[-] Training final model on 100% of data with {len(feature_cols)} features...")
    full_dataset = lgb.Dataset(X_train_full, label=np.log1p(y_train_full))
    
    production_model = lgb.train(
        params,
        full_dataset,
        num_boost_round=val_model.best_iteration if val_model.best_iteration > 0 else 186
    )
    
    # 6. Extract and Export Feature Importance (Fulfilling Assessment Requirements)
    print("[-] Executing feature importance explainability analysis...")
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": production_model.feature_importance(importance_type="gain")
    }).sort_values(by="importance", ascending=False)
    
    importance_path = os.path.join(data_dir, "feature_importance.csv")
    importance_df.to_csv(importance_path, index=False)
    print(f"[+] Feature importances successfully exported to: {importance_path}")
    
    # 7. Execute Inference on Hidden Test Set Window
    print("[-] Executing inference on hidden test set window...")
    test_preds_log = production_model.predict(X_test_full)
    test_preds = np.expm1(test_preds_log) # Invert the log transformation back to real sales scale
    test_preds = np.clip(test_preds, 0, None) # Clip any negative predictions safely
    
    # 8. Construct and Export Submission CSV File
    submission = pd.DataFrame({
        "id": test_complete["id"].astype(int),
        "sales": test_preds
    })
    
    output_path = os.path.join(data_dir, "submission.csv")
    submission.to_csv(output_path, index=False)
    
    print(f"\n[+] Inference Complete! Submission file successfully generated.")
    print(f"[+] File Saved To: {os.path.abspath(output_path)}")
    print(f"[+] Total Predictions Exported: {len(submission)}")

if __name__ == "__main__":
    run_submission_pipeline()