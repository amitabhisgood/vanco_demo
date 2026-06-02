import pandas as pd
import numpy as np

class TimeAwareValidation:
    def __init__(self, train_df: pd.DataFrame, val_days: int = 16):
        """
        Initializes the validation splitter.
        The competition test window spans exactly 16 days. 
        We mimic this by using the final 16 days of the training data as our validation set.
        """
        self.train_df = train_df
        self.val_days = val_days

    def train_test_split_by_date(self):
        """
        Executes a deterministic out-of-time validation split.
        """
        print("[-] Designing time-aware validation split...")
        
        # Determine the cutoff date based on the maximum date in the dataset
        max_date = self.train_df["date"].max()
        split_date = max_date - pd.Timedelta(days=self.val_days)
        
        # Split datasets based on the timeline boundary
        train_split = self.train_df[self.train_df["date"] <= split_date].copy()
        val_split = self.train_df[self.train_df["date"] > split_date].copy()
        
        print(f"[+] Split Point Date Boundary: {split_date.strftime('%Y-%m-%d')}")
        print(f"[+] Training Split Shape  : {train_split.shape} | Date Range: {train_split['date'].min().strftime('%Y-%m-%d')} to {train_split['date'].max().strftime('%Y-%m-%d')}")
        print(f"[+] Validation Split Shape: {val_split.shape} | Date Range: {val_split['date'].min().strftime('%Y-%m-%d')} to {val_split['date'].max().strftime('%Y-%m-%d')}")
        
        return train_split, val_split

    @staticmethod
    def calculate_rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculates Root Mean Squared Logarithmic Error (RMSLE).
        This is the official evaluation metric for the competition.
        """
        # Secure values against negative predictions
        y_pred = np.clip(y_pred, 0, None)
        
        log_true = np.log1p(y_true)
        log_pred = np.log1p(y_pred)
        
        return float(np.sqrt(np.mean((log_true - log_pred) ** 2)))