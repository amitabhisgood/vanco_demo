import pandas as pd
import numpy as np
import os

class GroceryDataPipeline:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def load_raw_tables(self):
        """Loads all individual relational CSV files from the data directory."""
        print("[-] Ingesting core data tables...")
        train = pd.read_csv(os.path.join(self.data_dir, "train.csv"), parse_dates=["date"])
        test = pd.read_csv(os.path.join(self.data_dir, "test.csv"), parse_dates=["date"])
        stores = pd.read_csv(os.path.join(self.data_dir, "stores.csv"))
        oil = pd.read_csv(os.path.join(self.data_dir, "oil.csv"), parse_dates=["date"])
        holidays = pd.read_csv(os.path.join(self.data_dir, "holidays_events.csv"), parse_dates=["date"])
        
        return train, test, stores, oil, holidays

    def process_oil_prices(self, oil_df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes the oil price table.
        Oil prices have missing values on weekends and holidays. We map 
        them to a complete date range and forward-fill to prevent data leakage.
        """
        # Create a continuous calendar timeline to catch missing days
        min_date, max_date = oil_df["date"].min(), oil_df["date"].max()
        full_timeline = pd.date_range(start=min_date, end=max_date, freq="D")
        oil_filled = pd.DataFrame({"date": full_timeline})
        
        # Merge and forward-fill gaps (imputing Monday with Friday's price)
        oil_filled = oil_filled.merge(oil_df, on="date", how="left")
        oil_filled["dcoilwtico"] = oil_filled["dcoilwtico"].ffill().bfill()
        return oil_filled

    def merge_pipeline(self, base_df: pd.DataFrame, stores: pd.DataFrame, oil: pd.DataFrame, holidays: pd.DataFrame) -> pd.DataFrame:
        """
        Executes a deterministic join across tables. 
        Ensures strict time-alignment to block data leakage during inference.
        """
        # 1. Merge Store Metadata
        df = base_df.merge(stores, on="store_nbr", how="left")
        
        # 2. Merge Economical Context (Oil Prices)
        df = df.merge(oil, on="date", how="left")
        
        # 3. Merge National/Local Holidays
        # Filtering transferred holidays to avoid misalignments
        holidays = holidays[holidays["transferred"] == False]
        
        # Group by date to handle multiple events happening on the same day safely
        daily_holidays = holidays.groupby("date").first().reset_index()
        daily_holidays = daily_holidays[["date", "type", "locale", "locale_name"]].rename(
            columns={"type": "holiday_type"}
        )
        
        df = df.merge(daily_holidays, on="date", how="left")
        
        # Fill missing values for days that are regular working days
        df["holiday_type"] = df["holiday_type"].fillna("WorkDay")
        df["locale"] = df["locale"].fillna("None")
        df["locale_name"] = df["locale_name"].fillna("None")
        
        return df

    def run(self):
        """Orchestrates the entire ingestion and merging workflow."""
        train, test, stores, oil, holidays = self.load_raw_tables()
        
        processed_oil = self.process_oil_prices(oil)
        
        print("[-] Joining relational dimensions safely...")
        final_train = self.merge_pipeline(train, stores, processed_oil, holidays)
        final_test = self.merge_pipeline(test, stores, processed_oil, holidays)
        
        print(f"[+] Success! Compiled Train Shape: {final_train.shape} | Test Shape: {final_test.shape}")
        return final_train, final_test

if __name__ == "__main__":
    # Test script locally assuming execution from grocery_forecasting directory
    pipeline = GroceryDataPipeline(data_dir="data")
    train_feat, test_feat = pipeline.run()