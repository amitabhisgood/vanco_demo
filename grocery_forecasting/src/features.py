import pandas as pd
import numpy as np


class FeatureEngineer:
    """
    Key architectural fix vs. previous versions:
    ─────────────────────────────────────────────
    The combined df (train + test concatenated) is sorted by
    [store_nbr, family, date] internally for lag/rolling computation,
    but the ORIGINAL ROW ORDER is restored before returning.
    This ensures compute.py's  iloc[:num_train_rows]  correctly
    recovers the training rows and  iloc[num_train_rows:]  the test rows.
    Scrambling that split was causing the model to train on test rows
    and validate on training rows — explaining early stopping at round ~60
    with a deceptively low-looking RMSE.
    """

    def __init__(self):
        self._family_target_enc       = None
        self._store_family_target_enc = None
        self._global_log_mean         = None

    # ------------------------------------------------------------------
    # 1. TEMPORAL  (no sort dependency — safe to run first)
    # ------------------------------------------------------------------

    def _temporal(self, df: pd.DataFrame) -> pd.DataFrame:
        df["year"]           = df["date"].dt.year
        df["month"]          = df["date"].dt.month
        df["day"]            = df["date"].dt.day
        df["dayofweek"]      = df["date"].dt.dayofweek
        df["weekofyear"]     = df["date"].dt.isocalendar().week.astype(int)
        df["quarter"]        = df["date"].dt.quarter
        df["is_weekend"]     = df["dayofweek"].isin([5, 6]).astype(int)
        df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
        df["is_month_end"]   = df["date"].dt.is_month_end.astype(int)
        return df

    # ------------------------------------------------------------------
    # 2. OIL  (date-level table → merge; never shifts across store rows)
    # ------------------------------------------------------------------

    def _oil(self, df: pd.DataFrame) -> pd.DataFrame:
        if "dcoilwtico" not in df.columns:
            return df

        oil = (
            df[["date", "dcoilwtico"]]
            .drop_duplicates("date")
            .sort_values("date")
            .set_index("date")
        )

        oil["oil_lag_7"]           = oil["dcoilwtico"].shift(7)
        oil["oil_lag_14"]          = oil["dcoilwtico"].shift(14)
        oil["oil_rolling_mean_7"]  = oil["dcoilwtico"].rolling(7,  min_periods=1).mean()
        oil["oil_rolling_mean_28"] = oil["dcoilwtico"].rolling(28, min_periods=1).mean()
        oil["oil_change_7d"]       = oil["dcoilwtico"] - oil["oil_lag_7"]
        oil["oil_change_14d"]      = oil["dcoilwtico"] - oil["oil_lag_14"]
        oil["oil_deviation_28"]    = oil["dcoilwtico"] - oil["oil_rolling_mean_28"]

        oil_cols = [c for c in oil.columns if c != "dcoilwtico"]
        df = df.merge(oil[oil_cols].reset_index(), on="date", how="left")
        return df

    # ------------------------------------------------------------------
    # 3. LAGS  — computed on a sorted working copy; result merged back
    #           by the original index so row order is preserved.
    # ------------------------------------------------------------------

    def _lags(self, df: pd.DataFrame) -> pd.DataFrame:

        work = (
            df[["store_nbr", "family", "date", "sales"]]
            .sort_values(["store_nbr", "family", "date"])
            .copy()
        )

        # KEY FIX: Forward-fill sales so test rows (NaN) carry the last
        # known training sale. Without this, lag_1 for Aug 17 sees Aug 16
        # as NaN/0, causing predictions to collapse across the test window.
        work["sales"] = (
            work.groupby(["store_nbr", "family"])["sales"]
            .transform(lambda x: x.ffill())
        )

        grouped = work.groupby(["store_nbr", "family"])

        lag_defs = [1, 7, 14, 16, 17, 21, 28, 30, 35, 42]

        for lag in lag_defs:
            col = f"lag_{lag}"
            work[col]              = grouped["sales"].shift(lag)
            work[f"{col}_missing"] = work[col].isna().astype(np.int8)
            work[col]              = work[col].fillna(0)

        lag_cols = (
            [f"lag_{l}"         for l in lag_defs] +
            [f"lag_{l}_missing" for l in lag_defs]
        )

        # Merge computed lags back onto the original-order df by index
        df = df.join(work[lag_cols])
        return df

    # ------------------------------------------------------------------
    # 4. ROLLING / EWM  — same pattern: sort → compute → join back
    # ------------------------------------------------------------------

    def _rolling(self, df: pd.DataFrame) -> pd.DataFrame:

        work = (
            df[["store_nbr", "family", "date",
                "lag_1", "lag_7", "lag_14",
                "lag_16", "lag_21", "lag_28", "lag_35", "lag_42"]]
            .sort_values(["store_nbr", "family", "date"])
            .copy()
        )

        g1  = work.groupby(["store_nbr", "family"])["lag_1"]
        g16 = work.groupby(["store_nbr", "family"])["lag_16"]

        # ── Family A: anchored at lag_1 ──
        for w in [7, 14, 28, 56]:
            work[f"rolling_mean_{w}"] = (
                g1.transform(lambda x, w=w: x.rolling(w, min_periods=1).mean())
            )

        work["rolling_std_7"]  = (
            g1.transform(lambda x: x.rolling(7,  min_periods=3).std().fillna(0))
        )
        work["rolling_std_28"] = (
            g1.transform(lambda x: x.rolling(28, min_periods=3).std().fillna(0))
        )
        work["rolling_max_7"]  = (
            g1.transform(lambda x: x.rolling(7, min_periods=1).max())
        )
        work["rolling_min_7"]  = (
            g1.transform(lambda x: x.rolling(7, min_periods=1).min())
        )

        # ── Family B: anchored at lag_16 (test-window safe) ──
        for w in [7, 14, 28]:
            work[f"rolling_mean_lag16_{w}"] = (
                g16.transform(lambda x, w=w: x.rolling(w, min_periods=1).mean())
            )

        work["rolling_std_lag16_14"] = (
            g16.transform(lambda x: x.rolling(14, min_periods=3).std().fillna(0))
        )

        # ── EWMs (fast / medium / slow) — both lag_1 and lag_16 anchors ──
        for alpha in [0.2, 0.5, 0.8]:
            tag = f"ewm_a{int(alpha * 10):02d}"
            work[f"{tag}_lag1"] = (
                g1.transform(lambda x, a=alpha: x.ewm(alpha=a, min_periods=1).mean())
            )
            work[f"{tag}_lag16"] = (
                g16.transform(lambda x, a=alpha: x.ewm(alpha=a, min_periods=1).mean())
            )

        # ── Same-weekday means ──
        work["same_weekday_mean_4w"]   = work[["lag_7","lag_14","lag_21","lag_28"]].mean(axis=1)
        work["same_weekday_mean_2w"]   = work[["lag_7","lag_14"]].mean(axis=1)
        work["same_weekday_mean_safe"] = work[["lag_21","lag_28","lag_35","lag_42"]].mean(axis=1)

        roll_cols = [c for c in work.columns
                     if c not in ["store_nbr","family","date",
                                  "lag_1","lag_7","lag_14",
                                  "lag_16","lag_21","lag_28","lag_35","lag_42"]]

        df = df.join(work[roll_cols])
        return df

    # ------------------------------------------------------------------
    # 5. PROMOTION FEATURES
    # ------------------------------------------------------------------

    def _promotions(self, df: pd.DataFrame) -> pd.DataFrame:
        if "onpromotion" not in df.columns:
            return df

        work = (
            df[["store_nbr", "family", "date", "onpromotion"]]
            .sort_values(["store_nbr", "family", "date"])
            .copy()
        )

        g = work.groupby(["store_nbr", "family"])["onpromotion"]

        for lag in [1, 7, 14]:
            work[f"promo_lag_{lag}"] = g.shift(lag).fillna(0)

        for w in [4, 8, 28]:
            work[f"promo_rolling_sum_{w}"] = (
                g.transform(
                    lambda x, w=w: x.shift(1).rolling(w, min_periods=1).sum()
                )
            )

        promo_cols = [c for c in work.columns
                      if c not in ["store_nbr","family","date","onpromotion"]]

        df = df.join(work[promo_cols])

        df["promo_weekend"] = df["onpromotion"] * df["is_weekend"]

        if "holiday_type" in df.columns:
            is_holiday = (df["holiday_type"].astype(str) != "WorkDay").astype(int)
            df["promo_holiday"] = (df["onpromotion"] * is_holiday).fillna(0)

        return df

    # ------------------------------------------------------------------
    # 6. STORE-LEVEL AGGREGATES
    # ------------------------------------------------------------------

    def _store_features(self, df: pd.DataFrame) -> pd.DataFrame:

        work = (
            df[["store_nbr", "family", "date", "sales"]]
            .sort_values(["store_nbr", "family", "date"])
            .copy()
        )

        g = work.groupby(["store_nbr", "family"])

        work["store_family_expanding_mean"] = (
            g["sales"]
            .transform(lambda x: x.expanding().mean().shift(1))
            .fillna(0)
        )

        shifted = g["sales"].transform(lambda x: x.shift(1)).fillna(0)
        work["_shifted"] = shifted

        family_date_sum = (
            work.groupby(["date", "family"])["_shifted"]
            .transform("sum")
            .replace(0, np.nan)
        )
        work["store_family_sales_share"] = (shifted / family_date_sum).fillna(0)

        store_cols = ["store_family_expanding_mean", "store_family_sales_share"]
        df = df.join(work[store_cols])
        return df

    # ------------------------------------------------------------------
    # 7. TARGET ENCODING  (fitted on explicit train-row slice)
    # ------------------------------------------------------------------

    def fit_target_encodings(self, train_df: pd.DataFrame) -> None:
        log_sales = np.log1p(train_df["sales"].clip(lower=0))
        self._global_log_mean = float(log_sales.mean())
        gmean = self._global_log_mean

        def _smooth(series):
            n    = len(series)
            mean = series.mean()
            k    = 1 / (1 + np.exp(-(n - 50) / 10))
            return k * mean + (1 - k) * gmean

        temp = train_df.assign(_ls=log_sales)

        self._family_target_enc = (
            temp.groupby("family")["_ls"].apply(_smooth)
        )
        self._store_family_target_enc = (
            temp.groupby(["store_nbr", "family"])["_ls"].apply(_smooth)
        )

    def _apply_target_encodings(self, df: pd.DataFrame) -> pd.DataFrame:
        fallback = self._global_log_mean if self._global_log_mean is not None else 0.0

        if self._family_target_enc is not None:
            df["family_target_enc"] = (
                df["family"]
                .map(self._family_target_enc)
                .fillna(fallback)
            )

        if self._store_family_target_enc is not None:
            idx = pd.MultiIndex.from_arrays(
                [df["store_nbr"], df["family"]]
            )
            df["store_family_target_enc"] = (
                self._store_family_target_enc
                .reindex(idx)
                .values
            )
            df["store_family_target_enc"] = (
                df["store_family_target_enc"].fillna(fallback)
            )

        return df

    # ------------------------------------------------------------------
    # 8. CATEGORICAL LABEL ENCODING
    # ------------------------------------------------------------------

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in ["family","type","city","state",
                    "holiday_type","locale","locale_name"]:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .fillna("Unknown")
                    .astype("category")
                    .cat.codes
                )
        return df

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def fit_transform(
        self,
        df: pd.DataFrame,
        is_train: bool = True,
        num_train_rows: int = None
    ) -> pd.DataFrame:

        print(
            f"[-] Engineering features "
            f"(Mode: {'Train' if is_train else 'Test'})..."
        )

        # Stamp the original position so we can restore order at the end
        df = df.reset_index(drop=True)
        df["_orig_idx"] = df.index

        # Ensure sales column exists for train rows.
        # Test rows intentionally keep NaN sales so the ffill in _lags()
        # carries forward the last known training sale instead of zero.
        if "sales" not in df.columns:
            df["sales"] = np.nan

        # Steps 1-6: each helper sorts internally, computes, joins back
        df = self._temporal(df)       # order-independent
        df = self._oil(df)            # date-level merge, order-independent
        df = self._lags(df)           # sort → ffill sales → compute → join by index
        df = self._rolling(df)        # sort → compute → join by index
        df = self._promotions(df)     # sort → compute → join by index
        df = self._store_features(df) # sort → compute → join by index

        # Step 7: target encoding on proper training rows only
        if is_train and num_train_rows is not None:
            # Identify original training rows by their _orig_idx
            train_mask = df["_orig_idx"] < num_train_rows
            self.fit_target_encodings(df[train_mask])
        elif is_train:
            self.fit_target_encodings(df)

        df = self._apply_target_encodings(df)

        # Step 8: label-encode categoricals
        df = self._encode_categoricals(df)

        # Restore original row order (critical for iloc split in compute.py)
        df = df.sort_values("_orig_idx").drop(columns=["_orig_idx"])
        df = df.reset_index(drop=True)

        # Global NaN / inf cleanup
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = (
            df[num_cols]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )

        print(f"[+] Feature matrix shape after engineering: {df.shape}")
        return df