# src/feature_engineering.py
import pandas as pd
import numpy as np
from typing import Dict, Optional

def _safe_rolling_std(series: pd.Series, window: int):
    return series.rolling(window=window, min_periods=max(1, window//2)).std()

def _load_index_volatility(path: str, window: int, col_close: str = "close"):
    """Load an external CSV (index), compute its rolling vol and return DataFrame with date index."""
    idx_df = pd.read_csv(path)
    idx_df["date"] = pd.to_datetime(idx_df["date"], utc=True, errors="coerce")
    idx_df = idx_df.sort_values("date").reset_index(drop=True)
    idx_df.set_index("date", inplace=True)
    idx_df["log_return"] = np.log(idx_df[col_close] / idx_df[col_close].shift(1))
    idx_df["idx_roll_vol"] = _safe_rolling_std(idx_df["log_return"], window)
    return idx_df[["idx_roll_vol"]].astype("float32")

def run(df: pd.DataFrame, params: Dict, external_index_paths: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """
    Builds features from the given dataframe that already contains many technical indicators.

    - Reuses numeric indicator columns directly
    - Adds intraday time features if requested
    - Adds rolling vol features (short/medium/long) computed on log_return
    - Optionally merges index volatility time series from external CSVs (paths provided)
    Input:
      df: preprocessed df from data_preprocessing.run (date as index)
      params: params dict from params.yaml -> expects rolling_windows and intraday keys
      external_index_paths: dict like {"NIFTY50": "data/external/nifty50.csv", ...}
    Returns:
      df with new engineered features (and dropped rows with NaNs)
    """
    print("⚙️ Running feature engineering ...")
    df_local = df.copy()

    # ensure date index
    if not isinstance(df_local.index, pd.DatetimeIndex):
        if "date" in df_local.columns:
            df_local["date"] = pd.to_datetime(df_local["date"], utc=True)
            df_local = df_local.set_index("date")
        else:
            raise ValueError("DataFrame needs a DatetimeIndex or a 'date' column.")

    # ensure numeric columns are floats (already float32 from preprocessing)
    numeric_cols = df_local.select_dtypes(include=["number"]).columns.tolist()
    # exclude 'target' from features
    if "target" in numeric_cols:
        numeric_cols.remove("target")

    # Intraday features
    if params.get("intraday", {}).get("include_time_of_day", True):
        # convert to local timezone for minute_of_day (IST). if tz-naive, dt.hour will still work.
        try:
            local_idx = df_local.index.tz_convert("Asia/Kolkata")
        except Exception:
            local_idx = df_local.index
        df_local["minute_of_day"] = (local_idx.hour * 60 + local_idx.minute).astype("int16")

    if params.get("intraday", {}).get("include_minutes_from_open", False):
        market_open_h, market_open_m = (9, 15)  # default market open 09:15 IST
        try:
            local_idx = df_local.index.tz_convert("Asia/Kolkata")
        except Exception:
            local_idx = df_local.index
        df_local["minutes_from_open"] = ((local_idx.hour - market_open_h) * 60 + (local_idx.minute - market_open_m)).astype("int16")

    # rolling vol features (on log_return)
    rw = params.get("rolling_windows", {})
    short = int(rw.get("short", 12))
    medium = int(rw.get("medium", 75))
    long = int(rw.get("long", 375))

    df_local["roll_vol_short"] = _safe_rolling_std(df_local["log_return"], short)
    df_local["roll_vol_medium"] = _safe_rolling_std(df_local["log_return"], medium)
    df_local["roll_vol_long"] = _safe_rolling_std(df_local["log_return"], long)

    # If user requested ATR/price measures but they are already present, keep them.
    # Cross-asset features: compute index rolling vol and merge
    if external_index_paths:
        for name, path in external_index_paths.items():
            try:
                idx_vol = _load_index_volatility(path, window=medium)
                # join on datetime index (inner join to keep aligned rows)
                df_local = df_local.join(idx_vol.rename(columns={"idx_roll_vol": f"{name}_roll_vol"}), how="left")
            except Exception as e:
                print(f"⚠️ Could not load index at {path} for {name}: {e}")

    # final cleaning: drop rows with NaNs in any of the new rolling features or target
    required_cols = ["roll_vol_short", "roll_vol_medium", "roll_vol_long", "target"]
    for c in required_cols:
        if c not in df_local.columns:
            raise ValueError(f"Expected column {c} after feature generation, but missing.")
    df_local = df_local.dropna(subset=required_cols).copy()

    print(f"✅ Feature engineering complete. Shape: {df_local.shape}")
    return df_local
