import pandas as pd
import numpy as np
from typing import List

EXPECTED_COLUMNS = [
    "date","open","high","low","close","volume","sma5","sma10","sma15","sma20",
    "ema5","ema10","ema15","ema20","upperband","middleband","lowerband","HT_TRENDLINE",
    "KAMA10","KAMA20","KAMA30","SAR","TRIMA5","TRIMA10","TRIMA20","ADX5","ADX10","ADX20",
    "APO","CCI5","CCI10","CCI15","macd510","macd520","macd1020","macd1520","macd1226",
    "MOM10","MOM15","MOM20","ROC5","ROC10","ROC20","PPO","RSI14","RSI8","slowk","slowd",
    "fastk","fastd","fastksr","fastdsr","ULTOSC","WILLR","ATR","Trange","TYPPRICE",
    "HT_DCPERIOD","BETA"
]

def _warn_cols(df_cols: List[str]):
    missing = [c for c in EXPECTED_COLUMNS if c not in df_cols]
    extra = [c for c in df_cols if c not in EXPECTED_COLUMNS and c != "target"]
    if missing:
        print(f"Warning: CSV is missing expected columns: {missing}")
    if extra:
        print(f"Notice: CSV contains extra columns: {extra}")

def _to_float32(df: pd.DataFrame, exclude: List[str] = None):
    exclude = exclude or []
    for col in df.columns:  
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]) or df[col].dtype == object:
            # attempt conversion
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
            except Exception:
                # leave as-is (could be non-numeric)
                pass
    return df

def run(raw_path: str,
        prediction_horizon_bars: int = 12,
        annualization_factor: float = 252 * 75,
        set_index: bool = True) -> pd.DataFrame:
    """
    Load a single company CSV and return a cleaned DataFrame:
      - parse timezone-aware 'date'
      - validate columns
      - convert numeric cols to float32
      - compute log returns
      - compute target = annualized realized volatility of next H bars (t+1 ... t+H)
    Args:
      raw_path: path to the company CSV
      prediction_horizon_bars: H
      annualization_factor: N for annualization (e.g., 252*75)
      set_index: whether to set date as DataFrame index
    Returns:
      pd.DataFrame with columns including 'log_return' and 'target'
    """
    print(f"Loading raw data from {raw_path} ...")
    df = pd.read_csv(raw_path)

    # normalize header casing/spaces if needed (assumes headers are exact as sample)
    cols = df.columns.tolist()
    _warn_cols(cols)

    # parse date with timezone awareness
    if "date" not in df.columns:
        raise ValueError("CSV must contain a 'date' column.")
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    if df["date"].isna().any():
        raise ValueError("Some date values could not be parsed. Check format.")

    # sort by date
    df = df.sort_values("date").reset_index(drop=True)

    # convert numeric columns to float32 to save memory
    df = _to_float32(df, exclude=["date"])

    # compute log returns from 'close'
    if "close" not in df.columns:
        raise ValueError("CSV must contain 'close' column.")
    df["log_return"] = np.log(df["close"] / df["close"].shift(1)).astype("float32")

    # compute target: realized vol over next H bars (t+1 .. t+H)
    H = int(prediction_horizon_bars)
    # rolling std ends at t+H, so shift back by H to align to t
    future_std = df["log_return"].rolling(window=H, min_periods=H).std().shift(-H)
    df["target"] = (future_std * np.sqrt(float(annualization_factor))).astype("float32")

    # drop rows where log_return or target are NA (beginning and tail)
    df = df.dropna(subset=["log_return", "target"]).reset_index(drop=True)

    if set_index:
        df = df.set_index("date")

    print(f"Preprocessing done. Result shape: {df.shape}")
    return df
