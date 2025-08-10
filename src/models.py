# src/models.py
import pickle
from typing import Any, Dict, Optional
import numpy as np

from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

# GARCH dependencies
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except Exception:
    ARCH_AVAILABLE = False


class BaselineMean:
    """Baseline model: predicts the mean target from training set for all points."""
    def __init__(self):
        self.mean_ = None

    def fit(self, X, y):
        self.mean_ = float(np.nanmean(y))
        return self

    def predict(self, X):
        if self.mean_ is None:
            raise RuntimeError("BaselineMean not fitted")
        n = len(X)
        return np.full(shape=(n,), fill_value=self.mean_, dtype="float32")


class GARCHWrapper:
    """
    A lightweight wrapper around arch.arch_model for a GARCH(p,q) model.
    Behavior:
      - fit(X, y): expects DataFrame/array that contains 'log_return' (used for fitting).
      - predict(X): returns forecasted annualized volatility for each row in X.
    Modes:
      - fast (default): fit once on the full series and produce a single H-step forecast (repeated for all rows).
      - rolling_refit (optional): for each time t, fit on data up to t and forecast H-steps ahead -> correct but SLOW.
    Params (via params dict):
      - p: int (GARCH p)
      - q: int (GARCH q)
      - dist: str (error distribution, e.g. 'normal', 't')
      - annualization_factor: float (used to annualize std from model's conditional volatility)
      - horizon: int (forecast horizon H)
      - rolling_refit: bool (default False)
      - rolling_window: int or None (if set, use only last rolling_window observations for each refit)
    """
    def __init__(self, params: Dict):
        if not ARCH_AVAILABLE:
            raise ImportError("arch package is required for GARCH. Install via pip install arch")
        self.p = int(params.get("p", 1))
        self.q = int(params.get("q", 1))
        self.dist = params.get("dist", "normal")
        self.annualization_factor = float(params.get("annualization_factor", 252 * 75))
        self.horizon = int(params.get("horizon", 12))
        self.rolling_refit = bool(params.get("rolling_refit", False))
        self.rolling_window = params.get("rolling_window", None)
        self.fitted = None
        self.last_forecast_variance = None  # variance forecast for horizon

    def fit(self, X, y=None):
        """
        X should be a DataFrame (or Series) that contains 'log_return' (or be a Series of returns).
        """
        # Accept Series of returns or DataFrame containing 'log_return'
        if hasattr(X, "get") and "log_return" in getattr(X, "columns", []):
            returns = X["log_return"].dropna().astype("float64")
        elif isinstance(X, (np.ndarray, list)):
            returns = np.asarray(X).astype("float64").ravel()
        else:
            # try interpret X as Series
            try:
                returns = X.squeeze().astype("float64").dropna()
            except Exception:
                raise ValueError("GARCHWrapper.fit received X in an unsupported format. Pass a DataFrame with 'log_return' or a Series/ndarray of returns.")

        # Fit single model on full series (fast mode)
        if not self.rolling_refit:
            am = arch_model(returns, vol="Garch", p=self.p, q=self.q, dist=self.dist, mean="Zero")
            self.fitted = am.fit(disp="off")
            # forecast H steps ahead from the end of the sample
            fc = self.fitted.forecast(horizon=self.horizon, reindex=False)
            # fc.variance is a DataFrame shape (1 x horizon) for forecast from last obs
            var = fc.variance.values[-1]  # take last row
            # take the horizon-th variance (indexed at horizon-1)
            var_h = float(var[self.horizon - 1])
            self.last_forecast_variance = var_h
        else:
            # Rolling refit mode: compute one forecast per time t (very slow)
            n = len(returns)
            horizon = self.horizon
            per_point_forecasts = np.full(n, np.nan, dtype="float32")
            # we will produce forecasts for indices 0..n-1 corresponding to fit on returns[:i+1] forecasts for i+1..i+horizon
            for i in range(horizon, n):
                start_idx = 0 if self.rolling_window is None else max(0, i + 1 - int(self.rolling_window))
                sample = returns[start_idx: i + 1]
                am = arch_model(sample, vol="Garch", p=self.p, q=self.q, dist=self.dist, mean="Zero")
                res = am.fit(disp="off")
                fc = res.forecast(horizon=horizon, reindex=False)
                var = fc.variance.values[-1]
                per_point_forecasts[i] = float(var[horizon - 1])
                # (optionally print progress)
                if (i - horizon) % 500 == 0:
                    print(f"  [garch] rolling refit progress: fitted up to idx {i}/{n}")
            # store per_point_forecasts (aligned to input returns). For predict we map them to X rows
            self.fitted = {"per_point_variance": per_point_forecasts}
            self.last_forecast_variance = float(np.nanmean(per_point_forecasts[~np.isnan(per_point_forecasts)]))

        return self

    def predict(self, X):
        """
        Returns annualized std forecasts aligned to length of X (one value per row).
        If rolling_refit=False -> returns constant forecast repeated.
        If rolling_refit=True -> attempts to return per-point forecasts where available, otherwise fills with last_forecast.
        Final output is volatility (std), annualized.
        """
        n = len(X)
        if not self.rolling_refit:
            # repeat last forecast variance for all rows
            if self.last_forecast_variance is None:
                raise RuntimeError("GARCHWrapper must be fitted before predict.")
            std = np.sqrt(self.last_forecast_variance) * np.sqrt(self.annualization_factor)
            return np.full(shape=(n,), fill_value=float(std), dtype="float32")
        else:
            per_var = self.fitted.get("per_point_variance")
            out = np.full(n, np.nan, dtype="float32")
            # per_var might be shorter than n if returns sample shorter; align to right
            length = len(per_var)
            out[:length] = np.sqrt(np.nan_to_num(per_var, nan=self.last_forecast_variance)) * np.sqrt(self.annualization_factor)
            # fill any remaining nans with last_forecast
            out = np.nan_to_num(out, nan=(np.sqrt(self.last_forecast_variance) * np.sqrt(self.annualization_factor)))
            return out


def train(model_name: str, df, params: Dict, horizon: int = 12):
    """
    Train a model. Expects df to include 'target' and have datetime index or date column.
    model_name must match a key in your params.yaml models section (ridge/lasso/random_forest/xgboost/baseline/garch)
    """
    print(f"🤖 Training model: {model_name}")
    # prepare X, y
    if "target" not in df.columns:
        raise ValueError("DataFrame must contain 'target' column for training.")
    X = df.drop(columns=["target"], errors="ignore").select_dtypes(include="number")
    y = df["target"].values

    # expose model-specific params (if present)
    model_params = params if isinstance(params, dict) else {}

    if model_name == "ridge":
        model = Ridge(**model_params)
    elif model_name == "lasso":
        model = Lasso(**model_params)
    elif model_name == "random_forest":
        model = RandomForestRegressor(**model_params)
    elif model_name == "xgboost":
        if XGBRegressor is None:
            raise ImportError("XGBoost not installed. Install xgboost to use this model.")
        model = XGBRegressor(**model_params)
    elif model_name == "baseline":
        model = BaselineMean()
    elif model_name == "garch":
        if not ARCH_AVAILABLE:
            raise ImportError("arch package not found. Install with `pip install arch` to use GARCH.")
        # build garch wrapper params and fit on log_return series (we need log_return column)
        garch_params = dict(
            p=int(model_params.get("p", 1)),
            q=int(model_params.get("q", 1)),
            dist=model_params.get("dist", "normal"),
            annualization_factor=model_params.get("annualization_factor", 252 * 75),
            horizon=horizon,
            rolling_refit=bool(model_params.get("rolling_refit", False)),
            rolling_window=model_params.get("rolling_window", None)
        )
        garch = GARCHWrapper(garch_params)
        # fit using the df (DataFrame expected to have 'log_return' column)
        garch.fit(df)
        model = garch
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    # For param-based models that support fit(X,y)
    if hasattr(model, "fit") and model_name not in ("garch", "baseline"):
        model.fit(X, y)
    elif model_name == "baseline":
        model.fit(X, y)

    return model


def save_model(model: Any, path: str):
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"💾 Model saved to: {path}")
