# src/backtest.py
import json
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
import os

def _max_drawdown(equity: np.ndarray) -> float:
    """
    Compute maximum drawdown (as fraction, e.g. 0.2 = 20%).
    """
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / peak
    return float(np.nanmax(dd)) if len(dd) > 0 else 0.0

def run(model, df: pd.DataFrame, backtest_params: Dict) -> Tuple[Dict, pd.Series]:
    """
    Realistic backtest using inverse-volatility sizing with execution costs.

    Returns:
      (summary_dict, equity_series)
      - summary_dict: dict of aggregated performance statistics (no equity curve)
      - equity_series: pd.Series indexed by df.index with equity over time
    """
    print("📈 Running realistic backtest...")

    # Prepare inputs
    X = df.drop(columns=["target"], errors="ignore").select_dtypes(include="number")
    if "close" not in df.columns:
        raise ValueError("DataFrame must include 'close' column for backtesting.")
    close = df["close"].astype("float64").values
    n = len(close)
    if n < 2:
        raise ValueError("Backtest requires at least 2 rows to compute returns.")

    # Predictions: expected annualized volatility (std)
    if model is None:
        raise ValueError("No model provided for backtest.")
    preds = model.predict(X)
    preds = np.asarray(preds, dtype="float64").ravel()
    if preds.shape[0] != n:
        # allow broadcasting if model returned a single value
        if preds.size == 1:
            preds = np.full(shape=(n,), fill_value=float(preds[0]), dtype="float64")
        else:
            raise ValueError("Length of model predictions must match number of rows in df or be scalar.")

    # Backtest params with sensible defaults
    initial_capital = float(backtest_params.get("initial_capital", 1_000_000.0))
    target_vol_daily = float(backtest_params.get("target_volatility", 0.02))  # daily target by default
    transaction_cost_bps = float(backtest_params.get("transaction_cost_bps", 5.0))  # bps
    slippage_bps = float(backtest_params.get("slippage_bps", 0.0))  # bps
    fixed_cost = float(backtest_params.get("fixed_cost", 0.0))  # absolute currency per trade
    leverage_cap = float(backtest_params.get("leverage_cap", 1.0))
    max_position_size = float(backtest_params.get("max_position_size", leverage_cap))  # fraction of capital
    min_position_size = float(backtest_params.get("min_position_size", 0.0))
    allow_short = bool(backtest_params.get("allow_short", False))
    annualization_factor = float(backtest_params.get("annualization_factor", 252.0 * 75.0))
    risk_free_rate = float(backtest_params.get("risk_free_rate", 0.0))

    # Derived conversions
    periods_per_day = annualization_factor / 252.0
    if periods_per_day <= 0:
        raise ValueError("annualization_factor must be > 0")
    # convert daily target_vol to per-period target vol
    target_vol_period = target_vol_daily / np.sqrt(periods_per_day)
    # convert preds (annualized) to per-period vol
    forecast_period_vol = preds / np.sqrt(annualization_factor)
    # floor tiny forecast vols to avoid extremely large allocations
    forecast_period_vol = np.maximum(forecast_period_vol, 1e-8)

    # desired position fraction per period
    raw_pos_frac = target_vol_period / forecast_period_vol
    pos_frac = raw_pos_frac.copy()
    pos_frac = np.clip(pos_frac, min_position_size, max_position_size)
    pos_frac = np.minimum(pos_frac, leverage_cap)
    if not allow_short:
        pos_frac = np.maximum(pos_frac, 0.0)

    # realized simple returns (close_t -> close_{t+1}); last period has no forward return (assume 0)
    realized_next = np.zeros(n, dtype="float64")
    realized_next[:-1] = close[1:] / close[:-1] - 1.0
    realized_next[-1] = 0.0

    # Initialize series
    equity = np.zeros(n, dtype="float64")
    equity[0] = initial_capital
    pos = np.zeros(n, dtype="float64")          # fraction of equity invested at start of period
    pos[0] = 0.0
    total_commissions = 0.0
    total_slippage = 0.0
    total_fixed = 0.0
    n_trades = 0

    # Iterate through periods
    for t in range(n - 1):
        desired_frac = float(pos_frac[t])
        prev_frac = float(pos[t])
        equity_t = float(equity[t])

        if equity_t <= 0:
            equity[t:] = 0.0
            pos[t:] = 0.0
            break

        delta_frac = desired_frac - prev_frac
        if abs(delta_frac) > 1e-12:
            trade_value = abs(delta_frac) * equity_t
            commission = trade_value * (transaction_cost_bps / 10000.0)
            slippage = trade_value * (slippage_bps / 10000.0)
            fixed = fixed_cost
            total_commissions += commission
            total_slippage += slippage
            total_fixed += fixed
            n_trades += 1
            costs = commission + slippage + fixed
        else:
            costs = 0.0

        # position value after rebalancing
        position_value = desired_frac * equity_t
        pnl = position_value * realized_next[t]
        net_pnl = pnl - costs

        equity[t + 1] = equity_t + net_pnl
        pos[t + 1] = desired_frac

        if not np.isfinite(equity[t + 1]) or equity[t + 1] <= 0:
            equity[t + 1:] = max(equity[t + 1], 0.0)
            pos[t + 1:] = 0.0
            break

    final_capital = float(equity[-1])
    total_periods = int(np.count_nonzero(~np.isnan(equity)))

    # compute period returns
    period_returns = np.zeros(max(0, n - 1), dtype="float64")
    for i in range(n - 1):
        if equity[i] > 0:
            period_returns[i] = (equity[i + 1] / equity[i]) - 1.0
        else:
            period_returns[i] = 0.0

    periods_per_year = annualization_factor
    mean_period_ret = float(np.mean(period_returns))
    std_period_ret = float(np.std(period_returns, ddof=1)) if len(period_returns) > 1 else 0.0
    if std_period_ret > 0:
        annualized_mean = mean_period_ret * periods_per_year
        annualized_std = std_period_ret * np.sqrt(periods_per_year)
        sharpe_ratio = float((annualized_mean - risk_free_rate) / annualized_std)
    else:
        sharpe_ratio = None

    # compute years spanned approx (in trading days)
    n_days = max(1.0, (len(period_returns) / periods_per_day))
    years = n_days / 252.0
    if years > 0 and initial_capital > 0:
        try:
            cagr = float((final_capital / initial_capital) ** (1.0 / years) - 1.0)
        except Exception:
            cagr = None
    else:
        cagr = None

    max_dd = _max_drawdown(equity)

    # Build summary (no equity curve)
    summary = {
        "final_capital": final_capital,
        "initial_capital": initial_capital,
        "return_pct": float((final_capital / initial_capital - 1.0) * 100.0),
        "n_periods": int(n - 1),
        "n_trades": int(n_trades),
        "avg_position_size": float(np.mean(pos)),
        "total_commissions": float(total_commissions),
        "total_slippage": float(total_slippage),
        "total_fixed_costs": float(total_fixed),
        "sharpe_ratio": sharpe_ratio,
        "cagr": cagr,
        "max_drawdown": max_dd
    }

    # create pandas Series for equity curve indexed by df index
    try:
        equity_index = df.index
        equity_series = pd.Series(equity, index=equity_index, name="equity")
    except Exception:
        equity_series = pd.Series(equity, name="equity")

    return summary, equity_series

def save_results(results: Dict, path: str):
    """
    Save summary JSON (does NOT include equity curve).
    """
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"📄 Backtest summary saved to: {path}")

def save_equity_curve(equity_series: pd.Series, path: str):
    """
    Save equity curve as CSV (datetime index preserved if present).
    """
    df_out = equity_series.to_frame().reset_index()
    df_out.columns = ["date", "equity"] if df_out.shape[1] == 2 else df_out.columns
    df_out.to_csv(path, index=False)
    print(f"📄 Equity curve saved to: {path}")

def save_equity_plot(equity_series: pd.Series, path: str, title: Optional[str] = None):
    """
    Save a PNG plot of the equity curve. Uses matplotlib, single plot.
    Does NOT set explicit colors/styles (respecting environment defaults).
    """
    if equity_series is None or equity_series.size == 0:
        print("⚠️ No equity series to plot.")
        return

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    # Plot time-series (matplotlib default colors)
    try:
        ax.plot(equity_series.index, equity_series.values)
    except Exception:
        # fallback to plotting by index (if index not datetime)
        ax.plot(equity_series.values)

    ax.set_title(title or "Equity Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.grid(True)

    # ensure target directory exists
    out_dir = os.path.dirname(path) if (path and isinstance(path, str)) else None
    if out_dir:
        import os as _os
        _os.makedirs(out_dir, exist_ok=True)

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"📄 Equity plot saved to: {path}")
