# main.py
import argparse
import os
from copy import deepcopy
import yaml
from typing import Dict

from src import data_preprocessing, feature_engineering, models as model_module, evaluation, backtest

# ---------------------------
# Config loader with profiles
# ---------------------------
def _deep_update(d: dict, u: dict) -> dict:
    """Recursively update dict d with u (returns new dict)."""
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            d[k] = _deep_update(d.get(k, {}), v)
        else:
            d[k] = deepcopy(v)
    return d

def load_params(path: str = "config/params.yaml", profile: str = None) -> Dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    base = cfg.get("base", {})
    if profile:
        profiles = cfg.get("profiles", {})
        prof = profiles.get(profile)
        if prof is None:
            raise ValueError(f"Profile '{profile}' not found in {path}. Available: {list(profiles.keys())}")
        merged = _deep_update(deepcopy(base), prof)
        return merged
    return base

# ---------------------------
# Utility helpers
# ---------------------------
def ensure_dirs(paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def build_external_index_paths(external_path: str, index_symbols: list) -> Dict[str, str]:
    res = {}
    for idx in index_symbols or []:
        p = os.path.join(external_path, f"{idx}.csv")
        if os.path.exists(p):
            res[idx] = p
        else:
            print(f"⚠️ External index file not found for {idx}: {p} (skipping)")
    return res

# ---------------------------
# Main pipeline
# ---------------------------
def run_pipeline(params: dict, override_symbol: str = None):
    # Resolve paths & symbol
    data_cfg = params.get("data", {})
    output_cfg = params.get("output", {})
    run_cfg = params.get("run", {})

    raw_dir = data_cfg.get("raw_path", "data/raw")
    interim_dir = data_cfg.get("interim_path", "data/interim")
    features_dir = data_cfg.get("features_path", "data/features")
    external_dir = data_cfg.get("external_path", "data/external")
    models_dir = output_cfg.get("models_path", "models")
    figures_dir = output_cfg.get("figures_path", "results/figures")
    metrics_dir = output_cfg.get("metrics_path", "results/metrics")
    backtest_dir = output_cfg.get("backtest_path", "results/backtest")

    ensure_dirs([interim_dir, features_dir, figures_dir, metrics_dir, backtest_dir, models_dir])

    symbol = override_symbol if override_symbol else data_cfg.get("symbol", "ICICIBANK")
    print(f"🔸 Running pipeline for symbol: {symbol}")

    raw_data_path = os.path.join(raw_dir, f"{symbol}.csv")
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Raw CSV not found: {raw_data_path}")

    # 1) Preprocessing
    print("🔹 Preprocessing data...")
    interim_df = data_preprocessing.run(
        raw_path=raw_data_path,
        prediction_horizon_bars=data_cfg.get("prediction_horizon_bars", 12),
        annualization_factor=data_cfg.get("annualization_factor", 252 * 75),
        set_index=True
    )

    if run_cfg.get("save_interim", True):
        interim_save_path = os.path.join(interim_dir, f"{symbol}_interim.parquet")
        interim_df.to_parquet(interim_save_path, engine=run_cfg.get("parquet_engine", "pyarrow"))
        print(f"📥 Interim saved -> {interim_save_path}")

    # 2) Feature engineering
    print("🔹 Feature engineering...")
    ext_index_paths = build_external_index_paths(external_dir, data_cfg.get("index_symbols", []))
    features_df = feature_engineering.run(
        df=interim_df,
        params=params.get("features", {}),
        external_index_paths=ext_index_paths if ext_index_paths else None
    )

    if run_cfg.get("save_features", True):
        features_save_path = os.path.join(features_dir, f"{symbol}_features.parquet")
        features_df.to_parquet(features_save_path, engine=run_cfg.get("parquet_engine", "pyarrow"))
        print(f"📥 Features saved -> {features_save_path}")

    # 3) Model training
    print("🔹 Training models...")
    trained_models = {}
    models_cfg = params.get("models", {})
    for model_name, model_params in models_cfg.items():
        try:
            print(f"  • Training {model_name} ...")
            mdl = model_module.train(model_name, features_df, model_params or {}, horizon=data_cfg.get("prediction_horizon_bars", 12))
            trained_models[model_name] = mdl
            model_path = os.path.join(models_dir, f"{symbol}_{model_name}.pkl")
            model_module.save_model(mdl, model_path)
        except Exception as e:
            print(f"⚠️ Skipping model '{model_name}' due to error: {e}")

    # 4) Evaluation
    print("🔹 Evaluating models...")
    eval_metrics = params.get("evaluation", {}).get("metrics", ["rmse", "mae", "mape"])
    for model_name, mdl in trained_models.items():
        try:
            metrics = evaluation.run(model=mdl, df=features_df, metrics=eval_metrics)
            metrics_path = os.path.join(metrics_dir, f"{symbol}_{model_name}_metrics.json")
            evaluation.save_metrics(metrics, metrics_path)
        except Exception as e:
            print(f"⚠️ Evaluation failed for {model_name}: {e}")

    # 5) Backtest (summary JSON + separate equity CSV + PNG plot)
    print("🔹 Running backtests...")
    backtest_params = params.get("evaluation", {}).get("economic_utility", {}).get("backtest", {})
    save_equity_curve_flag = backtest_params.get("save_equity_curve", True)
    save_equity_plot_flag = backtest_params.get("save_equity_plot", True)

    for model_name, mdl in trained_models.items():
        try:
            summary, equity_series = backtest.run(model=mdl, df=features_df, backtest_params=backtest_params)

            # save summary JSON
            bt_summary_path = os.path.join(backtest_dir, f"{symbol}_{model_name}_backtest.json")
            backtest.save_results(summary, bt_summary_path)

            # save equity curve CSV
            if save_equity_curve_flag and equity_series is not None:
                eq_path = os.path.join(backtest_dir, f"{symbol}_{model_name}_equity_curve.csv")
                backtest.save_equity_curve(equity_series, eq_path)

            # save equity plot PNG into figures_dir
            if save_equity_plot_flag and equity_series is not None:
                fig_path = os.path.join(figures_dir, f"{symbol}_{model_name}_equity_curve.png")
                backtest.save_equity_plot(equity_series, fig_path, title=f"{symbol} - {model_name} Equity Curve")

        except Exception as e:
            print(f"⚠️ Backtest failed for {model_name}: {e}")

    print("✅ Pipeline finished.")

# ---------------------------
# CLI
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Run intraday volatility forecasting pipeline")
    p.add_argument("--config", type=str, default="config/params.yaml", help="Path to params.yaml")
    p.add_argument("--profile", type=str, default=None, help="Config profile to load (e.g., debug, production)")
    p.add_argument("--symbol", type=str, default=None, help="Override symbol in config (e.g., ICICIBANK)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    params = load_params(path=args.config, profile=args.profile)
    run_pipeline(params, override_symbol=args.symbol)
