from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


OBS_COL = "airnow_pm25_obs_ug_m3"
NAQFC_COL = "naqfc_pm25_fcst_tplus1_ug_m3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one-hour-ahead PM2.5 predictions for NAQFC and saved Bayesian model summaries. "
            "This does not fit Stan and does not make plots."
        )
    )

    parser.add_argument("--csv", default="data/2024_FINAL.csv")
    parser.add_argument("--output-dir", default="results/evaluate_2024_naqfc_bayes")

    parser.add_argument("--linreg-json", default="linreg_fit_summary.json")
    parser.add_argument("--sinusoid-json", default="sinusoid_fit_summary.json")

    parser.add_argument(
        "--linreg-forecast-col",
        default="geoscf_pm25_fcst_tplus1_ug_m3",
        help="Forecast column used to recreate the saved linear model. Use the column the model was trained with.",
    )
    parser.add_argument(
        "--sinusoid-forecast-col",
        default="geoscf_pm25_fcst_tplus1_ug_m3",
        help="Forecast column used to recreate the saved sinusoidal model. Use the column the model was trained with.",
    )

    parser.add_argument(
        "--linreg-forecast-scale",
        type=float,
        default=200.0,
        help="Divide linear model forecast input by this. Old model.py used 200.",
    )
    parser.add_argument(
        "--sinusoid-forecast-scale",
        type=float,
        default=1.0,
        help="Divide sinusoidal model forecast input by this. Default is no scaling.",
    )

    return parser.parse_args()


def load_json(path: str | Path) -> dict:
    with open(path, "r") as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        raise TypeError(f"{path} did not contain a JSON object.")

    return obj


def require_number(obj: dict, key: str) -> float:
    if key not in obj:
        raise KeyError(f"Missing key {key}. Available keys: {list(obj.keys())}")

    val = obj[key]

    if not isinstance(val, (int, float)):
        raise TypeError(f"{key} should be numeric, got {type(val).__name__}")

    return float(val)


def require_vector(obj: dict, key: str) -> np.ndarray:
    if key not in obj:
        raise KeyError(f"Missing key {key}. Available keys: {list(obj.keys())}")

    val = obj[key]

    if not isinstance(val, list):
        raise TypeError(f"{key} should be a list, got {type(val).__name__}")

    return np.asarray(val, dtype=float)


def load_data(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if "timestamp_utc" not in df.columns:
        raise KeyError("CSV must contain timestamp_utc.")
    if OBS_COL not in df.columns:
        raise KeyError(f"CSV must contain {OBS_COL}.")
    if NAQFC_COL not in df.columns:
        raise KeyError(f"CSV must contain {NAQFC_COL}.")

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")

    if df["timestamp_utc"].isna().any():
        raise ValueError("Some timestamp_utc values could not be parsed.")

    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    df["month_id"] = df["timestamp_utc"].dt.month.astype(int)

    return df


def compute_metrics(pred: pd.DataFrame) -> dict:
    y = pred["y_true"].astype(float).to_numpy()
    yhat = pred["y_pred"].astype(float).to_numpy()
    resid = y - yhat

    denom = np.sum((y - np.mean(y)) ** 2)

    return {
        "model": pred["model"].iloc[0],
        "n": int(len(pred)),
        "rmse": float(np.sqrt(np.mean(resid**2))),
        "mae": float(np.mean(np.abs(resid))),
        "bias": float(np.mean(resid)),
        "r2": float(1.0 - np.sum(resid**2) / denom) if denom > 0 else np.nan,
        "y_true_mean": float(np.mean(y)),
        "y_pred_mean": float(np.mean(yhat)),
        "resid_sd": float(np.std(resid, ddof=0)),
    }


def residual_autocorr(pred: pd.DataFrame, max_lag: int = 24) -> pd.DataFrame:
    resid = pred["residual"].astype(float).to_numpy()
    rows = []

    for lag in range(1, max_lag + 1):
        if len(resid) <= lag:
            corr = np.nan
        else:
            a = resid[:-lag]
            b = resid[lag:]
            corr = np.corrcoef(a, b)[0, 1] if np.std(a) > 0 and np.std(b) > 0 else np.nan

        rows.append(
            {
                "model": pred["model"].iloc[0],
                "lag_hours": lag,
                "residual_autocorrelation": corr,
            }
        )

    return pd.DataFrame(rows)


def evaluate_naqfc(df: pd.DataFrame) -> pd.DataFrame:
    pred = pd.DataFrame(
        {
            "timestamp_utc": df["timestamp_utc"],
            "target_time_utc": df["timestamp_utc"].shift(-1),
            "model": "NAQFC_pred",
            "y_true": df[OBS_COL].shift(-1),
            "y_pred": df[NAQFC_COL],
        }
    ).dropna()

    pred["residual"] = pred["y_true"] - pred["y_pred"]
    return pred


def evaluate_linreg(
    df: pd.DataFrame,
    summary: dict,
    forecast_col: str,
    forecast_scale: float,
) -> pd.DataFrame:
    if forecast_col not in df.columns:
        raise KeyError(f"Missing forecast column: {forecast_col}")

    beta = require_vector(summary, "beta")
    alpha = require_vector(summary, "alpha")
    sigma = require_number(summary, "sigma")

    mu_obj = summary.get("mu")
    if not isinstance(mu_obj, dict) or "intercept" not in mu_obj:
        raise KeyError("linreg JSON must contain mu.intercept")

    intercept = float(mu_obj["intercept"])

    if len(alpha) != len(beta):
        raise ValueError(f"alpha length {len(alpha)} != beta length {len(beta)}")

    K = len(alpha)

    y = df[OBS_COL].astype(float).to_numpy()
    fcst_base = df[forecast_col].astype(float).to_numpy() / forecast_scale

    tplus3_col = forecast_col.replace("tplus1", "tplus3")

    if tplus3_col in df.columns:
        extra = df[tplus3_col].astype(float).to_numpy()[-2:] / forecast_scale
    else:
        extra = fcst_base[-2:]

    fcst = np.concatenate([fcst_base, extra])
    rows = []

    for n_stan in range(K, len(df)):
        i = n_stan - 1

        y_window = y[n_stan - K : n_stan]
        fcst_window = fcst[n_stan - K + 2 : n_stan + 2]

        if len(y_window) != K or len(fcst_window) != K:
            continue
        if not np.isfinite(y_window).all() or not np.isfinite(fcst_window).all():
            continue
        if not np.isfinite(y[i + 1]):
            continue

        y_pred = intercept + float(np.dot(y_window, alpha)) + float(np.dot(fcst_window, beta))
        y_true = y[i + 1]

        rows.append(
            {
                "timestamp_utc": df["timestamp_utc"].iloc[i],
                "target_time_utc": df["timestamp_utc"].iloc[i + 1],
                "model": "Linear_pred",
                "y_true": y_true,
                "y_pred": y_pred,
                "residual": y_true - y_pred,
                "sigma_summary": sigma,
                "forecast_col": forecast_col,
                "forecast_scale": forecast_scale,
                "K": K,
            }
        )

    return pd.DataFrame(rows)


def evaluate_sinusoid(
    df: pd.DataFrame,
    summary: dict,
    forecast_col: str,
    forecast_scale: float,
) -> pd.DataFrame:
    if forecast_col not in df.columns:
        raise KeyError(f"Missing forecast column: {forecast_col}")

    eta_0 = require_number(summary, "eta_0")
    delta_0 = require_number(summary, "delta_0")
    phi = require_number(summary, "phi")
    sigma = require_number(summary, "sigma")

    eta_month = require_vector(summary, "eta_month")
    delta_month = require_vector(summary, "delta_month")
    omega_month = require_vector(summary, "omega_month")
    alpha = require_vector(summary, "alpha")

    if len(eta_month) != 12 or len(delta_month) != 12 or len(omega_month) != 12:
        raise ValueError("Expected eta_month, delta_month, and omega_month to each have length 12.")

    K = len(alpha)

    y = df[OBS_COL].astype(float).to_numpy()
    fcst = df[forecast_col].astype(float).to_numpy() / forecast_scale
    month_id = df["month_id"].astype(int).to_numpy()

    rows = []

    for n_stan in range(1, len(df) - K + 1):
        i = n_stan - 1

        fcst_window = fcst[i : i + K]

        if len(fcst_window) != K:
            continue
        if not np.isfinite(fcst_window).all():
            continue
        if not np.isfinite(y[i + 1]):
            continue

        m = month_id[i] - 1

        seasonal = (
            delta_0
            + delta_month[m]
            + (eta_0 + eta_month[m])
            * math.sin(((2.0 * math.pi) / (7.0 * 24.0) + omega_month[m]) * n_stan + phi)
        )

        y_pred = seasonal + float(np.dot(fcst_window, alpha))
        y_true = y[i + 1]

        rows.append(
            {
                "timestamp_utc": df["timestamp_utc"].iloc[i],
                "target_time_utc": df["timestamp_utc"].iloc[i + 1],
                "model": "Sinusoidal_pred",
                "y_true": y_true,
                "y_pred": y_pred,
                "residual": y_true - y_pred,
                "sigma_summary": sigma,
                "forecast_col": forecast_col,
                "forecast_scale": forecast_scale,
                "K": K,
            }
        )

    return pd.DataFrame(rows)


def print_sanity_report(df: pd.DataFrame, preds: list[pd.DataFrame]) -> None:
    print("\nSanity report")
    print("-------------")
    print(f"Observed PM2.5 mean: {df[OBS_COL].mean():.4g}")
    print(f"Observed PM2.5 sd:   {df[OBS_COL].std(ddof=0):.4g}")
    print(f"Observed PM2.5 min:  {df[OBS_COL].min():.4g}")
    print(f"Observed PM2.5 max:  {df[OBS_COL].max():.4g}")

    for pred in preds:
        model = pred["model"].iloc[0]
        print(f"\n{model}")
        print(f"  n:          {len(pred)}")
        print(f"  pred mean:  {pred['y_pred'].mean():.4g}")
        print(f"  pred sd:    {pred['y_pred'].std(ddof=0):.4g}")
        print(f"  pred min:   {pred['y_pred'].min():.4g}")
        print(f"  pred max:   {pred['y_pred'].max():.4g}")
        print(f"  resid mean: {pred['residual'].mean():.4g}")


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.csv)

    linreg_summary = load_json(args.linreg_json)
    sinusoid_summary = load_json(args.sinusoid_json)

    preds = [
        evaluate_naqfc(df),
        evaluate_linreg(
            df=df,
            summary=linreg_summary,
            forecast_col=args.linreg_forecast_col,
            forecast_scale=args.linreg_forecast_scale,
        ),
        evaluate_sinusoid(
            df=df,
            summary=sinusoid_summary,
            forecast_col=args.sinusoid_forecast_col,
            forecast_scale=args.sinusoid_forecast_scale,
        ),
    ]

    preds = [p for p in preds if not p.empty]

    metrics = pd.DataFrame([compute_metrics(p) for p in preds]).sort_values("rmse").reset_index(drop=True)
    residual_acf = pd.concat([residual_autocorr(p) for p in preds], ignore_index=True)
    predictions = pd.concat(preds, ignore_index=True)

    predictions.to_csv(output_dir / "predictions_residuals.csv", index=False)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    residual_acf.to_csv(output_dir / "residual_autocorr.csv", index=False)

    with open(output_dir / "evaluation_summary.json", "w") as f:
        json.dump(
            {
                "csv": str(args.csv),
                "linreg_json": str(args.linreg_json),
                "sinusoid_json": str(args.sinusoid_json),
                "linreg_forecast_col": args.linreg_forecast_col,
                "sinusoid_forecast_col": args.sinusoid_forecast_col,
                "linreg_forecast_scale": args.linreg_forecast_scale,
                "sinusoid_forecast_scale": args.sinusoid_forecast_scale,
                "models": ["NAQFC_pred", "Linear_pred", "Sinusoidal_pred"],
                "task": "one_hour_ahead",
            },
            f,
            indent=2,
        )

    print_sanity_report(df, preds)

    print("\nMetrics")
    print("-------")
    print(metrics.to_string(index=False))

    print("\nWrote:")
    print(f"  {output_dir / 'predictions_residuals.csv'}")
    print(f"  {output_dir / 'metrics.csv'}")
    print(f"  {output_dir / 'residual_autocorr.csv'}")
    print(f"  {output_dir / 'evaluation_summary.json'}")


if __name__ == "__main__":
    main()