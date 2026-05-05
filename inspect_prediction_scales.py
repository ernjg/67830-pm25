from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect the scale of observed PM2.5 and each model prediction."
    )

    parser.add_argument(
        "--results-dir",
        default="results/evaluate_2024_saved_summaries",
    )
    parser.add_argument(
        "--pred-file",
        default=None,
        help="Defaults to <results-dir>/predictions_residuals.csv",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Defaults to <results-dir>/prediction_scale_summary.csv",
    )

    return parser.parse_args()


def safe_corr(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")

    mask = a.notna() & b.notna()

    if mask.sum() < 2:
        return np.nan

    if a[mask].std(ddof=0) == 0 or b[mask].std(ddof=0) == 0:
        return np.nan

    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def summarize_model(group: pd.DataFrame) -> dict:
    y = group["y_true"].astype(float)
    pred = group["y_pred"].astype(float)
    resid = group["residual"].astype(float)
    abs_error = resid.abs()

    ratio = pred / y.replace(0, np.nan)

    return {
        "model": group["model"].iloc[0],
        "n": int(len(group)),

        "obs_mean": float(y.mean()),
        "obs_sd": float(y.std(ddof=0)),
        "obs_min": float(y.min()),
        "obs_q01": float(y.quantile(0.01)),
        "obs_q05": float(y.quantile(0.05)),
        "obs_median": float(y.median()),
        "obs_q95": float(y.quantile(0.95)),
        "obs_q99": float(y.quantile(0.99)),
        "obs_max": float(y.max()),

        "pred_mean": float(pred.mean()),
        "pred_sd": float(pred.std(ddof=0)),
        "pred_min": float(pred.min()),
        "pred_q01": float(pred.quantile(0.01)),
        "pred_q05": float(pred.quantile(0.05)),
        "pred_median": float(pred.median()),
        "pred_q95": float(pred.quantile(0.95)),
        "pred_q99": float(pred.quantile(0.99)),
        "pred_max": float(pred.max()),

        "mean_pred_minus_obs": float(pred.mean() - y.mean()),
        "pred_mean_div_obs_mean": float(pred.mean() / y.mean()) if y.mean() != 0 else np.nan,
        "pred_sd_div_obs_sd": float(pred.std(ddof=0) / y.std(ddof=0)) if y.std(ddof=0) != 0 else np.nan,

        "ratio_pred_to_obs_median": float(ratio.median()),
        "ratio_pred_to_obs_q05": float(ratio.quantile(0.05)),
        "ratio_pred_to_obs_q95": float(ratio.quantile(0.95)),

        "rmse": float(np.sqrt(np.mean(resid**2))),
        "mae": float(abs_error.mean()),
        "bias": float(resid.mean()),
        "resid_sd": float(resid.std(ddof=0)),
        "corr_pred_obs": safe_corr(pred, y),
        "corr_abs_error_obs": safe_corr(abs_error, y),

        "n_obs_le_zero": int((y <= 0).sum()),
        "n_pred_le_zero": int((pred <= 0).sum()),
        "can_log_obs_without_shift": bool((y > 0).all()),
        "can_log_pred_without_shift": bool((pred > 0).all()),
    }


def main() -> None:
    args = parse_args()

    results_dir = Path(args.results_dir)
    pred_file = Path(args.pred_file) if args.pred_file else results_dir / "predictions_residuals.csv"
    output_csv = Path(args.output_csv) if args.output_csv else results_dir / "prediction_scale_summary.csv"

    if not pred_file.exists():
        raise FileNotFoundError(f"Missing predictions file: {pred_file}")

    df = pd.read_csv(pred_file)

    required = {"model", "y_true", "y_pred", "residual"}
    missing = required - set(df.columns)

    if missing:
        raise KeyError(f"Prediction file missing required columns: {missing}")

    summaries = []

    for _, group in df.groupby("model", sort=False):
        summaries.append(summarize_model(group))

    summary = pd.DataFrame(summaries)

    summary = summary.sort_values("rmse").reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_csv, index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)

    print("\nPrediction scale summary:")
    print(
        summary[
            [
                "model",
                "n",
                "obs_mean",
                "pred_mean",
                "pred_mean_div_obs_mean",
                "obs_sd",
                "pred_sd",
                "pred_sd_div_obs_sd",
                "obs_min",
                "pred_min",
                "obs_max",
                "pred_max",
                "rmse",
                "mae",
                "bias",
                "corr_pred_obs",
                "n_obs_le_zero",
                "n_pred_le_zero",
                "can_log_obs_without_shift",
                "can_log_pred_without_shift",
            ]
        ].to_string(index=False)
    )

    print(f"\nWrote {output_csv}")

    if (summary["n_obs_le_zero"] > 0).any() or (summary["n_pred_le_zero"] > 0).any():
        print("\nWarning:")
        print("  Some observed or predicted values are <= 0.")
        print("  Plain log(pm2.5) is undefined for those values.")
        print("  Before using log(pm2.5), you need either to filter nonpositive rows")
        print("  or use a shifted transform such as log(pm2.5 + c).")
        print("  Your earlier 2024 observed minimum was negative, so this matters.")


if __name__ == "__main__":
    main()