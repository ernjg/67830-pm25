from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODEL_ORDER = ["NAQFC_pred", "Linear_pred", "Sinusoidal_pred"]

DISPLAY_NAME = {
    "NAQFC_pred": "NAQFC",
    "Linear_pred": "Linear Model",
    "Sinusoidal_pred": "Sinusoidal Model",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize one-hour-ahead forecasts for NAQFC and Bayesian models on raw and log scales."
    )
    parser.add_argument(
        "--results-dir",
        default="results/evaluate_2024_naqfc_bayes",
    )
    parser.add_argument(
        "--pred-file",
        default=None,
        help="Defaults to <results-dir>/predictions_residuals.csv",
    )
    parser.add_argument(
        "--metrics-file",
        default=None,
        help="Defaults to <results-dir>/metrics.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to <results-dir>/plots",
    )
    parser.add_argument(
        "--window-start",
        default=None,
        help="Optional focused plot start, e.g. 2024-07-01",
    )
    parser.add_argument(
        "--window-end",
        default=None,
        help="Optional focused plot end, e.g. 2024-07-14",
    )
    parser.add_argument(
        "--context-days",
        type=int,
        default=7,
        help="If no window is given, use +/- this many days around the highest observed PM2.5.",
    )
    return parser.parse_args()


def pretty_name(model: str) -> str:
    return DISPLAY_NAME.get(model, model)


def load_data(pred_file: Path, metrics_file: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not pred_file.exists():
        raise FileNotFoundError(f"Missing file: {pred_file}")
    if not metrics_file.exists():
        raise FileNotFoundError(f"Missing file: {metrics_file}")

    preds = pd.read_csv(pred_file)
    metrics = pd.read_csv(metrics_file)

    preds["timestamp_utc"] = pd.to_datetime(preds["timestamp_utc"], utc=True, errors="coerce")
    preds["target_time_utc"] = pd.to_datetime(preds["target_time_utc"], utc=True, errors="coerce")
    preds["abs_error"] = preds["residual"].abs()

    preds = preds[preds["model"].isin(MODEL_ORDER)].copy()
    metrics = metrics[metrics["model"].isin(MODEL_ORDER)].copy()

    return preds, metrics


def choose_window(
    preds: pd.DataFrame,
    window_start: str | None,
    window_end: str | None,
    context_days: int,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    if window_start is not None and window_end is not None:
        return pd.Timestamp(window_start, tz="UTC"), pd.Timestamp(window_end, tz="UTC")

    ref = preds[preds["model"] == "Linear_pred"].copy()
    if ref.empty:
        ref = preds.copy()

    idx = ref["y_true"].idxmax()
    center = ref.loc[idx, "target_time_utc"]

    start = center - pd.Timedelta(days=context_days)
    end = center + pd.Timedelta(days=context_days)

    print("No window provided. Using a focused window around the highest observed PM2.5:")
    print(f"  start: {start}")
    print(f"  end:   {end}")

    return start, end


def get_observed_series(preds: pd.DataFrame, true_col: str) -> pd.DataFrame:
    obs = (
        preds[["target_time_utc", true_col]]
        .dropna()
        .drop_duplicates(subset=["target_time_utc"])
        .sort_values("target_time_utc")
        .reset_index(drop=True)
    )
    return obs


def compute_metrics_from_preds(preds: pd.DataFrame, true_col: str, pred_col: str) -> pd.DataFrame:
    rows = []

    for model in MODEL_ORDER:
        group = preds[preds["model"] == model].copy()
        if group.empty:
            continue

        y = group[true_col].to_numpy()
        yhat = group[pred_col].to_numpy()
        resid = y - yhat

        mask = np.isfinite(y) & np.isfinite(yhat)
        y = y[mask]
        yhat = yhat[mask]
        resid = resid[mask]

        if len(y) == 0:
            continue

        denom = np.sum((y - np.mean(y)) ** 2)

        rows.append(
            {
                "model": model,
                "display_name": pretty_name(model),
                "n": int(len(y)),
                "rmse": float(np.sqrt(np.mean(resid**2))),
                "mae": float(np.mean(np.abs(resid))),
                "bias": float(np.mean(resid)),
                "r2": float(1.0 - np.sum(resid**2) / denom) if denom > 0 else np.nan,
            }
        )

    return pd.DataFrame(rows)


def make_log_version(preds: pd.DataFrame) -> pd.DataFrame:
    log_preds = preds.copy()

    valid = (log_preds["y_true"] > 0) & (log_preds["y_pred"] > 0)

    log_preds["y_true_plot"] = np.where(valid, np.log(log_preds["y_true"]), np.nan)
    log_preds["y_pred_plot"] = np.where(valid, np.log(log_preds["y_pred"]), np.nan)
    log_preds["residual_plot"] = log_preds["y_true_plot"] - log_preds["y_pred_plot"]
    log_preds["abs_error_plot"] = np.abs(log_preds["residual_plot"])

    dropped = (
        log_preds.assign(valid_for_log=valid)
        .groupby("model", sort=False)["valid_for_log"]
        .agg(["sum", "count"])
        .reset_index()
    )
    dropped["excluded_for_log"] = dropped["count"] - dropped["sum"]

    print("\nRows usable for log(pm2.5) plots:")
    print(dropped[["model", "sum", "count", "excluded_for_log"]].to_string(index=False))

    return log_preds


def make_raw_version(preds: pd.DataFrame) -> pd.DataFrame:
    raw = preds.copy()
    raw["y_true_plot"] = raw["y_true"]
    raw["y_pred_plot"] = raw["y_pred"]
    raw["residual_plot"] = raw["residual"]
    raw["abs_error_plot"] = raw["abs_error"]
    return raw


def plot_metrics(metrics: pd.DataFrame, output_path: Path, figure_title: str, y_label: str) -> None:
    if metrics.empty:
        return

    order_map = {m: i for i, m in enumerate(MODEL_ORDER)}
    subset = metrics.copy()
    subset["order"] = subset["model"].map(order_map)
    subset = subset.sort_values("order")

    fig, axes = plt.subplots(3, 1, figsize=(8, 9))

    metric_specs = [
        ("rmse", "Root Mean Squared Error"),
        ("mae", "Mean Absolute Error"),
        ("bias", "Mean Error (Observed - Predicted)"),
    ]

    for ax, (metric, title) in zip(axes, metric_specs):
        if metric == "bias":
            ax.axhline(0.0, linewidth=1.0)

        ax.bar(subset["display_name"], subset[metric])
        ax.set_title(title)
        ax.set_ylabel(y_label)

    fig.suptitle(figure_title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_full_year_daily_overlay(
    preds: pd.DataFrame,
    output_path: Path,
    true_col: str,
    pred_col: str,
    y_label: str,
    figure_title: str,
) -> None:
    obs = get_observed_series(preds, true_col)
    if obs.empty:
        return

    obs_daily = (
        obs.set_index("target_time_utc")[[true_col]]
        .resample("1D")
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(obs_daily["target_time_utc"], obs_daily[true_col], label="Observed PM2.5", linewidth=1.8)

    for model in MODEL_ORDER:
        group = preds[preds["model"] == model].copy()
        group = group.dropna(subset=[pred_col])
        if group.empty:
            continue

        daily = (
            group.set_index("target_time_utc")[[pred_col]]
            .resample("1D")
            .mean()
            .reset_index()
        )

        ax.plot(
            daily["target_time_utc"],
            daily[pred_col],
            label=pretty_name(model),
            linewidth=1.0,
        )

    ax.set_title(figure_title)
    ax.set_xlabel("Date")
    ax.set_ylabel(y_label)
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_focused_hourly_overlay(
    preds: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    output_path: Path,
    true_col: str,
    pred_col: str,
    y_label: str,
    figure_title: str,
) -> None:
    subset = preds[(preds["target_time_utc"] >= start) & (preds["target_time_utc"] <= end)].copy()
    obs = get_observed_series(subset, true_col)
    if obs.empty:
        return

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(obs["target_time_utc"], obs[true_col], label="Observed PM2.5", linewidth=1.8)

    for model in MODEL_ORDER:
        group = subset[subset["model"] == model].sort_values("target_time_utc")
        group = group.dropna(subset=[pred_col])
        if group.empty:
            continue

        ax.plot(
            group["target_time_utc"],
            group[pred_col],
            label=pretty_name(model),
            linewidth=1.0,
        )

    ax.set_title(figure_title)
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel(y_label)
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_focused_residual_overlay(
    preds: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    output_path: Path,
    resid_col: str,
    y_label: str,
    figure_title: str,
) -> None:
    subset = preds[(preds["target_time_utc"] >= start) & (preds["target_time_utc"] <= end)].copy()

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.axhline(0.0, linewidth=1.0)

    for model in MODEL_ORDER:
        group = subset[subset["model"] == model].sort_values("target_time_utc")
        group = group.dropna(subset=[resid_col])
        if group.empty:
            continue

        ax.plot(
            group["target_time_utc"],
            group[resid_col],
            label=pretty_name(model),
            linewidth=1.0,
        )

    ax.set_title(figure_title)
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel(y_label)
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_residual_distribution_overlay(
    preds: pd.DataFrame,
    output_path: Path,
    resid_col: str,
    x_label: str,
    figure_title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    for model in MODEL_ORDER:
        group = preds[preds["model"] == model].dropna(subset=[resid_col])
        if group.empty:
            continue

        ax.hist(
            group[resid_col],
            bins=60,
            density=True,
            histtype="step",
            linewidth=1.5,
            label=pretty_name(model),
        )

    ax.axvline(0.0, linewidth=1.0)
    ax.set_title(figure_title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_shared_scatter_panels(
    preds: pd.DataFrame,
    output_path: Path,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    title: str,
    zero_line: bool = False,
    diagonal: bool = False,
) -> None:
    models = [m for m in MODEL_ORDER if m in set(preds["model"])]
    n = len(models)

    valid = preds[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        return

    fig, axes = plt.subplots(1, n, figsize=(4.8 * n, 4.8), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]

    if diagonal:
        lo = min(valid[x_col].min(), valid[y_col].min())
        hi = max(valid[x_col].max(), valid[y_col].max())

    for ax, model in zip(axes, models):
        group = preds[preds["model"] == model].dropna(subset=[x_col, y_col])
        if group.empty:
            continue

        ax.scatter(group[x_col], group[y_col], s=8, alpha=0.35)

        if zero_line:
            ax.axhline(0.0, linewidth=1.0)

        if diagonal:
            ax.plot([lo, hi], [lo, hi], linewidth=1.0)

        ax.set_title(pretty_name(model))
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_error_by_observed_bin(
    preds: pd.DataFrame,
    output_path: Path,
    true_col: str,
    abs_error_col: str,
    resid_col: str,
    xlabel: str,
    mae_ylabel: str,
    resid_ylabel: str,
    title_top: str,
    title_bottom: str,
) -> pd.DataFrame:
    work = preds.dropna(subset=[true_col, abs_error_col, resid_col]).copy()
    if work.empty:
        return pd.DataFrame()

    work["obs_bin"] = pd.qcut(work[true_col], q=10, duplicates="drop")

    grouped = (
        work.groupby(["model", "obs_bin"], observed=False)
        .agg(
            mean_observed=(true_col, "mean"),
            mean_abs_error=(abs_error_col, "mean"),
            mean_residual=(resid_col, "mean"),
            count=(true_col, "size"),
        )
        .reset_index()
    )

    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

    for model in MODEL_ORDER:
        group = grouped[grouped["model"] == model].sort_values("mean_observed")
        if group.empty:
            continue

        axes[0].plot(
            group["mean_observed"],
            group["mean_abs_error"],
            marker="o",
            linewidth=1.2,
            label=pretty_name(model),
        )

        axes[1].plot(
            group["mean_observed"],
            group["mean_residual"],
            marker="o",
            linewidth=1.2,
            label=pretty_name(model),
        )

    axes[0].set_title(title_top)
    axes[0].set_ylabel(mae_ylabel)
    axes[0].legend(fontsize=9)

    axes[1].axhline(0.0, linewidth=1.0)
    axes[1].set_title(title_bottom)
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(resid_ylabel)
    axes[1].legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return grouped


def make_plot_set(
    preds: pd.DataFrame,
    metrics: pd.DataFrame,
    output_dir: Path,
    prefix: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    true_col: str,
    pred_col: str,
    resid_col: str,
    abs_error_col: str,
    value_label: str,
    resid_label: str,
    abs_error_label: str,
    metrics_title: str,
    metrics_ylabel: str,
    full_year_title: str,
    focused_forecast_title: str,
    focused_resid_title: str,
    resid_dist_title: str,
    resid_dist_xlabel: str,
    obs_vs_pred_title: str,
    obs_vs_pred_xlabel: str,
    obs_vs_pred_ylabel: str,
    resid_vs_obs_title: str,
    resid_vs_obs_xlabel: str,
    resid_vs_obs_ylabel: str,
    abs_err_vs_obs_title: str,
    abs_err_vs_obs_xlabel: str,
    abs_err_vs_obs_ylabel: str,
    error_bin_top_title: str,
    error_bin_bottom_title: str,
    error_bin_xlabel: str,
    error_bin_mae_ylabel: str,
    error_bin_resid_ylabel: str,
) -> None:
    plot_metrics(metrics, output_dir / f"{prefix}_forecast_metrics.png", metrics_title, metrics_ylabel)

    plot_full_year_daily_overlay(
        preds=preds,
        output_path=output_dir / f"{prefix}_full_year_daily_overlay.png",
        true_col=true_col,
        pred_col=pred_col,
        y_label=value_label,
        figure_title=full_year_title,
    )

    plot_focused_hourly_overlay(
        preds=preds,
        start=start,
        end=end,
        output_path=output_dir / f"{prefix}_focused_hourly_overlay.png",
        true_col=true_col,
        pred_col=pred_col,
        y_label=value_label,
        figure_title=focused_forecast_title,
    )

    plot_focused_residual_overlay(
        preds=preds,
        start=start,
        end=end,
        output_path=output_dir / f"{prefix}_focused_residual_overlay.png",
        resid_col=resid_col,
        y_label=resid_label,
        figure_title=focused_resid_title,
    )

    plot_residual_distribution_overlay(
        preds=preds,
        output_path=output_dir / f"{prefix}_residual_distribution_overlay.png",
        resid_col=resid_col,
        x_label=resid_dist_xlabel,
        figure_title=resid_dist_title,
    )

    plot_shared_scatter_panels(
        preds=preds,
        output_path=output_dir / f"{prefix}_observed_vs_predicted_shared_axes.png",
        x_col=true_col,
        y_col=pred_col,
        x_label=obs_vs_pred_xlabel,
        y_label=obs_vs_pred_ylabel,
        title=obs_vs_pred_title,
        diagonal=True,
    )

    plot_shared_scatter_panels(
        preds=preds,
        output_path=output_dir / f"{prefix}_residual_vs_observed_shared_axes.png",
        x_col=true_col,
        y_col=resid_col,
        x_label=resid_vs_obs_xlabel,
        y_label=resid_vs_obs_ylabel,
        title=resid_vs_obs_title,
        zero_line=True,
    )

    plot_shared_scatter_panels(
        preds=preds,
        output_path=output_dir / f"{prefix}_absolute_error_vs_observed_shared_axes.png",
        x_col=true_col,
        y_col=abs_error_col,
        x_label=abs_err_vs_obs_xlabel,
        y_label=abs_err_vs_obs_ylabel,
        title=abs_err_vs_obs_title,
    )

    error_by_bin = plot_error_by_observed_bin(
        preds=preds,
        output_path=output_dir / f"{prefix}_error_by_observed_bin.png",
        true_col=true_col,
        abs_error_col=abs_error_col,
        resid_col=resid_col,
        xlabel=error_bin_xlabel,
        mae_ylabel=error_bin_mae_ylabel,
        resid_ylabel=error_bin_resid_ylabel,
        title_top=error_bin_top_title,
        title_bottom=error_bin_bottom_title,
    )
    if not error_by_bin.empty:
        error_by_bin.to_csv(output_dir / f"{prefix}_error_by_observed_bin.csv", index=False)


def main() -> None:
    args = parse_args()

    results_dir = Path(args.results_dir)
    pred_file = Path(args.pred_file) if args.pred_file else results_dir / "predictions_residuals.csv"
    metrics_file = Path(args.metrics_file) if args.metrics_file else results_dir / "metrics.csv"
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    preds, raw_metrics_from_file = load_data(pred_file, metrics_file)
    start, end = choose_window(preds, args.window_start, args.window_end, args.context_days)

    raw_preds = make_raw_version(preds)
    raw_metrics = compute_metrics_from_preds(raw_preds, "y_true_plot", "y_pred_plot")

    log_preds = make_log_version(preds)
    log_metrics = compute_metrics_from_preds(log_preds, "y_true_plot", "y_pred_plot")

    raw_metrics.to_csv(output_dir / "raw_metrics.csv", index=False)
    log_metrics.to_csv(output_dir / "log_metrics.csv", index=False)

    make_plot_set(
        preds=raw_preds,
        metrics=raw_metrics,
        output_dir=output_dir,
        prefix="raw",
        start=start,
        end=end,
        true_col="y_true_plot",
        pred_col="y_pred_plot",
        resid_col="residual_plot",
        abs_error_col="abs_error_plot",
        value_label="PM2.5 (µg/m³)",
        resid_label="Residual (Observed - Predicted, µg/m³)",
        abs_error_label="Absolute Error (µg/m³)",
        metrics_title="One-Hour-Ahead Forecast Performance",
        metrics_ylabel="PM2.5 (µg/m³)",
        full_year_title="Daily Mean One-Hour-Ahead Forecasts Across 2024",
        focused_forecast_title="Hourly One-Hour-Ahead Forecasts in a High-PM2.5 Window",
        focused_resid_title="Residuals in a High-PM2.5 Window",
        resid_dist_title="Residual Distributions",
        resid_dist_xlabel="Residual (Observed - Predicted, µg/m³)",
        obs_vs_pred_title="Observed vs Predicted PM2.5",
        obs_vs_pred_xlabel="Observed PM2.5 (µg/m³)",
        obs_vs_pred_ylabel="Predicted PM2.5 (µg/m³)",
        resid_vs_obs_title="Residual vs Observed PM2.5",
        resid_vs_obs_xlabel="Observed PM2.5 (µg/m³)",
        resid_vs_obs_ylabel="Residual (Observed - Predicted, µg/m³)",
        abs_err_vs_obs_title="Absolute Error vs Observed PM2.5",
        abs_err_vs_obs_xlabel="Observed PM2.5 (µg/m³)",
        abs_err_vs_obs_ylabel="Absolute Error (µg/m³)",
        error_bin_top_title="Mean Absolute Error by Observed PM2.5 Level",
        error_bin_bottom_title="Mean Residual by Observed PM2.5 Level",
        error_bin_xlabel="Mean Observed PM2.5 Within Bin (µg/m³)",
        error_bin_mae_ylabel="Mean Absolute Error (µg/m³)",
        error_bin_resid_ylabel="Mean Residual (µg/m³)",
    )

    make_plot_set(
        preds=log_preds,
        metrics=log_metrics,
        output_dir=output_dir,
        prefix="log",
        start=start,
        end=end,
        true_col="y_true_plot",
        pred_col="y_pred_plot",
        resid_col="residual_plot",
        abs_error_col="abs_error_plot",
        value_label="log(PM2.5)",
        resid_label="log(Observed PM2.5) - log(Predicted PM2.5)",
        abs_error_label="Absolute Log Error",
        metrics_title="One-Hour-Ahead Forecast Performance on the log(PM2.5) Scale",
        metrics_ylabel="log(PM2.5)",
        full_year_title="Daily Mean One-Hour-Ahead Forecasts Across 2024 on the log(PM2.5) Scale",
        focused_forecast_title="Hourly One-Hour-Ahead Forecasts in a High-PM2.5 Window on the log(PM2.5) Scale",
        focused_resid_title="Residuals in a High-PM2.5 Window on the log(PM2.5) Scale",
        resid_dist_title="Residual Distributions on the log(PM2.5) Scale",
        resid_dist_xlabel="log(Observed PM2.5) - log(Predicted PM2.5)",
        obs_vs_pred_title="Observed vs Predicted log(PM2.5)",
        obs_vs_pred_xlabel="Observed log(PM2.5)",
        obs_vs_pred_ylabel="Predicted log(PM2.5)",
        resid_vs_obs_title="Residual vs Observed log(PM2.5)",
        resid_vs_obs_xlabel="Observed log(PM2.5)",
        resid_vs_obs_ylabel="log(Observed PM2.5) - log(Predicted PM2.5)",
        abs_err_vs_obs_title="Absolute Log Error vs Observed log(PM2.5)",
        abs_err_vs_obs_xlabel="Observed log(PM2.5)",
        abs_err_vs_obs_ylabel="Absolute Log Error",
        error_bin_top_title="Mean Absolute Log Error by Observed log(PM2.5)",
        error_bin_bottom_title="Mean Log Residual by Observed log(PM2.5)",
        error_bin_xlabel="Mean Observed log(PM2.5) Within Bin",
        error_bin_mae_ylabel="Mean Absolute Log Error",
        error_bin_resid_ylabel="Mean Log Residual",
    )

    raw_metrics_from_file.to_csv(output_dir / "raw_metrics_from_evaluator.csv", index=False)

    print(f"Wrote plots to {output_dir}")
    print(f"Focused window: {start} to {end}")
    print("\nSaved raw-scale figures with prefix: raw_")
    print("Saved log-scale figures with prefix: log_")
    print("\nSaved tables:")
    print("  raw_metrics.csv")
    print("  log_metrics.csv")


if __name__ == "__main__":
    main()