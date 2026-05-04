from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FORECAST_SYSTEMS = {
    "geoscf": "geoscf_pm25_fcst_tplus{h}_ug_m3",
    "naqfc": "naqfc_pm25_fcst_tplus{h}_ug_m3",
    "cams": "cams_pm25_fcst_tplus{h}_ug_m3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add cyclical time features and forecast ensemble summary features "
            "to the combined PM2.5 dataset."
        )
    )

    parser.add_argument(
        "--input-csv",
        default="data/2023_combined_plus_naqfc_cams.csv",
    )
    parser.add_argument(
        "--output-csv",
        default="data/2023_combined_plus_naqfc_cams_derived.csv",
    )
    parser.add_argument(
        "--horizons",
        default="1,2,3",
        help="Comma-separated forecast horizons to process.",
    )

    return parser.parse_args()


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "timestamp_utc" not in out.columns:
        raise KeyError("Input CSV must contain timestamp_utc.")

    out["timestamp_utc"] = pd.to_datetime(
        out["timestamp_utc"],
        utc=True,
        errors="coerce",
    )

    if out["timestamp_utc"].isna().any():
        raise ValueError("Some timestamp_utc values could not be parsed.")

    out["hour_utc"] = out["timestamp_utc"].dt.hour
    out["day_of_year"] = out["timestamp_utc"].dt.dayofyear

    hour_angle = 2.0 * np.pi * out["hour_utc"] / 24.0
    day_angle = 2.0 * np.pi * (out["day_of_year"] - 1) / 365.0

    out["sin_hour_utc"] = np.sin(hour_angle)
    out["cos_hour_utc"] = np.cos(hour_angle)

    out["sin_day_of_year"] = np.sin(day_angle)
    out["cos_day_of_year"] = np.cos(day_angle)

    return out


def existing_forecast_cols(df: pd.DataFrame, horizon: int) -> dict[str, str]:
    cols = {}

    for system, pattern in FORECAST_SYSTEMS.items():
        col = pattern.format(h=horizon)

        if col in df.columns:
            cols[system] = col
        else:
            print(f"Skipping missing forecast column: {col}")

    if len(cols) < 2:
        print(
            f"Warning: only found {len(cols)} forecast system(s) for tplus{horizon}. "
            "Ensemble summary features will be limited."
        )

    return cols


def add_ensemble_features(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    out = df.copy()

    for h in horizons:
        cols_by_system = existing_forecast_cols(out, h)
        forecast_cols = list(cols_by_system.values())

        if not forecast_cols:
            continue

        values = out[forecast_cols].astype(float)

        out[f"pm25_fcst_mean_tplus{h}"] = values.mean(axis=1, skipna=True)
        out[f"pm25_fcst_sd_tplus{h}"] = values.std(axis=1, skipna=True, ddof=1)
        out[f"pm25_fcst_min_tplus{h}"] = values.min(axis=1, skipna=True)
        out[f"pm25_fcst_max_tplus{h}"] = values.max(axis=1, skipna=True)
        out[f"pm25_fcst_range_tplus{h}"] = (
            out[f"pm25_fcst_max_tplus{h}"] - out[f"pm25_fcst_min_tplus{h}"]
        )

        print(
            f"Added ensemble summary features for tplus{h} "
            f"using systems: {', '.join(cols_by_system)}"
        )

    return out


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)

    horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]

    df = pd.read_csv(input_path)

    out = add_time_features(df)
    out = add_ensemble_features(out, horizons)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    print(f"Wrote {output_path}")
    print(f"Shape: {out.shape}")


if __name__ == "__main__":
    main()