from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine monthly PM2.5 CSVs into one complete 2023 dataset."
    )
    parser.add_argument(
        "--input-dir",
        default="data",
        help="Directory containing monthly CSVs.",
    )
    parser.add_argument(
        "--pattern",
        default="*_23.csv",
        help="Glob pattern for monthly files, e.g. '*_23.csv'.",
    )
    parser.add_argument(
        "--output-csv",
        default="data/pm25_2023_combined.csv",
    )
    parser.add_argument(
        "--output-meta-json",
        default="data/pm25_2023_combined_metadata.json",
    )
    parser.add_argument(
        "--allow-missing-hours",
        action="store_true",
        help="Allow the combined 2023 dataset to have missing hours.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    paths = sorted(input_dir.glob(args.pattern))

    if not paths:
        raise FileNotFoundError(f"No files matched {input_dir / args.pattern}")

    frames = []
    used_files = []

    for path in paths:
        df = pd.read_csv(path)

        if "timestamp_utc" not in df.columns:
            print(f"Skipping {path}: no timestamp_utc column")
            continue

        df["timestamp_utc"] = pd.to_datetime(
            df["timestamp_utc"],
            utc=True,
            errors="coerce",
        )

        df = df[df["timestamp_utc"].dt.year == 2023].copy()

        if df.empty:
            print(f"Skipping {path}: no 2023 rows")
            continue

        frames.append(df)
        used_files.append(str(path))

    if not frames:
        raise ValueError("No usable 2023 rows found.")

    combined = pd.concat(frames, ignore_index=True)

    duplicate_mask = combined.duplicated("timestamp_utc", keep=False)
    if duplicate_mask.any():
        n_dupes = combined.loc[duplicate_mask, "timestamp_utc"].nunique()
        print(f"Found {n_dupes} duplicated timestamps. Keeping first occurrence.")

    combined = (
        combined.sort_values("timestamp_utc")
        .drop_duplicates("timestamp_utc", keep="first")
        .reset_index(drop=True)
    )

    expected = pd.date_range(
        "2023-01-01T00:00:00Z",
        "2023-12-31T23:00:00Z",
        freq="h",
    )

    actual = pd.DatetimeIndex(combined["timestamp_utc"])
    missing = expected.difference(actual)

    if len(missing) > 0:
        message = f"Combined dataset is missing {len(missing)} hourly timestamps in 2023."

        if args.allow_missing_hours:
            print(message)
        else:
            examples = ", ".join(str(ts) for ts in missing[:10])
            raise ValueError(
                message
                + "\nFirst missing timestamps: "
                + examples
                + "\nRerun with --allow-missing-hours if this is expected."
            )

    combined["hour_utc"] = combined["timestamp_utc"].dt.hour
    combined["day_of_year"] = combined["timestamp_utc"].dt.dayofyear

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_csv, index=False)

    meta = {
        "description": "Combined hourly 2023 PM2.5 dataset built from monthly files.",
        "input_files": used_files,
        "output_csv": str(output_csv),
        "n_rows": int(len(combined)),
        "first_timestamp_utc": str(combined["timestamp_utc"].min()),
        "last_timestamp_utc": str(combined["timestamp_utc"].max()),
        "missing_2023_hours": int(len(missing)),
        "join_key": "timestamp_utc",
    }

    meta_path = Path(args.output_meta_json)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote {output_csv}")
    print(f"Wrote {meta_path}")
    print(f"Shape: {combined.shape}")


if __name__ == "__main__":
    main()