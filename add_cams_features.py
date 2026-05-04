from __future__ import annotations

import argparse
import calendar
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


LA_URBAN_AREA_LAT = 33.9850
LA_URBAN_AREA_LON = -118.1224

CAMS_DATASET = "cams-global-atmospheric-composition-forecasts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download CAMS global PM2.5 forecasts from ADS in monthly batches "
            "and add non-leaking t+1/t+2/t+3 forecast features."
        )
    )

    parser.add_argument("--input-csv", default="data/2023_combined_plus_naqfc.csv")
    parser.add_argument("--output-csv", default="data/2023_combined_plus_naqfc_cams.csv")

    parser.add_argument(
        "--cache-dir",
        default="data/cache/cams_monthly_grib",
        help="Temporary cache for downloaded monthly CAMS GRIB files.",
    )
    parser.add_argument(
        "--forecast-table-cache",
        default="data/cache/cams_monthly_forecast_table_checkpoint.csv",
        help="Checkpoint CSV of extracted local CAMS forecast rows.",
    )

    parser.add_argument("--lat", type=float, default=LA_URBAN_AREA_LAT)
    parser.add_argument("--lon", type=float, default=LA_URBAN_AREA_LON)

    parser.add_argument(
        "--area-padding-deg",
        type=float,
        default=1.0,
        help="Download small lat/lon box around the target point.",
    )
    parser.add_argument(
        "--cycles",
        default="00,12",
        help="Comma-separated CAMS issue cycles.",
    )
    parser.add_argument(
        "--horizons",
        default="1,2,3",
        help="Comma-separated forecast horizons to add.",
    )
    parser.add_argument(
        "--max-lead-hour",
        type=int,
        default=15,
        help="Largest lead hour to request. Default 15 covers t+1/t+2/t+3 between 12-hour cycles.",
    )

    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)

    parser.add_argument(
        "--keep-cams-cache",
        action="store_true",
        help="Keep downloaded monthly CAMS GRIB files after extraction.",
    )

    return parser.parse_args()


def parse_utc_timestamp(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def haversine_km(
    lat1: float,
    lon1: float,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    earth_radius_km = 6371.0

    lat1r = np.radians(lat1)
    lon1r = np.radians(lon1)
    lat2r = np.radians(lat2)
    lon2r = np.radians(lon2)

    dlat = lat2r - lat1r
    dlon = lon2r - lon1r

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    )

    return 2.0 * earth_radius_km * np.arcsin(np.sqrt(a))


def cams_area(lat: float, lon: float, pad: float) -> list[float]:
    # ADS order: North, West, South, East
    return [lat + pad, lon - pad, lat - pad, lon + pad]


def month_start(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(year=ts.year, month=ts.month, day=1, tz="UTC")


def month_end(ts: pd.Timestamp) -> pd.Timestamp:
    last_day = calendar.monthrange(ts.year, ts.month)[1]
    return pd.Timestamp(year=ts.year, month=ts.month, day=last_day, hour=23, tz="UTC")


def iter_months(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    # Include previous month if needed because early Jan 1 can use previous-day forecasts.
    cur = month_start(start - pd.Timedelta(days=1))
    final = month_start(end)

    months = []
    while cur <= final:
        months.append((cur, month_end(cur)))
        if cur.month == 12:
            cur = pd.Timestamp(year=cur.year + 1, month=1, day=1, tz="UTC")
        else:
            cur = pd.Timestamp(year=cur.year, month=cur.month + 1, day=1, tz="UTC")

    return months


def cams_month_grib_path(cache_dir: Path, start_month: pd.Timestamp) -> Path:
    ym = start_month.strftime("%Y%m")
    return cache_dir / f"cams_pm25_{ym}.grib"


def retrieve_cams_month(
    cache_dir: Path,
    month_start_ts: pd.Timestamp,
    month_end_ts: pd.Timestamp,
    lat: float,
    lon: float,
    area_padding_deg: float,
    cycles: list[int],
    max_lead_hour: int,
) -> Path:
    import cdsapi

    grib_path = cams_month_grib_path(cache_dir, month_start_ts)

    if grib_path.exists() and grib_path.stat().st_size > 0:
        return grib_path

    date_range = f"{month_start_ts.strftime('%Y-%m-%d')}/{month_end_ts.strftime('%Y-%m-%d')}"
    times = [f"{cycle:02d}:00" for cycle in cycles]

    request = {
        "date": [date_range],
        "time": times,
        "leadtime_hour": [str(h) for h in range(0, max_lead_hour + 1)],
        "type": ["forecast"],
        "variable": ["particulate_matter_2.5um"],
        "data_format": "grib",
        "area": cams_area(lat, lon, area_padding_deg),
    }

    grib_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Requesting CAMS month {month_start_ts.strftime('%Y-%m')}, "
        f"cycles={times}, lead hours 0..{max_lead_hour}"
    )

    client = cdsapi.Client()
    client.retrieve(CAMS_DATASET, request).download(str(grib_path))

    return grib_path


def open_cams_grib(grib_path: Path) -> xr.Dataset:
    return xr.open_dataset(
        grib_path,
        engine="cfgrib",
        backend_kwargs={
            "indexpath": "",
            "errors": "ignore",
        },
    )


def extract_rows_from_month_grib(
    grib_path: Path,
    lat: float,
    lon: float,
) -> list[dict]:
    rows = []

    ds = open_cams_grib(grib_path)

    try:
        if not ds.data_vars:
            raise ValueError(f"No data variables found in {grib_path}")

        var_name = list(ds.data_vars)[0]
        da = ds[var_name]

        if "latitude" not in ds.coords or "longitude" not in ds.coords:
            raise ValueError(
                f"Could not find latitude/longitude coordinates in {grib_path}. "
                f"Available coords: {list(ds.coords)}"
            )

        lats = ds["latitude"].values
        lons = ds["longitude"].values

        target_lon = lon
        if np.nanmax(lons) > 180 and target_lon < 0:
            target_lon = target_lon % 360

        if lats.ndim == 1 and lons.ndim == 1:
            lon2d, lat2d = np.meshgrid(lons, lats)
            dists = haversine_km(lat, target_lon, lat2d, lon2d)
            lat_idx, lon_idx = np.unravel_index(np.nanargmin(dists), dists.shape)

            nearest_lat = float(lats[lat_idx])
            nearest_lon = float(lons[lon_idx])

            spatial_indexer = {
                "latitude": lat_idx,
                "longitude": lon_idx,
            }

        else:
            dists = haversine_km(lat, target_lon, lats, lons)
            nearest_ij = np.unravel_index(np.nanargmin(dists), dists.shape)

            nearest_lat = float(lats[nearest_ij])
            nearest_lon = float(lons[nearest_ij])

            y_dim, x_dim = ds["latitude"].dims
            y_idx, x_idx = nearest_ij

            spatial_indexer = {
                y_dim: y_idx,
                x_dim: x_idx,
            }

        print(
            f"Nearest CAMS grid point: lat={nearest_lat:.4f}, "
            f"lon={nearest_lon:.4f}"
        )

        selected = da.isel(spatial_indexer)

        for dim in list(selected.dims):
            if selected.sizes[dim] == 1:
                selected = selected.squeeze(dim=dim, drop=True)

        if "time" not in ds.coords:
            raise ValueError(f"CAMS GRIB missing time coordinate. Coords: {list(ds.coords)}")

        issue_times = pd.to_datetime(ds["time"].values, utc=True)

        if "step" in ds.coords:
            steps = pd.to_timedelta(ds["step"].values)
        elif "valid_time" in ds.coords and len(issue_times) == 1:
            valid_times = pd.to_datetime(ds["valid_time"].values, utc=True)
            steps = pd.to_timedelta(valid_times - issue_times[0])
        else:
            raise ValueError(
                f"Could not infer CAMS forecast steps. Coords: {list(ds.coords)}"
            )

        values = np.asarray(selected.values, dtype=float)

        # Expected shape after spatial selection: time x step
        if values.ndim == 1:
            if len(issue_times) == 1:
                values = values.reshape(1, -1)
            elif len(steps) == 1:
                values = values.reshape(-1, 1)

        if values.ndim != 2:
            raise ValueError(
                f"Expected selected CAMS values to be 2D time x step, got shape {values.shape}. "
                f"Selected dims: {selected.dims}. Coords: {list(ds.coords)}"
            )

        if values.shape[0] != len(issue_times) or values.shape[1] != len(steps):
            # Sometimes cfgrib orders dims as step x time.
            if values.shape[0] == len(steps) and values.shape[1] == len(issue_times):
                values = values.T
            else:
                raise ValueError(
                    f"Value/time mismatch: values shape {values.shape}, "
                    f"{len(issue_times)} issue times, {len(steps)} steps. "
                    f"Selected dims: {selected.dims}"
                )

        # CAMS PM2.5 is kg/m^3. Convert to ug/m^3.
        values_ug_m3 = values * 1e9

        for i, issue_time in enumerate(issue_times):
            issue_time = pd.Timestamp(issue_time)
            if issue_time.tzinfo is None:
                issue_time = issue_time.tz_localize("UTC")
            else:
                issue_time = issue_time.tz_convert("UTC")

            for j, step in enumerate(steps):
                valid_time = issue_time + step
                forecast_hour = int(round(pd.Timedelta(step).total_seconds() / 3600))

                rows.append(
                    {
                        "issue_time_utc": issue_time,
                        "valid_time_utc": valid_time,
                        "forecast_hour": forecast_hour,
                        "cams_pm25_ug_m3": float(values_ug_m3[i, j]),
                        "nearest_grid_lat": nearest_lat,
                        "nearest_grid_lon": nearest_lon,
                        "source_file": str(grib_path),
                    }
                )

        return rows

    finally:
        ds.close()


def write_checkpoint_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def load_checkpoint(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)

    if df.empty:
        return df

    df["issue_time_utc"] = pd.to_datetime(df["issue_time_utc"], utc=True)
    df["valid_time_utc"] = pd.to_datetime(df["valid_time_utc"], utc=True)

    print(f"Loaded {len(df)} CAMS checkpoint rows from {path}")

    return df


def build_forecast_table(
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_dir: Path,
    forecast_table_cache: Path,
    lat: float,
    lon: float,
    area_padding_deg: float,
    cycles: list[int],
    max_lead_hour: int,
    keep_cams_cache: bool,
) -> pd.DataFrame:
    months = iter_months(start, end)

    checkpoint_df = load_checkpoint(forecast_table_cache)
    rows = checkpoint_df.to_dict("records") if not checkpoint_df.empty else []

    done_months = set()
    if not checkpoint_df.empty and "source_file" in checkpoint_df.columns:
        done_months = {
            Path(p).stem.replace("cams_pm25_", "")
            for p in checkpoint_df["source_file"].dropna().unique()
        }

    for month_start_ts, month_end_ts in months:
        month_key = month_start_ts.strftime("%Y%m")

        if month_key in done_months:
            print(f"Skipping already extracted CAMS month: {month_key}")
            continue

        grib_path = None

        try:
            grib_path = retrieve_cams_month(
                cache_dir=cache_dir,
                month_start_ts=month_start_ts,
                month_end_ts=month_end_ts,
                lat=lat,
                lon=lon,
                area_padding_deg=area_padding_deg,
                cycles=cycles,
                max_lead_hour=max_lead_hour,
            )

            extracted = extract_rows_from_month_grib(
                grib_path=grib_path,
                lat=lat,
                lon=lon,
            )

            rows.extend(extracted)
            done_months.add(month_key)

            checkpoint = pd.DataFrame(rows)
            write_checkpoint_atomic(checkpoint, forecast_table_cache)

            print(f"Extracted {len(extracted)} CAMS rows for month {month_key}")
            print(f"Updated checkpoint: {forecast_table_cache}")

        except Exception as exc:
            print(f"Could not process CAMS month {month_key}: {exc}")

        finally:
            if not keep_cams_cache:
                if grib_path is not None and grib_path.exists():
                    grib_path.unlink(missing_ok=True)
                    print(f"Deleted cached CAMS GRIB: {grib_path}")

    if not rows:
        raise ValueError("No CAMS forecast values were extracted.")

    table = pd.DataFrame(rows)

    table["issue_time_utc"] = pd.to_datetime(table["issue_time_utc"], utc=True)
    table["valid_time_utc"] = pd.to_datetime(table["valid_time_utc"], utc=True)

    table["issue_time_utc"] = table["issue_time_utc"].dt.floor("h")
    table["valid_time_utc"] = table["valid_time_utc"].dt.floor("h")

    table = table[
        (table["valid_time_utc"] >= start)
        & (table["valid_time_utc"] <= end + pd.Timedelta(hours=3))
        & (table["forecast_hour"] > 0)
    ].copy()

    table = (
        table.sort_values(["valid_time_utc", "issue_time_utc"])
        .drop_duplicates(["valid_time_utc", "issue_time_utc"], keep="last")
        .reset_index(drop=True)
    )

    print(f"Extracted {len(table)} usable CAMS forecast rows.")

    if not table.empty:
        print(
            f"CAMS issue range: {table['issue_time_utc'].min()} "
            f"to {table['issue_time_utc'].max()}"
        )
        print(
            f"CAMS valid range: {table['valid_time_utc'].min()} "
            f"to {table['valid_time_utc'].max()}"
        )

    return table


def add_nonleaking_features(
    df: pd.DataFrame,
    forecast_table: pd.DataFrame,
    horizons: list[int],
) -> pd.DataFrame:
    out = df.copy()

    for horizon in horizons:
        col = f"cams_pm25_fcst_tplus{horizon}_ug_m3"

        targets = out[["timestamp_utc"]].drop_duplicates().copy()
        targets["valid_time_utc"] = targets["timestamp_utc"] + pd.Timedelta(hours=horizon)

        candidates = targets.merge(
            forecast_table[
                [
                    "issue_time_utc",
                    "valid_time_utc",
                    "cams_pm25_ug_m3",
                ]
            ],
            on="valid_time_utc",
            how="left",
        )

        candidates = candidates[
            candidates["issue_time_utc"].notna()
            & (candidates["issue_time_utc"] <= candidates["timestamp_utc"])
        ].copy()

        selected = (
            candidates.sort_values(["timestamp_utc", "issue_time_utc"])
            .drop_duplicates("timestamp_utc", keep="last")
            .rename(columns={"cams_pm25_ug_m3": col})
        )

        out = out.merge(
            selected[["timestamp_utc", col]],
            on="timestamp_utc",
            how="left",
        )

        print(f"{col}: missing {out[col].isna().sum()} rows")

    return out


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input_csv)

    if "timestamp_utc" not in df.columns:
        raise KeyError("Input CSV must have timestamp_utc.")

    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"],
        utc=True,
        errors="coerce",
    ).dt.floor("h")

    if df["timestamp_utc"].isna().any():
        raise ValueError("Some timestamp_utc values could not be parsed.")

    if args.start is not None:
        start = parse_utc_timestamp(args.start)
    else:
        start = df["timestamp_utc"].min()

    if args.end is not None:
        end = parse_utc_timestamp(args.end)
    else:
        end = df["timestamp_utc"].max()

    df = df[(df["timestamp_utc"] >= start) & (df["timestamp_utc"] <= end)].copy()

    if df.empty:
        raise ValueError("No input rows remain after applying start/end filters.")

    cycles = [int(x.strip()) for x in args.cycles.split(",") if x.strip()]
    horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]

    forecast_table = build_forecast_table(
        start=start,
        end=end,
        cache_dir=Path(args.cache_dir),
        forecast_table_cache=Path(args.forecast_table_cache),
        lat=args.lat,
        lon=args.lon,
        area_padding_deg=args.area_padding_deg,
        cycles=cycles,
        max_lead_hour=args.max_lead_hour,
        keep_cams_cache=args.keep_cams_cache,
    )

    out = add_nonleaking_features(
        df=df,
        forecast_table=forecast_table,
        horizons=horizons,
    )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    out = out.sort_values("timestamp_utc").reset_index(drop=True)
    out.to_csv(output_path, index=False)

    forecast_table_path = output_path.with_suffix(".cams_forecast_table.csv")
    forecast_table.to_csv(forecast_table_path, index=False)

    print(f"Wrote {output_path}")
    print(f"Wrote {forecast_table_path}")
    print(f"Shape: {out.shape}")


if __name__ == "__main__":
    main()