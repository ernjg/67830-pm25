from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests


AWS_BASE = "https://noaa-nws-naqfc-pds.s3.amazonaws.com"

LA_URBAN_AREA_LAT = 33.9850
LA_URBAN_AREA_LON = -118.1224


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download NAQFC/AQM PM2.5 forecast files from NOAA AWS and add "
            "non-leaking t+1/t+2/t+3 forecast features to an existing hourly PM2.5 dataset."
        )
    )

    parser.add_argument("--input-csv", default="data/2023_combined.csv")
    parser.add_argument("--output-csv", default="data/2023_combined_plus_naqfc.csv")

    parser.add_argument(
        "--cache-dir",
        default="data/cache/naqfc_grib",
        help="Temporary cache for downloaded GRIB2 files.",
    )
    parser.add_argument(
        "--forecast-table-cache",
        default="data/cache/naqfc_forecast_table_checkpoint.csv",
        help=(
            "Checkpoint CSV for extracted local NAQFC forecast rows. "
            "This is saved after each completed forecast cycle."
        ),
    )

    parser.add_argument("--lat", type=float, default=LA_URBAN_AREA_LAT)
    parser.add_argument("--lon", type=float, default=LA_URBAN_AREA_LON)

    parser.add_argument(
        "--cycles",
        default="06,12",
        help="Comma-separated forecast cycles to use.",
    )
    parser.add_argument(
        "--horizons",
        default="1,2,3",
        help="Comma-separated future forecast horizons to add.",
    )
    parser.add_argument(
        "--product",
        default="pm25",
        choices=["pm25", "pm25_bc"],
        help="Use raw PM2.5 or bias-corrected PM2.5.",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Optional start timestamp, e.g. 2023-01-01T00:00:00Z.",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="Optional end timestamp, e.g. 2023-01-03T23:00:00Z.",
    )
    parser.add_argument(
        "--keep-grib-cache",
        action="store_true",
        help="Keep downloaded GRIB2 files after extracting local values.",
    )

    return parser.parse_args()


def parse_utc_timestamp(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)

    if ts.tzinfo is None:
        return ts.tz_localize("UTC")

    return ts.tz_convert("UTC")


def aqm_version_for_issue_time(issue_time: pd.Timestamp) -> str:
    """
    NOAA's archive has AQMv6 for 2023 and early 2024, then AQMv7 after
    the May 2024 operational upgrade. The first AQMv7 date we observed
    in the archive is 2024-05-14.
    """
    switch_time = pd.Timestamp("2024-05-14T00:00:00Z")

    if issue_time >= switch_time:
        return "AQMv7"

    return "AQMv6"


def request_with_retries(
    url: str,
    timeout: int = 180,
    retries: int = 4,
) -> requests.Response:
    last_exc = None

    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=timeout)

            if response.status_code == 404:
                response.raise_for_status()

            if response.status_code in {429, 500, 502, 503, 504}:
                time.sleep(2**attempt)
                continue

            response.raise_for_status()
            return response

        except requests.RequestException as exc:
            last_exc = exc

            if attempt < retries - 1:
                time.sleep(2**attempt)

    raise last_exc


def naqfc_url(issue_time: pd.Timestamp, product: str) -> str:
    ymd = issue_time.strftime("%Y%m%d")
    cycle = issue_time.strftime("%H")
    version = aqm_version_for_issue_time(issue_time)

    if product == "pm25":
        filename = f"aqm.t{cycle}z.ave_1hr_pm25.{ymd}.227.grib2"
    else:
        filename = f"aqm.t{cycle}z.ave_1hr_pm25_bc.{ymd}.227.grib2"

    return f"{AWS_BASE}/{version}/CS/{ymd}/{cycle}/{filename}"


def local_naqfc_path(
    cache_dir: Path,
    issue_time: pd.Timestamp,
    product: str,
) -> Path:
    ymd = issue_time.strftime("%Y%m%d")
    cycle = issue_time.strftime("%H")
    version = aqm_version_for_issue_time(issue_time)

    if product == "pm25":
        filename = f"aqm.t{cycle}z.ave_1hr_pm25.{ymd}.227.grib2"
    else:
        filename = f"aqm.t{cycle}z.ave_1hr_pm25_bc.{ymd}.227.grib2"

    return cache_dir / version / ymd / cycle / filename


def download_naqfc_file(
    cache_dir: Path,
    issue_time: pd.Timestamp,
    product: str,
) -> Path | None:
    path = local_naqfc_path(cache_dir, issue_time, product)

    if path.exists() and path.stat().st_size > 0:
        return path

    url = naqfc_url(issue_time, product)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        print(f"Downloading {url}")
        response = request_with_retries(url)

        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(response.content)
        tmp.replace(path)

        return path

    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            print(f"Missing on server: {url}")
            return None

        raise


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


def extract_local_forecast_rows(
    grib_path: Path,
    issue_time: pd.Timestamp,
    lat: float,
    lon: float,
    product: str,
) -> list[dict]:
    import xarray as xr

    rows = []

    ds = xr.open_dataset(
        grib_path,
        engine="cfgrib",
        backend_kwargs={
            "indexpath": "",
            "errors": "ignore",
        },
    )

    try:
        if len(ds.data_vars) == 0:
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

        dists = haversine_km(lat, target_lon, lats, lons)
        nearest_ij = np.unravel_index(np.nanargmin(dists), dists.shape)

        nearest_lat = float(lats[nearest_ij])
        nearest_lon = float(lons[nearest_ij])

        print(
            f"Nearest NAQFC grid point: lat={nearest_lat:.4f}, "
            f"lon={nearest_lon:.4f}"
        )

        spatial_dims = ds["latitude"].dims
        if len(spatial_dims) != 2:
            raise ValueError(f"Expected 2D latitude grid, got dims {spatial_dims}")

        y_dim, x_dim = spatial_dims
        y_idx, x_idx = nearest_ij

        non_spatial_dims = [
            dim for dim in da.dims if dim not in {y_dim, x_dim}
        ]

        if len(non_spatial_dims) == 0:
            valid_time = pd.Timestamp(ds["valid_time"].values)

            if valid_time.tzinfo is None:
                valid_time = valid_time.tz_localize("UTC")
            else:
                valid_time = valid_time.tz_convert("UTC")

            forecast_hour = int(round((valid_time - issue_time).total_seconds() / 3600))
            value = float(da.isel({y_dim: y_idx, x_dim: x_idx}).values)

            rows.append(
                {
                    "aqm_version": aqm_version_for_issue_time(issue_time),
                    "product": product,
                    "extract_lat": lat,
                    "extract_lon": lon,
                    "nearest_grid_lat": nearest_lat,
                    "nearest_grid_lon": nearest_lon,
                    "issue_time_utc": issue_time,
                    "valid_time_utc": valid_time,
                    "forecast_hour": forecast_hour,
                    "naqfc_pm25_ug_m3": value,
                    "source_file": str(grib_path),
                }
            )

            return rows

        if len(non_spatial_dims) > 1:
            raise ValueError(
                f"Expected at most one forecast/time dimension, got {non_spatial_dims}"
            )

        time_dim = non_spatial_dims[0]

        if "valid_time" in ds.coords:
            valid_times = pd.to_datetime(ds["valid_time"].values, utc=True)
        elif "step" in ds.coords:
            steps = pd.to_timedelta(ds["step"].values)
            valid_times = pd.DatetimeIndex([issue_time + step for step in steps])
        else:
            raise ValueError(
                f"Could not infer valid times. Available coords: {list(ds.coords)}"
            )

        for k, valid_time in enumerate(valid_times):
            value = float(
                da.isel(
                    {
                        time_dim: k,
                        y_dim: y_idx,
                        x_dim: x_idx,
                    }
                ).values
            )

            valid_time = pd.Timestamp(valid_time)
            if valid_time.tzinfo is None:
                valid_time = valid_time.tz_localize("UTC")
            else:
                valid_time = valid_time.tz_convert("UTC")

            forecast_hour = int(round((valid_time - issue_time).total_seconds() / 3600))

            rows.append(
                {
                    "aqm_version": aqm_version_for_issue_time(issue_time),
                    "product": product,
                    "extract_lat": lat,
                    "extract_lon": lon,
                    "nearest_grid_lat": nearest_lat,
                    "nearest_grid_lon": nearest_lon,
                    "issue_time_utc": issue_time,
                    "valid_time_utc": valid_time,
                    "forecast_hour": forecast_hour,
                    "naqfc_pm25_ug_m3": value,
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


def load_checkpoint(
    path: Path,
    product: str,
    lat: float,
    lon: float,
) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)

    if df.empty:
        return df

    df["issue_time_utc"] = pd.to_datetime(df["issue_time_utc"], utc=True)
    df["valid_time_utc"] = pd.to_datetime(df["valid_time_utc"], utc=True)

    if "product" in df.columns:
        df = df[df["product"] == product].copy()

    if "extract_lat" in df.columns and "extract_lon" in df.columns:
        df = df[
            np.isclose(df["extract_lat"].astype(float), lat)
            & np.isclose(df["extract_lon"].astype(float), lon)
        ].copy()

    print(f"Loaded {len(df)} checkpoint rows from {path}")

    return df


def build_issue_times(
    start: pd.Timestamp,
    end: pd.Timestamp,
    cycles: list[int],
) -> list[pd.Timestamp]:
    first_day = start.floor("D") - pd.Timedelta(days=1)
    last_day = end.floor("D") + pd.Timedelta(days=1)

    issue_times = []

    for day in pd.date_range(first_day, last_day, freq="D", tz="UTC"):
        for cycle in cycles:
            issue_time = day + pd.Timedelta(hours=cycle)

            if issue_time <= end and issue_time >= start - pd.Timedelta(days=2):
                issue_times.append(issue_time)

    return sorted(issue_times)


def build_forecast_table(
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_dir: Path,
    forecast_table_cache: Path,
    lat: float,
    lon: float,
    cycles: list[int],
    product: str,
    keep_grib_cache: bool,
) -> pd.DataFrame:
    issue_times = build_issue_times(start, end, cycles)

    checkpoint_df = load_checkpoint(
        forecast_table_cache,
        product=product,
        lat=lat,
        lon=lon,
    )

    rows = checkpoint_df.to_dict("records") if not checkpoint_df.empty else []

    done_issues = set()
    if not checkpoint_df.empty:
        done_issues = set(
            checkpoint_df["issue_time_utc"].dt.floor("h").astype(str)
        )

    for issue_time in issue_times:
        issue_key = str(issue_time.floor("h"))

        if issue_key in done_issues:
            print(f"Skipping already extracted cycle: {issue_time}")
            continue

        path = download_naqfc_file(
            cache_dir=cache_dir,
            issue_time=issue_time,
            product=product,
        )

        if path is None:
            continue

        try:
            extracted = extract_local_forecast_rows(
                grib_path=path,
                issue_time=issue_time,
                lat=lat,
                lon=lon,
                product=product,
            )

            rows.extend(extracted)
            done_issues.add(issue_key)

            checkpoint = pd.DataFrame(rows)
            write_checkpoint_atomic(checkpoint, forecast_table_cache)

            print(f"Extracted {len(extracted)} messages from {path}")
            print(f"Updated checkpoint: {forecast_table_cache}")

        except Exception as exc:
            print(f"Could not read {path}: {exc}")

        finally:
            if not keep_grib_cache:
                try:
                    path.unlink(missing_ok=True)
                    print(f"Deleted cached GRIB: {path}")
                except Exception as exc:
                    print(f"Could not delete cached GRIB {path}: {exc}")

    if not rows:
        raise ValueError("No NAQFC forecast values were extracted.")

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

    print(f"Extracted {len(table)} usable forecast rows.")

    if not table.empty:
        print(
            f"Issue range: {table['issue_time_utc'].min()} "
            f"to {table['issue_time_utc'].max()}"
        )
        print(
            f"Valid range: {table['valid_time_utc'].min()} "
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
        col = f"naqfc_pm25_fcst_tplus{horizon}_ug_m3"

        targets = out[["timestamp_utc"]].drop_duplicates().copy()
        targets["valid_time_utc"] = targets["timestamp_utc"] + pd.Timedelta(hours=horizon)

        candidates = targets.merge(
            forecast_table[
                [
                    "issue_time_utc",
                    "valid_time_utc",
                    "naqfc_pm25_ug_m3",
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
            .rename(columns={"naqfc_pm25_ug_m3": col})
        )

        out = out.merge(
            selected[["timestamp_utc", col]],
            on="timestamp_utc",
            how="left",
        )

        missing = out[col].isna().sum()
        print(f"{col}: missing {missing} rows")

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
        cycles=cycles,
        product=args.product,
        keep_grib_cache=args.keep_grib_cache,
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

    forecast_table_path = output_path.with_suffix(".naqfc_forecast_table.csv")
    forecast_table.to_csv(forecast_table_path, index=False)

    print(f"Wrote {output_path}")
    print(f"Wrote {forecast_table_path}")
    print(f"Shape: {out.shape}")


if __name__ == "__main__":
    main()