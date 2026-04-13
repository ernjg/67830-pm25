from __future__ import annotations

import argparse
import io
import json
import shutil
import time
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import requests
from netCDF4 import Dataset


AIRNOW_BASE = "https://files.airnowtech.org/airnow"
GEOSCF_BASE = "https://portal.nccs.nasa.gov/datashare/gmao/geos-cf/v1/forecast"
GAZETTEER_URL = (
    "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/"
    "2023_Gazetteer/2023_Gaz_ua_national.zip"
)

EARTH_RADIUS_KM = 6371.0


@dataclass(frozen=True)
class Location:
    name: str
    lat: float
    lon: float


def utc_hour_range(start: datetime, end: datetime) -> Iterable[datetime]:
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(hours=1)


def haversine_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    lat1r = np.radians(lat1)
    lon1r = np.radians(lon1)
    lat2r = np.radians(lat2)
    lon2r = np.radians(lon2)
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def request_with_retries(url: str, timeout: int = 120, stream: bool = False, retries: int = 5) -> requests.Response:
    last_exc = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=timeout, stream=stream)
            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(1.7 ** attempt)
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(1.7 ** attempt)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Request failed: {url}")


class WorkingPM25Builder:
    """
    Output columns:
      - timestamp_utc
      - airnow_pm25_obs_ug_m3
      - geoscf_pm25_fcst_tplus1_ug_m3
      - geoscf_pm25_fcst_tplus2_ug_m3
      - geoscf_pm25_fcst_tplus3_ug_m3
      - hour_utc
      - day_of_year

    Design:
      - AirNow observation at hour t:
          mean PM2.5 over up to N nearest active measured PM2.5 observations
          within radius_km from the HourlyAQObs file for hour t.
      - GEOS-CF forecasts at hour t:
          PM2.5 forecasts for t+1, t+2, t+3 from the most recent 12Z cycle
          available at hour t.
    """

    def __init__(
        self,
        location_name: str,
        start: str,
        end: str,
        output_csv: str,
        output_meta_json: str,
        cache_dir: str,
        radius_km: float,
        max_pm25_monitors: int,
        force: bool,
        keep_raw_cache: bool,
        clear_all_cache_at_end: bool,
    ) -> None:
        self.start = pd.Timestamp(start, tz="UTC").to_pydatetime()
        self.end = pd.Timestamp(end, tz="UTC").to_pydatetime()
        if self.end < self.start:
            raise ValueError("end must be >= start")

        self.output_csv = Path(output_csv)
        self.output_meta_json = Path(output_meta_json)
        self.cache_dir = Path(cache_dir)

        self.radius_km = radius_km
        self.max_pm25_monitors = max_pm25_monitors
        self.force = force
        self.keep_raw_cache = keep_raw_cache
        self.clear_all_cache_at_end = clear_all_cache_at_end

        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        self.output_meta_json.parent.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.gazetteer_dir = self.cache_dir / "gazetteer"
        self.airnow_hourly_dir = self.cache_dir / "airnow_hourlyaqobs"
        self.geos_files_dir = self.cache_dir / "geoscf_files"
        self.geos_cycle_cache_dir = self.cache_dir / "geoscf_cycle_cache"

        for d in [self.gazetteer_dir, self.airnow_hourly_dir, self.geos_files_dir, self.geos_cycle_cache_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.location = self.get_location_info(location_name)

    def build(self) -> pd.DataFrame:
        print(f"Location: {self.location.name} ({self.location.lat:.4f}, {self.location.lon:.4f})")
        print("Building AirNow series...")
        airnow_df = self.build_airnow_hourly_series()

        print("Building GEOS-CF series...")
        geos_df = self.build_geoscf_hourly_series()

        df = airnow_df.merge(geos_df, on="timestamp_utc", how="inner")
        df = df.sort_values("timestamp_utc").reset_index(drop=True)
        df["hour_utc"] = df["timestamp_utc"].dt.hour
        df["day_of_year"] = df["timestamp_utc"].dt.dayofyear

        df.to_csv(self.output_csv, index=False)
        self.write_metadata()
        self.cleanup_cache()

        print(f"Wrote {self.output_csv}")
        print(f"Wrote {self.output_meta_json}")
        print(df.head())
        print(df.shape)
        return df

    def write_metadata(self) -> None:
        meta = {
            "location_name": self.location.name,
            "location_centroid": {"lat": self.location.lat, "lon": self.location.lon},
            "time_index": {
                "name": "timestamp_utc",
                "timezone": "UTC",
                "meaning": "hour beginning timestamp",
            },
            "columns": {
                "airnow_pm25_obs_ug_m3": {
                    "units": "ug/m3",
                    "description": "Mean PM2.5 over up to 10 nearest AirNow PM2.5 observations within 50 km from the HourlyAQObs file",
                },
                "geoscf_pm25_fcst_tplus1_ug_m3": {
                    "units": "ug/m3",
                    "description": "GEOS-CF PM2.5 forecast for t+1 from the most recent 12Z forecast cycle available at time t",
                },
                "geoscf_pm25_fcst_tplus2_ug_m3": {
                    "units": "ug/m3",
                    "description": "GEOS-CF PM2.5 forecast for t+2 from the most recent 12Z forecast cycle available at time t",
                },
                "geoscf_pm25_fcst_tplus3_ug_m3": {
                    "units": "ug/m3",
                    "description": "GEOS-CF PM2.5 forecast for t+3 from the most recent 12Z forecast cycle available at time t",
                },
                "hour_utc": {"units": "hour", "description": "UTC hour of day, 0 through 23"},
                "day_of_year": {"units": "day", "description": "Day of year, 1 through 366"},
            },
        }
        with open(self.output_meta_json, "w") as f:
            json.dump(meta, f, indent=2)

    def cleanup_cache(self) -> None:
        if self.clear_all_cache_at_end:
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir, ignore_errors=True)
            print(f"Cleared full cache directory: {self.cache_dir}")
            return

        if self.keep_raw_cache:
            return

        removed = []
        for path in [self.airnow_hourly_dir, self.geos_files_dir]:
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)
                removed.append(str(path))

        if removed:
            print("Cleared raw cache directories:")
            for p in removed:
                print(f"  - {p}")

    # ----------------------------
    # Gazetteer
    # ----------------------------

    def download_gazetteer(self) -> Path:
        txt_path = self.gazetteer_dir / "2023_Gaz_ua_national.txt"
        if txt_path.exists() and not self.force:
            return txt_path

        resp = request_with_retries(GAZETTEER_URL, timeout=120)
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            zf.extractall(self.gazetteer_dir)
        return txt_path

    def get_location_info(self, location_name: str) -> Location:
        txt_path = self.download_gazetteer()
        gaz = pd.read_csv(
            txt_path,
            delimiter="\t",
            skiprows=1,
            usecols=[1, 7, 8],
            names=["name", "lat", "lon"],
        )
        row = gaz.loc[gaz["name"] == location_name]
        if row.empty:
            close = (
                gaz.loc[gaz["name"].str.contains(location_name.split(",")[0], case=False, na=False), "name"]
                .head(5)
                .tolist()
            )
            raise ValueError(
                f"Could not find location '{location_name}' in gazetteer. Close matches: {close}"
            )
        vals = row.iloc[0]
        return Location(name=location_name, lat=float(vals["lat"]), lon=float(vals["lon"]))

    # ----------------------------
    # AirNow HourlyAQObs
    # ----------------------------

    def airnow_hourly_path(self, ts: datetime) -> Path:
        return self.airnow_hourly_dir / f"{ts:%Y%m%d%H}.csv"

    def download_airnow_hour(self, ts: datetime) -> pd.DataFrame:
        path = self.airnow_hourly_path(ts)
        if path.exists() and not self.force:
            return pd.read_csv(path)

        url = f"{AIRNOW_BASE}/{ts:%Y}/{ts:%Y%m%d}/HourlyAQObs_{ts:%Y%m%d%H}.dat"
        resp = request_with_retries(url, timeout=60)

        # The upstream BAMS code reads these files with the header row present.
        df = pd.read_csv(io.StringIO(resp.text))

        # Defensive fallback if the file is headerless for some reason.
        if "PM25" not in df.columns:
            cols = [
                "AQSID", "SiteName", "Status", "EPARegion", "Latitude", "Longitude", "Elevation",
                "GMTOffset", "CountryCode", "StateName", "ValidDate", "ValidTime", "DataSource",
                "ReportingArea_PipeDelimited", "OZONE_AQI", "PM10_AQI", "PM25_AQI", "NO2_AQI",
                "OZONE_Measured", "PM10_Measured", "PM25_Measured", "NO2_Measured", "PM25", "PM25_Unit",
                "OZONE", "OZONE_Unit", "NO2", "NO2_Unit", "CO", "CO_Unit", "SO2", "SO2_Unit", "PM10", "PM10_Unit",
            ]
            df = pd.read_csv(io.StringIO(resp.text), header=None, names=cols)

        df.to_csv(path, index=False)
        return df

    def parse_airnow_hour(self, ts: datetime) -> dict:
        df = self.download_airnow_hour(ts).copy()

        if df.empty:
            return {"timestamp_utc": pd.Timestamp(ts), "airnow_pm25_obs_ug_m3": np.nan}

        for col in ["Status", "AQSID", "PM25_Unit"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        if "AQSID" in df.columns:
            df["AQSID"] = df["AQSID"].astype(str).str.zfill(9)

        for col in ["Latitude", "Longitude", "PM25", "PM25_Measured"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        needed = {"Latitude", "Longitude", "PM25", "PM25_Measured", "PM25_Unit"}
        if not needed.issubset(df.columns):
            return {"timestamp_utc": pd.Timestamp(ts), "airnow_pm25_obs_ug_m3": np.nan}

        df = df[
            df["Latitude"].notna()
            & df["Longitude"].notna()
            & df["PM25"].notna()
            & (df["PM25_Measured"] == 1)
            & (df["PM25_Unit"].str.upper() == "UG/M3")
        ].copy()

        if "Status" in df.columns:
            df = df[df["Status"].str.upper() == "ACTIVE"].copy()

        if df.empty:
            return {"timestamp_utc": pd.Timestamp(ts), "airnow_pm25_obs_ug_m3": np.nan}

        dists = haversine_km(
            self.location.lat,
            self.location.lon,
            df["Latitude"].to_numpy(dtype=float),
            df["Longitude"].to_numpy(dtype=float),
        )
        df["dist_km"] = dists
        local_df = df[df["dist_km"] <= self.radius_km].sort_values("dist_km")

        if local_df.empty:
            local_df = df.sort_values("dist_km")

        local_df = local_df.head(self.max_pm25_monitors)

        return {
            "timestamp_utc": pd.Timestamp(ts),
            "airnow_pm25_obs_ug_m3": float(local_df["PM25"].mean()) if not local_df.empty else np.nan,
        }

    def build_airnow_hourly_series(self) -> pd.DataFrame:
        rows = []
        for ts in utc_hour_range(self.start, self.end):
            rows.append(self.parse_airnow_hour(ts))
        return pd.DataFrame(rows)

    # ----------------------------
    # GEOS-CF
    # ----------------------------

    def current_geos_issue_date(self, ts: datetime) -> date:
        if ts.hour >= 12:
            return ts.date()
        return (ts - timedelta(days=1)).date()

    def geos_cycle_cache_path(self, issue_date: date) -> Path:
        return self.geos_cycle_cache_dir / f"{issue_date:%Y%m%d}_12z.csv"

    def geos_file_dir(self, issue_date: date) -> Path:
        d = self.geos_files_dir / f"{issue_date:%Y%m%d}_12z"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def geos_file_path(self, issue_date: date, valid_start: datetime) -> Path:
        centered = valid_start + timedelta(minutes=30)
        filename = (
            "GEOS-CF.v01.fcst.aqc_tavg_1hr_g1440x721_v1."
            f"{issue_date:%Y%m%d}_12z+{centered:%Y%m%d_%H}30z.nc4"
        )
        return self.geos_file_dir(issue_date) / filename

    def geos_file_url(self, issue_date: date, valid_start: datetime) -> str:
        centered = valid_start + timedelta(minutes=30)
        filename = (
            "GEOS-CF.v01.fcst.aqc_tavg_1hr_g1440x721_v1."
            f"{issue_date:%Y%m%d}_12z+{centered:%Y%m%d_%H}30z.nc4"
        )
        return f"{GEOSCF_BASE}/Y{issue_date:%Y}/M{issue_date:%m}/D{issue_date:%d}/H12/{filename}"

    def download_geos_file(self, issue_date: date, valid_start: datetime) -> Path:
        path = self.geos_file_path(issue_date, valid_start)
        if path.exists() and not self.force:
            return path

        url = self.geos_file_url(issue_date, valid_start)
        resp = request_with_retries(url, timeout=180, stream=True)
        with open(path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
        return path

    def extract_geos_local_pm25(self, nc_path: Path) -> float:
        with Dataset(nc_path, "r") as ds:
            if "PM25_RH35_GCC" in ds.variables:
                var_name = "PM25_RH35_GCC"
            elif "pm25_rh35_gcc" in ds.variables:
                var_name = "pm25_rh35_gcc"
            else:
                raise KeyError(f"Could not find PM2.5 variable in {nc_path}")

            lat = np.array(ds.variables["lat"][:], dtype=float)
            lon = np.array(ds.variables["lon"][:], dtype=float)

            lat0 = self.location.lat
            lon0 = self.location.lon

            lat_mask = (lat >= lat0 - 1.0) & (lat <= lat0 + 1.0)
            lon_mask = (lon >= lon0 - 1.0) & (lon <= lon0 + 1.0)

            if not lat_mask.any() or not lon_mask.any():
                lat_idx = int(np.argmin(np.abs(lat - lat0)))
                lon_idx = int(np.argmin(np.abs(lon - lon0)))
                val = ds.variables[var_name][0, 0, lat_idx, lon_idx]
                return float(np.asarray(val))

            lat_inds = np.where(lat_mask)[0]
            lon_inds = np.where(lon_mask)[0]
            lat_slice = slice(int(lat_inds.min()), int(lat_inds.max()) + 1)
            lon_slice = slice(int(lon_inds.min()), int(lon_inds.max()) + 1)

            arr = ds.variables[var_name][0, 0, lat_slice, lon_slice]
            arr = np.asarray(np.ma.filled(arr, np.nan), dtype=float)

            sub_lat = lat[lat_slice]
            sub_lon = lon[lon_slice]
            lon2d, lat2d = np.meshgrid(sub_lon, sub_lat)
            dists = haversine_km(lat0, lon0, lat2d.ravel(), lon2d.ravel())
            keep = dists <= self.radius_km

            vals = arr.ravel()
            if keep.any() and not np.all(np.isnan(vals[keep])):
                return float(np.nanmean(vals[keep]))

            dmin = int(np.argmin(dists))
            return float(vals[dmin])

    def build_geos_cycle_summary(self, issue_date: date) -> Dict[datetime, float]:
        cache_path = self.geos_cycle_cache_path(issue_date)
        if cache_path.exists() and not self.force:
            df = pd.read_csv(cache_path, parse_dates=["valid_start_utc"])
            df["valid_start_utc"] = pd.to_datetime(df["valid_start_utc"], utc=True)
            return {ts.to_pydatetime(): float(val) for ts, val in zip(df["valid_start_utc"], df["pm25_ug_m3"])}

        issue_dt = datetime(issue_date.year, issue_date.month, issue_date.day, 12, 0, tzinfo=timezone.utc)
        rows = []

        # Need t+1, t+2, t+3 for every current hour in this cycle window,
        # so cache valid starts 13:00 same day through 14:00 next day.
        for k in range(1, 27):
            valid_start = issue_dt + timedelta(hours=k)
            nc_path = self.download_geos_file(issue_date, valid_start)
            pm25_val = self.extract_geos_local_pm25(nc_path)
            rows.append({"valid_start_utc": pd.Timestamp(valid_start), "pm25_ug_m3": pm25_val})

        df = pd.DataFrame(rows)
        df.to_csv(cache_path, index=False)
        return {ts.to_pydatetime(): float(val) for ts, val in zip(df["valid_start_utc"], df["pm25_ug_m3"])}

    def build_geoscf_hourly_series(self) -> pd.DataFrame:
        rows = []
        cycle_cache: Dict[date, Dict[datetime, float]] = {}

        for ts in utc_hour_range(self.start, self.end):
            issue_date = self.current_geos_issue_date(ts)
            if issue_date not in cycle_cache:
                print(f"  loading GEOS-CF cycle {issue_date} 12Z")
                cycle_cache[issue_date] = self.build_geos_cycle_summary(issue_date)

            cycle = cycle_cache[issue_date]
            t1 = ts + timedelta(hours=1)
            t2 = ts + timedelta(hours=2)
            t3 = ts + timedelta(hours=3)

            if not all(t in cycle for t in (t1, t2, t3)):
                continue

            rows.append(
                {
                    "timestamp_utc": pd.Timestamp(ts),
                    "geoscf_pm25_fcst_tplus1_ug_m3": cycle[t1],
                    "geoscf_pm25_fcst_tplus2_ug_m3": cycle[t2],
                    "geoscf_pm25_fcst_tplus3_ug_m3": cycle[t3],
                }
            )

        return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a simple hourly PM2.5 time series for one urban area.\n\n"
            "Columns:\n"
            "  - timestamp_utc\n"
            "  - airnow_pm25_obs_ug_m3\n"
            "  - geoscf_pm25_fcst_tplus1_ug_m3\n"
            "  - geoscf_pm25_fcst_tplus2_ug_m3\n"
            "  - geoscf_pm25_fcst_tplus3_ug_m3\n"
            "  - hour_utc\n"
            "  - day_of_year\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--location-name", default="Los Angeles--Long Beach--Anaheim, CA Urban Area")
    parser.add_argument("--start", default="2023-01-01T00:00:00Z")
    parser.add_argument("--end", default="2023-12-31T23:00:00Z")
    parser.add_argument("--output-csv", default="data/simple_pm25_2023.csv")
    parser.add_argument("--output-meta-json", default="data/simple_pm25_2023_metadata.json")
    parser.add_argument("--cache-dir", default="data/cache_working")
    parser.add_argument("--radius-km", type=float, default=50.0)
    parser.add_argument("--max-pm25-monitors", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--keep-raw-cache",
        action="store_true",
        help="Keep bulky downloaded raw files in the cache after the dataset is built",
    )
    parser.add_argument(
        "--clear-all-cache-at-end",
        action="store_true",
        help="Delete the entire cache directory after a successful run",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    builder = WorkingPM25Builder(
        location_name=args.location_name,
        start=args.start,
        end=args.end,
        output_csv=args.output_csv,
        output_meta_json=args.output_meta_json,
        cache_dir=args.cache_dir,
        radius_km=args.radius_km,
        max_pm25_monitors=args.max_pm25_monitors,
        force=args.force,
        keep_raw_cache=args.keep_raw_cache,
        clear_all_cache_at_end=args.clear_all_cache_at_end,
    )
    builder.build()


if __name__ == "__main__":
    main()
