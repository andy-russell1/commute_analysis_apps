from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from dx_core.geo.distance import haversine_vector_m
from dx_core.io.excel import read_table_from_path


DEFAULT_NAPTAN_DIR = Path(__file__).resolve().parents[2] / "data" / "reference" / "naptan"
LATITUDE_CANDIDATES = ["latitude", "lat", "stop_lat", "y", "y_coordinate"]
LONGITUDE_CANDIDATES = ["longitude", "lon", "lng", "stop_lon", "x", "x_coordinate"]
STOP_ID_CANDIDATES = ["atcocode", "naptancode", "id", "stop_id"]
STOP_NAME_CANDIDATES = ["commonname", "name", "stop_name", "label"]


def _normalise(name: str) -> str:
    return "".join(ch for ch in name.strip().lower() if ch.isalnum() or ch == "_")


def _find_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    lookup = {_normalise(col): col for col in df.columns}
    for candidate in candidates:
        key = _normalise(candidate)
        if key in lookup:
            return lookup[key]
    return None


def find_naptan_file(base_dir: Path = DEFAULT_NAPTAN_DIR) -> Optional[Path]:
    if not base_dir.exists():
        return None
    supported = {".csv", ".txt", ".xls", ".xlsx", ".parquet", ".feather"}
    candidates = [path for path in base_dir.iterdir() if path.is_file() and path.suffix.lower() in supported]
    return sorted(candidates)[0] if candidates else None


def _read_naptan(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".feather":
        return pd.read_feather(path)
    return read_table_from_path(path)


def load_naptan_stops(path: Optional[Path] = None) -> tuple[pd.DataFrame, Optional[str]]:
    naptan_path = path or find_naptan_file()
    if naptan_path is None:
        msg = "NaPTAN file not found. Place CSV/XLSX/Parquet under data/reference/naptan/ to enable transport distance."
        return pd.DataFrame(columns=["stop_id", "stop_name", "lat", "lon"]), msg

    try:
        raw = _read_naptan(naptan_path)
    except Exception as exc:
        return pd.DataFrame(columns=["stop_id", "stop_name", "lat", "lon"]), f"Failed to read NaPTAN: {exc}"

    lat_col = _find_column(raw, LATITUDE_CANDIDATES)
    lon_col = _find_column(raw, LONGITUDE_CANDIDATES)
    if lat_col is None or lon_col is None:
        msg = "NaPTAN missing lat/lon columns. Expected names like Latitude/Longitude or lat/lon."
        return pd.DataFrame(columns=["stop_id", "stop_name", "lat", "lon"]), msg

    id_col = _find_column(raw, STOP_ID_CANDIDATES)
    name_col = _find_column(raw, STOP_NAME_CANDIDATES)

    df = pd.DataFrame(
        {
            "stop_id": raw[id_col].astype(str) if id_col else "",
            "stop_name": raw[name_col].astype(str) if name_col else "",
            "lat": pd.to_numeric(raw[lat_col], errors="coerce"),
            "lon": pd.to_numeric(raw[lon_col], errors="coerce"),
        }
    )
    df = df.dropna(subset=["lat", "lon"])
    df = df[df["lat"].between(-90, 90) & df["lon"].between(-180, 180)]
    df = df.reset_index(drop=True)

    if df.empty:
        return pd.DataFrame(columns=["stop_id", "stop_name", "lat", "lon"]), "NaPTAN loaded but no valid stop coordinates found."

    return df, None


def nearest_stop_distances_m(sites_df: pd.DataFrame, stops_df: pd.DataFrame) -> pd.Series:
    if stops_df.empty:
        return pd.Series(np.nan, index=sites_df.index, dtype=float)

    stop_lats = stops_df["lat"].to_numpy(dtype=float)
    stop_lons = stops_df["lon"].to_numpy(dtype=float)
    output: list[float] = []

    for _, site in sites_df.iterrows():
        distances = haversine_vector_m(
            lat=float(site["lat"]),
            lon=float(site["lon"]),
            other_lats=stop_lats,
            other_lons=stop_lons,
        )
        output.append(float(np.min(distances)) if distances.size else float("nan"))

    return pd.Series(output, index=sites_df.index, dtype=float)
