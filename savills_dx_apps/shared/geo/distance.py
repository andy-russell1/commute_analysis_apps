from __future__ import annotations

import math

import numpy as np
import pandas as pd


EARTH_RADIUS_M = 6_371_000.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_M * c


def haversine_vector_m(lat: float, lon: float, other_lats: np.ndarray, other_lons: np.ndarray) -> np.ndarray:
    if other_lats.size == 0:
        return np.array([], dtype=float)
    lat1 = np.radians(lat)
    lon1 = np.radians(lon)
    lat2 = np.radians(other_lats)
    lon2 = np.radians(other_lons)
    d_lat = lat2 - lat1
    d_lon = lon2 - lon1
    a = np.sin(d_lat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(d_lon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return EARTH_RADIUS_M * c


def nearest_distance_m(lat: float, lon: float, candidates_df: pd.DataFrame, lat_col: str = "lat", lon_col: str = "lon") -> float:
    if candidates_df.empty:
        return float("nan")
    distances = haversine_vector_m(
        lat=lat,
        lon=lon,
        other_lats=candidates_df[lat_col].to_numpy(dtype=float),
        other_lons=candidates_df[lon_col].to_numpy(dtype=float),
    )
    return float(np.min(distances)) if distances.size else float("nan")
