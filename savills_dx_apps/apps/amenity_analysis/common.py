from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from dx_core.data.cache import ensure_cache_dir, make_hashed_key, read_json_cache, write_json_cache
from dx_core.data.naptan_loader import load_naptan_stops, nearest_stop_distances_m
from dx_core.data.osm_overpass import OverpassRateLimitError, fetch_pois, load_osm_tag_map
from dx_core.geo.buffers import build_combined_geojson, geojson_export_supported
from dx_core.geo.distance import EARTH_RADIUS_M, haversine_vector_m
from dx_core.scoring.amenity_index import (
    bucket_slug,
    count_column,
    nearest_distance_column,
    normalise_counts,
    normalise_weights,
)

try:
    import pydeck as pdk

    HAS_PYDECK = True
except Exception:  # pragma: no cover - optional rendering dependency
    HAS_PYDECK = False
    pdk = None


TRANSPORT_METRIC = "Public transport"

SITES_DF_KEY = "sites_df"
RESULTS_SUMMARY_KEY = "results_summary_df"
RESULTS_POI_KEY = "results_poi_df"
RESULTS_BY_RADIUS_KEY = "results_by_radius"
SELECTED_METRICS_KEY = "selected_metrics"
SELECTED_RADII_KEY = "selected_radii_m"
PRIMARY_RADIUS_KEY = "primary_radius_m"
WEIGHTS_RAW_KEY = "bucket_weights_raw"
WEIGHTS_NORM_KEY = "bucket_weights_norm"
ANALYSIS_MESSAGES_KEY = "analysis_messages"
CACHE_STATS_KEY = "cache_stats"
LAST_RUN_CONFIG_KEY = "last_run_config"
CONTROLS_VIEW_KEY = "controls_view"
EMBEDDED_MODE_KEY = "amenity_embedded_mode"
EMBEDDED_ROUTE_KEY = "amenity_embedded_route"

COMMUTE_COLOR_SCALE = [
    [0.0, "#1e8449"],
    [0.5, "#f1c40f"],
    [1.0, "#c0392b"],
]
POI_COLORS = {
    "Lunch & coffee": "#F59E0B",
    "Green": "#16A34A",
    "Fitness": "#2563EB",
}
RADIUS_COLORS = ["#1e8449", "#f1c40f", "#e67e22", "#c0392b", "#5b5f97"]
DENSITY_COLOR_RANGE = [
    [242, 251, 250],
    [214, 245, 240],
    [167, 234, 225],
    [92, 200, 190],
    [14, 159, 154],
    [11, 110, 106],
]


@lru_cache(maxsize=1)
def get_tag_map() -> dict[str, Any]:
    return load_osm_tag_map()


def get_amenity_bucket_names() -> list[str]:
    return list(get_tag_map().get("buckets", {}).keys())


def get_metric_options() -> list[str]:
    return [*get_amenity_bucket_names(), TRANSPORT_METRIC]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def office_name_from_address(address: str, office_id: str = "") -> str:
    name = str(address or "").split(",")[0].strip()
    if name:
        return name
    return str(office_id or "").strip()


def init_amenity_state(session_state: Any) -> None:
    metrics = get_metric_options()
    default_selected = list(metrics)
    defaults = {
        SITES_DF_KEY: pd.DataFrame(columns=["officeID", "address", "lat", "lon"]),
        RESULTS_SUMMARY_KEY: pd.DataFrame(),
        RESULTS_POI_KEY: pd.DataFrame(),
        RESULTS_BY_RADIUS_KEY: {},
        SELECTED_METRICS_KEY: default_selected,
        SELECTED_RADII_KEY: [1000],
        PRIMARY_RADIUS_KEY: 1000,
        WEIGHTS_RAW_KEY: {metric: 100.0 / max(len(default_selected), 1) for metric in default_selected},
        WEIGHTS_NORM_KEY: {metric: 1.0 / max(len(default_selected), 1) for metric in default_selected},
        ANALYSIS_MESSAGES_KEY: [],
        CACHE_STATS_KEY: {"hits": 0, "misses": 0},
        LAST_RUN_CONFIG_KEY: {},
        CONTROLS_VIEW_KEY: "Analysis controls",
        EMBEDDED_MODE_KEY: False,
        EMBEDDED_ROUTE_KEY: "app",
    }

    for key, value in defaults.items():
        if key not in session_state:
            session_state[key] = value


def safe_set_page_config(page_title: str, page_icon: str, layout: str = "wide") -> None:
    """Set page config where possible; ignore duplicate-call errors in embedded mode."""
    try:
        st.set_page_config(page_title=page_title, page_icon=page_icon, layout=layout)
    except Exception:
        return


def is_embedded_mode(session_state: Any) -> bool:
    return bool(session_state.get(EMBEDDED_MODE_KEY, False))


def set_embedded_mode(session_state: Any, enabled: bool) -> None:
    session_state[EMBEDDED_MODE_KEY] = bool(enabled)


def get_embedded_route(session_state: Any) -> str:
    return str(session_state.get(EMBEDDED_ROUTE_KEY, "app"))


def set_embedded_route(session_state: Any, route: str) -> None:
    session_state[EMBEDDED_ROUTE_KEY] = str(route)


def navigate_to(session_state: Any, route: str, standalone_page_path: str) -> None:
    """Navigate to an amenity page in embedded mode or standalone mode."""
    if is_embedded_mode(session_state):
        set_embedded_route(session_state, route)
        st.rerun()
    else:
        st.switch_page(standalone_page_path)


def clear_analysis_results(session_state: Any) -> None:
    session_state[RESULTS_SUMMARY_KEY] = pd.DataFrame()
    session_state[RESULTS_POI_KEY] = pd.DataFrame()
    session_state[RESULTS_BY_RADIUS_KEY] = {}
    session_state[ANALYSIS_MESSAGES_KEY] = []
    session_state[CACHE_STATS_KEY] = {"hits": 0, "misses": 0}
    session_state[LAST_RUN_CONFIG_KEY] = {}


@dataclass
class RadiusAnalysisRunResult:
    summary_df: pd.DataFrame
    poi_df: pd.DataFrame
    messages: list[str]
    cache_stats: dict[str, int]


@dataclass
class MultiRadiusRunResult:
    by_radius: dict[int, RadiusAnalysisRunResult]
    weights_normalised: dict[str, float]
    messages: list[str]
    cache_stats: dict[str, int]


def _cache_key(
    office_id: str,
    lat: float,
    lon: float,
    radius_m: int,
    selected_amenity_buckets: list[str],
    tag_map_version: int,
) -> str:
    payload = {
        "officeID": office_id,
        "lat": round(lat, 6),
        "lon": round(lon, 6),
        "radius_m": int(radius_m),
        "selected_buckets": sorted(selected_amenity_buckets),
        "tag_map_version": int(tag_map_version),
    }
    return make_hashed_key(payload)


def _empty_poi_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["bucket", "osm_type", "osm_id", "name", "poi_lat", "poi_lon", "tag_key", "tag_value"]
    )


def _compute_transport_normalised(summary_df: pd.DataFrame) -> pd.Series:
    transport_col = "nearest_public_transport_stop_distance_m"
    distance_series = pd.to_numeric(summary_df.get(transport_col), errors="coerce")

    if distance_series.isna().all():
        return pd.Series(np.full(len(summary_df), 0.5), index=summary_df.index)

    fallback_max = float(distance_series.max(skipna=True))
    filled = distance_series.fillna(fallback_max)
    min_distance = float(filled.min())
    max_distance = float(filled.max())
    if max_distance == min_distance:
        return pd.Series(np.full(len(summary_df), 0.5), index=summary_df.index)

    return (max_distance - filled) / (max_distance - min_distance)


def _apply_weighted_kpi(
    summary_df: pd.DataFrame,
    selected_metrics: list[str],
    weights_norm: dict[str, float],
) -> pd.DataFrame:
    out = summary_df.copy()
    kpi = np.zeros(len(out), dtype=float)

    for metric in selected_metrics:
        norm_col = f"normalised_{bucket_slug(metric)}"
        if metric == TRANSPORT_METRIC:
            out[norm_col] = _compute_transport_normalised(out)
        else:
            cnt_col = count_column(metric)
            if cnt_col not in out.columns:
                out[cnt_col] = 0
            out[norm_col] = normalise_counts(out[cnt_col].fillna(0).astype(float))

        kpi += out[norm_col].to_numpy(dtype=float) * float(weights_norm.get(metric, 0.0))

    out["amenity_kpi"] = np.round(kpi * 100.0, 1)
    return out


def _fetch_office_pois_at_max_radius(
    office_id: str,
    address: str,
    lat: float,
    lon: float,
    max_radius_m: int,
    amenity_buckets: list[str],
    tag_map: dict[str, Any],
    tag_map_version: int,
    cache_dir: Path,
    throttle_seconds: float,
) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    """Fetch (or cache-hit) POIs once per office at max radius, with distance annotations."""
    messages: list[str] = []
    cache_stats = {"hits": 0, "misses": 0}

    office_pois = _empty_poi_df()
    if not amenity_buckets:
        office_pois["distance_m"] = []
        return office_pois, messages, cache_stats

    key = _cache_key(
        office_id=office_id,
        lat=lat,
        lon=lon,
        radius_m=max_radius_m,
        selected_amenity_buckets=amenity_buckets,
        tag_map_version=tag_map_version,
    )
    cached = read_json_cache(cache_dir=cache_dir, key=key)

    office_name = office_name_from_address(address=address, office_id=office_id)

    if cached is not None and isinstance(cached.get("payload"), list):
        office_pois = pd.DataFrame(cached["payload"])
        cache_stats["hits"] += 1
    else:
        fetched_successfully = False
        try:
            office_pois = fetch_pois(
                lat=lat,
                lon=lon,
                radius_m=max_radius_m,
                selected_buckets_list=amenity_buckets,
                tag_map=tag_map,
            )
            fetched_successfully = True
        except OverpassRateLimitError as exc:
            office_pois = _empty_poi_df()
            retry_hint = (
                f" Retry after ~{int(exc.retry_after_seconds)}s."
                if exc.retry_after_seconds is not None
                else " Reduce sites/radii or retry shortly."
            )
            messages.append(
                f"Max radius {max_radius_m}m: Overpass rate limit for office {office_id}. "
                f"Using empty amenity results.{retry_hint}"
            )
        except requests.RequestException as exc:
            office_pois = _empty_poi_df()
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if status_code is not None:
                short_reason = f"HTTP {status_code}"
            else:
                short_reason = exc.__class__.__name__
            messages.append(
                f"Max radius {max_radius_m}m: Overpass request failed for office {office_id} ({short_reason}). "
                "Using empty amenity results for this office."
            )

        if fetched_successfully:
            write_json_cache(
                cache_dir=cache_dir,
                key=key,
                payload=office_pois.to_dict(orient="records"),
                metadata={
                    "officeID": office_id,
                    "radius_m": max_radius_m,
                    "selected_buckets": amenity_buckets,
                    "tag_map_version": tag_map_version,
                },
            )
        cache_stats["misses"] += 1
        time.sleep(throttle_seconds)

    if office_pois.empty:
        office_pois["distance_m"] = []
        return office_pois, messages, cache_stats

    dists = haversine_vector_m(
        lat=lat,
        lon=lon,
        other_lats=office_pois["poi_lat"].to_numpy(dtype=float),
        other_lons=office_pois["poi_lon"].to_numpy(dtype=float),
    )
    office_pois = office_pois.copy()
    office_pois["distance_m"] = dists
    office_pois["officeID"] = office_id
    office_pois["office_name"] = office_name
    office_pois["address"] = address
    office_pois["office_lat"] = lat
    office_pois["office_lon"] = lon
    return office_pois, messages, cache_stats


def run_multi_radius_analysis(
    sites_df: pd.DataFrame,
    selected_metrics: list[str],
    selected_radii_m: list[int],
    raw_weights: dict[str, float],
    throttle_seconds: float = 0.7,
) -> MultiRadiusRunResult:
    if not selected_metrics or not selected_radii_m or sites_df.empty:
        return MultiRadiusRunResult(
            by_radius={},
            weights_normalised=normalise_weights(raw_weights, selected_metrics),
            messages=["Select at least one metric and one radius, and load valid sites."],
            cache_stats={"hits": 0, "misses": 0},
        )

    radii = sorted({int(radius) for radius in selected_radii_m})
    max_radius = int(max(radii))
    amenity_buckets = [metric for metric in selected_metrics if metric in get_amenity_bucket_names()]
    include_transport = TRANSPORT_METRIC in selected_metrics
    weights_norm = normalise_weights(raw_weights, selected_metrics)

    tag_map = get_tag_map()
    tag_map_version = int(tag_map.get("version", 1))
    cache_dir = ensure_cache_dir(_project_root() / ".cache" / "amenity_analysis")

    office_base_rows: list[dict[str, Any]] = []
    office_pois_by_office: dict[str, pd.DataFrame] = {}

    by_radius: dict[int, RadiusAnalysisRunResult] = {}
    all_messages: list[str] = []
    total_hits = 0
    total_misses = 0

    for _, site in sites_df.iterrows():
        office_id = str(site["officeID"])
        address = str(site["address"])
        office_name = office_name_from_address(address=address, office_id=office_id)
        lat = float(site["lat"])
        lon = float(site["lon"])

        office_base_rows.append(
            {
                "officeID": office_id,
                "office_name": office_name,
                "address": address,
                "lat": lat,
                "lon": lon,
            }
        )

        office_pois, messages, cache_stats = _fetch_office_pois_at_max_radius(
            office_id=office_id,
            address=address,
            lat=lat,
            lon=lon,
            max_radius_m=max_radius,
            amenity_buckets=amenity_buckets,
            tag_map=tag_map,
            tag_map_version=tag_map_version,
            cache_dir=cache_dir,
            throttle_seconds=throttle_seconds,
        )
        office_pois_by_office[office_id] = office_pois
        all_messages.extend(messages)
        total_hits += int(cache_stats.get("hits", 0))
        total_misses += int(cache_stats.get("misses", 0))

    base_summary_df = pd.DataFrame(office_base_rows)

    transport_col = "nearest_public_transport_stop_distance_m"
    if include_transport:
        stops_df, msg = load_naptan_stops()
        if msg:
            all_messages.append(msg)
        base_summary_df[transport_col] = (
            nearest_stop_distances_m(base_summary_df[["lat", "lon"]], stops_df) if not stops_df.empty else np.nan
        )
    else:
        base_summary_df[transport_col] = np.nan

    for radius in radii:
        summary_rows: list[dict[str, Any]] = []
        poi_frames: list[pd.DataFrame] = []

        for _, site in base_summary_df.iterrows():
            office_id = str(site["officeID"])
            office_pois_max = office_pois_by_office.get(office_id, _empty_poi_df())

            if office_pois_max.empty:
                office_pois_radius = office_pois_max.copy()
            else:
                office_pois_radius = office_pois_max[office_pois_max["distance_m"] <= float(radius)].copy()
                if not office_pois_radius.empty:
                    poi_frames.append(office_pois_radius)

            row: dict[str, Any] = {
                "officeID": office_id,
                "office_name": str(site["office_name"]),
                "address": str(site["address"]),
                "lat": float(site["lat"]),
                "lon": float(site["lon"]),
                transport_col: float(site[transport_col]) if pd.notna(site[transport_col]) else float("nan"),
            }
            for bucket in amenity_buckets:
                bucket_df = (
                    office_pois_radius[office_pois_radius["bucket"] == bucket]
                    if not office_pois_radius.empty
                    else pd.DataFrame()
                )
                row[count_column(bucket)] = int(len(bucket_df))
                row[nearest_distance_column(bucket)] = (
                    float(bucket_df["distance_m"].min()) if not bucket_df.empty else float("nan")
                )
            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)
        summary_df = _apply_weighted_kpi(
            summary_df=summary_df,
            selected_metrics=selected_metrics,
            weights_norm=weights_norm,
        )
        poi_df = pd.concat(poi_frames, ignore_index=True) if poi_frames else pd.DataFrame()

        by_radius[radius] = RadiusAnalysisRunResult(
            summary_df=summary_df,
            poi_df=poi_df,
            messages=[],
            cache_stats={"hits": 0, "misses": 0},
        )

    return MultiRadiusRunResult(
        by_radius=by_radius,
        weights_normalised=weights_norm,
        messages=all_messages,
        cache_stats={"hits": total_hits, "misses": total_misses},
    )


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def build_geojson_bytes(summary_df: pd.DataFrame, poi_df: pd.DataFrame, radius_m: int) -> tuple[Optional[bytes], str]:
    supported, message = geojson_export_supported()
    if not supported:
        return None, message
    if summary_df.empty:
        return None, "No results available for GeoJSON export."

    feature_collection = build_combined_geojson(
        sites_df=summary_df[["officeID", "address", "lat", "lon"]],
        poi_df=poi_df,
        radius_m=radius_m,
    )
    if feature_collection is None:
        return None, "GeoJSON export failed."
    return json.dumps(feature_collection).encode("utf-8"), "GeoJSON ready."


def build_cia_style_map(summary_df: pd.DataFrame, poi_df: pd.DataFrame, title: str) -> Optional[go.Figure]:
    if summary_df.empty:
        return None

    fig = go.Figure()

    if not poi_df.empty:
        for bucket in sorted(poi_df["bucket"].dropna().unique()):
            bucket_df = poi_df[poi_df["bucket"] == bucket]
            color = POI_COLORS.get(str(bucket), "#5b5f97")
            fig.add_scattermapbox(
                lat=bucket_df["poi_lat"],
                lon=bucket_df["poi_lon"],
                mode="markers",
                marker=dict(size=9, color=color, opacity=0.75),
                name=str(bucket),
                text=bucket_df["name"].replace("", np.nan).fillna(bucket),
                customdata=np.stack(
                    [
                        bucket_df.get("office_name", bucket_df["officeID"]).astype(str),
                        bucket_df["distance_m"].round(1).astype(str),
                    ],
                    axis=1,
                ),
                hovertemplate="POI: %{text}<br>Bucket: "
                + str(bucket)
                + "<br>Office: %{customdata[0]}<br>Distance (m): %{customdata[1]}<extra></extra>",
            )

    office_scores = pd.to_numeric(summary_df.get("amenity_kpi"), errors="coerce").fillna(50.0)
    fig.add_scattermapbox(
        lat=summary_df["lat"],
        lon=summary_df["lon"],
        mode="markers",
        marker=dict(
            size=24,
            color=office_scores,
            colorscale=[[0.0, "#374151"], [1.0, "#10B981"]],
            cmin=0,
            cmax=100,
            opacity=0.95,
            showscale=True,
            colorbar=dict(title="Overall score"),
            line=dict(color="#ffffff", width=2),
        ),
        name="Office",
        text=summary_df.get("office_name", summary_df["officeID"]).astype(str),
        customdata=np.stack(
            [
                summary_df["address"].astype(str),
                summary_df["amenity_kpi"].round(1).astype(str),
            ],
            axis=1,
        ),
        hovertemplate="Office: %{text}<br>Address: %{customdata[0]}<br>Overall score: %{customdata[1]}<extra></extra>",
    )

    center_lat = float(summary_df["lat"].mean())
    center_lon = float(summary_df["lon"].mean())
    fig.update_layout(
        title=title,
        mapbox_style="carto-positron",
        mapbox_zoom=10,
        mapbox_center={"lat": center_lat, "lon": center_lon},
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=0.01, xanchor="left", x=0.01),
        dragmode="zoom",
        height=720,
    )
    return fig


def _circle_points(lat: float, lon: float, radius_m: int, steps: int = 72) -> tuple[list[float], list[float]]:
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    angular_distance = radius_m / EARTH_RADIUS_M

    lats: list[float] = []
    lons: list[float] = []

    for i in range(steps + 1):
        bearing = 2.0 * math.pi * (i / steps)
        sin_lat = math.sin(lat_rad)
        cos_lat = math.cos(lat_rad)
        sin_ang = math.sin(angular_distance)
        cos_ang = math.cos(angular_distance)

        lat2 = math.asin(sin_lat * cos_ang + cos_lat * sin_ang * math.cos(bearing))
        lon2 = lon_rad + math.atan2(
            math.sin(bearing) * sin_ang * cos_lat,
            cos_ang - sin_lat * math.sin(lat2),
        )

        lats.append(math.degrees(lat2))
        lons.append(math.degrees(lon2))

    return lats, lons


def build_location_drilldown_map(
    office_row: pd.Series,
    poi_df: pd.DataFrame,
    radii_m: list[int],
    title: str,
) -> go.Figure:
    fig = go.Figure()

    lat = float(office_row["lat"])
    lon = float(office_row["lon"])

    if not poi_df.empty:
        for bucket in sorted(poi_df["bucket"].dropna().unique()):
            bucket_df = poi_df[poi_df["bucket"] == bucket]
            color = POI_COLORS.get(str(bucket), "#5b5f97")
            fig.add_scattermapbox(
                lat=bucket_df["poi_lat"],
                lon=bucket_df["poi_lon"],
                mode="markers",
                marker=dict(size=9, color=color, opacity=0.75),
                name=str(bucket),
                text=bucket_df["name"].replace("", np.nan).fillna(bucket),
                hovertemplate="POI: %{text}<br>Bucket: " + str(bucket) + "<extra></extra>",
            )

    for idx, radius in enumerate(sorted(radii_m)):
        circle_lats, circle_lons = _circle_points(lat=lat, lon=lon, radius_m=int(radius))
        fig.add_scattermapbox(
            lat=circle_lats,
            lon=circle_lons,
            mode="lines",
            line=dict(width=2, color=RADIUS_COLORS[idx % len(RADIUS_COLORS)]),
            name=f"{radius}m radius",
            hoverinfo="skip",
        )

    office_color = _rgb_css(_rgb_for_score(pd.to_numeric(office_row.get("amenity_kpi"), errors="coerce")))
    fig.add_scattermapbox(
        lat=[lat],
        lon=[lon],
        mode="markers",
        marker=dict(size=26, color=office_color, opacity=0.95, line=dict(color="#ffffff", width=2)),
        name="Selected office",
        text=[str(office_row.get("office_name", office_row["officeID"]))],
        customdata=[[str(office_row["address"]), str(round(float(office_row.get("amenity_kpi", np.nan)), 1))]],
        hovertemplate="Office: %{text}<br>Address: %{customdata[0]}<br>Overall score: %{customdata[1]}<extra></extra>",
    )

    fig.update_layout(
        title=title,
        mapbox_style="carto-positron",
        mapbox_zoom=12,
        mapbox_center={"lat": lat, "lon": lon},
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=0.01, xanchor="left", x=0.01),
        dragmode="zoom",
        height=760,
    )
    return fig


def build_office_scores(summary_df: pd.DataFrame, selected_metrics: list[str]) -> pd.DataFrame:
    """Build a consolidated office score table for mapping and insight panels."""
    if summary_df.empty:
        return pd.DataFrame(
            columns=[
                "officeID",
                "office_name",
                "lat",
                "lon",
                "address",
                "total_score",
            ]
        )

    office_scores = pd.DataFrame(
        {
            "officeID": summary_df["officeID"].astype(str),
            "office_name": summary_df.get("office_name", summary_df["address"]).astype(str),
            "lat": pd.to_numeric(summary_df["lat"], errors="coerce"),
            "lon": pd.to_numeric(summary_df["lon"], errors="coerce"),
            "address": summary_df["address"].astype(str),
            "total_score": pd.to_numeric(summary_df["amenity_kpi"], errors="coerce"),
        }
    )
    office_scores["office_name"] = office_scores["office_name"].replace("", np.nan).fillna(
        office_scores["officeID"]
    )

    for metric in selected_metrics:
        slug = bucket_slug(metric)
        norm_col = f"normalised_{slug}"
        subscore_col = f"subscore_{slug}"
        if norm_col in summary_df.columns:
            office_scores[subscore_col] = pd.to_numeric(summary_df[norm_col], errors="coerce").fillna(0.0)
        elif metric == TRANSPORT_METRIC:
            office_scores[subscore_col] = _compute_transport_normalised(summary_df).fillna(0.0)
        else:
            office_scores[subscore_col] = 0.0

        count_src_col = count_column(metric)
        count_out_col = f"count_{slug}"
        if count_src_col in summary_df.columns:
            office_scores[count_out_col] = (
                pd.to_numeric(summary_df[count_src_col], errors="coerce").fillna(0).astype(int)
            )
        elif metric == TRANSPORT_METRIC:
            office_scores[count_out_col] = np.nan
        else:
            office_scores[count_out_col] = 0

    transport_col = "nearest_public_transport_stop_distance_m"
    if transport_col in summary_df.columns:
        office_scores[transport_col] = pd.to_numeric(summary_df[transport_col], errors="coerce")

    return office_scores


def _rgb_for_score(score: float | int | None) -> list[int]:
    """Map score (0-100) to low-grey -> high-green RGB color."""
    if score is None or pd.isna(score):
        return [156, 163, 175]
    s = float(np.clip(score, 0.0, 100.0))
    t = s / 100.0
    low = [55, 65, 81]   # #374151
    high = [16, 185, 129]  # #10B981
    r = low[0] + (high[0] - low[0]) * t
    g = low[1] + (high[1] - low[1]) * t
    b = low[2] + (high[2] - low[2]) * t
    return [int(r), int(g), int(b)]


def _rgb_css(rgb: list[int]) -> str:
    """Return CSS rgb() string from integer RGB triplet."""
    return f"rgb({int(rgb[0])},{int(rgb[1])},{int(rgb[2])})"


def _top_two_categories_for_office(office_row: pd.Series, selected_categories: list[str]) -> str:
    scored: list[tuple[str, float]] = []
    for category in selected_categories:
        slug = bucket_slug(category)
        sub_col = f"subscore_{slug}"
        if sub_col in office_row.index:
            scored.append((category, float(office_row[sub_col])))
    if not scored:
        return "N/A"
    top = sorted(scored, key=lambda item: item[1], reverse=True)[:2]
    return ", ".join(name for name, _ in top)


def build_amenity_points(
    poi_df: pd.DataFrame,
    office_scores_df: pd.DataFrame,
    selected_categories: list[str],
    weights_norm: dict[str, float],
) -> pd.DataFrame:
    """Build amenity point table with consistent schema for maps and comparison visuals."""
    columns = [
        "officeID",
        "office_name",
        "category",
        "name",
        "lat",
        "lon",
        "distance_m",
        "weight_contribution",
    ]
    if poi_df.empty:
        return pd.DataFrame(columns=columns)

    points = pd.DataFrame(
        {
            "officeID": poi_df["officeID"].astype(str),
            "office_name": poi_df.get("office_name", poi_df["officeID"]).astype(str),
            "category": poi_df["bucket"].astype(str),
            "name": poi_df.get("name", "").astype(str),
            "lat": pd.to_numeric(poi_df["poi_lat"], errors="coerce"),
            "lon": pd.to_numeric(poi_df["poi_lon"], errors="coerce"),
            "distance_m": pd.to_numeric(poi_df["distance_m"], errors="coerce"),
        }
    )
    points = points[points["category"].isin(selected_categories)].dropna(subset=["lat", "lon"]).copy()
    if points.empty:
        return pd.DataFrame(columns=columns)

    office_counts = office_scores_df.copy()
    for category in selected_categories:
        slug = bucket_slug(category)
        count_col = f"count_{slug}"
        if count_col not in office_counts.columns:
            office_counts[count_col] = 0

    weight_contributions: list[float] = []
    for _, row in points.iterrows():
        category = str(row["category"])
        office_id = str(row["officeID"])
        slug = bucket_slug(category)
        count_col = f"count_{slug}"
        office_row = office_counts[office_counts["officeID"].astype(str) == office_id]
        category_count = float(office_row.iloc[0][count_col]) if not office_row.empty else 0.0
        weight = float(weights_norm.get(category, 0.0))
        weight_contributions.append(weight / category_count if category_count > 0 else 0.0)

    points["weight_contribution"] = weight_contributions
    points["name"] = points["name"].replace("", np.nan).fillna(points["category"])
    return points[columns].reset_index(drop=True)


def _build_ring_paths(office_scores_df: pd.DataFrame, ring_radii: list[int]) -> pd.DataFrame:
    """Create ring paths for each office/radius combination for pydeck PathLayer."""
    ring_rows: list[dict[str, Any]] = []
    safe_radii = sorted({int(radius) for radius in ring_radii if int(radius) > 0})
    if not safe_radii:
        return pd.DataFrame(columns=["officeID", "radius_m", "path", "color"])

    for _, office in office_scores_df.iterrows():
        lat = float(office["lat"])
        lon = float(office["lon"])
        for ridx, radius_m in enumerate(safe_radii):
            lats, lons = _circle_points(lat=lat, lon=lon, radius_m=radius_m, steps=72)
            path = [[float(lon_val), float(lat_val)] for lon_val, lat_val in zip(lons, lats)]
            ring_rows.append(
                {
                    "officeID": str(office["officeID"]),
                    "radius_m": int(radius_m),
                    "path": path,
                    "color": RADIUS_COLORS[ridx % len(RADIUS_COLORS)],
                }
            )
    return pd.DataFrame(ring_rows)


def _hex_to_rgb_tuple(color_hex: str) -> tuple[int, int, int]:
    raw = color_hex.lstrip("#")
    if len(raw) != 6:
        return (91, 95, 151)
    return tuple(int(raw[i : i + 2], 16) for i in (0, 2, 4))


def _prepare_office_map_frame(office_scores_df: pd.DataFrame, selected_categories: list[str]) -> pd.DataFrame:
    map_df = office_scores_df.copy()
    map_df["office_color"] = map_df["total_score"].apply(_rgb_for_score)
    map_df["top_categories"] = map_df.apply(
        lambda row: _top_two_categories_for_office(row, selected_categories), axis=1
    )
    return map_df


def _mercator_y(lat_deg: float) -> float:
    """Convert latitude in degrees to Web Mercator Y."""
    clamped = max(min(float(lat_deg), 85.0511), -85.0511)
    lat_rad = math.radians(clamped)
    return math.log(math.tan((math.pi / 4.0) + (lat_rad / 2.0)))


def _fit_view_state(office_df: pd.DataFrame, pad_m: int):
    """Build a map view that fits all offices with a radius pad."""
    if office_df.empty:
        return pdk.ViewState(latitude=51.5074, longitude=-0.1278, zoom=10, pitch=0)

    lats = pd.to_numeric(office_df["lat"], errors="coerce").dropna().to_numpy(dtype=float)
    lons = pd.to_numeric(office_df["lon"], errors="coerce").dropna().to_numpy(dtype=float)
    if len(lats) == 0 or len(lons) == 0:
        return pdk.ViewState(latitude=51.5074, longitude=-0.1278, zoom=10, pitch=0)

    safe_pad = max(int(pad_m), 0)
    north = -90.0
    south = 90.0
    east = -180.0
    west = 180.0
    for lat, lon in zip(lats, lons):
        lat_pad = safe_pad / 111_320.0
        cos_lat = max(abs(math.cos(math.radians(lat))), 0.15)
        lon_pad = safe_pad / (111_320.0 * cos_lat)
        north = max(north, lat + lat_pad)
        south = min(south, lat - lat_pad)
        east = max(east, lon + lon_pad)
        west = min(west, lon - lon_pad)

    center_lat = (north + south) / 2.0
    center_lon = (east + west) / 2.0
    lon_span = max(east - west, 1e-4)
    mercator_span = max(abs(_mercator_y(north) - _mercator_y(south)), 1e-6)

    viewport_w_px = 1100.0
    viewport_h_px = 650.0
    zoom_lon = math.log2((360.0 * viewport_w_px) / (256.0 * lon_span))
    zoom_lat = math.log2((2.0 * math.pi * viewport_h_px) / (256.0 * mercator_span))
    zoom = min(zoom_lon, zoom_lat) - 0.85
    if len(lats) == 1:
        # Single-office views feel too tight with strict bounds; step back one more level.
        zoom -= 0.9
    zoom = float(max(2.0, min(13.0, zoom)))
    return pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=zoom, pitch=0)


def render_points_map(
    office_scores_df: pd.DataFrame,
    amenity_points_df: pd.DataFrame,
    radius_m: int,
    selected_categories: list[str],
    ring_radii: Optional[list[int]] = None,
):
    """Render points map deck (amenity points + office points + radius rings)."""
    if not HAS_PYDECK:
        return None
    if office_scores_df.empty:
        return None

    office_df = _prepare_office_map_frame(office_scores_df, selected_categories)
    points_df = amenity_points_df[amenity_points_df["category"].isin(selected_categories)].copy()
    if not points_df.empty:
        points_df["cat_color"] = points_df["category"].map(
            lambda c: list(_hex_to_rgb_tuple(POI_COLORS.get(str(c), "#5b5f97")))
        )

    rings_df = _build_ring_paths(
        office_scores_df=office_df,
        ring_radii=ring_radii if ring_radii is not None else [int(radius_m)],
    )

    layers: list[Any] = []
    if not points_df.empty:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=points_df,
                get_position="[lon, lat]",
                get_fill_color="cat_color",
                get_radius=18,
                radius_min_pixels=2,
                radius_max_pixels=6,
                opacity=0.45,
                pickable=True,
            )
        )
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=office_df,
            get_position="[lon, lat]",
            get_fill_color="office_color",
            get_line_color=[255, 255, 255],
            line_width_min_pixels=2,
            stroked=True,
            get_radius=50,
            radius_min_pixels=7,
            radius_max_pixels=14,
            opacity=0.95,
            pickable=True,
        )
    )
    if not rings_df.empty:
        layers.append(
            pdk.Layer(
                "PathLayer",
                data=rings_df,
                get_path="path",
                get_color=[120, 120, 120],
                width_min_pixels=1,
                pickable=False,
            )
        )

    safe_ring_radii = ring_radii if ring_radii is not None else [int(radius_m)]
    max_ring = max([int(radius) for radius in safe_ring_radii], default=int(radius_m))
    view_state = _fit_view_state(office_df=office_df, pad_m=max_ring + 500)
    return pdk.Deck(
        map_provider="carto",
        map_style="light",
        initial_view_state=view_state,
        layers=layers,
        tooltip={
            "html": "<b>{office_name}</b><br/>ID: {officeID}<br/>Address: {address}<br/>Overall score: {total_score}<br/>Top categories: {top_categories}"
        },
    )


def render_density_map(
    office_scores_df: pd.DataFrame,
    amenity_points_df: pd.DataFrame,
    radius_m: int,
    selected_categories: list[str],
    density_type: str = "Hex",
    ring_radii: Optional[list[int]] = None,
):
    """Render density map deck (Hexagon or Heatmap) with office points and radius rings."""
    if not HAS_PYDECK:
        return None
    if office_scores_df.empty:
        return None

    office_df = _prepare_office_map_frame(office_scores_df, selected_categories)
    points_df = amenity_points_df[amenity_points_df["category"].isin(selected_categories)].copy()
    rings_df = _build_ring_paths(
        office_scores_df=office_df,
        ring_radii=ring_radii if ring_radii is not None else [int(radius_m)],
    )

    is_hex = str(density_type).lower() == "hex"
    layers: list[Any] = []
    if not points_df.empty:
        if not is_hex:
            layers.append(
                pdk.Layer(
                    "HeatmapLayer",
                    data=points_df,
                    get_position="[lon, lat]",
                    get_weight="weight_contribution",
                    radius_pixels=45,
                    intensity=1.0,
                    threshold=0.05,
                    color_range=DENSITY_COLOR_RANGE,
                )
            )
        else:
            layers.append(
                pdk.Layer(
                    "HexagonLayer",
                    data=points_df,
                    get_position="[lon, lat]",
                    elevation_scale=10,
                    elevation_range=[0, 180],
                    radius=60,
                    extruded=False,
                    coverage=0.65,
                    color_range=DENSITY_COLOR_RANGE,
                    pickable=False,
                    auto_highlight=False,
                )
            )

    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=office_df,
            get_position="[lon, lat]",
            get_fill_color="office_color",
            get_line_color=[255, 255, 255],
            line_width_min_pixels=2,
            stroked=True,
            get_radius=55,
            radius_min_pixels=8,
            radius_max_pixels=14,
            opacity=0.98,
            pickable=True,
        )
    )
    if not rings_df.empty:
        layers.append(
            pdk.Layer(
                "PathLayer",
                data=rings_df,
                get_path="path",
                get_color=[110, 110, 110],
                width_min_pixels=1,
                pickable=False,
            )
        )

    safe_ring_radii = ring_radii if ring_radii is not None else [int(radius_m)]
    max_ring = max([int(radius) for radius in safe_ring_radii], default=int(radius_m))
    view_state = _fit_view_state(office_df=office_df, pad_m=max_ring + 500)
    tooltip = {
        "html": "<b>{office_name}</b><br/>ID: {officeID}<br/>Address: {address}<br/>Overall score: {total_score}<br/>Top categories: {top_categories}"
    }

    return pdk.Deck(
        map_provider="carto",
        map_style="light",
        initial_view_state=view_state,
        layers=layers,
        tooltip=tooltip,
    )


def render_tradeoff_panel(
    office_scores_df: pd.DataFrame,
    selected_office_id: str,
    selected_categories: list[str],
) -> None:
    """Render single-office trade-off chart against best office and portfolio average."""
    if office_scores_df.empty:
        st.info("No office scores available.")
        return

    office_df = office_scores_df.copy()
    office_df["officeID"] = office_df["officeID"].astype(str)
    selected = office_df[office_df["officeID"] == str(selected_office_id)]
    if selected.empty:
        st.info("Selected office is not present in current results.")
        return
    selected_row = selected.iloc[0]

    best_idx = office_df["total_score"].astype(float).idxmax()
    best_row = office_df.loc[best_idx]
    avg_row = office_df.mean(numeric_only=True)

    chart_rows: list[dict[str, Any]] = []
    for category in selected_categories:
        slug = bucket_slug(category)
        sub_col = f"subscore_{slug}"
        chart_rows.append(
            {
                "Category": category,
                "Selected office": float(selected_row.get(sub_col, 0.0)),
                "Best office": float(best_row.get(sub_col, 0.0)),
                "Portfolio average": float(avg_row.get(sub_col, 0.0)),
            }
        )
    chart_df = pd.DataFrame(chart_rows)
    if chart_df.empty:
        st.info("No category subscores available.")
        return

    long_df = chart_df.melt(
        id_vars=["Category"],
        value_vars=["Selected office", "Best office", "Portfolio average"],
        var_name="Series",
        value_name="Subscore",
    )
    fig = px.bar(
        long_df,
        x="Category",
        y="Subscore",
        color="Series",
        barmode="group",
        color_discrete_sequence=["#2563eb", "#059669", "#94a3b8"],
        title="Category trade-off (subscores 0-1)",
    )
    fig.update_yaxes(range=[0, 1])
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0), height=320)
    st.plotly_chart(fig, use_container_width=True)

    diff_df = chart_df.copy()
    diff_df["Gap vs best"] = (diff_df["Selected office"] - diff_df["Best office"]).round(3)
    diff_df["Gap vs average"] = (diff_df["Selected office"] - diff_df["Portfolio average"]).round(3)
    st.dataframe(diff_df[["Category", "Gap vs best", "Gap vs average"]], use_container_width=True)

    st.caption(
        "Selected: {0} | Best: {1}".format(
            str(selected_row.get("office_name", selected_row["officeID"])),
            str(best_row.get("office_name", best_row["officeID"])),
        )
    )


def _comparison_count_table(
    office_a: pd.Series,
    office_b: pd.Series,
    selected_categories: list[str],
    office_a_name: str,
    office_b_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for category in selected_categories:
        slug = bucket_slug(category)
        count_col = f"count_{slug}"
        a_count = float(office_a.get(count_col, 0.0))
        b_count = float(office_b.get(count_col, 0.0))
        rows.append(
            {
                "Category": category,
                f"{office_a_name} count": int(a_count) if not pd.isna(a_count) else 0,
                f"{office_b_name} count": int(b_count) if not pd.isna(b_count) else 0,
                f"Delta ({office_a_name}-{office_b_name})": int(a_count - b_count)
                if not (pd.isna(a_count) or pd.isna(b_count))
                else 0,
            }
        )
    return pd.DataFrame(rows)


def render_comparison_panel(
    office_scores_df: pd.DataFrame,
    office_a_id: str,
    office_b_id: str,
    selected_categories: list[str],
    amenity_points_df: Optional[pd.DataFrame] = None,
    top_n: int = 0,
) -> None:
    """Render comparison mode visuals between two offices.

    Legacy args `amenity_points_df` and `top_n` are accepted but unused.
    """
    if office_scores_df.empty:
        st.info("No office scores available.")
        return

    scores = office_scores_df.copy()
    scores["officeID"] = scores["officeID"].astype(str)
    office_a = scores[scores["officeID"] == str(office_a_id)]
    office_b = scores[scores["officeID"] == str(office_b_id)]
    if office_a.empty or office_b.empty:
        st.info("Selected comparison offices are not available.")
        return

    row_a = office_a.iloc[0]
    row_b = office_b.iloc[0]
    office_a_name = str(row_a.get("office_name", office_a_id))
    office_b_name = str(row_b.get("office_name", office_b_id))

    a_score = float(row_a.get("total_score", np.nan))
    b_score = float(row_b.get("total_score", np.nan))
    delta = a_score - b_score

    m1, m2, m3 = st.columns(3)
    m1.metric(office_a_name, f"{a_score:.1f}")
    m2.metric(office_b_name, f"{b_score:.1f}")
    m3.metric(f"Delta ({office_a_name} - {office_b_name})", f"{delta:+.1f}")

    delta_rows: list[dict[str, Any]] = []
    for category in selected_categories:
        slug = bucket_slug(category)
        sub_col = f"subscore_{slug}"
        delta_rows.append(
            {
                "Category": category,
                "Delta": float(row_a.get(sub_col, 0.0)) - float(row_b.get(sub_col, 0.0)),
            }
        )
    delta_df = pd.DataFrame(delta_rows)

    fig = px.bar(
        delta_df,
        x="Category",
        y="Delta",
        color="Delta",
        color_continuous_scale=["#c0392b", "#f1c40f", "#1e8449"],
        title=f"Category deltas ({office_a_name} - {office_b_name})",
    )
    fig.update_layout(margin=dict(l=0, r=0, t=45, b=0), height=280, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    count_df = _comparison_count_table(
        office_a=row_a,
        office_b=row_b,
        selected_categories=selected_categories,
        office_a_name=office_a_name,
        office_b_name=office_b_name,
    )
    st.dataframe(count_df, use_container_width=True)
