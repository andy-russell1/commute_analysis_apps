from __future__ import annotations

from typing import Any, Optional, Tuple

import pandas as pd

try:
    import geopandas as gpd
    from shapely.geometry import Point, mapping

    HAS_GEO_EXPORT = True
except Exception:
    HAS_GEO_EXPORT = False
    gpd = None
    Point = None
    mapping = None


def geojson_export_supported() -> Tuple[bool, str]:
    if HAS_GEO_EXPORT:
        return True, "GeoJSON export is available."
    return False, "GeoJSON export requires geopandas/shapely."


def build_buffers_geojson(sites_df: pd.DataFrame, radius_m: int) -> Optional[dict[str, Any]]:
    if not HAS_GEO_EXPORT:
        return None

    points = [Point(lon, lat) for lat, lon in zip(sites_df["lat"], sites_df["lon"])]
    gdf = gpd.GeoDataFrame(sites_df[["officeID", "address"]].copy(), geometry=points, crs="EPSG:4326")
    projected = gdf.to_crs(epsg=3857)
    projected["geometry"] = projected.buffer(radius_m)
    reprojected = projected.to_crs(epsg=4326)

    features = []
    for _, row in reprojected.iterrows():
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "feature_type": "office_buffer",
                    "officeID": row["officeID"],
                    "address": row["address"],
                    "radius_m": radius_m,
                },
                "geometry": mapping(row.geometry),
            }
        )
    return {"type": "FeatureCollection", "features": features}


def build_pois_geojson(poi_df: pd.DataFrame) -> Optional[dict[str, Any]]:
    if not HAS_GEO_EXPORT:
        return None
    if poi_df.empty:
        return {"type": "FeatureCollection", "features": []}

    points = [Point(lon, lat) for lat, lon in zip(poi_df["poi_lat"], poi_df["poi_lon"])]
    gdf = gpd.GeoDataFrame(poi_df.copy(), geometry=points, crs="EPSG:4326")

    features = []
    for _, row in gdf.iterrows():
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "feature_type": "poi",
                    "officeID": row.get("officeID"),
                    "bucket": row.get("bucket"),
                    "name": row.get("name"),
                    "distance_m": row.get("distance_m"),
                    "osm_id": row.get("osm_id"),
                    "osm_type": row.get("osm_type"),
                },
                "geometry": mapping(row.geometry),
            }
        )
    return {"type": "FeatureCollection", "features": features}


def build_combined_geojson(sites_df: pd.DataFrame, poi_df: pd.DataFrame, radius_m: int) -> Optional[dict[str, Any]]:
    buffers = build_buffers_geojson(sites_df=sites_df, radius_m=radius_m)
    pois = build_pois_geojson(poi_df=poi_df)
    if buffers is None or pois is None:
        return None
    return {"type": "FeatureCollection", "features": [*buffers["features"], *pois["features"]]}
