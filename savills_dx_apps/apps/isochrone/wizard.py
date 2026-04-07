from __future__ import annotations

import hashlib
from typing import Optional

import geopandas as gpd
import pandas as pd
import streamlit as st
import folium
from folium.plugins import MarkerCluster
import streamlit.components.v1 as components

from shared.runtime.downloads import df_to_csv_bytes
from shared.ui.kpi import render_kpi_strip
from shared.ui.page_header import render_page_header
from apps.isochrone.io import load_isochrones_from_zip, validate_isochrone_zip
from shared.runtime.models import AppArtifacts, AppMetadata, AppPlugin, UploadPayload
from shared.runtime.paths import DATA_DIR


ISOCHRONE_COLORS = [
    "#2ecc71",
    "#f1c40f",
    "#f39c12",
    "#e67e22",
    "#e74c3c",
    "#c0392b",
]
OFFICE_NAME_COL = "Office___A"
OFFICE_LAT_COL = "Office___L"
OFFICE_LON_COL = "Office___2"
CORE_BANDS = [30.0, 45.0, 60.0]
CORE_MODE_LABEL = "Core (30 / 45 / 60)"
EXTENDED_MODE_LABEL = "Extended (all uploaded bands)"
MAP_RENDER_VERSION = "lad-boundaries-hidden-by-default-v2"

POP_AGE_COLS = [
    "Aged 15 to 19 years",
    "Aged 20 to 24 years",
    "Aged 25 to 29 years",
    "Aged 30 to 34 years",
    "Aged 35 to 39 years",
    "Aged 40 to 44 years",
    "Aged 45 to 49 years",
    "Aged 50 to 54 years",
    "Aged 55 to 59 years",
    "Aged 60 to 64 years",
]


@st.cache_data(show_spinner=False)
def load_population_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_lookup_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str)


@st.cache_data(show_spinner=False)
def load_lad_geojson(path: str) -> gpd.GeoDataFrame:
    return gpd.read_file(path)


@st.cache_data(show_spinner=False)
def load_lad_population(pop_path: str, lookup_path: str) -> pd.DataFrame:
    pop_df = load_population_csv(pop_path)
    lookup_df = load_lookup_csv(lookup_path)
    return _aggregate_population_by_lad(pop_df, lookup_df)


def _find_oa_column(df: pd.DataFrame) -> str:
    cols = {str(c).lower(): c for c in df.columns}
    for key in ("2021 output area", "output area", "oa21cd", "oa21"):
        if key in cols:
            return cols[key]
    if "mnemonic" in cols:
        return cols["mnemonic"]
    raise KeyError("No output area column found in population CSV.")


def _aggregate_population_by_lad(pop_df: pd.DataFrame, lookup_df: pd.DataFrame) -> pd.DataFrame:
    oa_col = _find_oa_column(pop_df)
    pop = pop_df.copy()
    pop[oa_col] = pop[oa_col].astype(str).str.strip()

    for col in POP_AGE_COLS + ["Total"]:
        if col in pop.columns:
            pop[col] = pd.to_numeric(pop[col], errors="coerce").fillna(0)

    lookup = lookup_df.copy()
    lookup["OA21CD"] = lookup["OA21CD"].astype(str).str.strip()

    merged = pop.merge(lookup, left_on=oa_col, right_on="OA21CD", how="left")
    if "LAD24CD" not in merged.columns:
        raise KeyError("Lookup file is missing LAD24CD.")

    value_cols = [c for c in POP_AGE_COLS + ["Total"] if c in merged.columns]
    agg = (
        merged.groupby(["LAD24CD", "LAD24NM"], dropna=False)[value_cols]
        .sum()
        .reset_index()
    )
    return agg


def _find_office_col(iso: gpd.GeoDataFrame) -> Optional[str]:
    candidates = [
        "address",
        "Office_Name",
        "Office",
        "OfficeName",
        "Office___Na",
        "Offie___Na",
        OFFICE_NAME_COL,
        "officeID",
        "office_id",
        "officeid",
    ]
    cols = {str(c).lower(): c for c in iso.columns}
    for col in candidates:
        if col in iso.columns:
            return col
        lower = str(col).lower()
        if lower in cols:
            return cols[lower]
    return None


def _get_col_case_insensitive(df: pd.DataFrame, name: str) -> Optional[str]:
    cols = {str(c).lower(): c for c in df.columns}
    return cols.get(name.lower())


def _find_office_lat_lon_cols(iso: gpd.GeoDataFrame) -> Optional[tuple[str, str]]:
    cols = [str(c) for c in iso.columns]
    lat_candidates = [c for c in cols if "office" in c.lower() and "lat" in c.lower()]
    lon_candidates = [
        c for c in cols if "office" in c.lower() and ("lon" in c.lower() or "long" in c.lower())
    ]
    if lat_candidates and lon_candidates:
        return lat_candidates[0], lon_candidates[0]
    return None


def _age_columns(gdf: gpd.GeoDataFrame) -> tuple[Optional[list[str]], Optional[str]]:
    age_cols = [c for c in POP_AGE_COLS if c in gdf.columns]
    if len(age_cols) == len(POP_AGE_COLS):
        return age_cols, ", ".join(age_cols)
    return None, None


def _format_band_minutes(time_col: str, value: object) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if time_col == "Query_Isoc":
        return numeric / 60.0
    return numeric


def _minutes_series(df: pd.DataFrame, time_col: str) -> pd.Series:
    values = pd.to_numeric(df[time_col], errors="coerce")
    if time_col == "Query_Isoc":
        return values / 60.0
    return values


def _normalise_band_value(value: float) -> float:
    return round(float(value), 6)


def _available_band_minutes(iso: gpd.GeoDataFrame, time_col: str) -> list[float]:
    if time_col not in iso.columns:
        return []
    minutes = _minutes_series(iso, time_col)
    minutes = minutes[pd.to_numeric(minutes, errors="coerce").notna()]
    minutes = minutes[minutes >= 0]
    return sorted({_normalise_band_value(value) for value in minutes.tolist()})


def _resolve_display_bands(available_bands: list[float], analysis_mode: str) -> tuple[list[float], list[float]]:
    if analysis_mode == CORE_MODE_LABEL:
        core = [_normalise_band_value(value) for value in CORE_BANDS]
        available_set = {_normalise_band_value(value) for value in available_bands}
        display = [value for value in core if value in available_set]
        missing = [value for value in core if value not in available_set]
        return display, missing
    return list(available_bands), []


def _format_minutes_label(value: float) -> str:
    numeric = float(value)
    if numeric.is_integer():
        return "{0:.0f}".format(numeric)
    return "{0:g}".format(numeric)


def _format_band_title(value: float) -> str:
    return "Residents within {0} min".format(_format_minutes_label(value))


def _format_interval_band_label(current_band: float, previous_band: Optional[float]) -> str:
    if previous_band is None:
        return "<={0} min".format(_format_minutes_label(current_band))
    return "{0}-{1} min".format(_format_minutes_label(previous_band), _format_minutes_label(current_band))


def _format_transport_label(value: object) -> str:
    label = str(value).replace("_", " ").replace("+", " and ").title()
    return label.replace(" And ", " and ")


def _mix_channel(start: int, end: int, ratio: float) -> int:
    return int(round(start + (end - start) * ratio))


def _mix_hex(start: str, end: str, ratio: float) -> str:
    start = start.lstrip("#")
    end = end.lstrip("#")
    ratio = max(0.0, min(1.0, float(ratio)))
    channels = [
        _mix_channel(int(start[index:index + 2], 16), int(end[index:index + 2], 16), ratio)
        for index in (0, 2, 4)
    ]
    return "#{0:02x}{1:02x}{2:02x}".format(*channels)


def _build_band_palette(bands: list[float]) -> dict[float, str]:
    if not bands:
        return {}

    green = "#2ecc71"
    amber = "#f1c40f"
    red = "#c0392b"
    count = len(bands)
    colors: list[str] = []
    if count == 1:
        colors = [green]
    else:
        for index in range(count):
            position = index / float(count - 1)
            if position <= 0.5:
                color = _mix_hex(green, amber, position / 0.5 if position else 0.0)
            else:
                color = _mix_hex(amber, red, (position - 0.5) / 0.5)
            colors.append(color)
    return {float(band): color for band, color in zip(bands, colors)}


def _format_increment_note(current_value: float, previous_value: Optional[float], previous_band: Optional[float]) -> str:
    if previous_value is None or previous_band is None:
        return "Fastest available band"
    delta = current_value - previous_value
    if abs(delta) < 0.5:
        return "No material change vs {0} min".format(_format_minutes_label(previous_band))
    sign = "+" if delta >= 0 else "-"
    return "{0}{1:,.0f} vs {2} min".format(sign, abs(delta), _format_minutes_label(previous_band))


def _build_band_summary_items(pop_counts: dict[float, float]) -> list[dict[str, object]]:
    ordered_bands = sorted(pop_counts)
    items: list[dict[str, object]] = []
    previous_band: Optional[float] = None
    previous_value: Optional[float] = None
    palette = _build_band_palette(ordered_bands)
    for band in ordered_bands:
        current_value = float(pop_counts.get(band, 0.0))
        items.append(
            {
                "band": float(band),
                "label": _format_band_title(band),
                "value": "{0:,.0f}".format(current_value),
                "note": _format_increment_note(current_value, previous_value, previous_band),
                "color": palette.get(float(band), ISOCHRONE_COLORS[0]),
            }
        )
        previous_band = float(band)
        previous_value = current_value
    return items


def _render_population_summary(pop_counts: dict[float, float], analysis_mode: str) -> None:
    if not pop_counts:
        return

    ordered_counts = {float(band): float(pop_counts[band]) for band in sorted(pop_counts)}
    if analysis_mode == CORE_MODE_LABEL:
        render_kpi_strip(
            [(_format_band_title(band), "{0:,.0f}".format(value)) for band, value in ordered_counts.items()],
            columns=max(len(ordered_counts), 1),
        )
        return

    items = _build_band_summary_items(ordered_counts)
    render_kpi_strip(
        [
            (item["label"], item["value"], item["note"], item["color"])
            for item in items
        ],
        columns=4,
    )


def _build_interval_band_geometries(iso: gpd.GeoDataFrame, bands: list[float]) -> list[tuple[float, Optional[float], object]]:
    intervals: list[tuple[float, Optional[float], object]] = []
    previous_band: Optional[float] = None
    previous_union = None

    for band in sorted(bands):
        cumulative = iso[iso["_minutes"] <= float(band)]
        if cumulative.empty:
            previous_band = float(band)
            continue

        current_union = cumulative.geometry.unary_union
        if current_union is None or current_union.is_empty:
            previous_band = float(band)
            continue

        current_union = current_union.buffer(0)
        interval_geom = current_union
        if previous_union is not None and not previous_union.is_empty:
            interval_geom = current_union.difference(previous_union)
        if interval_geom is None or interval_geom.is_empty:
            previous_band = float(band)
            previous_union = current_union
            continue

        intervals.append((float(band), previous_band, interval_geom))
        previous_band = float(band)
        previous_union = current_union

    return intervals


def _compute_map_geometry_payload(
    isochrones: gpd.GeoDataFrame,
    tran: str,
    display_bands: Optional[list[float]] = None,
) -> dict[str, object]:
    iso = isochrones.copy()
    time_col = "Query_Time" if "Query_Time" in iso.columns else "Query_Isoc"
    if time_col not in iso.columns:
        raise ValueError("Isochrone data is missing 'Query_Time' or 'Query_Isoc'")

    iso["_minutes"] = _minutes_series(iso, time_col).apply(
        lambda value: _normalise_band_value(value) if pd.notna(value) else value
    )
    iso = iso[pd.to_numeric(iso["_minutes"], errors="coerce").notna()].copy()

    if "Query_Tran" in iso.columns:
        iso = iso[iso["Query_Tran"] == tran]
    else:
        raise ValueError("Isochrone data is missing 'Query_Tran'")

    iso_all = iso.copy()
    if display_bands is not None:
        allowed = {_normalise_band_value(value) for value in display_bands}
        iso = iso[iso["_minutes"].isin(allowed)].copy()

    if iso_all.empty:
        raise ValueError("No isochrones found for Query_Tran = '{0}'".format(tran))

    times_sorted = sorted({_normalise_band_value(value) for value in iso["_minutes"].tolist()})
    minx, miny, maxx, maxy = iso_all.total_bounds
    has_bounds = all(pd.notna(value) for value in [minx, miny, maxx, maxy]) and minx != maxx and miny != maxy
    center = iso_all.geometry.unary_union.centroid

    return {
        "color_map": _build_band_palette(times_sorted),
        "interval_geometries": _build_interval_band_geometries(iso, times_sorted),
        "bounds": ((float(miny), float(minx)), (float(maxy), float(maxx))) if has_bounds else None,
        "center": (float(center.y), float(center.x)),
    }


def _cache_namespace(name: str) -> dict:
    cache = st.session_state.setdefault("_isochrone_cache", {})
    return cache.setdefault(name, {})


def _ensure_total_population(lad_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    lad = lad_gdf.copy()
    if "Total" in lad.columns:
        lad["Total"] = pd.to_numeric(lad["Total"], errors="coerce").fillna(0)
        return lad
    age_cols = [c for c in POP_AGE_COLS if c in lad.columns]
    if not age_cols:
        raise ValueError("Population data is missing a Total column and age columns.")
    for c in age_cols:
        lad[c] = pd.to_numeric(lad[c], errors="coerce").fillna(0)
    lad["Total"] = lad[age_cols].sum(axis=1)
    return lad


def _compute_population_within_bands(
    lad_gdf: gpd.GeoDataFrame,
    iso_filtered: gpd.GeoDataFrame,
    time_col: str,
    thresholds: list[float],
) -> dict[float, float]:
    lad = _ensure_total_population(lad_gdf)
    if lad.crs is None:
        raise ValueError("LAD boundaries are missing CRS.")

    lad_proj = lad.to_crs("EPSG:27700")
    iso_proj = iso_filtered.to_crs("EPSG:27700")

    if time_col == "Query_Isoc":
        iso_proj["_minutes"] = pd.to_numeric(iso_proj[time_col], errors="coerce") / 60.0
    else:
        iso_proj["_minutes"] = pd.to_numeric(iso_proj[time_col], errors="coerce")

    lad_proj["lad_area"] = lad_proj.geometry.area

    results: dict[float, float] = {}
    for threshold in thresholds:
        band = iso_proj[iso_proj["_minutes"] <= threshold].copy()
        if band.empty:
            results[threshold] = 0.0
            continue
        union_geom = band.unary_union
        band_gdf = gpd.GeoDataFrame({"geometry": [union_geom]}, crs=iso_proj.crs)
        inter = gpd.overlay(lad_proj, band_gdf, how="intersection")
        if inter.empty:
            results[threshold] = 0.0
            continue
        inter_area = inter.geometry.area
        inter["area_frac"] = inter_area / inter["lad_area"]
        inter["pop_within"] = inter["Total"] * inter["area_frac"]
        results[threshold] = float(inter["pop_within"].sum())
    return results


def build_folium_map(
    gdf: gpd.GeoDataFrame,
    isochrones: gpd.GeoDataFrame,
    tran: str,
    display_bands: Optional[list[float]] = None,
    show_markers: bool = True,
    show_office_marker: bool = True,
    postcode_col: str = "Postcode District",
    office_point: Optional[tuple] = None,
    lad_boundaries: Optional[gpd.GeoDataFrame] = None,
    show_isochrones: bool = True,
    map_payload: Optional[dict[str, object]] = None,
) -> folium.Map:
    gdf = gdf.copy()
    payload = map_payload or _compute_map_geometry_payload(
        isochrones=isochrones,
        tran=tran,
        display_bands=display_bands,
    )
    center = payload["center"]
    m = folium.Map(location=[center[0], center[1]], zoom_start=11)

    folium.TileLayer(
        "CartoDB positron",
        name="CartoDB Positron",
        attr="Map tiles by Carto, under CC BY 3.0 - Map data OpenStreetMap contributors",
    ).add_to(m)

    if lad_boundaries is not None and not lad_boundaries.empty:
        lad_group = folium.FeatureGroup(name="LAD boundaries", show=False)
        folium.GeoJson(
            lad_boundaries,
            style_function=lambda _: {"fillOpacity": 0.0, "color": "#5f6368", "weight": 0.6},
        ).add_to(lad_group)
        lad_group.add_to(m)

    bounds = payload.get("bounds")
    if bounds:
        m.fit_bounds([[bounds[0][0], bounds[0][1]], [bounds[1][0], bounds[1][1]]])

    interval_geometries = payload.get("interval_geometries", [])
    color_map = payload.get("color_map", {})
    if show_isochrones and interval_geometries:
        iso_group = folium.FeatureGroup(name="Isochrone Bands", show=True)
        for band_minutes, previous_band, geometry in reversed(interval_geometries):
            color = color_map.get(float(band_minutes), ISOCHRONE_COLORS[0])
            folium.GeoJson(
                geometry,
                style_function=lambda x, color=color: {
                    "fillColor": color,
                    "color": color,
                    "weight": 1,
                    "fillOpacity": 0.44,
                },
                tooltip="{0} - {1}".format(
                    _format_transport_label(tran),
                    _format_interval_band_label(band_minutes, previous_band),
                ),
            ).add_to(iso_group)
        iso_group.add_to(m)

    if show_markers:
        marker_cluster = MarkerCluster(name="Postcodes").add_to(m)
        for _, r in gdf.iterrows():
            lat = r.get("lat")
            lon = r.get("lon")
            if pd.isna(lat) or pd.isna(lon):
                continue
            label = r.get(postcode_col, "")
            folium.CircleMarker(
                location=[lat, lon],
                radius=3,
                color="#2c3e50",
                fill=True,
                fill_opacity=0.8,
                tooltip=label,
            ).add_to(marker_cluster)

    if show_office_marker and office_point:
        folium.Marker(
            location=[office_point[0], office_point[1]],
            icon=folium.Icon(color="red", icon="briefcase", prefix="fa"),
            tooltip="Office",
        ).add_to(m)

    folium.LayerControl(collapsed=True, position="bottomright").add_to(m)
    return m


def folium_to_html(m: folium.Map) -> str:
    return m.get_root().render()


class IsochronePlugin(AppPlugin):
    metadata = AppMetadata(
        id="isochrone",
        name="Isochrone Analysis",
        description="Upload a zipped shapefile of isochrones and explore coverage.",
        accepted_upload_types=["zip"],
        upload_label="Upload isochrone ZIP",
        upload_help="ZIP should contain .shp, .dbf, and .shx files.",
    )

    def validate(self, upload: UploadPayload) -> None:
        if upload.ext != "zip":
            raise ValueError("Isochrone Analysis expects a ZIP file.")
        validate_isochrone_zip(upload.bytes_data)

    def build(self, upload: UploadPayload, log) -> AppArtifacts:
        log("Reading isochrones from ZIP")
        isochrones = load_isochrones_from_zip(upload.bytes_data)
        if isochrones.crs is not None and str(isochrones.crs).lower() != "epsg:4326":
            log("Reprojecting to EPSG:4326")
            isochrones = isochrones.to_crs("EPSG:4326")

        pop_path = str(DATA_DIR / "census" / "population per output area.csv")
        lookup_path = str(DATA_DIR / "lookup" / "oa21_lad24_lookup.csv")
        lad_path = str(DATA_DIR / "geo" / "lad_uk_2024.geojson")

        log("Loading LAD boundaries")
        lad_gdf = load_lad_geojson(lad_path)
        lad_pop = load_lad_population(pop_path, lookup_path)

        if "LAD24CD" not in lad_gdf.columns:
            raise ValueError("LAD geojson is missing LAD24CD.")

        lad_gdf = lad_gdf.merge(lad_pop, on="LAD24CD", how="left")
        if lad_gdf.crs is None or str(lad_gdf.crs).lower() != "epsg:4326":
            lad_gdf = lad_gdf.to_crs("EPSG:4326")

        lad_points = lad_gdf.to_crs("EPSG:27700")
        lad_points["geometry"] = lad_points.geometry.centroid
        lad_points = lad_points.to_crs("EPSG:4326")

        return {
            "isochrones": isochrones,
            "lad_gdf": lad_gdf,
            "lad_points": lad_points,
            "upload_signature": hashlib.md5(upload.bytes_data).hexdigest(),
        }

    def render(self, artifacts: AppArtifacts) -> None:
        isochrones = artifacts["isochrones"]
        lad_gdf = artifacts["lad_gdf"]
        lad_points = artifacts["lad_points"]
        upload_signature = artifacts.get("upload_signature", "")

        render_page_header("Isochrone Travel Time Analysis")

        show_markers = True
        label_col = "LAD24NM"

        office_col = _get_col_case_insensitive(isochrones, "address") or OFFICE_NAME_COL
        iso_filtered = isochrones
        office_value = None
        office_col_name = None
        if office_col in isochrones.columns:
            office_col_name = office_col
            office_series = isochrones[office_col].astype(str).str.strip()
            office_values = sorted(office_series.dropna().unique())
            if office_values:
                office_value = st.sidebar.selectbox("Office", office_values)
                iso_filtered = isochrones[office_series == office_value].copy()
        else:
            fallback_col = _find_office_col(isochrones)
            if fallback_col:
                office_col_name = fallback_col
                office_series = isochrones[fallback_col].astype(str).str.strip()
                office_values = sorted(office_series.dropna().unique())
                if office_values:
                    office_value = st.sidebar.selectbox("Office", office_values)
                    iso_filtered = isochrones[office_series == office_value].copy()
            else:
                st.warning("No office column found; showing all isochrones.")

        if iso_filtered.empty:
            st.error("No isochrones found for the selected office.")
            return

        iso_tmp = iso_filtered.copy()
        if "Query_Tran" not in iso_tmp.columns:
            st.error("Isochrone data is missing 'Query_Tran'.")
            return

        transports = sorted([str(x) for x in iso_tmp["Query_Tran"].dropna().unique()])
        if not transports:
            st.error("No transport modes found after filtering.")
            return

        transport_labels = {t: _format_transport_label(t) for t in transports}
        transport_options = [transport_labels[t] for t in transports]
        selected_label = st.sidebar.selectbox("Transport Mode", transport_options)
        tran = next((t for t, label in transport_labels.items() if label == selected_label), transports[0])

        time_col = "Query_Time" if "Query_Time" in isochrones.columns else "Query_Isoc"
        iso_mode = iso_filtered[iso_filtered["Query_Tran"] == tran].copy()
        available_bands = _available_band_minutes(iso_mode, time_col)
        if not available_bands:
            st.error("No valid isochrone time bands found for the selected office and transport mode.")
            return

        core_band_set = {_normalise_band_value(value) for value in CORE_BANDS}
        has_non_core_bands = any(_normalise_band_value(value) not in core_band_set for value in available_bands)
        if has_non_core_bands:
            analysis_mode = st.sidebar.radio(
                "Analysis Mode",
                options=[CORE_MODE_LABEL, EXTENDED_MODE_LABEL],
                index=0,
            )
        else:
            analysis_mode = CORE_MODE_LABEL
        display_bands, missing_core_bands = _resolve_display_bands(available_bands, analysis_mode)
        if not display_bands:
            st.warning(
                "Core mode requires uploaded {0} min bands. Switch to Extended to analyse the available uploaded bands.".format(
                    ", ".join(_format_minutes_label(value) for value in CORE_BANDS)
                )
            )
            return

        if analysis_mode == CORE_MODE_LABEL and missing_core_bands:
            st.caption(
                "Uploaded file is missing core bands: {0}. Showing available core bands only.".format(
                    ", ".join("{0} min".format(_format_minutes_label(value)) for value in missing_core_bands)
                )
            )

        if analysis_mode == EXTENDED_MODE_LABEL:
            map_display_bands = st.sidebar.multiselect(
                "Map Layers",
                options=display_bands,
                default=display_bands,
                format_func=lambda value: "{0} min".format(_format_minutes_label(value)),
            )
        else:
            map_display_bands = list(display_bands)

        population_cache = _cache_namespace("population_counts")
        population_key = (
            upload_signature,
            str(office_value or ""),
            str(tran),
            time_col,
            tuple(display_bands),
        )
        if population_key in population_cache:
            pop_counts = population_cache[population_key]
        else:
            with st.spinner("Calculating residents within travel time bands..."):
                try:
                    pop_counts = _compute_population_within_bands(
                        lad_gdf=lad_gdf,
                        iso_filtered=iso_mode,
                        time_col=time_col,
                        thresholds=display_bands,
                    )
                except Exception as exc:
                    pop_counts = {}
                    st.warning("Unable to compute resident KPIs: {0}".format(exc))
            population_cache[population_key] = pop_counts

        visible_pop_counts = {
            float(band): float(pop_counts[band])
            for band in map_display_bands
            if band in pop_counts
        }
        if visible_pop_counts:
            _render_population_summary(visible_pop_counts, analysis_mode)
        elif pop_counts:
            st.caption("No KPI bands selected. Choose one or more map layers to show resident KPIs.")

        commuters_df = None
        pop_label = None
        commuter_cache = _cache_namespace("commuter_table")
        commuter_key = (
            upload_signature,
            str(tran),
            time_col,
            str(office_col_name or ""),
            tuple(display_bands),
        )
        if commuter_key in commuter_cache:
            cached_commuter = commuter_cache[commuter_key]
            commuters_df = cached_commuter.get("df")
            pop_label = cached_commuter.get("label")
        elif time_col in isochrones.columns and office_col_name:
            iso_all = isochrones.copy()
            if "Query_Tran" in iso_all.columns:
                iso_all = iso_all[iso_all["Query_Tran"] == tran]
            pop_label = "OA21 population by age"
            if not iso_all.empty:
                age_cols, _ = _age_columns(lad_gdf)
                if age_cols is None:
                    st.warning("No age columns found for commuter bands table.")
                else:
                    with st.spinner("Preparing commuter bands table..."):
                        lad_calc = lad_gdf[[*age_cols, "geometry"]].copy()
                        for c in age_cols:
                            lad_calc[c] = pd.to_numeric(lad_calc[c], errors="coerce").fillna(0)
                        lad_calc = lad_calc.to_crs("EPSG:27700")
                        lad_calc["lad_area"] = lad_calc.geometry.area

                        iso_calc = iso_all[[office_col_name, time_col, "geometry"]].copy()
                        iso_calc = iso_calc.to_crs("EPSG:27700")

                        try:
                            inter = gpd.overlay(iso_calc, lad_calc, how="intersection")
                        except Exception as exc:
                            st.warning("Commuter band table skipped: {0}".format(exc))
                            inter = None

                        if inter is not None and not inter.empty:
                            inter_area = inter.geometry.area
                            inter["area_frac"] = inter_area / inter["lad_area"]
                            for c in age_cols:
                                inter[c] = inter[c] * inter["area_frac"]

                            agg = (
                                inter.groupby([office_col_name, time_col])[age_cols]
                                .sum()
                                .reset_index()
                            )
                            agg = agg.rename(columns={office_col_name: "Office", time_col: "Band"})
                            agg["Band (mins)"] = agg["Band"].apply(lambda v: _format_band_minutes(time_col, v))
                            if agg["Band (mins)"].notna().any():
                                agg["Band (mins)"] = agg["Band (mins)"].apply(_normalise_band_value)
                                allowed_bands = {_normalise_band_value(value) for value in display_bands}
                                agg = agg[agg["Band (mins)"].isin(allowed_bands)].copy()
                                agg = agg.sort_values(by=["Office", "Band (mins)"])
                            else:
                                agg = agg.sort_values(by=["Office", "Band"])
                            for c in age_cols:
                                agg[c] = pd.to_numeric(agg[c], errors="coerce").round(0)
                            agg = agg.drop(columns=["Band"])
                            agg = agg.rename(
                                columns={
                                    "Aged 15 to 19 years": "Age 15-19",
                                    "Aged 20 to 24 years": "Age 20-24",
                                    "Aged 25 to 29 years": "Age 25-29",
                                    "Aged 30 to 34 years": "Age 30-34",
                                    "Aged 35 to 39 years": "Age 35-39",
                                    "Aged 40 to 44 years": "Age 40-44",
                                    "Aged 45 to 49 years": "Age 45-49",
                                    "Aged 50 to 54 years": "Age 50-54",
                                    "Aged 55 to 59 years": "Age 55-59",
                                    "Aged 60 to 64 years": "Age 60-64",
                                }
                            )
                            commuters_df = agg.reset_index(drop=True)
            commuter_cache[commuter_key] = {"df": commuters_df, "label": pop_label}

        office_point = None
        lat_col = OFFICE_LAT_COL if OFFICE_LAT_COL in iso_filtered.columns else _get_col_case_insensitive(iso_filtered, OFFICE_LAT_COL)
        lon_col = OFFICE_LON_COL if OFFICE_LON_COL in iso_filtered.columns else _get_col_case_insensitive(iso_filtered, OFFICE_LON_COL)
        if lat_col and lon_col:
            lat_val = pd.to_numeric(iso_filtered[lat_col], errors="coerce").dropna()
            lon_val = pd.to_numeric(iso_filtered[lon_col], errors="coerce").dropna()
            if not lat_val.empty and not lon_val.empty:
                office_point = (float(lat_val.iloc[0]), float(lon_val.iloc[0]))
        else:
            office_latlon = _find_office_lat_lon_cols(iso_filtered)
            if office_latlon:
                lat_col, lon_col = office_latlon
                lat_val = pd.to_numeric(iso_filtered[lat_col], errors="coerce").dropna()
                lon_val = pd.to_numeric(iso_filtered[lon_col], errors="coerce").dropna()
                if not lat_val.empty and not lon_val.empty:
                    office_point = (float(lat_val.iloc[0]), float(lon_val.iloc[0]))

        map_cache = _cache_namespace("map_html")
        map_geometry_cache = _cache_namespace("map_geometry")
        geometry_key = (
            MAP_RENDER_VERSION,
            upload_signature,
            str(office_value or ""),
            str(tran),
            tuple(map_display_bands),
        )
        if geometry_key in map_geometry_cache:
            map_payload = map_geometry_cache[geometry_key]
        else:
            with st.spinner("Preparing map geometry..."):
                try:
                    map_payload = _compute_map_geometry_payload(
                        isochrones=iso_filtered,
                        tran=tran,
                        display_bands=map_display_bands,
                    )
                except Exception as exc:
                    st.error("Map geometry prep failed: {0}".format(exc))
                    return
            map_geometry_cache[geometry_key] = map_payload

        map_key = (
            MAP_RENDER_VERSION,
            upload_signature,
            str(office_value or ""),
            str(tran),
            tuple(map_display_bands),
            bool(show_markers),
            bool(office_point),
            office_point,
        )
        if map_key in map_cache:
            html = map_cache[map_key]
        else:
            with st.spinner("Building map..."):
                try:
                    m = build_folium_map(
                        gdf=lad_points,
                        isochrones=iso_filtered,
                        tran=tran,
                        display_bands=map_display_bands,
                        show_markers=show_markers,
                        show_office_marker=True,
                        postcode_col=label_col,
                        office_point=office_point,
                        lad_boundaries=lad_gdf,
                        show_isochrones=True,
                        map_payload=map_payload,
                    )
                    html = folium_to_html(m)
                except Exception as exc:
                    st.error("Map build failed: {0}".format(exc))
                    return
            map_cache[map_key] = html

        st.subheader("Map")
        if not map_display_bands:
            st.caption("No isochrone layers selected. Map is showing the office and postcode markers only.")
        components.html(html, height=860, scrolling=True)

        st.subheader("Commuters within bands (all offices)")
        if commuters_df is None or commuters_df.empty:
            st.info("No commuter band table available for the current selection.")
        else:
            if pop_label:
                st.caption("Population source: {0}".format(pop_label))
            st.dataframe(commuters_df, use_container_width=True)
            st.download_button(
                "Download commuter bands CSV",
                data=df_to_csv_bytes(commuters_df),
                file_name="commuter_bands.csv",
                mime="text/csv",
            )

        st.markdown("---")
        st.caption(
            "Data source: ONS Output Area (2021) population by age, joined to LAD24 via OA21-LAD24 lookup."
        )
        st.caption(
            "Files: assets/data/census/population per output area.csv; assets/data/lookup/oa21_lad24_lookup.csv; "
            "assets/data/geo/lad_uk_2024.geojson."
        )


PLUGIN = IsochronePlugin()
