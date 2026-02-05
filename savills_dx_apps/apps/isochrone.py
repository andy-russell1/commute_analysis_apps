from __future__ import annotations

from typing import Optional

import geopandas as gpd
import pandas as pd
import streamlit as st
import folium
from folium.plugins import MarkerCluster
import streamlit.components.v1 as components

from core.downloads import df_to_csv_bytes
from core.isochrone import load_isochrones_from_zip, validate_isochrone_zip
from core.models import AppArtifacts, AppMetadata, AppPlugin, UploadPayload
from core.paths import DATA_DIR


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
    drop_driving_train: bool = True,
    show_markers: bool = True,
    show_office_marker: bool = True,
    postcode_col: str = "Postcode District",
    office_point: Optional[tuple] = None,
    lad_boundaries: Optional[gpd.GeoDataFrame] = None,
    show_isochrones: bool = True,
) -> folium.Map:
    gdf = gdf.copy()
    iso = isochrones.copy()

    if drop_driving_train and "Query_Tran" in iso.columns:
        iso = iso[iso["Query_Tran"] != "driving+train"]

    time_col = "Query_Time" if "Query_Time" in iso.columns else "Query_Isoc"
    if time_col not in iso.columns:
        raise ValueError("Isochrone data is missing 'Query_Time' or 'Query_Isoc'")

    times_all = [t for t in iso[time_col].dropna().unique().tolist()]

    def _time_key(val):
        try:
            return (0, float(val))
        except (TypeError, ValueError):
            return (1, str(val))

    times_sorted = sorted(times_all, key=_time_key)
    color_map = {
        str(t): ISOCHRONE_COLORS[i % len(ISOCHRONE_COLORS)] for i, t in enumerate(times_sorted)
    }

    if "Query_Tran" in iso.columns:
        iso = iso[iso["Query_Tran"] == tran]
    else:
        raise ValueError("Isochrone data is missing 'Query_Tran'")

    if iso.empty:
        raise ValueError("No isochrones found for Query_Tran = '{0}'".format(tran))

    center = iso.geometry.unary_union.centroid
    m = folium.Map(location=[center.y, center.x], zoom_start=11)

    folium.TileLayer(
        "CartoDB positron",
        name="CartoDB Positron",
        attr="Map tiles by Carto, under CC BY 3.0 - Map data OpenStreetMap contributors",
    ).add_to(m)

    if lad_boundaries is not None and not lad_boundaries.empty:
        folium.GeoJson(
            lad_boundaries,
            name="LAD boundaries",
            style_function=lambda _: {"fillOpacity": 0.0, "color": "#5f6368", "weight": 0.6},
        ).add_to(m)

    iso = iso.sort_values(by=time_col, ascending=False)

    if show_isochrones:
        iso_group = folium.FeatureGroup(name="Isochrone Bands", show=True)
        for _, row in iso.iterrows():
            time_val = row.get(time_col, "")
            color = color_map.get(str(time_val), ISOCHRONE_COLORS[0])
            folium.GeoJson(
                row.geometry,
                style_function=lambda x, color=color: {
                    "fillColor": color,
                    "color": color,
                    "weight": 1,
                    "fillOpacity": 0.2,
                },
                tooltip="{0} - {1} mins".format(tran, time_val),
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
        }

    def render(self, artifacts: AppArtifacts) -> None:
        isochrones = artifacts["isochrones"]
        lad_gdf = artifacts["lad_gdf"]
        lad_points = artifacts["lad_points"]

        st.title("Isochrone Travel Time Analysis")

        show_boundaries = st.session_state.get("show_lad_boundaries", True)
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
        if "Query_Tran" in iso_tmp.columns:
            iso_tmp = iso_tmp[iso_tmp["Query_Tran"] != "driving+train"]

        if "Query_Tran" not in iso_tmp.columns:
            st.error("Isochrone data is missing 'Query_Tran'.")
            return

        transports = sorted([str(x) for x in iso_tmp["Query_Tran"].dropna().unique()])
        if not transports:
            st.error("No transport modes found after filtering.")
            return

        transport_labels = {t: t.replace("_", " ").title() for t in transports}
        transport_options = [transport_labels[t] for t in transports]
        selected_label = st.sidebar.selectbox("Transport Mode", transport_options)
        tran = next((t for t, label in transport_labels.items() if label == selected_label), transports[0])

        time_col = "Query_Time" if "Query_Time" in isochrones.columns else "Query_Isoc"
        with st.spinner("Calculating residents within travel time bands..."):
            try:
                pop_counts = _compute_population_within_bands(
                    lad_gdf=lad_gdf,
                    iso_filtered=iso_filtered[iso_filtered["Query_Tran"] == tran],
                    time_col=time_col,
                    thresholds=[30.0, 45.0, 60.0],
                )
            except Exception as exc:
                pop_counts = {}
                st.warning("Unable to compute resident KPIs: {0}".format(exc))

        if pop_counts:
            kpi_cols = st.columns(3)
            kpi_cols[0].metric("Residents within 30 min", "{0:,.0f}".format(pop_counts.get(30.0, 0.0)))
            kpi_cols[1].metric("Residents within 45 min", "{0:,.0f}".format(pop_counts.get(45.0, 0.0)))
            kpi_cols[2].metric("Residents within 60 min", "{0:,.0f}".format(pop_counts.get(60.0, 0.0)))

        commuters_df = None
        pop_label = None
        if time_col in isochrones.columns and office_col_name:
            iso_all = isochrones.copy()
            if "Query_Tran" in iso_all.columns:
                iso_all = iso_all[iso_all["Query_Tran"] != "driving+train"]
                iso_all = iso_all[iso_all["Query_Tran"] == tran]
            if not iso_all.empty:
                age_cols, _ = _age_columns(lad_gdf)
                pop_label = "OA21 population by age"
                if age_cols is None:
                    st.warning("No age columns found for commuter bands table.")
                else:
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

        with st.spinner("Building map..."):
            try:
                m = build_folium_map(
                    gdf=lad_points,
                    isochrones=iso_filtered,
                    tran=tran,
                    drop_driving_train=True,
                    show_markers=show_markers,
                    show_office_marker=True,
                    postcode_col=label_col,
                    office_point=office_point,
                    lad_boundaries=lad_gdf if show_boundaries else None,
                    show_isochrones=True,
                )
                html = folium_to_html(m)
            except Exception as exc:
                st.error("Map build failed: {0}".format(exc))
                return

        st.subheader("Map")
        components.html(html, height=860, scrolling=True)
        st.checkbox("Show LAD Boundaries", value=show_boundaries, key="show_lad_boundaries")

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
