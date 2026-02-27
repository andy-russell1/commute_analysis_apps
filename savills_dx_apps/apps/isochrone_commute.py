from __future__ import annotations

import io
import zipfile
from typing import Optional

import geopandas as gpd
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.commute import filter_travel_time_valid, match_columns
from core.downloads import df_to_csv_bytes
from core.isochrone import load_isochrones_from_zip, validate_isochrone_zip
from core.models import AppArtifacts, AppMetadata, AppPlugin, UploadPayload


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

TRAVEL_THRESHOLDS = [30.0, 45.0, 60.0]


def _norm_name(value: str) -> str:
    text = str(value).strip().lower()
    for old in ["-", "_", "/", "(", ")", ".", ","]:
        text = text.replace(old, " ")
    return " ".join(text.split())


def _find_col_case_insensitive(df: pd.DataFrame, name: str) -> Optional[str]:
    cols = {str(c).lower(): c for c in df.columns}
    return cols.get(name.lower())


def _find_any_column(df: pd.DataFrame, candidates: list[str], required: bool = True) -> Optional[str]:
    norm_cols = {_norm_name(c): c for c in df.columns}
    for cand in candidates:
        key = _norm_name(cand)
        if key in norm_cols:
            return norm_cols[key]
    if required:
        raise KeyError(
            "Could not match required column from {0}. Available columns: {1}".format(
                candidates,
                list(df.columns),
            )
        )
    return None


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


def _find_office_lat_lon_cols(iso: gpd.GeoDataFrame) -> Optional[tuple[str, str]]:
    cols = [str(c) for c in iso.columns]
    lat_candidates = [c for c in cols if "office" in c.lower() and "lat" in c.lower()]
    lon_candidates = [c for c in cols if "office" in c.lower() and ("lon" in c.lower() or "long" in c.lower())]
    if lat_candidates and lon_candidates:
        return lat_candidates[0], lon_candidates[0]
    return None


def _find_office_id_col(iso: gpd.GeoDataFrame) -> Optional[str]:
    candidates = [
        "officeid",
        "office_id",
        "office id",
        "OfficeID",
        "Office ID",
        "officeID",
    ]
    cols = {str(c).lower(): c for c in iso.columns}
    for candidate in candidates:
        if candidate in iso.columns:
            return candidate
        key = str(candidate).lower()
        if key in cols:
            return cols[key]
    return None


def _delta_vs_benchmark_percent(current: int, benchmark: int) -> Optional[float]:
    if benchmark == 0:
        if current == 0:
            return 0.0
        return None
    return ((float(current) - float(benchmark)) / float(benchmark)) * 100.0


def _render_worker_kpis(
    selected_counts: dict[float, int],
    benchmark_counts: dict[float, int],
    total_workers: int,
) -> None:
    style = """
    <style>
    .iso-kpi-label {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }
    .iso-kpi-value-row {
        display: flex;
        align-items: baseline;
        gap: 0.4rem;
        line-height: 1;
    }
    .iso-kpi-main {
        font-size: 2.4rem;
        font-weight: 700;
        color: #f3f4f6;
    }
    .iso-kpi-share {
        font-size: 0.95rem;
        color: rgba(229, 231, 235, 0.7);
    }
    .iso-kpi-delta {
        margin-top: 0.45rem;
        font-size: 1rem;
        font-weight: 600;
    }
    .iso-kpi-delta-up {
        color: #00d26a;
    }
    .iso-kpi-delta-down {
        color: #ef5350;
    }
    .iso-kpi-delta-flat {
        color: rgba(229, 231, 235, 0.8);
    }
    </style>
    """
    st.markdown(style, unsafe_allow_html=True)

    band_meta = [
        (30.0, "Workers within 30 min"),
        (45.0, "Workers within 45 min"),
        (60.0, "Workers within 60 min"),
    ]
    cols = st.columns(3)

    for col, (threshold, label) in zip(cols, band_meta):
        current = int(selected_counts.get(threshold, 0))
        benchmark = int(benchmark_counts.get(threshold, 0))
        share = (float(current) / float(total_workers) * 100.0) if total_workers > 0 else 0.0
        delta_pct = _delta_vs_benchmark_percent(current=current, benchmark=benchmark)

        if delta_pct is None:
            delta_class = "iso-kpi-delta-flat"
            delta_text = "→ N/A vs benchmark"
        elif delta_pct > 0:
            delta_class = "iso-kpi-delta-up"
            delta_text = "↑ {0:.1f}% vs benchmark".format(delta_pct)
        elif delta_pct < 0:
            delta_class = "iso-kpi-delta-down"
            delta_text = "↓ {0:.1f}% vs benchmark".format(abs(delta_pct))
        else:
            delta_class = "iso-kpi-delta-flat"
            delta_text = "→ 0.0% vs benchmark"

        with col:
            st.markdown(
                """
                <div class="iso-kpi-label">{label}</div>
                <div class="iso-kpi-value-row">
                    <span class="iso-kpi-main">{count}</span>
                    <span class="iso-kpi-share">({share:.1f}%)</span>
                </div>
                <div class="iso-kpi-delta {delta_class}">{delta_text}</div>
                """.format(
                    label=label,
                    count="{0:,}".format(current),
                    share=share,
                    delta_class=delta_class,
                    delta_text=delta_text,
                ),
                unsafe_allow_html=True,
            )


def _extract_combined_upload(zip_payload_bytes: bytes) -> tuple[str, bytes, str, bytes]:
    with zipfile.ZipFile(io.BytesIO(zip_payload_bytes)) as zf:
        files = [n for n in zf.namelist() if not n.endswith("/")]
        iso_candidates = [n for n in files if n.lower().endswith(".zip")]
        employee_candidates = [n for n in files if n.lower().endswith((".csv", ".xls", ".xlsx"))]

        if not iso_candidates:
            raise ValueError("Combined upload is missing an isochrone ZIP file.")
        if not employee_candidates:
            raise ValueError("Combined upload is missing an employee file (CSV/XLS/XLSX).")

        isochrone_name = iso_candidates[0]
        employee_name = employee_candidates[0]
        isochrone_bytes = zf.read(isochrone_name)
        employee_bytes = zf.read(employee_name)

    return isochrone_name, isochrone_bytes, employee_name, employee_bytes


def _read_tabular(file_name: str, bytes_data: bytes) -> pd.DataFrame:
    ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
    if ext == "csv":
        try:
            return pd.read_csv(io.BytesIO(bytes_data))
        except UnicodeDecodeError:
            return pd.read_csv(io.BytesIO(bytes_data), encoding="latin-1")
    if ext in {"xls", "xlsx"}:
        return pd.read_excel(io.BytesIO(bytes_data))
    raise ValueError("Unsupported employee file type: {0}".format(file_name))


def _extract_worker_points(df_raw: pd.DataFrame) -> pd.DataFrame:
    try:
        cols = match_columns(df_raw)
        df_valid = filter_travel_time_valid(df_raw, cols)
    except Exception:
        df_valid = pd.DataFrame()

    if not df_valid.empty:
        keep = ["employeeID", "lat", "lon", "travel_time_min"]
        for optional in ["postcode", "city", "country"]:
            if optional in df_valid.columns:
                keep.append(optional)
        keep = [c for c in keep if c in df_valid.columns]
        points = df_valid[keep].copy()
        points["employeeID"] = points["employeeID"].astype(str).str.strip()
        points["lat"] = pd.to_numeric(points["lat"], errors="coerce")
        points["lon"] = pd.to_numeric(points["lon"], errors="coerce")
        if "travel_time_min" in points.columns:
            points["travel_time_min"] = pd.to_numeric(points["travel_time_min"], errors="coerce")
            points = points.sort_values("travel_time_min", ascending=True, na_position="last")
        points = points[~points["employeeID"].str.lower().isin(["", "nan", "none"])]
        points = points.dropna(subset=["employeeID", "lat", "lon"])
        points = points.drop_duplicates(subset=["employeeID"], keep="first")
        return points

    lat_col = _find_any_column(
        df_raw,
        ["lat", "latitude", "employee lat", "employee latitude", "employee - lat"],
        required=True,
    )
    lon_col = _find_any_column(
        df_raw,
        ["lon", "long", "longitude", "employee lon", "employee long", "employee longitude", "employee - long"],
        required=True,
    )
    emp_col = _find_any_column(
        df_raw,
        ["employeeid", "employee id", "worker id", "id", "person id"],
        required=False,
    )
    postcode_col = _find_any_column(df_raw, ["postcode", "postal code"], required=False)
    city_col = _find_any_column(df_raw, ["city"], required=False)
    country_col = _find_any_column(df_raw, ["country"], required=False)
    travel_time_col = _find_any_column(
        df_raw,
        ["travel_time_min", "travel time (mins)", "travel time", "commute time", "commute_time_min"],
        required=False,
    )

    points = pd.DataFrame()
    if emp_col:
        points["employeeID"] = df_raw[emp_col].astype(str).str.strip()
    else:
        points["employeeID"] = pd.Series(range(1, len(df_raw) + 1)).astype(str)

    points["lat"] = pd.to_numeric(df_raw[lat_col], errors="coerce")
    points["lon"] = pd.to_numeric(df_raw[lon_col], errors="coerce")

    if postcode_col:
        points["postcode"] = df_raw[postcode_col]
    if city_col:
        points["city"] = df_raw[city_col]
    if country_col:
        points["country"] = df_raw[country_col]
    if travel_time_col:
        points["travel_time_min"] = pd.to_numeric(df_raw[travel_time_col], errors="coerce")

    points = points[~points["employeeID"].str.lower().isin(["", "nan", "none"])]
    points = points.dropna(subset=["employeeID", "lat", "lon"]).copy()
    if "travel_time_min" in points.columns:
        points = points.sort_values("travel_time_min", ascending=True, na_position="last")
    points = points.drop_duplicates(subset=["employeeID"], keep="first")
    return points


def _minutes_from_iso(iso: gpd.GeoDataFrame, time_col: str) -> pd.Series:
    values = pd.to_numeric(iso[time_col], errors="coerce")
    if time_col == "Query_Isoc":
        return values / 60.0
    return values


def _worker_counts_within_thresholds(
    workers: gpd.GeoDataFrame,
    isochrones: gpd.GeoDataFrame,
    time_col: str,
    thresholds: list[float],
) -> dict[float, int]:
    if workers.empty or isochrones.empty:
        return {t: 0 for t in thresholds}

    workers_proj = workers.to_crs("EPSG:27700").copy()
    iso_proj = isochrones.to_crs("EPSG:27700").copy()
    iso_proj["_minutes"] = _minutes_from_iso(iso_proj, time_col)

    results: dict[float, int] = {}
    for threshold in thresholds:
        within = iso_proj[iso_proj["_minutes"] <= threshold]
        if within.empty:
            results[threshold] = 0
            continue
        union_geom = within.unary_union
        mask = workers_proj.geometry.intersects(union_geom)
        results[threshold] = int(workers_proj.loc[mask, "employeeID"].nunique())
    return results


def _format_mode_label(mode: str) -> str:
    return str(mode).replace("_", " ").replace("+", " + ").title()


def _build_worker_band_table(
    workers: gpd.GeoDataFrame,
    isochrones: gpd.GeoDataFrame,
    office_col: str,
    mode: str,
    time_col: str,
    thresholds: list[float],
) -> pd.DataFrame:
    rows = []
    office_values = sorted(isochrones[office_col].dropna().astype(str).str.strip().unique().tolist())
    for office in office_values:
        iso_office = isochrones[
            (isochrones[office_col].astype(str).str.strip() == office)
            & (isochrones["Query_Tran"].astype(str) == str(mode))
        ].copy()
        counts = _worker_counts_within_thresholds(
            workers=workers,
            isochrones=iso_office,
            time_col=time_col,
            thresholds=thresholds,
        )
        total_workers = int(workers["employeeID"].nunique())
        rows.append(
            {
                "Office": office,
                "Workers <=30 min": counts.get(30.0, 0),
                "Workers <=45 min": counts.get(45.0, 0),
                "Workers <=60 min": counts.get(60.0, 0),
                "Total Workers": total_workers,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["Share <=30 min (%)"] = (
        (out["Workers <=30 min"] / out["Total Workers"] * 100).where(out["Total Workers"] > 0, 0).round(1)
    )
    out["Share <=45 min (%)"] = (
        (out["Workers <=45 min"] / out["Total Workers"] * 100).where(out["Total Workers"] > 0, 0).round(1)
    )
    out["Share <=60 min (%)"] = (
        (out["Workers <=60 min"] / out["Total Workers"] * 100).where(out["Total Workers"] > 0, 0).round(1)
    )
    return out.sort_values("Office").reset_index(drop=True)


def _hex_to_rgba(color: str, alpha: float) -> str:
    value = str(color).lstrip("#")
    if len(value) != 6:
        return "rgba(44,62,80,{0})".format(alpha)
    r = int(value[0:2], 16)
    g = int(value[2:4], 16)
    b = int(value[4:6], 16)
    return "rgba({0},{1},{2},{3})".format(r, g, b, alpha)


def _darken_hex(color: str, factor: float = 0.82) -> str:
    value = str(color).lstrip("#")
    if len(value) != 6:
        return color
    factor = max(0.0, min(1.0, float(factor)))
    r = int(int(value[0:2], 16) * factor)
    g = int(int(value[2:4], 16) * factor)
    b = int(int(value[4:6], 16) * factor)
    return "#{0:02x}{1:02x}{2:02x}".format(r, g, b)


def _iter_polygons(geometry) -> list:
    if geometry is None or geometry.is_empty:
        return []

    geom_type = geometry.geom_type
    if geom_type == "Polygon":
        return [geometry]
    if geom_type == "MultiPolygon":
        return [g for g in geometry.geoms if g and not g.is_empty]
    if geom_type == "GeometryCollection":
        out = []
        for g in geometry.geoms:
            out.extend(_iter_polygons(g))
        return out
    return []


def _build_map(
    workers: gpd.GeoDataFrame,
    isochrones: gpd.GeoDataFrame,
    tran: str,
    time_col: str,
    office_point: Optional[tuple[float, float]],
) -> go.Figure:
    iso = isochrones.copy()
    if iso.crs is None or str(iso.crs).lower() != "epsg:4326":
        iso = iso.to_crs("EPSG:4326")
    iso["_minutes"] = _minutes_from_iso(iso, time_col)
    iso = iso[pd.to_numeric(iso["_minutes"], errors="coerce").notna()].copy()
    iso["_band_key"] = pd.to_numeric(iso["_minutes"], errors="coerce").round(6)

    workers_map = workers.copy()
    workers_map["lat"] = pd.to_numeric(workers_map["lat"], errors="coerce")
    workers_map["lon"] = pd.to_numeric(workers_map["lon"], errors="coerce")
    workers_map = workers_map.dropna(subset=["lat", "lon"]).copy()

    band_keys = (
        pd.to_numeric(iso["_band_key"], errors="coerce")
        .dropna()
        .sort_values(ascending=True)
        .unique()
        .tolist()
    )
    color_map = {
        float(band): ISOCHRONE_COLORS[i % len(ISOCHRONE_COLORS)] for i, band in enumerate(band_keys)
    }

    # Assign each worker to the smallest (fastest) band they fall within.
    workers_map["_band_key"] = pd.NA
    if not workers_map.empty and band_keys:
        iso_proj = iso.to_crs("EPSG:27700").copy()
        workers_proj = workers_map.to_crs("EPSG:27700").copy()
        workers_proj["_band_key"] = pd.NA

        for band in band_keys:
            band_iso = iso_proj[pd.to_numeric(iso_proj["_band_key"], errors="coerce") == float(band)]
            if band_iso.empty:
                continue
            union_geom = band_iso.geometry.unary_union
            if union_geom is None or union_geom.is_empty:
                continue
            mask = workers_proj["_band_key"].isna() & workers_proj.geometry.intersects(union_geom)
            workers_proj.loc[mask, "_band_key"] = float(band)

        workers_map["_band_key"] = workers_proj["_band_key"].values

    workers_map["_point_color"] = workers_map["_band_key"].apply(
        lambda val: _darken_hex(color_map.get(float(val), "#4b5563"))
        if pd.notna(val)
        else "#4b5563"
    )

    fig = go.Figure()
    if not workers_map.empty:
        fig.add_trace(
            go.Scattermapbox(
                lat=workers_map["lat"],
                lon=workers_map["lon"],
                mode="markers",
                marker=dict(size=6, color=workers_map["_point_color"].tolist(), opacity=0.78),
                name="Workers",
                hoverinfo="skip",
                hovertemplate=None,
            )
        )

    # Draw larger bands first so smaller/faster bands remain visible above.
    for band in sorted(band_keys, reverse=True):
        color = color_map.get(float(band), ISOCHRONE_COLORS[0])
        band_iso = iso[pd.to_numeric(iso["_band_key"], errors="coerce") == float(band)]
        union_geom = band_iso.geometry.unary_union
        polygons = _iter_polygons(union_geom)
        for j, poly in enumerate(polygons):
            lons, lats = poly.exterior.coords.xy
            fig.add_trace(
                go.Scattermapbox(
                    lon=list(lons),
                    lat=list(lats),
                    mode="lines",
                    fill="toself",
                    fillcolor=_hex_to_rgba(color, 0.18),
                    line=dict(color=color, width=1.4),
                    name="Isochrone {0:.0f} min".format(float(band)),
                    legendgroup="iso_{0:.0f}".format(float(band)),
                    showlegend=(j == 0),
                    hovertemplate="{0} - {1:.0f} mins<extra></extra>".format(_format_mode_label(tran), float(band)),
                )
            )

    if office_point:
        fig.add_scattermapbox(
            lat=[office_point[0]],
            lon=[office_point[1]],
            mode="markers",
            marker=dict(size=25, color="darkred", opacity=0.82),
            name="Office",
            hovertext=["Office"],
            hoverinfo="text",
        )

    center = iso.geometry.unary_union.centroid
    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_zoom=10,
        mapbox_center=dict(lat=float(center.y), lon=float(center.x)),
        margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=0.01, xanchor="left", x=0.01),
        dragmode="zoom",
    )
    return fig


class IsochroneCommutePlugin(AppPlugin):
    metadata = AppMetadata(
        id="isochrone_commute",
        name="Isochrone + Commute Impact",
        description="Combine isochrone ZIPs with geocoded worker data to quantify workforce reach by travel time.",
        accepted_upload_types=["zip"],
        upload_label="Upload combined ZIP",
        upload_help="Upload a combined ZIP created by the app from isochrones ZIP + employee file.",
    )

    def validate(self, upload: UploadPayload) -> None:
        if upload.ext != "zip":
            raise ValueError("Isochrone + Commute Impact expects a ZIP upload.")

        _, isochrone_bytes, employee_name, employee_bytes = _extract_combined_upload(upload.bytes_data)
        validate_isochrone_zip(isochrone_bytes)

        employee_df = _read_tabular(employee_name, employee_bytes)
        points = _extract_worker_points(employee_df)
        if points.empty:
            raise ValueError("No geocoded worker points found in the employee file.")

    def build(self, upload: UploadPayload, log) -> AppArtifacts:
        isochrone_name, isochrone_bytes, employee_name, employee_bytes = _extract_combined_upload(upload.bytes_data)

        log("Reading isochrones from ZIP")
        isochrones = load_isochrones_from_zip(isochrone_bytes)
        if isochrones.crs is not None and str(isochrones.crs).lower() != "epsg:4326":
            isochrones = isochrones.to_crs("EPSG:4326")

        log("Reading employee data")
        employee_df = _read_tabular(employee_name, employee_bytes)
        workers_df = _extract_worker_points(employee_df)
        if workers_df.empty:
            raise ValueError("No geocoded worker points found in the employee file.")

        workers_df["lat"] = pd.to_numeric(workers_df["lat"], errors="coerce")
        workers_df["lon"] = pd.to_numeric(workers_df["lon"], errors="coerce")
        workers_df = workers_df.dropna(subset=["lat", "lon"]).copy()
        workers_gdf = gpd.GeoDataFrame(
            workers_df,
            geometry=gpd.points_from_xy(workers_df["lon"], workers_df["lat"]),
            crs="EPSG:4326",
        )

        return {
            "isochrones": isochrones,
            "workers": workers_gdf,
            "isochrone_name": isochrone_name,
            "employee_name": employee_name,
        }

    def render(self, artifacts: AppArtifacts) -> None:
        isochrones = artifacts["isochrones"]
        workers = artifacts["workers"]
        isochrone_name = artifacts.get("isochrone_name", "")
        employee_name = artifacts.get("employee_name", "")

        st.title("Isochrone + Commute Impact Assessment")
        if isochrone_name or employee_name:
            st.caption(
                "Inputs: {0} | {1}".format(
                    isochrone_name or "N/A",
                    employee_name or "N/A",
                )
            )

        if workers.empty:
            st.error("No valid geocoded worker rows were found.")
            return

        office_col = _find_col_case_insensitive(isochrones, "address") or OFFICE_NAME_COL
        iso_filtered = isochrones
        office_col_name = None
        if office_col in isochrones.columns:
            office_col_name = office_col
        else:
            fallback_col = _find_office_col(isochrones)
            if fallback_col:
                office_col_name = fallback_col

        if not office_col_name:
            st.error("Isochrone file is missing an office column (for example 'address' or office name field).")
            return

        office_series = isochrones[office_col_name].astype(str).str.strip()
        office_values = sorted(office_series.dropna().unique().tolist())
        if not office_values:
            st.error("No office values found in isochrone file.")
            return

        office_id_col = _find_office_id_col(isochrones)
        benchmark_office_value = None
        benchmark_is_office_id_1 = False
        if office_id_col:
            office_id_series = isochrones[office_id_col].astype(str).str.strip()
            benchmark_names = (
                isochrones.loc[office_id_series == "1", office_col_name]
                .astype(str)
                .str.strip()
                .dropna()
                .unique()
                .tolist()
            )
            if benchmark_names:
                benchmark_office_value = sorted(benchmark_names)[0]
                benchmark_is_office_id_1 = True
        if benchmark_office_value is None:
            if "1" in office_values:
                benchmark_office_value = "1"
                benchmark_is_office_id_1 = True
            else:
                benchmark_office_value = office_values[0]

        office_value = st.sidebar.selectbox("Office", office_values)
        iso_filtered = isochrones[office_series == office_value].copy()

        if "Query_Tran" not in iso_filtered.columns:
            st.error("Isochrone data is missing 'Query_Tran'.")
            return
        transport_values = sorted(iso_filtered["Query_Tran"].dropna().astype(str).unique().tolist())
        if not transport_values:
            st.error("No transport modes found in isochrone data.")
            return

        transport_labels = {t: _format_mode_label(t) for t in transport_values}
        selected_transport_label = st.sidebar.selectbox(
            "Transport Mode",
            [transport_labels[t] for t in transport_values],
        )
        transport = next(
            (t for t, label in transport_labels.items() if label == selected_transport_label),
            transport_values[0],
        )

        time_col = "Query_Time" if "Query_Time" in isochrones.columns else "Query_Isoc"
        if time_col not in isochrones.columns:
            st.error("Isochrone data is missing 'Query_Time' or 'Query_Isoc'.")
            return

        iso_mode = iso_filtered[iso_filtered["Query_Tran"].astype(str) == str(transport)].copy()
        if iso_mode.empty:
            st.error("No isochrone records found for selected office and transport mode.")
            return

        counts = _worker_counts_within_thresholds(
            workers=workers,
            isochrones=iso_mode,
            time_col=time_col,
            thresholds=TRAVEL_THRESHOLDS,
        )
        total_workers = int(workers["employeeID"].nunique())

        benchmark_mode = isochrones[
            (office_series == str(benchmark_office_value))
            & (isochrones["Query_Tran"].astype(str) == str(transport))
        ].copy()
        benchmark_counts = _worker_counts_within_thresholds(
            workers=workers,
            isochrones=benchmark_mode,
            time_col=time_col,
            thresholds=TRAVEL_THRESHOLDS,
        )

        _render_worker_kpis(
            selected_counts=counts,
            benchmark_counts=benchmark_counts,
            total_workers=total_workers,
        )
        if benchmark_is_office_id_1:
            st.caption("Benchmark office (officeID=1): {0}".format(benchmark_office_value))
        else:
            st.caption("Benchmark office: {0}".format(benchmark_office_value))

        selected_band_table = pd.DataFrame(
            [
                {
                    "Band": "<=30 min",
                    "Workers": counts.get(30.0, 0),
                    "Share (%)": round((counts.get(30.0, 0) / total_workers) * 100, 1) if total_workers else 0.0,
                },
                {
                    "Band": "<=45 min",
                    "Workers": counts.get(45.0, 0),
                    "Share (%)": round((counts.get(45.0, 0) / total_workers) * 100, 1) if total_workers else 0.0,
                },
                {
                    "Band": "<=60 min",
                    "Workers": counts.get(60.0, 0),
                    "Share (%)": round((counts.get(60.0, 0) / total_workers) * 100, 1) if total_workers else 0.0,
                },
            ]
        )

        office_point = None
        lat_col = OFFICE_LAT_COL if OFFICE_LAT_COL in iso_filtered.columns else _find_col_case_insensitive(iso_filtered, OFFICE_LAT_COL)
        lon_col = OFFICE_LON_COL if OFFICE_LON_COL in iso_filtered.columns else _find_col_case_insensitive(iso_filtered, OFFICE_LON_COL)
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

        st.subheader("Map")
        map_fig = _build_map(
            workers=workers,
            isochrones=iso_mode,
            tran=transport,
            time_col=time_col,
            office_point=office_point,
        )
        st.plotly_chart(map_fig, use_container_width=True, config={"scrollZoom": True})

        st.subheader("Selected office: workers within bands")
        st.dataframe(selected_band_table, use_container_width=True, hide_index=True)
        st.download_button(
            "Download selected office worker bands CSV",
            data=df_to_csv_bytes(selected_band_table),
            file_name="selected_office_worker_bands.csv",
            mime="text/csv",
        )

        st.subheader("All offices: workers within bands ({0})".format(_format_mode_label(transport)))
        all_offices_table = _build_worker_band_table(
            workers=workers,
            isochrones=isochrones,
            office_col=office_col_name,
            mode=transport,
            time_col=time_col,
            thresholds=TRAVEL_THRESHOLDS,
        )
        if all_offices_table.empty:
            st.info("No office-level worker band table available.")
        else:
            st.dataframe(all_offices_table, use_container_width=True, hide_index=True)
            st.download_button(
                "Download all offices worker bands CSV",
                data=df_to_csv_bytes(all_offices_table),
                file_name="all_offices_worker_bands.csv",
                mime="text/csv",
            )

        st.markdown("---")
        st.caption(
            "Worker counts are based on geocoded employee points spatially intersected with uploaded isochrones."
        )


PLUGIN = IsochroneCommutePlugin()
