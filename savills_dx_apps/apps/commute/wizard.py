from __future__ import annotations

import io
import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from apps.commute.logic import filter_travel_time_valid, match_columns
from apps.commute.metrics import (
    explore_table,
    office_stats,
    select_office_method,
    threshold_bands,
    wide_table,
    wide_table_all_offices,
)
from shared.runtime.downloads import df_to_csv_bytes
from shared.runtime.models import AppArtifacts, AppMetadata, AppPlugin, UploadPayload
from shared.runtime.paths import LOGO_DIR
from shared.ui.kpi import render_kpi_strip


BEST_LABEL = "Best"
DRIVING_EMISSIONS_KGCO2E_PER_KM = 0.122544
TRAIN_EMISSIONS_KGCO2E_PER_KM = 0.02483
PUBLIC_TRANSPORT_EMISSIONS_KGCO2E_PER_KM = 0.053477
DRIVING_TRAIN_TRAIN_SHARE = 0.80
DRIVING_TRAIN_EMISSIONS_KGCO2E_PER_KM = (
    (1.0 - DRIVING_TRAIN_TRAIN_SHARE) * DRIVING_EMISSIONS_KGCO2E_PER_KM
    + DRIVING_TRAIN_TRAIN_SHARE * TRAIN_EMISSIONS_KGCO2E_PER_KM
)
MODE_EMISSIONS_KGCO2E_PER_KM = {
    "cycling": 0.0,
    "driving": DRIVING_EMISSIONS_KGCO2E_PER_KM,
    "public_transport": PUBLIC_TRANSPORT_EMISSIONS_KGCO2E_PER_KM,
    "driving+train": DRIVING_TRAIN_EMISSIONS_KGCO2E_PER_KM,
}


def _render_commute_kpi_strip(items: list[tuple[object, object]], *, columns: int = 4) -> None:
    render_kpi_strip(items, columns=columns)
COMMUTE_COLOR_SCALE = [
    [0.0, "#1e8449"],  # low travel time
    [0.5, "#f1c40f"],  # midpoint = yellow/amber
    [1.0, "#c0392b"],  # high travel time
]


def drop_fully_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    keep_cols = []
    for c in df.columns:
        s = df[c]
        if s.dropna().empty:
            continue
        s2 = s.astype(str).str.strip()
        s2 = s2.replace("nan", "", regex=False)
        if s2.eq("").all():
            continue
        keep_cols.append(c)

    return df[keep_cols].copy()


def _extract_driving_distance_km(df: pd.DataFrame, cols) -> pd.DataFrame:
    d = df.copy()
    d[cols.metric] = d[cols.metric].astype(str).str.strip().str.lower()
    d = d[d[cols.metric] == "distance"].copy()
    d[cols.value] = pd.to_numeric(d[cols.value], errors="coerce") / 1000.0
    d[cols.method] = d[cols.method].astype(str).str.strip().str.lower()
    d = d[d[cols.method] == "driving"].copy()

    if cols.query_dt and cols.query_dt in d.columns:
        d[cols.query_dt] = pd.to_datetime(d[cols.query_dt], errors="coerce")
        d = d.sort_values(cols.query_dt, ascending=False)

    required = [cols.emp_id, cols.office_id, cols.value]
    d = d.dropna(subset=required).copy()
    d[cols.emp_id] = d[cols.emp_id].astype(str)
    d[cols.office_id] = d[cols.office_id].astype(str)

    d = d.drop_duplicates(subset=[cols.office_id, cols.emp_id], keep="first")
    out = d.rename(
        columns={
            cols.emp_id: "employeeID",
            cols.office_id: "officeID",
            cols.value: "driving_distance_km",
        }
    )[["employeeID", "officeID", "driving_distance_km"]].copy()
    out["driving_distance_km"] = pd.to_numeric(out["driving_distance_km"], errors="coerce").fillna(0.0)
    return out


def _emissions_factor_for_mode(mode: str) -> float:
    key = str(mode).strip().lower()
    return float(MODE_EMISSIONS_KGCO2E_PER_KM.get(key, 0.0))


def _add_journey_emissions(df_valid: pd.DataFrame, driving_distance_km: pd.DataFrame) -> pd.DataFrame:
    d = df_valid.copy()
    d["employeeID"] = d["employeeID"].astype(str)
    d["officeID"] = d["officeID"].astype(str)
    distances = driving_distance_km.copy()
    distances["employeeID"] = distances["employeeID"].astype(str)
    distances["officeID"] = distances["officeID"].astype(str)
    d = d.merge(distances, on=["officeID", "employeeID"], how="left")
    d["driving_distance_km"] = pd.to_numeric(d["driving_distance_km"], errors="coerce").fillna(0.0)
    d["emissions_factor"] = d["method"].map(_emissions_factor_for_mode).astype(float)
    d["journey_kgco2e"] = d["driving_distance_km"] * d["emissions_factor"]
    return d


def _ensure_journey_emissions(
    df_valid: pd.DataFrame,
    driving_distance_km: pd.DataFrame | None = None,
) -> pd.DataFrame:
    d = df_valid.copy()
    d["employeeID"] = d["employeeID"].astype(str)
    d["officeID"] = d["officeID"].astype(str)

    if "driving_distance_km" not in d.columns:
        if isinstance(driving_distance_km, pd.DataFrame) and not driving_distance_km.empty:
            distances = driving_distance_km.copy()
            distances["employeeID"] = distances["employeeID"].astype(str)
            distances["officeID"] = distances["officeID"].astype(str)
            d = d.merge(distances, on=["officeID", "employeeID"], how="left")
        else:
            d["driving_distance_km"] = 0.0

    d["driving_distance_km"] = pd.to_numeric(d["driving_distance_km"], errors="coerce").fillna(0.0)
    d["emissions_factor"] = d["method"].map(_emissions_factor_for_mode).astype(float)
    d["journey_kgco2e"] = pd.to_numeric(
        d.get("journey_kgco2e", d["driving_distance_km"] * d["emissions_factor"]),
        errors="coerce",
    )
    return d


def _emissions_stats_by_office(
    df_valid: pd.DataFrame,
    offices: list[dict],
    method: str,
    min_time: float,
    max_time: float,
    *,
    best_label: str,
) -> pd.DataFrame:
    rows = []
    for o in offices:
        oid = str(o.get("officeID"))
        d = select_office_method(
            df_valid,
            oid,
            method,
            best_label=best_label,
            min_time=min_time,
            max_time=max_time,
        ).copy()
        if d.empty:
            rows.append(
                {
                    "officeID": oid,
                    "Office ID": oid,
                    "Office": str(o.get("address", oid)).split(",")[0].strip() or oid,
                    "Employee Count": 0,
                    "Avg (kgCO2e)": np.nan,
                }
            )
            continue

        d["journey_kgco2e"] = pd.to_numeric(d.get("journey_kgco2e"), errors="coerce")
        rows.append(
            {
                "officeID": oid,
                "Office ID": oid,
                "Office": str(o.get("address", oid)).split(",")[0].strip() or oid,
                "Employee Count": int(len(d)),
                "Avg (kgCO2e)": float(d["journey_kgco2e"].dropna().mean()) if len(d) else np.nan,
            }
        )

    return pd.DataFrame(rows)


def _employee_emissions_for_selection(
    df_valid: pd.DataFrame,
    office_id: str,
    method: str,
    min_time: float,
    max_time: float,
    *,
    best_label: str,
) -> pd.DataFrame:
    d = select_office_method(
        df_valid,
        office_id,
        method,
        best_label=best_label,
        min_time=min_time,
        max_time=max_time,
    ).copy()
    if d.empty:
        return pd.DataFrame(columns=["employeeID", "driving_distance_km", "journey_kgco2e"])

    d["driving_distance_km"] = pd.to_numeric(d.get("driving_distance_km"), errors="coerce").fillna(0.0)
    d["journey_kgco2e"] = pd.to_numeric(d.get("journey_kgco2e"), errors="coerce")
    if d["journey_kgco2e"].isna().all():
        factor_series = (
            d["best_method"].map(_emissions_factor_for_mode).astype(float)
            if "best_method" in d.columns
            else d["method"].map(_emissions_factor_for_mode).astype(float)
        )
        d["journey_kgco2e"] = d["driving_distance_km"] * factor_series
    return d[["employeeID", "driving_distance_km", "journey_kgco2e"]].copy()


def emissions_bar_figure(emissions_df: pd.DataFrame, method_label: str, office_order: list[str] | None = None) -> go.Figure:
    dfp = emissions_df.copy()
    dfp["Avg (kgCO2e)"] = pd.to_numeric(dfp.get("Avg (kgCO2e)"), errors="coerce")
    dfp["Employee Count"] = pd.to_numeric(dfp.get("Employee Count"), errors="coerce").fillna(0).astype(int)
    dfp = dfp.dropna(subset=["Avg (kgCO2e)"]).copy()
    if office_order:
        order_index = {name: idx for idx, name in enumerate(office_order)}
        dfp["__order"] = dfp["Office"].map(order_index).fillna(len(order_index)).astype(int)
        dfp = dfp.sort_values("__order", ascending=True).drop(columns="__order")
    else:
        dfp = dfp.sort_values("Avg (kgCO2e)", ascending=False)

    fig = go.Figure(
        go.Bar(
            y=dfp["Office"].astype(str),
            x=dfp["Avg (kgCO2e)"],
            orientation="h",
            marker=dict(color="#2f855a"),
            text=dfp["Avg (kgCO2e)"].round(2),
            texttemplate="%{text:.2f}",
            textposition="outside",
            customdata=dfp["Employee Count"],
            hovertemplate="%{y}<br>Avg kgCO2e: %{x:.2f}<br>Employees: %{customdata}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Average emissions by office - {0}".format(method_label),
        xaxis=dict(title="Avg kgCO2e", showgrid=True, zeroline=False),
        yaxis=dict(title="", automargin=True, ticklabeloverflow="allow"),
        margin=dict(l=180, r=40, t=60, b=60),
        height=max(380, 45 * len(dfp) + 130),
    )
    return fig


def threshold_stacked_bar_figure(
    bands_df: pd.DataFrame,
    method_label: str,
    office_order: list[str] | None = None,
) -> go.Figure:
    label_col = "Office" if "Office" in bands_df.columns else "officeShort"

    dfp = bands_df.copy()
    for c in ["<=30 min", "30-45 min", "45-60 min", ">60 min"]:
        if c not in dfp.columns:
            dfp[c] = 0.0
        dfp[c] = pd.to_numeric(dfp[c], errors="coerce").fillna(0.0)

    if office_order:
        order_index = {name: idx for idx, name in enumerate(office_order)}
        dfp["__order"] = dfp[label_col].map(order_index).fillna(len(order_index)).astype(int)
        dfp = dfp.sort_values("__order", ascending=True).drop(columns="__order")

    y = dfp[label_col].astype(str).tolist()
    total_emp = dfp.get("Total Employees", pd.Series([0] * len(y))).astype(float)

    gt60_emp = (total_emp * dfp[">60 min"] / 100).round(0).astype(int)
    b45_60_emp = (total_emp * dfp["45-60 min"] / 100).round(0).astype(int)
    b30_45_emp = (total_emp * dfp["30-45 min"] / 100).round(0).astype(int)
    le30_emp = (total_emp * dfp["<=30 min"] / 100).round(0).astype(int)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="<=30 min",
            y=y,
            x=dfp["<=30 min"],
            orientation="h",
            marker=dict(color="#1e8449"),
            customdata=le30_emp,
            text=le30_emp,
            texttemplate="%{text}",
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(color="white"),
            hovertemplate="%{y}<br>Employees: %{customdata}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="30-45 min",
            y=y,
            x=dfp["30-45 min"],
            orientation="h",
            marker=dict(color="#f1c40f"),
            customdata=b30_45_emp,
            text=b30_45_emp,
            texttemplate="%{text}",
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(color="black"),
            hovertemplate="%{y}<br>Employees: %{customdata}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="45-60 min",
            y=y,
            x=dfp["45-60 min"],
            orientation="h",
            marker=dict(color="#e67e22"),
            customdata=b45_60_emp,
            text=b45_60_emp,
            texttemplate="%{text}",
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(color="white"),
            hovertemplate="%{y}<br>Employees: %{customdata}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name=">60 min",
            y=y,
            x=dfp[">60 min"],
            orientation="h",
            marker=dict(color="#c0392b"),
            customdata=gt60_emp,
            text=gt60_emp,
            texttemplate="%{text}",
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(color="white"),
            hovertemplate="%{y}<br>Employees: %{customdata}<extra></extra>",
        )
    )

    fig.update_layout(
        barmode="stack",
        title="Employee commute time distribution - {0}".format(method_label),
        xaxis=dict(
            title="Employees (%)",
            range=[0, 100],
            ticksuffix="%",
            showgrid=True,
            zeroline=False,
        ),
        yaxis=dict(title="", automargin=True, ticklabeloverflow="allow"),
        legend=dict(
            title="",
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5,
            traceorder="normal",
        ),
        margin=dict(l=180, r=40, t=60, b=90),
        height=max(380, 45 * len(y) + 150),
    )

    return fig


def employee_scatter_map(
    df_emp: pd.DataFrame,
    office_obj: dict,
    title: str,
    *,
    metric_col: str | None = None,
    metric_label: str = "Time (mins)",
):
    d = df_emp.copy()

    d["lat"] = pd.to_numeric(d.get("lat"), errors="coerce")
    d["lon"] = pd.to_numeric(d.get("lon"), errors="coerce")
    color_col = metric_col or ("Travel Time (mins)" if "Travel Time (mins)" in d.columns else "travel_time_min")
    d[color_col] = pd.to_numeric(d.get(color_col), errors="coerce")
    d = d.dropna(subset=["lat", "lon", color_col]).copy()

    if d.empty:
        return None

    emp_id_col = "Employee ID" if "Employee ID" in d.columns else "employeeID"

    hover_cols = [emp_id_col, color_col]
    for old_c, new_c in [
        ("postcode", "Postcode"),
        ("city", "City"),
        ("country", "Country"),
        ("best_method", "Best Method"),
        ("driving_distance_km", "Driving Distance (km)"),
        ("journey_kgco2e", "Emissions (kgCO2e)"),
    ]:
        if new_c in d.columns:
            hover_cols.append(new_c)
        elif old_c in d.columns:
            hover_cols.append(old_c)

    fig = px.scatter_mapbox(
        d,
        lat="lat",
        lon="lon",
        color=color_col,
        hover_data=hover_cols,
        zoom=9,
        height=720,
        title=title,
        color_continuous_scale=COMMUTE_COLOR_SCALE,
    )

    fig.update_coloraxes(
        colorbar=dict(
            title=metric_label,
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.15,
            yanchor="top",
            len=0.6,
            thickness=12,
        ),
    )

    fig.add_scattermapbox(
        lat=[office_obj["lat"]],
        lon=[office_obj["lon"]],
        mode="markers",
        marker=dict(size=25, color="darkred", opacity=0.8),
        name="Office",
        hovertext=["Office: {0}".format(office_obj["address"])],
        hoverinfo="text",
    )

    fig.update_layout(
        mapbox_style="carto-positron",
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=0.01, xanchor="left", x=0.01),
        dragmode="zoom",
    )
    return fig


def office_metric_scatter_map(
    stats_df: pd.DataFrame,
    offices: list[dict],
    title: str,
    *,
    metric_col: str,
    metric_label: str,
):
    if stats_df is None or stats_df.empty:
        return None

    office_points = pd.DataFrame(offices).copy()
    if office_points.empty:
        return None

    office_points = office_points.rename(
        columns={
            "officeID": "Office ID",
            "address": "Office Full",
        }
    )
    office_points["Office"] = office_points["Office Full"].astype(str).str.split(",").str[0].str.strip()
    office_points["lat"] = pd.to_numeric(office_points.get("lat"), errors="coerce")
    office_points["lon"] = pd.to_numeric(office_points.get("lon"), errors="coerce")

    stats_cols = ["Office ID", metric_col, "Employee Count"]
    stats_points = stats_df[[c for c in stats_cols if c in stats_df.columns]].copy()
    points = office_points.merge(stats_points, on="Office ID", how="left")
    points[metric_col] = pd.to_numeric(points.get(metric_col), errors="coerce")
    points["Employee Count"] = pd.to_numeric(points.get("Employee Count"), errors="coerce")
    points = points.dropna(subset=["lat", "lon"]).copy()
    if points.empty:
        return None

    with_data = points.dropna(subset=[metric_col]).copy()
    fig = None
    if not with_data.empty:
        fig = px.scatter_mapbox(
            with_data,
            lat="lat",
            lon="lon",
            color=metric_col,
            hover_data={
                "Office": True,
                metric_col: ":.2f",
                "lat": False,
                "lon": False,
            },
            zoom=9,
            height=720,
            title=title,
            color_continuous_scale=COMMUTE_COLOR_SCALE,
        )
        fig.update_traces(marker=dict(size=20, opacity=0.8), hoverlabel=dict(font=dict(size=16)))
    else:
        fig = go.Figure()
        fig.update_layout(title=title, height=720)

    without_data = points[points[metric_col].isna()].copy()
    if not without_data.empty:
        fig.add_scattermapbox(
            lat=without_data["lat"],
            lon=without_data["lon"],
            mode="markers",
            marker=dict(size=20, color="#9ca3af", opacity=0.8),
            name="No median available",
            text=without_data["Office"],
            hovertemplate="Office: %{text}<br>{0}: N/A<extra></extra>".format(metric_label),
            hoverlabel=dict(font=dict(size=16)),
        )

    fig.update_coloraxes(
        colorbar=dict(
            title=metric_label,
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.15,
            yanchor="top",
            len=0.6,
            thickness=12,
        ),
    )
    fig.update_layout(
        mapbox_style="carto-positron",
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=0.01, xanchor="left", x=0.01),
        dragmode="zoom",
    )
    return fig


class CommutePlugin(AppPlugin):
    metadata = AppMetadata(
        id="commute",
        name="Commute Analysis",
        description="Upload a Successful.csv export and explore commute metrics.",
        accepted_upload_types=["csv"],
        upload_label="Upload Successful.csv",
        upload_help="Expected format: long table with Employee, Office, Metric, Value, and method columns.",
    )

    def validate(self, upload: UploadPayload) -> None:
        if upload.ext != "csv":
            raise ValueError("Commute Analysis expects a CSV file.")
        df = pd.read_csv(io.BytesIO(upload.bytes_data))
        if df.empty:
            raise ValueError("Uploaded CSV is empty.")
        cols = match_columns(df)
        df_valid = filter_travel_time_valid(df, cols)
        if df_valid.empty:
            raise ValueError("No valid travel_time rows found after cleaning.")

    def build(self, upload: UploadPayload, log) -> AppArtifacts:
        log("Reading CSV")
        df = pd.read_csv(io.BytesIO(upload.bytes_data))
        cols = match_columns(df)
        log("Filtering travel_time rows")
        df_valid = filter_travel_time_valid(df, cols)
        if df_valid.empty:
            raise ValueError("No valid travel_time rows found after cleaning.")
        log("Extracting driving distances")
        driving_distance_km = _extract_driving_distance_km(df, cols)
        log("Calculating journey emissions")
        df_valid = _add_journey_emissions(df_valid, driving_distance_km)

        log("Preparing office list")
        off_tbl = (
            df_valid[["officeID", "officeAddress", "officeLat", "officeLon"]]
            .drop_duplicates(subset=["officeID"])
            .copy()
        )
        off_tbl["officeLat"] = pd.to_numeric(off_tbl["officeLat"], errors="coerce")
        off_tbl["officeLon"] = pd.to_numeric(off_tbl["officeLon"], errors="coerce")
        off_tbl = off_tbl.dropna(subset=["officeLat", "officeLon"]).copy()

        offices = [
            {
                "officeID": str(r["officeID"]),
                "address": str(r["officeAddress"]),
                "lat": float(r["officeLat"]),
                "lon": float(r["officeLon"]),
            }
            for _, r in off_tbl.iterrows()
        ]

        if not offices:
            raise ValueError("No offices found (missing office lat/lon). Check your Successful.csv.")

        methods = sorted(df_valid["method"].dropna().astype(str).unique().tolist())
        if not methods:
            raise ValueError("No transport methods found in travel_time data.")

        return {
            "df_valid": df_valid,
            "offices": offices,
            "methods": methods,
            "upload_name": upload.name,
        }

    def render(self, artifacts: AppArtifacts) -> None:
        df_valid = artifacts["df_valid"]
        df_valid = _ensure_journey_emissions(df_valid, artifacts.get("driving_distance_km"))
        offices = artifacts["offices"]
        methods = artifacts["methods"]
        upload_name = artifacts.get("upload_name")

        def _method_time_series(df: pd.DataFrame, method: str, best_label: str) -> pd.Series:
            d = df.copy()
            d["travel_time_min"] = pd.to_numeric(d["travel_time_min"], errors="coerce")
            if method == best_label:
                if d.empty:
                    return pd.Series(dtype="float64")
                d = d.sort_values("travel_time_min", ascending=True, na_position="last")
                d = d.drop_duplicates(subset=["officeID", "employeeID"], keep="first")
                return d["travel_time_min"].dropna()
            return d.loc[d["method"].astype(str) == str(method), "travel_time_min"].dropna()

        def _format_method_label(value: str) -> str:
            if value == BEST_LABEL:
                return BEST_LABEL
            label = str(value).replace("_", " ").replace("+", " + ").strip()
            return label.title()

        style_block = """
            <style>
            .source-caption {
                font-size: 0.875rem;
                color: rgba(49, 51, 63, 0.6);
                margin-bottom: 0.35rem;
            }
            .kc-logo img {
                margin-left: auto;
                margin-right: 0;
            }
            .kc-logo {
                display: flex;
                justify-content: flex-end;
                align-items: flex-start;
            }
            @media screen {
                .kc-logo img {
                    margin-top: -20mm;
                }
            }
            [data-testid="stMetricValue"] {
                white-space: normal !important;
                overflow: visible !important;
                text-overflow: unset !important;
                word-break: break-word;
                max-width: none !important;
            }
            </style>
            """
        st.markdown(style_block, unsafe_allow_html=True)
        office_lookup = {o["officeID"]: o for o in offices}
        office_ids = [o["officeID"] for o in offices]
        minimum_median_office_id = "1" if "1" in office_ids else office_ids[0]

        def _minimum_median(stats_df: pd.DataFrame) -> float | None:
            minimum_median_row = stats_df[stats_df["officeID"].astype(str) == str(minimum_median_office_id)]
            if minimum_median_row.empty:
                return None
            minimum_median_values = pd.to_numeric(minimum_median_row["Median (mins)"], errors="coerce").dropna()
            if minimum_median_values.empty:
                return None
            return float(minimum_median_values.iloc[0])

        header_cols = st.columns([5, 2])
        with header_cols[0]:
            st.markdown(
                '<h1 class="print-title" style="font-size: 1.7rem; margin-bottom: 0.2rem;">Commute Impact Assessment</h1>',
                unsafe_allow_html=True,
            )
        if upload_name:
            st.markdown(
                '<div class="print-hide source-caption">Source: {0}</div>'.format(upload_name),
                unsafe_allow_html=True,
            )

        st.sidebar.divider()
        with st.sidebar.container():
            st.header("Controls")
            office_id = st.selectbox(
                "Office",
                options=office_ids,
                index=0,
                format_func=lambda oid: office_lookup[oid]["address"],
                key="office_select",
            )
            method = st.selectbox(
                "Transport Method",
                [BEST_LABEL] + methods,
                index=0,
                key="method_select",
                format_func=_format_method_label,
            )
            kpi_mode = st.selectbox(
                "KPI Metric",
                ["Travel Time", "Emissions"],
                index=0,
                key="kpi_metric_mode",
            )

            tt_series = _method_time_series(df_valid, method, BEST_LABEL)
            range_max = int(math.ceil(tt_series.max())) if not tt_series.empty else 90
            range_max = max(range_max, 90)
            default_max = min(90, range_max)

            if st.session_state.get("tt_range_method") != method:
                st.session_state["travel_time_range"] = (0, int(default_max))
                st.session_state["tt_range_method"] = method
            if "travel_time_range" not in st.session_state:
                st.session_state["travel_time_range"] = (0, int(default_max))
            current_min, current_max = st.session_state["travel_time_range"]
            current_min = max(0, min(int(current_min), int(range_max)))
            current_max = max(current_min, min(int(current_max), int(range_max)))
            st.session_state["travel_time_range"] = (current_min, current_max)

            travel_time_range = st.slider(
                "Travel time range (mins)",
                min_value=0,
                max_value=range_max,
                step=1,
                key="travel_time_range",
            )
            min_time, max_time = travel_time_range

        office_obj = office_lookup[office_id]
        param_line = "Office: {0} | Transport Mode: {1} | Time range: {2}-{3} mins".format(
            office_obj["address"],
            _format_method_label(method),
            int(min_time),
            int(max_time),
        )
        st.markdown('<div class="source-caption">{0}</div>'.format(param_line), unsafe_allow_html=True)

        df_valid_range = df_valid.copy()
        df_valid_range["travel_time_min"] = pd.to_numeric(df_valid_range["travel_time_min"], errors="coerce")
        df_valid_range = df_valid_range[
            df_valid_range["travel_time_min"].between(min_time, max_time, inclusive="both")
        ].copy()
        stats_df = office_stats(
            df_valid,
            offices,
            method=method,
            best_label=BEST_LABEL,
            min_time=min_time,
            max_time=max_time,
        )
        emissions_stats = _emissions_stats_by_office(
            df_valid=df_valid,
            offices=offices,
            method=method,
            min_time=min_time,
            max_time=max_time,
            best_label=BEST_LABEL,
        )

        tab_explore, tab_compare, tab_downloads = st.tabs(["Explore", "Compare", "Downloads + PDF"])

        with tab_explore:
            office_obj = office_lookup[office_id]

            emp_tbl = explore_table(
                df_valid,
                office_id=office_id,
                method=method,
                best_label=BEST_LABEL,
                min_time=min_time,
                max_time=max_time,
            )
            if kpi_mode == "Emissions":
                emp_emissions_df = _employee_emissions_for_selection(
                    df_valid=df_valid,
                    office_id=office_id,
                    method=method,
                    min_time=min_time,
                    max_time=max_time,
                    best_label=BEST_LABEL,
                ).rename(
                    columns={
                        "employeeID": "Employee ID",
                        "driving_distance_km": "Driving Distance (km)",
                        "journey_kgco2e": "Emissions (kgCO2e)",
                    }
                )
                emp_tbl = emp_tbl.merge(emp_emissions_df, on="Employee ID", how="left")
                emp_tbl["Driving Distance (km)"] = pd.to_numeric(
                    emp_tbl["Driving Distance (km)"], errors="coerce"
                ).round(2)
                emp_tbl["Emissions (kgCO2e)"] = pd.to_numeric(emp_tbl["Emissions (kgCO2e)"], errors="coerce").round(2)
            emp_tbl = drop_fully_empty_columns(emp_tbl)

            st.subheader("Office comparison")
            st.markdown('<div class="kpi-block">', unsafe_allow_html=True)
            if not stats_df.empty:
                sample_size = len(emp_tbl)

                if kpi_mode == "Emissions":
                    valid_emissions = emissions_stats.dropna(subset=["Avg (kgCO2e)"])
                    if not valid_emissions.empty:
                        best_emissions = valid_emissions["Avg (kgCO2e)"].min()
                        best_office = valid_emissions[valid_emissions["Avg (kgCO2e)"] == best_emissions][
                            "Office"
                        ].values[0]

                        baseline_series = valid_emissions[
                            valid_emissions["officeID"].astype(str) == str(minimum_median_office_id)
                        ]["Avg (kgCO2e)"]
                        current_series = valid_emissions[valid_emissions["officeID"].astype(str) == str(office_id)][
                            "Avg (kgCO2e)"
                        ]
                        average_value = "N/A"
                        if not current_series.empty:
                            average_value = "{0:.2f} kg".format(float(current_series.iloc[0]))
                        delta_display = "N/A"
                        if not baseline_series.empty and not current_series.empty:
                            delta_value = float(current_series.iloc[0] - best_emissions)#baseline_series.iloc[0])
                            delta_display = "{0:+.2f} kg".format(delta_value)
                        _render_commute_kpi_strip(
                            [
                                ("Lowest Emissions Office", "{0}".format(best_office)),
                                ("Sample Size", "{0:,}".format(int(sample_size))),
                                ("Average kgCO2e", average_value),
                                ("∆ Best Performing Office", delta_display),
                            ]
                        )
                    else:
                        _render_commute_kpi_strip(
                            [
                                ("Lowest Emissions Office", "N/A"),
                                ("Sample Size", "{0:,}".format(int(sample_size))),
                                ("Average kgCO2e", "N/A"),
                                ("∆ Best Performing Office", "N/A"),
                            ]
                        )
                else:
                    best_median = stats_df["Median (mins)"].min()
                    best_office = stats_df[stats_df["Median (mins)"] == best_median]["Office"].values[0]
                    avg_median = stats_df["Median (mins)"].mean()
                    minimum_median = best_median

                    delta_display = "N/A"
                    current_median = stats_df[stats_df["officeID"].astype(str) == str(office_id)]["Median (mins)"].values
                    if len(current_median) > 0 and minimum_median is not None:
                        delta_value = current_median[0] - minimum_median
                        delta_display = "{0:+.1f} min".format(delta_value)
                    _render_commute_kpi_strip(
                        [
                            ("Best Performing Office", "{0}".format(best_office)),
                            ("Sample Size", "{0:,}".format(int(sample_size))),
                            ("Average Median Time", "{0:.1f} min".format(avg_median)),
                            ("∆ Best Performing Office", delta_display),
                        ]
                    )
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="print-map">', unsafe_allow_html=True)
            show_office_map = bool(st.session_state.get("explore_map_toggle_offices", False))
            toggle_label = (
                "Switch from Office to Employee View"
                if show_office_map
                else "Switch from Employee to Office View"
            )
            if show_office_map:
                if kpi_mode == "Emissions":
                    fig_map = office_metric_scatter_map(
                        stats_df=emissions_stats,
                        offices=offices,
                        title="Office average emissions - {0}".format(_format_method_label(method)),
                        metric_col="Avg (kgCO2e)",
                        metric_label="Avg kgCO2e",
                    )
                else:
                    fig_map = office_metric_scatter_map(
                        stats_df=stats_df,
                        offices=offices,
                        title="Office median commute times - {0}".format(_format_method_label(method)),
                        metric_col="Median (mins)",
                        metric_label="Median (mins)",
                    )
                if fig_map is None:
                    st.info("No mappable office points available.")
                else:
                    st.plotly_chart(fig_map, use_container_width=True, config={"scrollZoom": True})
            else:
                if kpi_mode == "Emissions":
                    fig_map = employee_scatter_map(
                        emp_tbl,
                        office_obj,
                        title="Employees - {0} - {1}".format(office_obj["address"], method),
                        metric_col="Emissions (kgCO2e)",
                        metric_label="kgCO2e",
                    )
                else:
                    fig_map = employee_scatter_map(
                        emp_tbl,
                        office_obj,
                        title="Employees - {0} - {1}".format(office_obj["address"], method),
                    )
                if fig_map is None:
                    st.info("No mappable employee points (missing lat/lon).")
                else:
                    st.plotly_chart(fig_map, use_container_width=True, config={"scrollZoom": True})
            st.toggle(
                toggle_label,
                key="explore_map_toggle_offices",
            )
            st.markdown("</div>", unsafe_allow_html=True)

            with st.expander("Employee-level table", expanded=False):
                display_emp_tbl = emp_tbl.drop(columns=["lat", "lon"], errors="ignore")
                st.dataframe(display_emp_tbl, use_container_width=True, height=520)

                st.download_button(
                    "Download employee table (CSV)",
                    data=df_to_csv_bytes(emp_tbl),
                    file_name="employee_table.csv",
                    mime="text/csv",
                )

        with tab_compare:
            st.subheader("Office comparison")
            if not stats_df.empty:
                emp_tbl = explore_table(
                    df_valid,
                    office_id=office_id,
                    method=method,
                    best_label=BEST_LABEL,
                    min_time=min_time,
                    max_time=max_time,
                )
                sample_size = len(emp_tbl)

                if kpi_mode == "Emissions":
                    valid_emissions = emissions_stats.dropna(subset=["Avg (kgCO2e)"])
                    if not valid_emissions.empty:
                        best_emissions = valid_emissions["Avg (kgCO2e)"].min()
                        best_office = valid_emissions[valid_emissions["Avg (kgCO2e)"] == best_emissions][
                            "Office"
                        ].values[0]

                        baseline_series = valid_emissions[
                            valid_emissions["officeID"].astype(str) == str(minimum_median_office_id)
                        ]["Avg (kgCO2e)"]
                        current_series = valid_emissions[valid_emissions["officeID"].astype(str) == str(office_id)][
                            "Avg (kgCO2e)"
                        ]
                        average_value = "N/A"
                        if not current_series.empty:
                            average_value = "{0:.2f} kg".format(float(current_series.iloc[0]))
                        delta_display = "N/A"
                        if not baseline_series.empty and not current_series.empty:
                            delta_value = float(current_series.iloc[0] - baseline_series.iloc[0])
                            delta_display = "{0:+.2f} kg".format(delta_value)
                        _render_commute_kpi_strip(
                            [
                                ("Lowest Emissions Office", "{0}".format(best_office)),
                                ("Sample Size", "{0:,}".format(int(sample_size))),
                                ("Average kgCO2e", average_value),
                                ("∆ Best Performing Office", delta_display),
                            ]
                        )
                    else:
                        _render_commute_kpi_strip(
                            [
                                ("Lowest Emissions Office", "N/A"),
                                ("Sample Size", "{0:,}".format(int(sample_size))),
                                ("Average kgCO2e", "N/A"),
                                ("∆ Best Performing Office", "N/A"),
                            ]
                        )
                else:
                    best_median = stats_df["Median (mins)"].min()
                    best_office = stats_df[stats_df["Median (mins)"] == best_median]["Office"].values[0]
                    avg_median = stats_df["Median (mins)"].mean()
                    minimum_median = best_median

                    delta_display = "N/A"
                    current_median = stats_df[stats_df["officeID"].astype(str) == str(office_id)]["Median (mins)"].values
                    if len(current_median) > 0 and minimum_median is not None:
                        delta_value = current_median[0] - minimum_median
                        delta_display = "{0:+.1f} min".format(delta_value)
                    _render_commute_kpi_strip(
                        [
                            ("Best Performing Office", "{0}".format(best_office)),
                            ("Sample Size", "{0:,}".format(int(sample_size))),
                            ("Average Median Time", "{0:.1f} min".format(avg_median)),
                            ("∆ Best Performing Office", delta_display),
                        ]
                    )

            st.divider()

            if kpi_mode == "Emissions":
                office_order = (
                    emissions_stats.sort_values("Avg (kgCO2e)", ascending=False, na_position="last")["Office"].tolist()
                    if not emissions_stats.empty
                    else []
                )
                fig = emissions_bar_figure(emissions_stats, method_label=method, office_order=office_order)
                st.plotly_chart(fig, use_container_width=True)

                st.divider()
                st.subheader("Office statistics")

                display_stats = emissions_stats[["Office", "Avg (kgCO2e)"]].copy()
                display_stats = display_stats.sort_values("Avg (kgCO2e)", ascending=True, na_position="last")
                current_office_short = office_lookup[office_id]["address"].split(",")[0].strip()
                current_office_avg = emissions_stats[emissions_stats["Office"] == current_office_short][
                    "Avg (kgCO2e)"
                ].values
                if len(current_office_avg) > 0:
                    current_office_avg = float(current_office_avg[0])
                    display_stats["vs Current (kgCO2e)"] = (display_stats["Avg (kgCO2e)"] - current_office_avg).round(2)
                else:
                    display_stats["vs Current (kgCO2e)"] = 0.0

                display_stats = display_stats.copy()
                display_stats["Avg (kgCO2e)"] = display_stats["Avg (kgCO2e)"].apply(
                    lambda x: "{0:.2f}".format(x) if pd.notna(x) else ""
                )
                display_stats["vs Current (kgCO2e)"] = display_stats["vs Current (kgCO2e)"].apply(
                    lambda x: "{0:+.2f}".format(x) if pd.notna(x) else ""
                )
                numeric_vs_current = pd.to_numeric(
                    display_stats["vs Current (kgCO2e)"].str.replace("+", "", regex=False), errors="coerce"
                )
            else:
                bands_df = threshold_bands(
                    df_valid,
                    offices,
                    method=method,
                    best_label=BEST_LABEL,
                    min_time=min_time,
                    max_time=max_time,
                )
                office_order = (
                    stats_df.sort_values("Median (mins)", ascending=False, na_position="last")["Office"].tolist()
                    if not stats_df.empty
                    else []
                )
                fig = threshold_stacked_bar_figure(bands_df, method_label=method, office_order=office_order)
                st.plotly_chart(fig, use_container_width=True)

                st.divider()
                st.subheader("Office statistics")

                display_stats = stats_df[["Office", "Median (mins)"]].copy()
                display_stats = display_stats.sort_values("Median (mins)", ascending=True, na_position="last")
                display_stats = display_stats.rename(columns={"Median (mins)": "Avg (mins)"})
                display_stats["Avg (mins)"] = display_stats["Avg (mins)"].round(1)

                current_office_short = office_lookup[office_id]["address"].split(",")[0].strip()
                current_office_avg = stats_df[stats_df["Office"] == current_office_short]["Median (mins)"].values
                if len(current_office_avg) > 0:
                    current_office_avg = current_office_avg[0]
                    display_stats["vs Current (mins)"] = (display_stats["Avg (mins)"] - current_office_avg).round(1)
                else:
                    display_stats["vs Current (mins)"] = 0.0

                display_stats = display_stats.copy()
                display_stats["Avg (mins)"] = display_stats["Avg (mins)"].apply(
                    lambda x: "{0:.1f}".format(x) if pd.notna(x) else ""
                )
                display_stats["vs Current (mins)"] = display_stats["vs Current (mins)"].apply(
                    lambda x: "{0:+.1f}".format(x) if pd.notna(x) else ""
                )
                numeric_vs_current = pd.to_numeric(
                    display_stats["vs Current (mins)"].str.replace("+", "", regex=False), errors="coerce"
                )

            def color_gradient_vs(val):
                if pd.isna(val) or val == "":
                    return ""
                try:
                    float_val = float(val.replace("+", ""))
                except Exception:
                    return ""

                if float_val == 0:
                    return ""

                negative_vals = numeric_vs_current[numeric_vs_current < 0]
                positive_vals = numeric_vs_current[numeric_vs_current > 0]

                def interp_color(start_rgb, end_rgb, t):
                    t = max(0.0, min(1.0, t))
                    r = int(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * t)
                    g = int(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * t)
                    b = int(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * t)
                    return r, g, b

                if float_val < 0:
                    if negative_vals.empty:
                        return ""
                    min_neg = negative_vals.min()
                    intensity = abs(float_val) / abs(min_neg) if min_neg != 0 else 1.0
                    r, g, b = interp_color((232, 245, 233), (30, 132, 73), intensity)
                else:
                    if positive_vals.empty:
                        return ""
                    max_pos = positive_vals.max()
                    intensity = float_val / max_pos if max_pos != 0 else 1.0
                    r, g, b = interp_color((253, 236, 234), (192, 57, 43), intensity)

                text_color = "#1f2937" if intensity < 0.55 else "white"

                return "background-color: rgb({0}, {1}, {2}); color: {3}; font-weight: bold".format(
                    r,
                    g,
                    b,
                    text_color,
                )

            vs_col = "vs Current (kgCO2e)" if "vs Current (kgCO2e)" in display_stats.columns else "vs Current (mins)"
            styled_stats = display_stats.style.applymap(color_gradient_vs, subset=[vs_col])
            st.dataframe(styled_stats, use_container_width=True, hide_index=True)

        with tab_downloads:
            st.subheader("Downloads")

            st.markdown("**Table (employees by methods) - Current Office**")
            wide = wide_table(df_valid_range, office_id=office_id, methods=methods)
            wide = drop_fully_empty_columns(wide)

            st.download_button(
                "Download table (CSV) - Current Office",
                data=df_to_csv_bytes(wide),
                file_name="office_table.csv",
                mime="text/csv",
            )

            with st.expander("View table", expanded=False):
                st.dataframe(wide, use_container_width=True, height=420)

            st.divider()
            st.markdown("**Master**")
            wide_all = wide_table_all_offices(df_valid_range, offices=offices, methods=methods)
            wide_all = drop_fully_empty_columns(wide_all)

            st.download_button(
                "Download table - all offices (CSV)",
                data=df_to_csv_bytes(wide_all),
                file_name="wide_all_offices_table.csv",
                mime="text/csv",
            )

            with st.expander("View table", expanded=False):
                st.dataframe(wide_all, use_container_width=True, height=420)




PLUGIN = CommutePlugin()
