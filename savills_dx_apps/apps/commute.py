from __future__ import annotations

import io
import math

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.commute import filter_travel_time_valid, match_columns
from core.commute_metrics import (
    explore_table,
    office_stats,
    threshold_bands,
    wide_table,
    wide_table_all_offices,
)
from core.downloads import df_to_csv_bytes
from core.models import AppArtifacts, AppMetadata, AppPlugin, UploadPayload
from core.paths import LOGO_DIR


BEST_LABEL = "Best"


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


def employee_scatter_map(df_emp: pd.DataFrame, office_obj: dict, title: str):
    d = df_emp.copy()

    d["lat"] = pd.to_numeric(d.get("lat"), errors="coerce")
    d["lon"] = pd.to_numeric(d.get("lon"), errors="coerce")
    tt_col = "Travel Time (mins)" if "Travel Time (mins)" in d.columns else "travel_time_min"
    d["travel_time_min"] = pd.to_numeric(d.get(tt_col), errors="coerce")
    d = d.dropna(subset=["lat", "lon", "travel_time_min"]).copy()

    if d.empty:
        return None

    emp_id_col = "Employee ID" if "Employee ID" in d.columns else "employeeID"
    tt_col = "Travel Time (mins)" if "Travel Time (mins)" in d.columns else "travel_time_min"

    hover_cols = [emp_id_col, tt_col]
    for old_c, new_c in [
        ("postcode", "Postcode"),
        ("city", "City"),
        ("country", "Country"),
        ("best_method", "Best Method"),
    ]:
        if new_c in d.columns:
            hover_cols.append(new_c)
        elif old_c in d.columns:
            hover_cols.append(old_c)

    fig = px.scatter_mapbox(
        d,
        lat="lat",
        lon="lon",
        color=tt_col,
        hover_data=hover_cols,
        zoom=9,
        height=720,
        title=title,
        color_continuous_scale="RdYlGn_r",
    )

    fig.update_coloraxes(
        colorbar=dict(
            title="Time (mins)",
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

        header_cols = st.columns([5, 2])
        with header_cols[0]:
            st.markdown(
                '<h1 class="print-title" style="font-size: 1.7rem; margin-bottom: 0.2rem;">Commute Impact Assessment</h1>',
                unsafe_allow_html=True,
            )
        with header_cols[1]:
            kc_logo = LOGO_DIR / "Knowledge Cubed.png"
            if kc_logo.exists():
                st.markdown('<div class="kc-logo">', unsafe_allow_html=True)
                st.image(str(kc_logo), width=220)
                st.markdown("</div>", unsafe_allow_html=True)
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
            method = st.selectbox("Method", [BEST_LABEL] + methods, index=0, key="method_select")

            tt_series = _method_time_series(df_valid, method, BEST_LABEL)
            range_max = int(math.ceil(tt_series.max())) if not tt_series.empty else 90
            range_max = max(range_max, 90)
            default_max = min(90, range_max)

            if st.session_state.get("tt_range_method") != method:
                st.session_state["travel_time_range"] = (0, int(default_max))
                st.session_state["tt_range_method"] = method
            if "travel_time_range" not in st.session_state:
                st.session_state["travel_time_range"] = (0, int(default_max))

            travel_time_range = st.slider(
                "Travel time range (mins)",
                min_value=0,
                max_value=range_max,
                value=st.session_state["travel_time_range"],
                step=1,
                key="travel_time_range",
            )
            min_time, max_time = travel_time_range

        df_valid_range = df_valid.copy()
        df_valid_range["travel_time_min"] = pd.to_numeric(df_valid_range["travel_time_min"], errors="coerce")
        df_valid_range = df_valid_range[
            df_valid_range["travel_time_min"].between(min_time, max_time, inclusive="both")
        ].copy()

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
            emp_tbl = drop_fully_empty_columns(emp_tbl)

            st.subheader("Office comparison")
            st.markdown('<div class="kpi-block">', unsafe_allow_html=True)
            stats_df = office_stats(
                df_valid,
                offices,
                method=method,
                best_label=BEST_LABEL,
                min_time=min_time,
                max_time=max_time,
            )
            if not stats_df.empty:
                col1, col2, col3, col4 = st.columns(4)

                best_median = stats_df["Median (mins)"].min()
                best_office = stats_df[stats_df["Median (mins)"] == best_median]["Office"].values[0]
                sample_size = len(emp_tbl)
                avg_median = stats_df["Median (mins)"].mean()

                col1.metric("Best Performing Office", "{0}".format(best_office))
                col2.metric("Sample Size", "{0:,}".format(int(sample_size)))
                col3.metric("Average Median Time", "{0:.1f} min".format(avg_median))

                current_office_short = office_lookup[office_id]["address"].split(",")[0].strip()
                current_median = stats_df[stats_df["Office"] == current_office_short]["Median (mins)"].values
                if len(current_median) > 0:
                    delta_value = current_median[0] - best_median
                    col4.metric("Average time vs best performing office", "{0:+.1f} min".format(delta_value))
                else:
                    col4.metric("Average time vs best performing office", "N/A")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="print-map">', unsafe_allow_html=True)
            fig_map = employee_scatter_map(
                emp_tbl,
                office_obj,
                title="Employees - {0} - {1}".format(office_obj["address"], method),
            )
            if fig_map is None:
                st.info("No mappable employee points (missing lat/lon).")
            else:
                st.plotly_chart(fig_map, use_container_width=True, config={"scrollZoom": True})
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
            stats_df = office_stats(
                df_valid,
                offices,
                method=method,
                best_label=BEST_LABEL,
                min_time=min_time,
                max_time=max_time,
            )

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
                col1, col2, col3, col4 = st.columns(4)

                best_median = stats_df["Median (mins)"].min()
                best_office = stats_df[stats_df["Median (mins)"] == best_median]["Office"].values[0]
                avg_median = stats_df["Median (mins)"].mean()

                col1.metric("Best Performing Office", "{0}".format(best_office))
                col2.metric("Sample Size", "{0:,}".format(int(sample_size)))
                col3.metric("Average Median Time", "{0:.1f} min".format(avg_median))

                current_office_short = office_lookup[office_id]["address"].split(",")[0].strip()
                current_median = stats_df[stats_df["Office"] == current_office_short]["Median (mins)"].values
                if len(current_median) > 0:
                    delta_value = current_median[0] - best_median
                    col4.metric("Average time vs best performing office", "{0:+.1f} min".format(delta_value))
                else:
                    col4.metric("Average time vs best performing office", "N/A")

            st.divider()

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

            styled_stats = display_stats.style.applymap(color_gradient_vs, subset=["vs Current (mins)"])
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
