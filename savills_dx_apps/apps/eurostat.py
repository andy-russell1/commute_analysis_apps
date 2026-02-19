from __future__ import annotations

import branca.colormap as cm
import folium
import geopandas as gpd
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from core.downloads import df_to_csv_bytes
from core.models import AppArtifacts, AppMetadata, AppPlugin, UploadPayload
from core.paths import EUROSTAT_BOUNDARY_LOOKUP_PATH, EUROSTAT_WORKBOOK_PATH


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"year"}
    return [
        c
        for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and str(c).strip().lower() not in excluded
    ]


def geo_bucket(value: object) -> str:
    code = str(value).strip().upper()
    if len(code) == 2 and code.isalpha():
        return "country"
    if "_" in code or code in {"EA21", "EU27_2020"}:
        return "aggregate"
    if len(code) >= 3 and code[:2].isalpha() and code.isalnum():
        return "region"
    return "unknown"


def choose_preferred_age(df: pd.DataFrame) -> str | None:
    if "age" not in df.columns:
        return None
    age_values = set(df["age"].dropna().astype(str))
    preferred_order = ["Y15-74", "Y20-64", "Y15-64", "Y25-64", "Y15-59"]
    for age_code in preferred_order:
        if age_code in age_values:
            return age_code
    return sorted(age_values)[0] if age_values else None


def canonical_employment_total(df: pd.DataFrame, sheet_name: str) -> tuple[float | None, str]:
    if "employment_volume" not in df.columns:
        return None, "No employment volume metric in this sheet."

    local = df.copy()
    local["employment_volume"] = pd.to_numeric(local["employment_volume"], errors="coerce")
    local = local[local["employment_volume"].notna()].copy()
    if local.empty:
        return None, "No non-null employment volume values."

    if "geo" in local.columns:
        local["geo"] = local["geo"].astype(str).str.strip().str.upper()
        local = local[local["geo"].map(geo_bucket) == "country"].copy()
        if local.empty:
            return None, "No country-level rows available."

    age_code = choose_preferred_age(local)
    if age_code and "age" in local.columns:
        local = local[local["age"].astype(str) == age_code].copy()
        if local.empty:
            return None, "No rows found for preferred age band {0}.".format(age_code)

    if sheet_name == "occupation_country" and "isco08" in local.columns:
        if (local["isco08"].astype(str) == "TOTAL").any():
            local = local[local["isco08"].astype(str) == "TOTAL"].copy()
        else:
            return None, "No ISCO TOTAL row found for stable occupation total."

    if sheet_name == "industry_nuts2" and "nace_r2" in local.columns:
        if (local["nace_r2"].astype(str) == "TOTAL").any():
            local = local[local["nace_r2"].astype(str) == "TOTAL"].copy()
        else:
            return None, "No NACE TOTAL row found for stable industry total."

    if "geo" in local.columns:
        local = local.sort_values("employment_volume", ascending=False).drop_duplicates(subset=["geo"], keep="first")

    total = float(local["employment_volume"].sum())
    method = "Baseline: country-only, non-overlapping, age={0}".format(age_code if age_code else "n/a")
    return total, method


def apply_geo_view_filter(df: pd.DataFrame, geo_view: str) -> pd.DataFrame:
    if "geo" not in df.columns:
        return df
    local = df.copy()
    local["geo"] = local["geo"].astype(str).str.strip().str.upper()
    buckets = local["geo"].map(geo_bucket)
    if geo_view == "Country":
        return local[buckets == "country"].copy()
    return local[buckets == "region"].copy()


def fill_granular_defaults(df: pd.DataFrame) -> pd.DataFrame:
    local = df.copy()
    if "geo" in local.columns:
        local["geo"] = local["geo"].astype(str).str.strip().str.upper()
    if "region" in local.columns and "geo" in local.columns:
        region = local["region"].astype("string").str.strip()
        local["region"] = region.mask(region.eq("") | region.isna(), local["geo"])
    if "year" in local.columns:
        local["year"] = pd.to_numeric(local["year"], errors="coerce").fillna(2024)
    if "sex_group" in local.columns and "sex" in local.columns:
        sex_group = local["sex_group"].astype("string").str.strip()
        sex = local["sex"].astype("string").str.strip()
        local["sex_group"] = sex_group.mask(sex_group.eq("") | sex_group.isna(), sex)
    if "age_group" in local.columns and "age" in local.columns:
        age_group = local["age_group"].astype("string").str.strip()
        age = local["age"].astype("string").str.strip()
        local["age_group"] = age_group.mask(age_group.eq("") | age_group.isna(), age)
    return local


def single_select_filter(
    df: pd.DataFrame,
    column: str,
    label: str,
    key: str,
    all_label: str = "All",
) -> tuple[pd.DataFrame, str]:
    if column not in df.columns:
        return df, all_label
    options = sorted(df[column].dropna().astype(str).unique().tolist())
    if not options:
        return df, all_label
    selected = st.selectbox(label, [all_label] + options, index=0, key=key)
    if selected == all_label:
        return df, selected
    return df[df[column].astype(str) == selected].copy(), selected


def map_subset_for_geo_view(gdf: gpd.GeoDataFrame, geo_view: str) -> gpd.GeoDataFrame:
    if geo_view == "Country":
        return gdf[gdf["geo_class"] == "country"].copy()
    return gdf[gdf["geo_class"] == "nuts"].copy()


def build_map_df(
    df: pd.DataFrame,
    boundaries: gpd.GeoDataFrame,
    metric: str,
    agg: str,
    geo_view: str,
) -> tuple[gpd.GeoDataFrame, list[str]]:
    local = df.copy()
    local["geo"] = local["geo"].astype(str).str.strip().str.upper()
    grouped = local.groupby("geo", dropna=False)[metric].agg(agg).reset_index()
    mapped_col = "{0}_{1}".format(metric, agg)
    grouped.rename(columns={metric: mapped_col}, inplace=True)

    grouped_with_geometry = grouped.merge(boundaries[["geo", "geometry"]], on="geo", how="left")
    source_geos = set(grouped["geo"].tolist())
    matched_geos = set(grouped_with_geometry.loc[grouped_with_geometry["geometry"].notna(), "geo"].tolist())
    unmatched = sorted(source_geos - matched_geos)

    display = map_subset_for_geo_view(boundaries, geo_view)
    display = display[display["geo"].isin(grouped["geo"])].copy()
    display = display.merge(grouped[["geo", mapped_col]], on="geo", how="left")
    plot_df = gpd.GeoDataFrame(display, geometry="geometry", crs="EPSG:4326")
    return plot_df, unmatched


def build_step_colormap(values: pd.Series) -> cm.StepColormap:
    clean = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if clean.empty:
        return cm.StepColormap(colors=["#fff5f0", "#fb6a4a", "#67000d"], vmin=0, vmax=1)
    quantiles = clean.quantile([0, 0.2, 0.4, 0.6, 0.8, 1]).tolist()
    uniq: list[float] = []
    for q in quantiles:
        if not uniq or q > uniq[-1]:
            uniq.append(float(q))
    if len(uniq) < 2:
        uniq = [float(clean.min()), float(clean.max()) + 1e-9]
    return cm.StepColormap(
        colors=["#fff5f0", "#fcbba1", "#fc9272", "#fb6a4a", "#de2d26", "#a50f15"],
        vmin=uniq[0],
        vmax=uniq[-1],
        index=uniq,
    )


def focus_europe_extent(gdf: gpd.GeoDataFrame, metric_col: str) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    minx, miny, maxx, maxy = (-30.0, 25.0, 50.0, 75.0)
    parts = gdf.explode(index_parts=False).copy()
    centroids = parts.geometry.centroid
    keep = centroids.x.between(minx, maxx) & centroids.y.between(miny, maxy)
    focused = parts[keep].copy()
    if focused.empty:
        return gdf
    group_cols = [c for c in ["geo", "geo_class", "nuts_level", metric_col] if c in focused.columns]
    if not group_cols:
        return focused
    focused = focused.dissolve(by=group_cols, as_index=False)
    return gpd.GeoDataFrame(focused, geometry="geometry", crs=gdf.crs)


def folium_map_html(gdf: gpd.GeoDataFrame, metric_col: str, title: str) -> str:
    valid_geom = gdf[gdf.geometry.notna()].copy()
    if valid_geom.empty:
        return folium.Map(location=[54, 15], zoom_start=4).get_root().render()

    render_geom = focus_europe_extent(valid_geom, metric_col)
    if render_geom.empty:
        render_geom = valid_geom

    data_geom = render_geom[render_geom[metric_col].notna()].copy() if metric_col in render_geom.columns else render_geom
    bounds_target = data_geom if not data_geom.empty else render_geom
    bounds = bounds_target.total_bounds
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2

    m = folium.Map(location=[center_lat, center_lon], zoom_start=4, tiles="CartoDB positron")
    colormap = build_step_colormap(render_geom[metric_col])
    colormap.caption = title

    folium.GeoJson(
        render_geom,
        style_function=lambda feature: {
            "fillColor": (
                colormap(feature["properties"].get(metric_col))
                if feature["properties"].get(metric_col) is not None
                and pd.notna(feature["properties"].get(metric_col))
                else "#d1d5db"
            ),
            "color": "#4b5563",
            "weight": 0.5,
            "fillOpacity": 0.7,
        },
        tooltip=folium.GeoJsonTooltip(fields=["geo", metric_col], aliases=["Geo", metric_col], localize=True),
    ).add_to(m)
    colormap.add_to(m)
    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
    return m.get_root().render()


class EurostatPlugin(AppPlugin):
    metadata = AppMetadata(
        id="eurostat",
        name="Eurostat",
        description="Explore the Eurostat workbook by sheet with filters, tables, and choropleth maps.",
        accepted_upload_types=[],
        upload_label="",
        upload_help="",
        requires_upload=False,
    )

    def validate(self, upload: UploadPayload) -> None:
        if not EUROSTAT_WORKBOOK_PATH.exists():
            raise ValueError("Workbook not found: {0}".format(EUROSTAT_WORKBOOK_PATH))
        if not EUROSTAT_BOUNDARY_LOOKUP_PATH.exists():
            raise ValueError("Boundary lookup not found: {0}".format(EUROSTAT_BOUNDARY_LOOKUP_PATH))
        expected = {
            "granular_sex_age",
            "unemployment_by_edu",
            "industry_nuts2",
            "occupation_country",
        }
        xls = pd.ExcelFile(EUROSTAT_WORKBOOK_PATH)
        missing = sorted(expected - set(xls.sheet_names))
        if missing:
            raise ValueError("Eurostat workbook missing expected sheets: {0}".format(", ".join(missing)))

    def build(self, upload: UploadPayload, log) -> AppArtifacts:
        log("Loading Eurostat workbook")
        xls = pd.ExcelFile(EUROSTAT_WORKBOOK_PATH)
        sheets = {sheet: pd.read_excel(EUROSTAT_WORKBOOK_PATH, sheet_name=sheet) for sheet in xls.sheet_names}
        log("Loading boundary lookup")
        boundaries = gpd.read_file(EUROSTAT_BOUNDARY_LOOKUP_PATH)
        boundaries["geo"] = boundaries["geo"].astype(str).str.strip().str.upper()
        if boundaries.crs is None or str(boundaries.crs).lower() != "epsg:4326":
            boundaries = boundaries.to_crs("EPSG:4326")
        return {"sheets": sheets, "boundaries": boundaries}

    def render(self, artifacts: AppArtifacts) -> None:
        sheets = artifacts["sheets"]
        boundaries = artifacts["boundaries"]

        st.title("Eurostat Workbook Explorer")
        st.caption("Source workbook: {0}".format(EUROSTAT_WORKBOOK_PATH))
        st.caption("Note: `employment_volume` is reported in thousand persons (Eurostat unit `THS_PER`).")
        st.success("Loaded {0} sheets.".format(len(sheets)))

        tabs = st.tabs(list(sheets.keys()))
        for (sheet_name, df), tab in zip(sheets.items(), tabs):
            with tab:
                st.subheader(sheet_name)

                if sheet_name == "granular_sex_age":
                    df = fill_granular_defaults(df)

                geo_view = "Country" if sheet_name == "occupation_country" else st.radio(
                    "Geography view",
                    ["Country", "Regions"],
                    horizontal=True,
                    key="{0}_geo_view".format(sheet_name),
                )

                filtered = apply_geo_view_filter(df, geo_view)

                if sheet_name == "granular_sex_age":
                    filtered, _ = single_select_filter(filtered, "age_group", "Age group", key="{0}_age_group_filter".format(sheet_name))

                if sheet_name == "unemployment_by_edu":
                    filtered, _ = single_select_filter(filtered, "age_group", "Age group", key="{0}_age_group_filter".format(sheet_name))
                    edu_col = "education_level" if "education_level" in filtered.columns else "isced11"
                    if edu_col in filtered.columns:
                        edu_options = sorted(filtered[edu_col].dropna().astype(str).unique().tolist())
                        selected_edu = st.selectbox(
                            "Education level filter",
                            ["All"] + edu_options,
                            index=0,
                            key="{0}_edu_filter".format(sheet_name),
                        )
                        if selected_edu != "All":
                            filtered = filtered[filtered[edu_col].astype(str) == selected_edu].copy()

                if sheet_name == "industry_nuts2":
                    filtered, _ = single_select_filter(filtered, "age_group", "Age group", key="{0}_age_group_filter".format(sheet_name))
                    ind_col = "industry" if "industry" in filtered.columns else "nace_r2"
                    if ind_col in filtered.columns:
                        ind_options = sorted(filtered[ind_col].dropna().astype(str).unique().tolist())
                        selected_ind = st.selectbox(
                            "Industry filter",
                            ["All"] + ind_options,
                            index=0,
                            key="{0}_industry_filter".format(sheet_name),
                        )
                        if selected_ind != "All":
                            filtered = filtered[filtered[ind_col].astype(str) == selected_ind].copy()

                if sheet_name == "occupation_country":
                    filtered, _ = single_select_filter(filtered, "age_group", "Age group", key="{0}_age_group_filter".format(sheet_name))
                    role_col = "job_role" if "job_role" in filtered.columns else "isco08"
                    if role_col in filtered.columns:
                        role_options = sorted(filtered[role_col].dropna().astype(str).unique().tolist())
                        if "TOTAL" in role_options:
                            role_choices = ["TOTAL", "All"] + [r for r in role_options if r != "TOTAL"]
                        else:
                            role_choices = ["All"] + role_options
                        selected_role = st.selectbox(
                            "Job role filter",
                            role_choices,
                            index=0,
                            key="{0}_role_filter".format(sheet_name),
                        )
                        if selected_role != "All":
                            filtered = filtered[filtered[role_col].astype(str) == selected_role].copy()
                        else:
                            st.warning("All roles mixes TOTAL with occupation subgroups and can overstate summed employment.")

                c1, c2, c3 = st.columns(3)
                if sheet_name == "granular_sex_age":
                    avg_emp_rate = pd.to_numeric(filtered.get("employment_rate"), errors="coerce").mean()
                    canonical_total, canonical_note = canonical_employment_total(df, sheet_name)
                    avg_unemp_rate = pd.to_numeric(filtered.get("unemployment_rate"), errors="coerce").mean()
                    c1.metric("Avg Employment Rate", "n/a" if pd.isna(avg_emp_rate) else "{0:.2f}%".format(avg_emp_rate))
                    c2.metric(
                        "Employment Total (thousand persons)",
                        "n/a" if canonical_total is None else "{0:,.0f}".format(canonical_total),
                    )
                    c3.metric("Avg Unemployment Rate", "n/a" if pd.isna(avg_unemp_rate) else "{0:.2f}%".format(avg_unemp_rate))
                    st.caption(canonical_note)
                elif sheet_name in {"industry_nuts2", "occupation_country"}:
                    canonical_total, canonical_note = canonical_employment_total(df, sheet_name)
                    c1.metric("Rows", "{0:,}".format(len(filtered)))
                    c2.metric("Unique geo", "{0:,}".format(filtered["geo"].nunique()) if "geo" in filtered.columns else "n/a")
                    c3.metric(
                        "Employment Total (thousand persons)",
                        "n/a" if canonical_total is None else "{0:,.0f}".format(canonical_total),
                    )
                    st.caption(canonical_note)
                elif sheet_name == "unemployment_by_edu":
                    avg_unemp_rate = pd.to_numeric(filtered.get("unemployment_rate"), errors="coerce").mean()
                    c1.metric("Rows", "{0:,}".format(len(filtered)))
                    c2.metric("Unique geo", "{0:,}".format(filtered["geo"].nunique()) if "geo" in filtered.columns else "n/a")
                    c3.metric("Avg Unemployment Rate", "n/a" if pd.isna(avg_unemp_rate) else "{0:.2f}%".format(avg_unemp_rate))
                else:
                    c1.metric("Rows", "{0:,}".format(len(filtered)))
                    c2.metric("Columns", "{0:,}".format(len(filtered.columns)))
                    c3.metric("Unique geo", "{0:,}".format(filtered["geo"].nunique()) if "geo" in filtered.columns else "n/a")

                st.caption("Base rows in sheet: {0:,}".format(len(df)))
                view_tab, map_tab = st.tabs(["Data", "Map"])

                with view_tab:
                    st.dataframe(filtered, use_container_width=True, height=420)
                    st.download_button(
                        label="Download {0} as CSV".format(sheet_name),
                        data=df_to_csv_bytes(filtered),
                        file_name="{0}.csv".format(sheet_name),
                        mime="text/csv",
                    )

                with map_tab:
                    if "geo" not in filtered.columns:
                        st.warning("No `geo` column found in this sheet.")
                        continue
                    if filtered.empty:
                        st.warning("No rows left after selected filters.")
                        continue
                    numeric_cols = get_numeric_columns(filtered)
                    if not numeric_cols:
                        st.warning("No numeric columns available for choropleth mapping.")
                        continue

                    ctrl1, ctrl2 = st.columns(2)
                    metric = ctrl1.selectbox("Metric", numeric_cols, index=0, key="{0}_metric".format(sheet_name))
                    agg = ctrl2.selectbox("Aggregation", ["mean", "sum", "median"], index=0, key="{0}_agg".format(sheet_name))

                    plot_df, unmatched = build_map_df(filtered, boundaries, metric, agg, geo_view)
                    mapped_col = "{0}_{1}".format(metric, agg)
                    if plot_df.empty or mapped_col not in plot_df.columns:
                        st.warning("No mappable rows for the selected configuration.")
                        st.caption("Check that selected metric has non-null values.")
                        continue

                    map_title = "{0}: {1}".format(sheet_name, mapped_col)
                    if metric == "employment_volume":
                        map_title += " (thousand persons)"
                    components.html(folium_map_html(plot_df, mapped_col, map_title), height=720, scrolling=True)
                    st.caption(
                        "Mapped polygons: {0:,} | Non-null metric rows: {1:,}".format(
                            len(plot_df),
                            int(plot_df[mapped_col].notna().sum()),
                        )
                    )
                    if unmatched:
                        st.warning("Unmatched geos ({0}): {1}".format(len(unmatched), ", ".join(unmatched[:20])))
                    else:
                        st.success("All geo codes matched to boundaries for this metric view.")

                    export_gdf = plot_df[["geo", "geo_class", "nuts_level", mapped_col, "geometry"]].copy()
                    st.download_button(
                        label="Download mapped GeoJSON ({0})".format(sheet_name),
                        data=export_gdf.to_json().encode("utf-8"),
                        file_name="{0}_{1}_mapped.geojson".format(sheet_name, mapped_col),
                        mime="application/geo+json",
                    )


PLUGIN = EurostatPlugin()
