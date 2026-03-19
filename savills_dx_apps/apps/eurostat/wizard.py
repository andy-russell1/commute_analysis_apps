from __future__ import annotations

import branca.colormap as cm
import folium
import geopandas as gpd
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from shared.runtime.downloads import df_to_csv_bytes
from shared.runtime.models import AppArtifacts, AppMetadata, AppPlugin, UploadPayload
from shared.runtime.paths import (
    EUROSTAT_BOUNDARY_LOOKUP_PATH,
    EUROSTAT_UK_NUTS1_BOUNDARY_PATH,
    EUROSTAT_WORKBOOK_PATH,
    OXECON_DOWNLOAD_PATH,
)
from shared.ui.kpi import render_kpi_strip
from shared.ui.page_header import render_page_header


OXECON_LOCATION_MAP = {
    "Aggregate - United Kingdom": ("UK", "United Kingdom"),
    "East": ("UKH", "East of England"),
    "London": ("UKI", "London"),
    "North East": ("UKC", "North East (England)"),
    "North West": ("UKD", "North West (England)"),
    "Northern Ireland": ("UKN", "Northern Ireland"),
    "Scotland": ("UKM", "Scotland"),
    "South East": ("UKJ", "South East (England)"),
    "South West": ("UKK", "South West (England)"),
    "Wales": ("UKL", "Wales"),
    "West Midlands": ("UKG", "West Midlands (England)"),
    "Yorkshire and the Humber": ("UKE", "Yorkshire and the Humber"),
}


@st.cache_data(show_spinner=False)
def load_workbook(path: str) -> dict[str, pd.DataFrame]:
    xls = pd.ExcelFile(path)
    return {sheet: pd.read_excel(path, sheet_name=sheet) for sheet in xls.sheet_names}


@st.cache_data(show_spinner=False)
def load_boundaries(path: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    gdf["geo"] = gdf["geo"].astype(str).str.strip().str.upper()
    if gdf.crs is None or str(gdf.crs).lower() != "epsg:4326":
        gdf = gdf.to_crs("EPSG:4326")
    return gdf


@st.cache_data(show_spinner=False)
def load_oxecon_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    year_cols = [col for col in df.columns if str(col).strip().isdigit()]
    if not year_cols:
        raise ValueError("Oxecon download missing year columns.")

    local = df.rename(
        columns={
            "Location": "location",
            "Location code": "oxecon_location_code",
            "Indicator": "indicator",
            "Sector": "sector",
            "Units": "units",
            "Scale": "scale",
            "Measurement": "measurement",
            "Source": "source",
            "Seasonally adjusted": "seasonally_adjusted",
            "Historical end": "historical_end",
            "Date of last update": "date_of_last_update",
            "Indicator code": "indicator_code",
        }
    ).copy()
    local = local.melt(
        id_vars=[col for col in local.columns if col not in year_cols],
        value_vars=year_cols,
        var_name="year",
        value_name="value",
    )
    local["year"] = pd.to_numeric(local["year"], errors="coerce")
    local["value"] = pd.to_numeric(local["value"], errors="coerce")
    local["location"] = local["location"].astype(str).str.strip()

    geo_region = local["location"].map(OXECON_LOCATION_MAP)
    local["geo"] = geo_region.map(lambda item: item[0] if isinstance(item, tuple) else None)
    local["region"] = geo_region.map(lambda item: item[1] if isinstance(item, tuple) else None)
    local["geo"] = local["geo"].astype("string").str.upper()
    local["region"] = local["region"].astype("string")

    local["seasonally_adjusted"] = local["seasonally_adjusted"].astype("string").str.upper()
    local["date_of_last_update"] = pd.to_datetime(local["date_of_last_update"], errors="coerce")

    ordered = [
        "geo",
        "region",
        "location",
        "oxecon_location_code",
        "year",
        "indicator",
        "sector",
        "value",
        "units",
        "scale",
        "measurement",
        "source",
        "seasonally_adjusted",
        "historical_end",
        "date_of_last_update",
        "indicator_code",
    ]
    return local[ordered].copy()


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


def filtered_employment_total(df: pd.DataFrame) -> tuple[float | None, str]:
    if "employment_volume" not in df.columns:
        return None, "No employment volume metric in current filtered view."
    values = pd.to_numeric(df["employment_volume"], errors="coerce")
    values = values[values.notna()]
    if values.empty:
        return None, "No non-null employment volume values in current filtered view."
    return float(values.sum()), "Dynamic: total for current filters."


def apply_geo_view_filter(df: pd.DataFrame, geo_view: str) -> pd.DataFrame:
    if "geo" not in df.columns:
        return df
    local = df.copy()
    local["geo"] = local["geo"].astype(str).str.strip().str.upper()
    buckets = local["geo"].map(geo_bucket)
    if geo_view == "Country":
        return local[buckets == "country"].copy()
    return local[buckets == "region"].copy()


def geo_code_length(value: object) -> int:
    code = str(value).strip().upper()
    return len(code)


def region_level_to_nuts_level(region_code_len: int | None) -> int | None:
    mapping = {3: 1, 4: 2, 5: 3}
    return mapping.get(region_code_len)


def region_level_options(df: pd.DataFrame) -> list[int]:
    if "geo" not in df.columns:
        return []
    local = df.copy()
    local["geo"] = local["geo"].astype(str).str.strip().str.upper()
    local = local[local["geo"].map(geo_bucket) == "region"].copy()
    if local.empty:
        return []
    lengths = sorted(local["geo"].map(geo_code_length).dropna().astype(int).unique().tolist())
    return [length for length in lengths if length in {3, 4, 5}]


def apply_region_level_filter(df: pd.DataFrame, region_code_len: int | None) -> pd.DataFrame:
    if region_code_len is None or "geo" not in df.columns:
        return df
    local = df.copy()
    local["geo"] = local["geo"].astype(str).str.strip().str.upper()
    return local[local["geo"].map(geo_code_length) == int(region_code_len)].copy()


def region_level_label(region_code_len: int | None) -> str:
    nuts_level = region_level_to_nuts_level(region_code_len)
    if nuts_level is not None:
        return "NUTS{0} Regions".format(nuts_level)
    return "Unknown"


def region_code_len_from_label(geography_label: str) -> int | None:
    mapping = {
        "NUTS1 Regions": 3,
        "NUTS2 Regions": 4,
        "NUTS3 Regions": 5,
    }
    return mapping.get(geography_label)


def geography_level_options(df: pd.DataFrame) -> list[str]:
    if "geo" not in df.columns:
        return ["Country"]
    local = df.copy()
    local["geo"] = local["geo"].astype(str).str.strip().str.upper()
    buckets = local["geo"].map(geo_bucket)

    options: list[str] = []
    if (buckets == "country").any():
        options.append("Country")
    for region_len in region_level_options(local):
        options.append(region_level_label(region_len))
    return options if options else ["Country"]


def apply_geography_level_filter(df: pd.DataFrame, geography_label: str) -> tuple[pd.DataFrame, int | None]:
    if geography_label == "Country":
        return apply_geo_view_filter(df, "Country"), None
    region_len = region_code_len_from_label(geography_label)
    filtered = apply_geo_view_filter(df, "Regions")
    if region_len is None:
        return filtered, None
    return apply_region_level_filter(filtered, region_len), region_len


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


def enforce_total_sex(df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
    if sheet_name == "granular_sex_age" or "sex" not in df.columns:
        return df
    local = df.copy()
    sex = local["sex"].astype(str).str.strip().str.upper()
    if (sex == "T").any():
        local = local[sex == "T"].copy()
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


def age_filter_with_overview(
    df: pd.DataFrame,
    sheet_name: str,
    key: str,
) -> tuple[pd.DataFrame, str, str]:
    if "age_group" not in df.columns:
        return df, "All", ""
    options = sorted(df["age_group"].dropna().astype(str).unique().tolist())
    if not options:
        return df, "All", ""

    selected = st.selectbox("Age group", ["All"] + options, index=0, key=key)
    if selected != "All":
        return df[df["age_group"].astype(str) == selected].copy(), selected, ""

    if "age" not in df.columns:
        return df, selected, ""

    preferred_age = choose_preferred_age(df)
    if not preferred_age:
        return df, selected, ""

    age_series = df["age"].astype(str)
    overview = df[age_series == preferred_age].copy()
    if overview.empty:
        return df, selected, ""

    note = "All ages uses baseline non-overlapping band: {0}".format(preferred_age)
    return overview, selected, note


def map_subset_for_geo_view(gdf: gpd.GeoDataFrame, geography_label: str) -> gpd.GeoDataFrame:
    if geography_label == "Country":
        return gdf[gdf["geo_class"] == "country"].copy()
    region_len = region_code_len_from_label(geography_label)
    if region_len is None:
        return gdf[gdf["geo_class"] == "nuts"].copy()
    nuts_level = region_level_to_nuts_level(region_len)
    if nuts_level is None or "nuts_level" not in gdf.columns:
        return gdf[gdf["geo_class"] == "nuts"].copy()
    return gdf[(gdf["geo_class"] == "nuts") & (gdf["nuts_level"] == nuts_level)].copy()


def build_map_df(
    df: pd.DataFrame,
    boundaries: gpd.GeoDataFrame,
    metric: str,
    agg: str,
    geography_label: str,
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

    display = map_subset_for_geo_view(boundaries, geography_label)
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


def combine_boundaries(*frames: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    prepared: list[gpd.GeoDataFrame] = []
    for frame in frames:
        local = frame.copy()
        if "geo" in local.columns:
            local["geo"] = local["geo"].astype(str).str.strip().str.upper()
        prepared.append(local)

    combined = pd.concat(prepared, ignore_index=True, sort=False)
    combined = combined.drop_duplicates(subset=["geo"], keep="first").copy()
    return gpd.GeoDataFrame(combined, geometry="geometry", crs="EPSG:4326")


def default_oxecon_sector(options: list[str]) -> str:
    preferred = ["Whole Economy", "Total"]
    for candidate in preferred:
        if candidate in options:
            return candidate
    return options[0] if options else "All"


OXECON_VIEW_FAMILY_ORDER = [
    "Headline labour market",
    "Sector jobs",
    "Sector GVA",
    "Occupation profile",
]

OXECON_VIEW_FAMILY_MAP = {
    "Headline labour market": [
        "Claimant count unemployment",
        "Claimant count unemployment rate",
        "ILO unemployment",
        "ILO unemployment rate",
        "People based employment",
        "Residence employment rate",
        "Resident employment",
    ],
    "Sector jobs": [
        "Employee jobs",
        "Employment - jobs based",
        "Self employed jobs",
    ],
    "Sector GVA": [
        "GVA by Sector",
    ],
    "Occupation profile": [
        "Administrative and secretarial",
        "Associate professional and technical",
        "Elementary",
        "Managers and senior professionals",
        "Personal services",
        "Process, plant and machine operatives",
        "Professionals",
        "Sales and customer services",
        "Skilled trades",
    ],
}

OXECON_COMPARISON_MODE_LABELS = {
    "raw_value": "Raw value",
    "rank_desc": "Rank",
    "delta_vs_uk": "Delta vs UK",
    "index_vs_uk_100": "Index vs UK=100",
}


def oxecon_indicator_family(indicator: str) -> str:
    indicator_name = str(indicator).strip()
    for family in OXECON_VIEW_FAMILY_ORDER:
        if indicator_name in OXECON_VIEW_FAMILY_MAP.get(family, []):
            return family
    return "Headline labour market"


def oxecon_family_to_indicators(df: pd.DataFrame, family: str) -> list[str]:
    available = set(df["indicator"].dropna().astype(str).unique().tolist())
    ordered = [
        indicator
        for indicator in OXECON_VIEW_FAMILY_MAP.get(family, [])
        if indicator in available
    ]
    if ordered:
        return ordered
    return sorted(indicator for indicator in available if oxecon_indicator_family(indicator) == family)


def oxecon_family_options(df: pd.DataFrame) -> list[str]:
    return [family for family in OXECON_VIEW_FAMILY_ORDER if oxecon_family_to_indicators(df, family)]


def oxecon_sector_is_required(family: str) -> bool:
    return family in {"Sector jobs", "Sector GVA"}


def oxecon_default_indicator(indicators: list[str]) -> str:
    preferred = [
        "Resident employment",
        "Residence employment rate",
        "Employment - jobs based",
        "GVA by Sector",
        "Managers and senior professionals",
    ]
    for candidate in preferred:
        if candidate in indicators:
            return candidate
    return indicators[0] if indicators else ""


def oxecon_format_value(value: object, units: str | None = None) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "n/a"
    units_text = str(units).strip().lower() if units is not None else ""
    if "rate" in units_text or "%" in units_text or "percent" in units_text:
        return "{0:.2f}%".format(float(numeric))
    if abs(float(numeric)) >= 100:
        return "{0:,.0f}".format(float(numeric))
    return "{0:,.2f}".format(float(numeric))


def oxecon_build_comparison_table(
    df: pd.DataFrame,
    indicator: str,
    year: int,
    sector: str | None,
) -> pd.DataFrame:
    local = df[
        (df["indicator"].astype(str) == indicator)
        & (pd.to_numeric(df["year"], errors="coerce") == int(year))
    ].copy()
    if sector:
        local = local[local["sector"].astype(str) == sector].copy()

    local["geo"] = local["geo"].astype(str).str.strip().str.upper()
    local["value"] = pd.to_numeric(local["value"], errors="coerce")
    local["region"] = local["region"].astype("string").fillna(local["location"].astype("string"))

    # The UK aggregate is used as the benchmark, while only NUTS1 regions are displayed.
    uk_rows = local[local["geo"] == "UK"].copy()
    uk_benchmark = float(uk_rows["value"].dropna().iloc[0]) if not uk_rows["value"].dropna().empty else None

    region_rows = local[local["geo"].map(geo_bucket) == "region"].copy()
    region_rows = apply_region_level_filter(region_rows, 3)
    if region_rows.empty:
        return pd.DataFrame(
            columns=[
                "geo",
                "region",
                "raw_value",
                "uk_benchmark",
                "delta_vs_uk",
                "index_vs_uk_100",
                "rank_desc",
                "units",
                "historical_end",
                "indicator",
                "sector",
            ]
        )

    comparison = region_rows[
        ["geo", "region", "value", "units", "historical_end", "indicator", "sector"]
    ].copy()
    comparison.rename(columns={"value": "raw_value", "UK benchmark": "uk_benchmark"}, inplace=True)
    comparison["uk_benchmark"] = uk_benchmark
    comparison["delta_vs_uk"] = (
        comparison["raw_value"] - comparison["uk_benchmark"] if uk_benchmark is not None else pd.NA
    )
    comparison["index_vs_uk_100"] = (
        comparison["raw_value"] / comparison["uk_benchmark"] * 100
        if uk_benchmark not in {None, 0}
        else pd.NA
    )
    comparison["rank_desc"] = (
        comparison["raw_value"].rank(method="min", ascending=False).astype("Int64")
    )
    comparison["sector"] = (
        comparison["sector"].astype("string").fillna("Whole Economy")
    )
    comparison["historical_end"] = comparison["historical_end"].astype("string")
    comparison = comparison.sort_values(["rank_desc", "region"], ascending=[True, True]).reset_index(drop=True)
    return comparison


def oxecon_prepare_display_metric(
    comparison_table: pd.DataFrame,
    comparison_mode: str,
) -> tuple[pd.DataFrame, str, str]:
    local = comparison_table.copy()
    local["display_value"] = pd.to_numeric(local[comparison_mode], errors="coerce")
    mode_label = OXECON_COMPARISON_MODE_LABELS.get(comparison_mode, "Raw value")
    return local, "display_value", mode_label


def folium_map_html_oxecon(
    gdf: gpd.GeoDataFrame,
    metric_col: str,
    title: str,
    metric_alias: str,
) -> str:
    valid_geom = gdf[gdf.geometry.notna()].copy()
    if valid_geom.empty:
        return folium.Map(location=[54, -2], zoom_start=5).get_root().render()

    render_geom = focus_europe_extent(valid_geom, metric_col)
    if render_geom.empty:
        render_geom = valid_geom

    data_geom = render_geom[render_geom[metric_col].notna()].copy() if metric_col in render_geom.columns else render_geom
    bounds_target = data_geom if not data_geom.empty else render_geom
    bounds = bounds_target.total_bounds
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2

    tooltip_fields = [
        field
        for field in [
            "region",
            metric_col,
            "raw_value",
            "rank_desc",
            "uk_benchmark",
            "delta_vs_uk",
            "index_vs_uk_100",
            "units",
            "historical_end",
        ]
        if field in render_geom.columns
    ]
    tooltip_aliases = {
        "region": "Region",
        metric_col: metric_alias,
        "raw_value": "Raw value",
        "rank_desc": "Rank",
        "uk_benchmark": "UK benchmark",
        "delta_vs_uk": "Delta vs UK",
        "index_vs_uk_100": "Index vs UK=100",
        "units": "Units",
        "historical_end": "Historical end",
    }

    m = folium.Map(location=[center_lat, center_lon], zoom_start=5, tiles="CartoDB positron")
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
            "weight": 0.7,
            "fillOpacity": 0.75,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=tooltip_fields,
            aliases=[tooltip_aliases[field] for field in tooltip_fields],
            localize=True,
        ),
    ).add_to(m)
    colormap.add_to(m)
    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
    return m.get_root().render()


def render_oxecon_tab(df: pd.DataFrame, boundaries: gpd.GeoDataFrame) -> None:
    st.subheader("oxecon_nuts1")
    st.caption("Source file: {0}".format(OXECON_DOWNLOAD_PATH))
    st.caption("UK Oxford Economics regional extract normalized onto mappable UK NUTS1 geocodes.")

    control_cols = st.columns([1.2, 1.4, 0.8, 1.0])
    family_options = oxecon_family_options(df)
    selected_family = control_cols[0].selectbox(
        "View family",
        family_options,
        index=0,
        key="oxecon_view_family",
    )

    indicator_options = oxecon_family_to_indicators(df, selected_family)
    default_indicator = oxecon_default_indicator(indicator_options)
    selected_indicator = control_cols[1].selectbox(
        "Indicator",
        indicator_options,
        index=indicator_options.index(default_indicator) if default_indicator in indicator_options else 0,
        key="oxecon_indicator",
    )

    indicator_df = df[df["indicator"].astype(str) == selected_indicator].copy()
    year_options = sorted(indicator_df["year"].dropna().astype(int).unique().tolist(), reverse=True)
    selected_year = control_cols[2].selectbox("Year", year_options, index=0, key="oxecon_year")

    selected_sector: str | None = None
    scoped_df = indicator_df[indicator_df["year"].astype(int) == int(selected_year)].copy()
    if oxecon_sector_is_required(selected_family):
        sector_options = sorted(scoped_df["sector"].dropna().astype(str).unique().tolist())
        if not sector_options:
            st.warning("No sector options are available for the selected indicator and year.")
            return
        default_sector = default_oxecon_sector(sector_options)
        selected_sector = control_cols[3].selectbox(
            "Sector",
            sector_options,
            index=sector_options.index(default_sector) if default_sector in sector_options else 0,
            key="oxecon_sector",
        )
    else:
        control_cols[3].caption("Sector fixed to whole-economy series for this view.")

    comparison_mode = st.radio(
        "Comparison mode",
        options=list(OXECON_COMPARISON_MODE_LABELS.keys()),
        format_func=lambda key: OXECON_COMPARISON_MODE_LABELS[key],
        horizontal=True,
        key="oxecon_comparison_mode",
    )

    comparison_table = oxecon_build_comparison_table(df, selected_indicator, int(selected_year), selected_sector)
    if comparison_table.empty:
        st.warning("No Oxecon rows are available for the selected view.")
        return

    comparison_table, display_metric, mode_label = oxecon_prepare_display_metric(comparison_table, comparison_mode)
    filtered_export = comparison_table.copy()

    units = ", ".join(sorted(comparison_table["units"].dropna().astype(str).unique().tolist()))
    historical_ends = sorted(comparison_table["historical_end"].dropna().astype(str).unique().tolist())
    benchmark_series = pd.to_numeric(comparison_table["uk_benchmark"], errors="coerce").dropna()
    benchmark_value = float(benchmark_series.iloc[0]) if not benchmark_series.empty else None
    ranked = comparison_table.dropna(subset=["display_value"]).sort_values("display_value", ascending=False)
    top_row = ranked.iloc[0] if not ranked.empty else None
    bottom_row = ranked.iloc[-1] if not ranked.empty else None

    render_kpi_strip(
        [
            ("Regions shown", "{0:,}".format(len(comparison_table))),
            ("UK benchmark", oxecon_format_value(benchmark_value, units)),
            (
                "Highest displayed",
                "n/a" if top_row is None else str(top_row["region"]),
                None if top_row is None else oxecon_format_value(top_row["display_value"], units if comparison_mode == "raw_value" else None),
            ),
            (
                "Lowest displayed",
                "n/a" if bottom_row is None else str(bottom_row["region"]),
                None if bottom_row is None else oxecon_format_value(bottom_row["display_value"], units if comparison_mode == "raw_value" else None),
            ),
        ]
    )

    context_bits = [
        "{0} in {1}".format(selected_indicator, selected_year),
        "mapped as {0}".format(mode_label.lower()),
        "UK aggregate used as the benchmark",
        "geography fixed to UK NUTS1 regions",
    ]
    if selected_sector:
        context_bits.insert(1, selected_sector)
    st.caption(" | ".join(context_bits))
    if units:
        st.caption("Units: {0}".format(units))
    if historical_ends:
        st.caption("Historical end available in table: {0}".format(", ".join(historical_ends)))

    plot_df, unmatched = build_map_df(comparison_table, boundaries, display_metric, "mean", "NUTS1 Regions")
    mapped_col = "{0}_mean".format(display_metric)
    if plot_df.empty or mapped_col not in plot_df.columns:
        st.warning("No mappable rows for the selected Oxecon configuration.")
        return

    plot_df = plot_df.merge(
        comparison_table[
            [
                "geo",
                "region",
                "raw_value",
                "rank_desc",
                "uk_benchmark",
                "delta_vs_uk",
                "index_vs_uk_100",
                "units",
                "historical_end",
            ]
        ],
        on="geo",
        how="left",
    )
    plot_df["display_value"] = plot_df[mapped_col]

    map_title = "oxecon_nuts1: {0} | {1} | {2}".format(selected_indicator, mode_label, selected_year)
    if selected_sector:
        map_title += " | {0}".format(selected_sector)
    components.html(
        folium_map_html_oxecon(plot_df, "display_value", map_title, mode_label),
        height=720,
        scrolling=True,
    )
    st.caption(
        "Map shows {0} for {1} UK NUTS1 regions.".format(
            mode_label.lower(),
            int(plot_df["display_value"].notna().sum()),
        )
    )
    if unmatched:
        st.warning("Unmatched geos ({0}): {1}".format(len(unmatched), ", ".join(unmatched[:20])))
    else:
        st.success("All geo codes matched to boundaries for this Oxecon view.")

    analysis_table = comparison_table[
        [
            "region",
            "display_value",
            "raw_value",
            "rank_desc",
            "uk_benchmark",
            "delta_vs_uk",
            "index_vs_uk_100",
            "units",
            "historical_end",
        ]
    ].copy()
    analysis_table.rename(
        columns={
            "region": "Region",
            "display_value": "Displayed value",
            "raw_value": "Raw value",
            "rank_desc": "Rank",
            "uk_benchmark": "UK benchmark",
            "delta_vs_uk": "Delta vs UK",
            "index_vs_uk_100": "Index vs UK=100",
            "units": "Units",
            "historical_end": "Historical end",
        },
        inplace=True,
    )
    st.dataframe(analysis_table, use_container_width=True, height=420)

    download_cols = st.columns(2)
    download_cols[0].download_button(
        label="Download filtered CSV",
        data=df_to_csv_bytes(filtered_export),
        file_name="oxecon_nuts1_filtered.csv",
        mime="text/csv",
    )
    export_cols = [
        col
        for col in [
            "geo",
            "region",
            "geo_class",
            "nuts_level",
            "display_value",
            "raw_value",
            "rank_desc",
            "uk_benchmark",
            "delta_vs_uk",
            "index_vs_uk_100",
            "units",
            "historical_end",
            "geometry",
        ]
        if col in plot_df.columns
    ]
    export_gdf = plot_df[export_cols].copy()
    download_cols[1].download_button(
        label="Download mapped GeoJSON",
        data=export_gdf.to_json().encode("utf-8"),
        file_name="oxecon_nuts1_mapped.geojson",
        mime="application/geo+json",
    )


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
        if not OXECON_DOWNLOAD_PATH.exists():
            raise ValueError("Oxecon download not found: {0}".format(OXECON_DOWNLOAD_PATH))
        if not EUROSTAT_UK_NUTS1_BOUNDARY_PATH.exists():
            raise ValueError("UK NUTS1 boundary lookup not found: {0}".format(EUROSTAT_UK_NUTS1_BOUNDARY_PATH))
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
        sheets = load_workbook(str(EUROSTAT_WORKBOOK_PATH))
        log("Loading boundary lookup")
        boundaries = load_boundaries(str(EUROSTAT_BOUNDARY_LOOKUP_PATH))
        log("Loading UK NUTS1 boundary lookup")
        uk_nuts1_boundaries = load_boundaries(str(EUROSTAT_UK_NUTS1_BOUNDARY_PATH))
        log("Loading Oxecon extract")
        oxecon = load_oxecon_data(str(OXECON_DOWNLOAD_PATH))
        combined_boundaries = combine_boundaries(boundaries, uk_nuts1_boundaries)
        return {"sheets": sheets, "boundaries": combined_boundaries, "oxecon": oxecon}

    def render(self, artifacts: AppArtifacts) -> None:
        sheets = artifacts["sheets"]
        boundaries = artifacts["boundaries"]
        oxecon = artifacts["oxecon"]

        render_page_header("Eurostat Workbook Explorer")
        st.caption("Source workbook: {0}".format(EUROSTAT_WORKBOOK_PATH))
        st.caption("Note: `employment_volume` is reported in thousand persons (Eurostat unit `THS_PER`).")
        st.success("Loaded {0} sheets.".format(len(sheets)))

        tab_labels = list(sheets.keys()) + ["oxecon_nuts1"]
        tabs = st.tabs(tab_labels)
        for tab_label, tab in zip(tab_labels, tabs):
            with tab:
                if tab_label == "oxecon_nuts1":
                    render_oxecon_tab(oxecon, boundaries)
                    continue

                sheet_name = tab_label
                df = sheets[sheet_name]
                st.subheader(sheet_name)

                if sheet_name == "granular_sex_age":
                    df = fill_granular_defaults(df)

                preprocessed = enforce_total_sex(df, sheet_name)
                geo_options = ["Country"] if sheet_name == "occupation_country" else geography_level_options(preprocessed)
                default_geo_option = "NUTS2 Regions" if "NUTS2 Regions" in geo_options else geo_options[0]
                geography_label = st.radio(
                    "Geography level",
                    geo_options,
                    horizontal=True,
                    index=geo_options.index(default_geo_option),
                    key="{0}_geo_level".format(sheet_name),
                )
                filtered, selected_region_len = apply_geography_level_filter(preprocessed, geography_label)
                age_note = ""

                if sheet_name == "granular_sex_age":
                    filtered, _, age_note = age_filter_with_overview(
                        filtered,
                        sheet_name,
                        key="{0}_age_group_filter".format(sheet_name),
                    )

                if sheet_name == "unemployment_by_edu":
                    filtered, _, age_note = age_filter_with_overview(
                        filtered,
                        sheet_name,
                        key="{0}_age_group_filter".format(sheet_name),
                    )
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
                    filtered, _, age_note = age_filter_with_overview(
                        filtered,
                        sheet_name,
                        key="{0}_age_group_filter".format(sheet_name),
                    )
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
                    filtered, _, age_note = age_filter_with_overview(
                        filtered,
                        sheet_name,
                        key="{0}_age_group_filter".format(sheet_name),
                    )
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
                        role_series = filtered[role_col].astype(str)
                        if selected_role == "All":
                            if "TOTAL" in role_options:
                                filtered = filtered[role_series != "TOTAL"].copy()
                        else:
                            filtered = filtered[role_series == selected_role].copy()

                if sheet_name == "granular_sex_age":
                    avg_emp_rate = pd.to_numeric(filtered.get("employment_rate"), errors="coerce").mean()
                    dynamic_total, dynamic_note = filtered_employment_total(filtered)
                    avg_unemp_rate = pd.to_numeric(filtered.get("unemployment_rate"), errors="coerce").mean()
                    render_kpi_strip(
                        [
                            ("Avg Employment Rate", "n/a" if pd.isna(avg_emp_rate) else "{0:.2f}%".format(avg_emp_rate)),
                            (
                                "Employment Total (thousand persons)",
                                "n/a" if dynamic_total is None else "{0:,.0f}".format(dynamic_total),
                            ),
                            ("Avg Unemployment Rate", "n/a" if pd.isna(avg_unemp_rate) else "{0:.2f}%".format(avg_unemp_rate)),
                        ],
                        columns=3,
                    )
                    st.caption(dynamic_note)
                elif sheet_name in {"industry_nuts2", "occupation_country"}:
                    dynamic_total, dynamic_note = filtered_employment_total(filtered)
                    render_kpi_strip(
                        [
                            ("Rows", "{0:,}".format(len(filtered))),
                            ("Unique geo", "{0:,}".format(filtered["geo"].nunique()) if "geo" in filtered.columns else "n/a"),
                            (
                                "Employment Total (thousand persons)",
                                "n/a" if dynamic_total is None else "{0:,.0f}".format(dynamic_total),
                            ),
                        ],
                        columns=3,
                    )
                    st.caption(dynamic_note)
                elif sheet_name == "unemployment_by_edu":
                    avg_unemp_rate = pd.to_numeric(filtered.get("unemployment_rate"), errors="coerce").mean()
                    render_kpi_strip(
                        [
                            ("Rows", "{0:,}".format(len(filtered))),
                            ("Unique geo", "{0:,}".format(filtered["geo"].nunique()) if "geo" in filtered.columns else "n/a"),
                            ("Avg Unemployment Rate", "n/a" if pd.isna(avg_unemp_rate) else "{0:.2f}%".format(avg_unemp_rate)),
                        ],
                        columns=3,
                    )
                else:
                    render_kpi_strip(
                        [
                            ("Rows", "{0:,}".format(len(filtered))),
                            ("Columns", "{0:,}".format(len(filtered.columns))),
                            ("Unique geo", "{0:,}".format(filtered["geo"].nunique()) if "geo" in filtered.columns else "n/a"),
                        ],
                        columns=3,
                    )

                if age_note:
                    st.caption(age_note)
                if selected_region_len is not None:
                    st.caption("Region view aligned to {0}.".format(geography_label))
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

                    plot_df, unmatched = build_map_df(filtered, boundaries, metric, agg, geography_label)
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
