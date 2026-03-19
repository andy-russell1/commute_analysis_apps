from __future__ import annotations

from typing import Any

import branca.colormap as cm
import folium
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from shared.runtime.downloads import df_to_csv_bytes
from shared.ui.kpi import render_kpi_strip
from shared.ui.page_header import render_page_header

from .config import CONFIG, LONDON_WIDE_METRICS, TALENT_METRICS
from .data import TalentAnalyticsBundle, list_available_clients, load_bundle, load_uploaded_bundle
from .geography import constituent_authority_lists


SAVILLS_COLOUR_SCALE = [
    [0.0, "#edf2f7"],
    [0.2, "#d7e3ea"],
    [0.4, "#a8c1c0"],
    [0.65, "#6f968f"],
    [1.0, "#262a43"],
]

TALENT_MAP_COLOURS = [
    "#ffffff",
    "#fff5f0",
    "#fee0d2",
    "#fcbba1",
    "#fc9272",
    "#fb6a4a",
    "#ef3b2c",
    "#cb181d",
    "#99000d",
]

PAGE_SEQUENCE = [
    {
        "key": "talent",
        "label": "1. Talent",
        "description": "Role demand across the custom London geography groups.",
    },
    {
        "key": "competition",
        "label": "2. Competition",
        "description": "London-wide hiring company and skills context for the supplied basket.",
    },
    {
        "key": "industry",
        "label": "3. Industry",
        "description": "Industry landscape for the combined London role basket.",
    },
    {
        "key": "demographics",
        "label": "4. Demographics",
        "description": "Public-data demographic context aggregated to the same custom geography contract.",
    },
    {
        "key": "methodology",
        "label": "5. Methodology",
        "description": "Scope notes, assumptions, and pack documentation.",
    },
]


def _apply_local_styles() -> None:
    st.markdown(
        """
        <style>
          .ta-note {
            border-left: 4px solid #4a9a8d;
            background: rgba(74, 154, 141, 0.08);
            padding: 0.9rem 1rem;
            border-radius: 0 14px 14px 0;
            color: #243341;
            margin: 0.75rem 0 1rem;
          }
          div[data-testid="stRadio"] > div[role="radiogroup"] {
            gap: 0.45rem;
          }
          div[data-testid="stRadio"] > div[role="radiogroup"] label {
            border: 1px solid #d7e3ea;
            border-radius: 999px;
            padding: 0.35rem 0.8rem;
            background: #f7fafc;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _page_labels() -> list[str]:
    return [page["label"] for page in PAGE_SEQUENCE]


def _get_active_page_label() -> str:
    page_labels = _page_labels()
    default_label = page_labels[0]
    current_label = str(st.session_state.get("talent_analytics_active_page", default_label))
    if current_label not in page_labels:
        current_label = default_label
    st.session_state["talent_analytics_active_page"] = current_label
    return current_label


def _format_number(value: Any) -> str:
    if value is None or pd.isna(value):
        return "Not available"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if numeric.is_integer():
        return f"{int(numeric):,}"
    return f"{numeric:,.1f}"


def _format_metric_value(value: Any, format_type: str) -> str:
    if value is None or pd.isna(value):
        return "Not available"
    numeric = float(value)
    if format_type == "percentage_1dp":
        return f"{numeric:.1%}"
    if format_type == "percentage_0dp":
        return f"{numeric:.0%}"
    if format_type == "decimal_1dp":
        return f"{numeric:,.1f}"
    return _format_number(numeric)


def _metric_options(metric_map: dict[str, str]) -> list[str]:
    return list(metric_map.keys())


def _role_options(bundle: TalentAnalyticsBundle) -> list[str]:
    role_frame = bundle.target_roles.sort_values("role_rank")
    return role_frame["role_label"].tolist()


def _selected_role_codes(bundle: TalentAnalyticsBundle, selected_role_labels: list[str]) -> list[str]:
    role_frame = bundle.target_roles.set_index("role_label")
    return role_frame.loc[selected_role_labels, "soc_code_4d"].tolist()


def _aggregate_talent(bundle: TalentAnalyticsBundle, selected_role_codes: list[str], metric: str) -> pd.DataFrame:
    filtered = bundle.postings_by_geography[bundle.postings_by_geography["soc_code_4d"].isin(selected_role_codes)].copy()
    if filtered.empty:
        return pd.DataFrame()
    filtered[metric] = pd.to_numeric(filtered[metric], errors="coerce")
    grouped = (
        filtered.groupby(
            [
                "custom_geography_key",
                "custom_geography_name",
                "constituent_authorities",
            ],
            as_index=False,
        )
        .agg(
            {
                metric: "sum",
                "soc_code_4d": pd.Series.nunique,
            }
        )
        .rename(columns={"soc_code_4d": "roles_in_scope"})
    )
    geo = bundle.london_custom_groups[
        ["custom_geography_key", "custom_geography_name", "lad_count", "geometry"]
    ].merge(grouped, on=["custom_geography_key", "custom_geography_name"], how="left")
    geo[metric] = pd.to_numeric(geo[metric], errors="coerce")
    return geo.sort_values(metric, ascending=False, na_position="last")


def _build_step_colormap(values: pd.Series) -> cm.StepColormap:
    clean = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if clean.empty:
        return cm.StepColormap(colors=TALENT_MAP_COLOURS, vmin=0, vmax=1)
    quantiles = clean.quantile([0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]).tolist()
    uniq: list[float] = []
    for value in quantiles:
        if not uniq or value > uniq[-1]:
            uniq.append(value)
    if len(uniq) < 2:
        uniq = [float(clean.min()), float(clean.max()) + 1e-9]
    return cm.StepColormap(colors=TALENT_MAP_COLOURS, vmin=uniq[0], vmax=uniq[-1], index=uniq)


def _render_talent_map(data: pd.DataFrame, metric: str, metric_label: str) -> None:
    plot_df = data.copy()
    plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
    data_geographies = plot_df[pd.notna(plot_df[metric])].copy()
    map_target = data_geographies if not data_geographies.empty else plot_df
    if map_target.empty:
        st.info("No geography rows are available for the current map selection.")
        return

    merged_geometry = map_target.geometry.union_all() if hasattr(map_target.geometry, "union_all") else map_target.geometry.unary_union
    center = merged_geometry.centroid
    map_object = folium.Map(location=[center.y, center.x], zoom_start=9)
    folium.TileLayer(
        "CartoDB positron",
        name="CartoDB Positron",
        attr="Map tiles by Carto, under CC BY 3.0 - Map data OpenStreetMap contributors",
        no_wrap=True,
    ).add_to(map_object)

    colormap = _build_step_colormap(plot_df[metric])
    colormap.caption = metric_label

    folium.GeoJson(
        plot_df,
        style_function=lambda feature: {
            "fillColor": (
                colormap(feature["properties"].get(metric))
                if feature["properties"].get(metric) is not None and pd.notna(feature["properties"].get(metric))
                else "#dfe7ef"
            ),
            "color": "#ffffff",
            "weight": 1.0,
            "fillOpacity": 0.72,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["custom_geography_name", metric, "constituent_authorities", "roles_in_scope"],
            aliases=["Geography group", metric_label, "Constituent authorities", "Roles represented"],
            localize=True,
            sticky=False,
        ),
    ).add_to(map_object)

    colormap.add_to(map_object)
    bounds = map_target.total_bounds
    leaflet_bounds = [[float(bounds[1]), float(bounds[0])], [float(bounds[3]), float(bounds[2])]]
    map_name = map_object.get_name()
    map_object.get_root().script.add_child(
        folium.Element(
            f"""
            setTimeout(function() {{
                {map_name}.invalidateSize();
                {map_name}.fitBounds({leaflet_bounds}, {{padding: [20, 20], maxZoom: 10}});
            }}, 250);
            """
        )
    )
    components.html(map_object.get_root().render(), height=640, scrolling=False)


def _build_rank_chart(df: pd.DataFrame, label_col: str, metric: str, metric_label: str, top_n: int, title: str) -> go.Figure:
    plot_df = df.head(top_n).copy()
    fig = px.bar(
        plot_df.sort_values(metric, ascending=True),
        x=metric,
        y=label_col,
        orientation="h",
        text=metric,
        color=metric,
        color_continuous_scale=SAVILLS_COLOUR_SCALE,
        labels={metric: metric_label, label_col: ""},
        title=title,
    )
    fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside", cliponaxis=False)
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=10, r=20, t=56, b=10),
        font=dict(color="#262a43"),
        coloraxis_showscale=False,
        yaxis=dict(categoryorder="array", categoryarray=plot_df.sort_values(metric, ascending=True)[label_col].tolist()),
    )
    return fig


def _render_demographics_map(data: pd.DataFrame, metric_label: str) -> None:
    plot_df = data.copy()
    plot_df["value"] = pd.to_numeric(plot_df["value"], errors="coerce")
    data_geographies = plot_df[pd.notna(plot_df["value"])].copy()
    map_target = data_geographies if not data_geographies.empty else plot_df
    if map_target.empty:
        st.info("No geography rows are available for the current map selection.")
        return

    merged_geometry = map_target.geometry.union_all() if hasattr(map_target.geometry, "union_all") else map_target.geometry.unary_union
    center = merged_geometry.centroid
    map_object = folium.Map(location=[center.y, center.x], zoom_start=9)
    folium.TileLayer(
        "CartoDB positron",
        name="CartoDB Positron",
        attr="Map tiles by Carto, under CC BY 3.0 - Map data OpenStreetMap contributors",
        no_wrap=True,
    ).add_to(map_object)

    colormap = _build_step_colormap(plot_df["value"])
    colormap.caption = metric_label

    folium.GeoJson(
        plot_df,
        style_function=lambda feature: {
            "fillColor": (
                colormap(feature["properties"].get("value"))
                if feature["properties"].get("value") is not None and pd.notna(feature["properties"].get("value"))
                else "#dfe7ef"
            ),
            "color": "#ffffff",
            "weight": 1.0,
            "fillOpacity": 0.72,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["custom_geography_name", "value_display", "constituent_authorities"],
            aliases=["Geography group", metric_label, "Constituent authorities"],
            localize=False,
            sticky=False,
        ),
    ).add_to(map_object)

    colormap.add_to(map_object)
    bounds = map_target.total_bounds
    leaflet_bounds = [[float(bounds[1]), float(bounds[0])], [float(bounds[3]), float(bounds[2])]]
    map_name = map_object.get_name()
    map_object.get_root().script.add_child(
        folium.Element(
            f"""
            setTimeout(function() {{
                {map_name}.invalidateSize();
                {map_name}.fitBounds({leaflet_bounds}, {{padding: [20, 20], maxZoom: 10}});
            }}, 250);
            """
        )
    )
    components.html(map_object.get_root().render(), height=640, scrolling=False)


def _download_button(label: str, df: pd.DataFrame, filename: str, key: str) -> None:
    st.download_button(
        label,
        data=df_to_csv_bytes(df),
        file_name=filename,
        mime="text/csv",
        key=key,
    )


def _build_unique_display_table(df: pd.DataFrame, column_order: list[str], rename_map: dict[str, str]) -> pd.DataFrame:
    selected = df[column_order].copy().rename(columns=rename_map)
    unique_names: list[str] = []
    counts: dict[str, int] = {}
    for column in selected.columns:
        seen = counts.get(column, 0)
        counts[column] = seen + 1
        unique_names.append(column if seen == 0 else f"{column} ({seen + 1})")
    selected.columns = unique_names
    return selected


def _set_active_page(label: str) -> None:
    st.session_state["talent_analytics_active_page"] = label


def _render_sidebar_navigation() -> str:
    active_label = _get_active_page_label()
    st.sidebar.subheader("Navigation")
    for page in PAGE_SEQUENCE:
        is_active = page["label"] == active_label
        st.sidebar.button(
            page["label"],
            key=f"talent_sidebar_nav_{page['key']}",
            use_container_width=True,
            disabled=is_active,
            on_click=_set_active_page,
            args=(page["label"],),
        )
    st.sidebar.divider()
    return active_label


def _render_page_status(active_label: str) -> str:
    page_labels = _page_labels()
    active_page = next(page for page in PAGE_SEQUENCE if page["label"] == active_label)
    active_index = page_labels.index(active_label)
    st.caption(f"Page {active_index + 1} of {len(PAGE_SEQUENCE)}. {active_page['description']}")
    return str(active_page["key"])


def _render_sidebar_data_source(clients: list[str]) -> tuple[Any, str | None]:
    st.sidebar.subheader("Data Source")
    upload = st.sidebar.file_uploader(
        "Upload private client pack (.zip)",
        type=["zip"],
        help=(
            "Upload a zip containing one client folder with the five required Talent Analytics CSV files. "
            "Shared geography and ONS / Nomis assets stay in the repository, while private client data is supplied at runtime."
        ),
        key="talent_analytics_sidebar_upload",
    )

    selected_client = None
    if clients:
        default_index = clients.index(CONFIG.default_client_id) if CONFIG.default_client_id in clients else 0
        with st.sidebar.expander("Checked-in client pack", expanded=False):
            selected_client = st.selectbox(
                "Checked-in client pack",
                options=clients,
                index=default_index,
                format_func=lambda value: value.replace("_", " ").title(),
                help="This is optional. An uploaded zip takes priority if you provide one.",
                key="talent_analytics_sidebar_client",
            )

    return upload, selected_client


def _render_scope_cards(bundle: TalentAnalyticsBundle) -> None:
    client_name = bundle.manifest.get("client", bundle.client_id)
    analysis_period = bundle.manifest.get("analysis_period", "Not stated")
    render_kpi_strip(
        [
            ("Client", _format_number(client_name), "Default client loaded from the app-ready pack"),
            ("Analysis period", _format_number(analysis_period), "Validated manual export window in the supplied pack"),
            (
                "Talent geographies",
                _format_number(bundle.london_custom_groups.shape[0]),
                "Custom London groups supplied for the PoC",
            ),
            (
                "Role basket",
                _format_number(bundle.target_roles.shape[0]),
                "Selected 4-digit SOC roles from anchor company analysis",
            ),
        ],
        columns=4,
    )


def _render_talent_tab(bundle: TalentAnalyticsBundle, default_role: str) -> None:
    st.subheader("Talent")
    st.caption("Demand is mapped only across the supplied grouped/custom London geographies. Borough-level rows are not inferred.")

    control_cols = st.columns([2.6, 1.4, 1.2])
    with control_cols[0]:
        role_selection = st.selectbox(
            "Role",
            options=_role_options(bundle),
            index=_role_options(bundle).index(default_role) if default_role in _role_options(bundle) else 0,
            help="The supplied postings file supports role-level views across the 10-role basket at custom London geography level.",
        )
    with control_cols[1]:
        metric = st.selectbox(
            "Talent metric",
            options=_metric_options(TALENT_METRICS),
            format_func=lambda value: TALENT_METRICS[value],
        )
    with control_cols[2]:
        include_zero_rows = st.toggle("Show empty rows", value=False, help="Include geographies with no value for the selected metric.")

    selected_codes = _selected_role_codes(bundle, [role_selection])
    talent_df = _aggregate_talent(bundle, selected_codes, metric)
    if talent_df.empty:
        st.info("No real postings rows are available for the current role selection.")
        return

    metric_label = TALENT_METRICS[metric]
    ranked_df = talent_df.drop(columns=["geometry"]).copy()
    ranked_df["rank"] = ranked_df[metric].rank(method="dense", ascending=False).astype("Int64")
    if not include_zero_rows:
        ranked_df = ranked_df[ranked_df[metric].fillna(0) > 0].copy()

    if ranked_df.empty:
        st.info("All selected geographies are empty for the current metric, so the view is intentionally blank.")
        return

    top_row = ranked_df.iloc[0]
    bottom_row = ranked_df.sort_values(metric, ascending=True).iloc[0]
    top_three_share = ranked_df.head(3)[metric].sum() / ranked_df[metric].sum() if ranked_df[metric].sum() else 0.0

    render_kpi_strip(
        [
            ("Highest geography", _format_number(top_row[metric]), str(top_row["custom_geography_name"])),
            ("Lowest geography", _format_number(bottom_row[metric]), str(bottom_row["custom_geography_name"])),
            ("Top 3 share", _format_number(f"{top_three_share:.0%}"), "Share of the selected metric across populated groups"),
        ],
        columns=3,
    )

    st.markdown(
        f"""
        <div class="ta-note">
          For <strong>{role_selection}</strong>, <strong>{top_row['custom_geography_name']}</strong> leads on
          {metric_label.lower()}, while <strong>{bottom_row['custom_geography_name']}</strong> sits at the bottom of
          the supplied grouped geography view. The top three custom groups account for {top_three_share:.0%} of the
          populated total for this role in the current cut of the pack.
        </div>
        """,
        unsafe_allow_html=True,
    )

    _render_talent_map(
        talent_df[talent_df["custom_geography_key"].isin(ranked_df["custom_geography_key"])].copy(),
        metric,
        metric_label,
    )

    display_df = ranked_df[
        [
            "rank",
            "custom_geography_name",
            "constituent_authorities",
            "roles_in_scope",
            metric,
        ]
    ].rename(
        columns={
            "custom_geography_name": "Geography group",
            "constituent_authorities": "Constituent authorities",
            "roles_in_scope": "Roles represented",
            metric: metric_label,
        }
    )
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    _download_button(
        "Download geography table",
        display_df,
        "talent_geography_rankings.csv",
        key="download_talent_table",
    )


def _render_competition_tab(bundle: TalentAnalyticsBundle) -> None:
    st.subheader("Competition")
    st.caption("The supplied company and skills files are London-wide combined outputs for the full role basket, not role-segmented exports.")

    st.markdown(
        """
        <div class="ta-note">
          Role-specific competition filtering is not available in this PoC because the supplied London-wide files do not
          contain role-level breakdown columns. The view below stays faithful to the combined basket export.
        </div>
        """,
        unsafe_allow_html=True,
    )

    controls = st.columns([1.5, 1.2, 1.2])
    with controls[0]:
        st.selectbox(
            "Role scope",
            options=["Combined supplied role basket"],
            disabled=True,
            help="The uploaded company and skills files are basket-level only.",
        )
    with controls[1]:
        metric = st.selectbox(
            "Competition metric",
            options=_metric_options(LONDON_WIDE_METRICS),
            format_func=lambda value: LONDON_WIDE_METRICS[value],
            key="competition_metric",
        )
    with controls[2]:
        top_n = st.slider("Top N", min_value=5, max_value=25, value=10, step=5, key="competition_top_n")

    metric_label = LONDON_WIDE_METRICS[metric]
    companies = bundle.company_rankings.sort_values(metric, ascending=False).copy()
    skills = bundle.skills.sort_values(metric, ascending=False).copy()
    company_top = companies.iloc[0]
    skill_top = skills.iloc[0]

    render_kpi_strip(
        [
            ("Top hiring company", _format_number(company_top[metric]), str(company_top["company_name"])),
            ("Top skill", _format_number(skill_top[metric]), str(skill_top["skill_or_qualification"])),
            (
                "Competition scope",
                _format_number(bundle.company_rankings["geography_name"].iloc[0]),
                "London-wide combined role basket",
            ),
        ],
        columns=3,
    )

    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.plotly_chart(
            _build_rank_chart(companies, "company_name", metric, metric_label, top_n, "Top hiring companies"),
            use_container_width=True,
        )
        company_table = _build_unique_display_table(
            companies.head(top_n),
            ["rank_london", "company_name", metric, "median_annual_advertised_salary"],
            {
                "rank_london": "Rank",
                "company_name": "Company",
                metric: metric_label,
                "median_annual_advertised_salary": "Median advertised salary",
            },
        )
        st.dataframe(company_table, use_container_width=True, hide_index=True)
        _download_button(
            "Download company table",
            company_table,
            "competition_companies_london.csv",
            key="download_companies",
        )

    with chart_cols[1]:
        st.plotly_chart(
            _build_rank_chart(skills, "skill_or_qualification", metric, metric_label, top_n, "Top skills and qualifications"),
            use_container_width=True,
        )
        skill_table = _build_unique_display_table(
            skills.head(top_n),
            ["rank_london", "skill_or_qualification", metric, "number_of_companies_posting"],
            {
                "rank_london": "Rank",
                "skill_or_qualification": "Skill or qualification",
                metric: metric_label,
                "number_of_companies_posting": "Companies posting",
            },
        )
        st.dataframe(skill_table, use_container_width=True, hide_index=True)
        _download_button(
            "Download skills table",
            skill_table,
            "competition_skills_london.csv",
            key="download_skills",
        )


def _render_industry_tab(bundle: TalentAnalyticsBundle) -> None:
    st.subheader("Industry")
    st.caption("Industry context is provided only for the combined London basket in the supplied pack.")

    controls = st.columns([1.5, 1.2, 1.2])
    with controls[0]:
        st.selectbox(
            "Role scope",
            options=["Combined supplied role basket"],
            disabled=True,
            help="The uploaded industry file is not split by role.",
        )
    with controls[1]:
        metric = st.selectbox(
            "Industry metric",
            options=_metric_options(LONDON_WIDE_METRICS),
            format_func=lambda value: LONDON_WIDE_METRICS[value],
            key="industry_metric",
        )
    with controls[2]:
        top_n = st.slider("Top industries", min_value=5, max_value=25, value=10, step=5, key="industry_top_n")

    metric_label = LONDON_WIDE_METRICS[metric]
    industries = bundle.industry_landscape.sort_values(metric, ascending=False).copy()
    top_industry = industries.iloc[0]

    render_kpi_strip(
        [
            ("Top industry", _format_number(top_industry[metric]), str(top_industry["industry_name"])),
            ("Industries in file", _format_number(industries.shape[0]), "Distinct industry rows supplied in the PoC pack"),
            ("Scope", _format_number(industries["geography_name"].iloc[0]), "Combined London industry landscape"),
        ],
        columns=3,
    )

    st.markdown(
        """
        <div class="ta-note">
          Industry granularity is limited to the supplied London-wide output. This PoC does not infer borough-level or
          role-specific industry demand where the real file does not provide it.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.plotly_chart(
        _build_rank_chart(industries, "industry_name", metric, metric_label, top_n, "Top industries in the London basket"),
        use_container_width=True,
    )

    industry_table = _build_unique_display_table(
        industries.head(top_n),
        [
            "rank_london",
            "sic_code",
            "industry_name",
            metric,
            "number_of_companies_posting",
            "median_annual_advertised_salary",
        ],
        {
            "rank_london": "Rank",
            "sic_code": "SIC code",
            "industry_name": "Industry",
            metric: metric_label,
            "number_of_companies_posting": "Companies posting",
            "median_annual_advertised_salary": "Median advertised salary",
        },
    )
    st.dataframe(industry_table, use_container_width=True, hide_index=True)
    _download_button(
        "Download industry table",
        industry_table,
        "industry_landscape_london.csv",
        key="download_industry",
    )


def _render_demographics_tab(bundle: TalentAnalyticsBundle) -> None:
    st.subheader("Demographics")
    st.caption("Public-data demographics are preprocessed from live Nomis / ONS extracts at LAD level, then aggregated to the 20 custom London regions before the app loads.")

    metadata = bundle.demographics_metadata[bundle.demographics_metadata["is_active"]].copy()
    if metadata.empty or bundle.demographics_by_custom_geography.empty:
        st.info("No processed demographics outputs are available in the current shared Talent Analytics data pack.")
        return

    family_options = (
        metadata.groupby("metric_family", as_index=False)["display_order"].min().sort_values("display_order")["metric_family"].tolist()
    )
    geography_lists = constituent_authority_lists(bundle.geography_lookup)

    control_cols = st.columns([1.3, 1.7])
    with control_cols[0]:
        selected_family = st.selectbox("Metric family", options=family_options)

    family_metadata = metadata[metadata["metric_family"] == selected_family].sort_values("display_order").reset_index(drop=True)
    with control_cols[1]:
        selected_metric = st.selectbox(
            "Metric",
            options=family_metadata["metric_key"].tolist(),
            format_func=lambda value: str(
                family_metadata.loc[family_metadata["metric_key"] == value, "short_label"].iloc[0]
            ),
        )

    selected_meta = family_metadata.loc[family_metadata["metric_key"] == selected_metric].iloc[0]
    metric_rows = bundle.demographics_by_custom_geography[
        bundle.demographics_by_custom_geography["metric_key"] == selected_metric
    ].copy()
    metric_rows = metric_rows.merge(
        geography_lists,
        on=["custom_geography_key", "custom_geography_name", "display_order"],
        how="left",
    )
    geo_df = bundle.london_custom_groups[
        ["custom_geography_key", "custom_geography_name", "display_order", "geometry"]
    ].merge(
        metric_rows,
        on=["custom_geography_key", "custom_geography_name", "display_order"],
        how="left",
    )
    geo_df["value"] = pd.to_numeric(geo_df["value"], errors="coerce")
    geo_df["value_display"] = geo_df["value"].apply(
        lambda value: _format_metric_value(value, str(selected_meta["format_type"]))
    )

    ranked = geo_df.drop(columns=["geometry"]).copy()
    ranked = ranked[pd.notna(ranked["value"])].copy()
    if ranked.empty:
        st.info("The processed demographics file does not contain rows for the selected metric.")
        return

    ascending = str(selected_meta.get("default_sort_direction", "desc")).lower() == "asc"
    ranked = ranked.sort_values(["value", "display_order"], ascending=[ascending, True], na_position="last").reset_index(drop=True)
    ranked["rank"] = range(1, len(ranked) + 1)

    top_row = ranked.iloc[0]
    bottom_row = ranked.iloc[-1]
    period_label = str(ranked["period"].dropna().iloc[0]) if ranked["period"].notna().any() else "Not stated"

    render_kpi_strip(
        [
            ("Period", period_label, str(selected_meta["source_dataset"])),
            ("Leading geography", top_row["value_display"], str(top_row["custom_geography_name"])),
            ("Trailing geography", bottom_row["value_display"], str(bottom_row["custom_geography_name"])),
        ],
        columns=3,
    )

    st.markdown(
        f"""
        <div class="ta-note">
          {selected_meta['source_note']} Count metrics are summed across boroughs, while rates and shares use explicit
          denominator-weighted aggregation in preprocessing rather than unweighted borough averages.
        </div>
        """,
        unsafe_allow_html=True,
    )

    _render_demographics_map(geo_df, str(selected_meta["metric_label"]))

    display_df = ranked[
        [
            "rank",
            "custom_geography_name",
            "constituent_authorities",
            "value_display",
            "period",
            "source_dataset",
        ]
    ].rename(
        columns={
            "custom_geography_name": "Geography group",
            "constituent_authorities": "Constituent authorities",
            "value_display": str(selected_meta["metric_label"]),
            "period": "Period",
            "source_dataset": "Source dataset",
        }
    )
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    _download_button(
        "Download demographics view",
        display_df,
        f"{selected_metric}_custom_geography.csv",
        key=f"download_{selected_metric}",
    )


def _render_methodology_tab(bundle: TalentAnalyticsBundle) -> None:
    st.subheader("Methodology")
    st.markdown(
        """
        - Any checked-in public view of this app is limited to shared geography and shared ONS / Nomis outputs only.
        - The Talent map uses grouped/custom London geographies supplied in the pack, not a full one-row-per-borough production dataset.
        - The target role basket is based on selected 4-digit SOC roles from the anchor company analysis.
        - Competition, skills, and industry views are London-wide combined-basket outputs for the 20-authority geography scope.
        - Demographics now loads from processed shared Nomis / ONS outputs keyed to the same custom geography contract as the Lightcast pack.
        - ONS / Nomis metrics are extracted at LAD level, validated, and aggregated to the 20 custom London groups in preprocessing before the app loads.
        - Count metrics are summed, while labour-market and profile rates use explicit denominator-weighted aggregation to preserve defensible combined values.
        """
    )

    notes = bundle.manifest.get("notes", [])
    if notes:
        st.markdown("**Pack notes**")
        for note in notes:
            st.write(f"- {note}")

    with st.expander("Pack README excerpt", expanded=False):
        st.text(bundle.readme)


def _load_bundle_or_show_error(client_id: str) -> TalentAnalyticsBundle | None:
    try:
        return load_bundle(client_id)
    except FileNotFoundError as exc:
        st.error(str(exc))
    except Exception as exc:  # pragma: no cover - defensive UI path
        st.error(f"Talent Analytics could not load the supplied pack: {exc}")
    return None


def _load_uploaded_bundle_or_show_error(upload_name: str, upload_bytes: bytes) -> TalentAnalyticsBundle | None:
    try:
        return load_uploaded_bundle(upload_name, upload_bytes)
    except FileNotFoundError as exc:
        st.error(str(exc))
    except Exception as exc:  # pragma: no cover - defensive UI path
        st.error(f"Talent Analytics could not load the uploaded client pack: {exc}")
    return None


def run_talent_analytics() -> None:
    _apply_local_styles()
    render_page_header(
        "Talent Analytics",
        "Client-showable PoC built only from the supplied app-ready pack.",
    )

    clients = list_available_clients()
    active_label = _render_sidebar_navigation()
    upload, selected_client = _render_sidebar_data_source(clients)

    bundle = None
    if upload is not None:
        bundle = _load_uploaded_bundle_or_show_error(upload.name, upload.getvalue())
    elif selected_client is not None:
        bundle = _load_bundle_or_show_error(selected_client)
    else:
        st.info(
            "Upload a private Talent Analytics client zip to continue. "
            "The repository now keeps only shared geography and shared ONS / Nomis outputs."
        )
        return

    if bundle is None:
        return

    pack_source = upload.name if upload is not None else selected_client
    if pack_source:
        st.sidebar.caption(f"Loaded pack: {pack_source}")

    with st.expander("Current client pack summary", expanded=False):
        _render_scope_cards(bundle)

    default_role = _role_options(bundle)[0]
    active_page = _render_page_status(active_label)

    if active_page == "talent":
        _render_talent_tab(bundle, default_role)
    elif active_page == "competition":
        _render_competition_tab(bundle)
    elif active_page == "industry":
        _render_industry_tab(bundle)
    elif active_page == "demographics":
        _render_demographics_tab(bundle)
    else:
        _render_methodology_tab(bundle)
