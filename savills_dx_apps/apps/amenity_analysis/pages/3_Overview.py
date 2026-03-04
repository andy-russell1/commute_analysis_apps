from __future__ import annotations

import pandas as pd
import streamlit as st

from apps.amenity_analysis.common import (
    CACHE_STATS_KEY,
    HAS_PYDECK,
    PRIMARY_RADIUS_KEY,
    RESULTS_BY_RADIUS_KEY,
    SELECTED_METRICS_KEY,
    TRANSPORT_METRIC,
    WEIGHTS_NORM_KEY,
    build_amenity_points,
    build_cia_style_map,
    build_office_scores,
    get_amenity_bucket_names,
    init_amenity_state,
    navigate_to,
    render_comparison_panel,
    render_density_map,
    render_tradeoff_panel,
    safe_set_page_config,
)
from dx_core.scoring.amenity_index import bucket_slug


safe_set_page_config(page_title="Amenity Analysis - Overview", page_icon="📊", layout="wide")
init_amenity_state(st.session_state)

st.title("Overview")

results_by_radius = st.session_state[RESULTS_BY_RADIUS_KEY]
if not results_by_radius:
    st.info("Run analysis from Controls page first.")
    if st.button("Open Controls"):
        navigate_to(st.session_state, route="controls", standalone_page_path="pages/2_Controls.py")
    st.stop()

available_radii = sorted(results_by_radius.keys())
selected_radius = int(st.session_state.get(PRIMARY_RADIUS_KEY, available_radii[0]))
if selected_radius not in available_radii:
    selected_radius = available_radii[0]

radius_result = results_by_radius[selected_radius]
summary_df = radius_result["summary_df"]
poi_df = radius_result["poi_df"]
selected_metrics = st.session_state.get(SELECTED_METRICS_KEY, [])
weights_norm = st.session_state.get(WEIGHTS_NORM_KEY, {})

if summary_df.empty:
    st.warning("No office results available for this radius.")
    st.stop()

office_scores_df = build_office_scores(summary_df=summary_df, selected_metrics=selected_metrics)
amenity_categories = [metric for metric in selected_metrics if metric in get_amenity_bucket_names()]
if not amenity_categories:
    amenity_categories = sorted(set(poi_df.get("bucket", pd.Series(dtype=str)).dropna().astype(str).tolist()))

amenity_points_df = build_amenity_points(
    poi_df=poi_df,
    office_scores_df=office_scores_df,
    selected_categories=amenity_categories,
    weights_norm=weights_norm,
)

density_mode = str(st.session_state.get("overview_density_mode", "Hex"))
if density_mode not in {"Hex", "Heatmap"}:
    density_mode = "Hex"

# Keep checkbox state as the single source of truth to avoid one-run lag/inconsistent renders.
for category in amenity_categories:
    checkbox_key = f"overview_metric_checkbox_{bucket_slug(category)}"
    if checkbox_key not in st.session_state:
        st.session_state[checkbox_key] = True

selected_map_categories = [
    category
    for category in amenity_categories
    if bool(st.session_state.get(f"overview_metric_checkbox_{bucket_slug(category)}", False))
]
st.session_state["overview_map_categories"] = list(selected_map_categories)

filtered_points_df = amenity_points_df[amenity_points_df["category"].isin(selected_map_categories)].copy()
if not filtered_points_df.empty:
    # Map display should not double-count the same POI across overlapping office catchments.
    map_points_df = (
        filtered_points_df.groupby(["category", "name", "lat", "lon"], as_index=False)
        .agg(
            distance_m=("distance_m", "min"),
            weight_contribution=("weight_contribution", "max"),
            officeID=("officeID", "first"),
            office_name=("office_name", "first"),
        )
    )
else:
    map_points_df = filtered_points_df

overview_tab, office_insights_tab, comparison_tab = st.tabs(["Overview", "Office Insights", "Comparison"])

with overview_tab:
    st.subheader("Overview")

    m1, m2, m3, m4 = st.columns(4)
    best_idx = office_scores_df["total_score"].astype(float).idxmax()
    best_office = str(office_scores_df.loc[best_idx, "office_name"])
    best_score = float(office_scores_df.loc[best_idx, "total_score"])
    m1.metric("Best office", best_office)
    m2.metric("Best score", f"{best_score:.1f}")
    m3.metric("Radius", f"{selected_radius} m")
    m4.metric("Unique amenities shown", len(map_points_df))

    st.caption(
        f"Density map radius: {selected_radius}m. Categories shown: {', '.join(selected_map_categories) if selected_map_categories else 'None'}"
    )
    st.caption("Map shows unique POIs across selected offices. KPI scoring still uses per-office catchments.")

    if HAS_PYDECK:
        deck = render_density_map(
            office_scores_df=office_scores_df,
            amenity_points_df=map_points_df,
            radius_m=int(selected_radius),
            selected_categories=selected_map_categories,
            density_type=density_mode,
        )
        if deck is not None:
            st.pydeck_chart(deck, use_container_width=True)
        else:
            fallback = build_cia_style_map(
                summary_df=summary_df,
                poi_df=poi_df[poi_df["bucket"].isin(selected_map_categories)].copy() if not poi_df.empty else poi_df,
                title=f"Amenity KPI map ({selected_radius}m)",
            )
            if fallback is not None:
                st.plotly_chart(fallback, use_container_width=True, config={"scrollZoom": True})
    else:
        fallback = build_cia_style_map(
            summary_df=summary_df,
            poi_df=poi_df[poi_df["bucket"].isin(selected_map_categories)].copy() if not poi_df.empty else poi_df,
            title=f"Amenity KPI map ({selected_radius}m)",
        )
        if fallback is not None:
            st.plotly_chart(fallback, use_container_width=True, config={"scrollZoom": True})

    st.markdown("### Map display controls")
    controls_left, controls_right = st.columns([1.0, 2.0])
    with controls_left:
        st.radio("Density type", options=["Hex", "Heatmap"], horizontal=True, key="overview_density_mode")
    with controls_right:
        st.markdown("Metrics to display on map")
        selected_from_checks: list[str] = []
        checkbox_columns = st.columns(max(len(amenity_categories), 1))
        for idx, category in enumerate(amenity_categories):
            with checkbox_columns[idx % len(checkbox_columns)]:
                if st.checkbox(category, key=f"overview_metric_checkbox_{bucket_slug(category)}"):
                    selected_from_checks.append(category)
        st.session_state["overview_map_categories"] = selected_from_checks
        if not selected_from_checks:
            st.caption("No map metrics selected.")

    st.markdown("### Office insights snapshot")
    snapshot_office_ids = office_scores_df["officeID"].astype(str).tolist()
    snapshot_office_name_map = dict(
        zip(office_scores_df["officeID"].astype(str), office_scores_df["office_name"].astype(str))
    )
    snapshot_office = st.selectbox(
        "Office",
        options=snapshot_office_ids,
        format_func=lambda oid: snapshot_office_name_map.get(str(oid), str(oid)),
        key="overview_snapshot_office",
    )
    render_tradeoff_panel(
        office_scores_df=office_scores_df,
        selected_office_id=str(snapshot_office),
        selected_categories=selected_map_categories,
    )

    st.markdown("### Results table")
    display_df = office_scores_df[["office_name", "total_score"]].copy()
    display_df = display_df.rename(columns={"office_name": "Office", "total_score": "Amenity Index"})

    transport_col = "nearest_public_transport_stop_distance_m"
    if TRANSPORT_METRIC in selected_metrics and transport_col in office_scores_df.columns:
        display_df["Public transport distance (m)"] = pd.to_numeric(
            office_scores_df[transport_col], errors="coerce"
        ).round(1)

    for category in amenity_categories:
        slug = bucket_slug(category)
        count_col = f"count_{slug}"
        sub_col = f"subscore_{slug}"
        if count_col in office_scores_df.columns:
            display_df[f"{category} count"] = pd.to_numeric(
                office_scores_df[count_col], errors="coerce"
            ).fillna(0).astype(int)
        if sub_col in office_scores_df.columns:
            display_df[f"{category} subscore"] = pd.to_numeric(
                office_scores_df[sub_col], errors="coerce"
            ).round(3)

    st.dataframe(display_df.fillna("Not available").sort_values("Amenity Index", ascending=False), use_container_width=True)

    if TRANSPORT_METRIC in selected_metrics and transport_col in office_scores_df.columns:
        if pd.to_numeric(office_scores_df[transport_col], errors="coerce").isna().all():
            st.warning(
                "Public transport is selected but NaPTAN data was not found, so transport values are not available."
            )

    cache_stats = st.session_state[CACHE_STATS_KEY]
    st.caption(
        f"Cache reused {int(cache_stats.get('hits', 0))} office queries and ran {int(cache_stats.get('misses', 0))} new queries."
    )

with office_insights_tab:
    st.subheader("Office Insights")
    current_categories = st.session_state.get("overview_map_categories", amenity_categories)
    current_categories = [category for category in current_categories if category in amenity_categories]

    office_ids = office_scores_df["officeID"].astype(str).tolist()
    office_name_map = dict(zip(office_scores_df["officeID"].astype(str), office_scores_df["office_name"].astype(str)))
    selected_office = st.selectbox(
        "Selected office",
        options=office_ids,
        format_func=lambda oid: office_name_map.get(str(oid), str(oid)),
        key="overview_selected_office",
    )

    render_tradeoff_panel(
        office_scores_df=office_scores_df,
        selected_office_id=str(selected_office),
        selected_categories=current_categories,
    )

    selected_office_row = office_scores_df[office_scores_df["officeID"].astype(str) == str(selected_office)]
    if not selected_office_row.empty:
        row = selected_office_row.iloc[0]
        detail_rows: list[dict[str, object]] = []
        for category in amenity_categories:
            slug = bucket_slug(category)
            count_val = pd.to_numeric(row.get(f"count_{slug}"), errors="coerce")
            subscore_val = pd.to_numeric(row.get(f"subscore_{slug}"), errors="coerce")
            nearest_val = pd.to_numeric(row.get(f"nearest_distance_m_{slug}"), errors="coerce")
            detail_rows.append(
                {
                    "Metric": category,
                    "Count within radius": int(count_val) if pd.notna(count_val) else 0,
                    "Nearest distance (m)": round(float(nearest_val), 1) if pd.notna(nearest_val) else "Not available",
                    "Subscore (0-1)": round(float(subscore_val), 3) if pd.notna(subscore_val) else 0.0,
                }
            )

        transport_col = "nearest_public_transport_stop_distance_m"
        if TRANSPORT_METRIC in selected_metrics:
            transport_distance = pd.to_numeric(row.get(transport_col), errors="coerce")
            detail_rows.append(
                {
                    "Metric": TRANSPORT_METRIC,
                    "Count within radius": "N/A",
                    "Subscore (0-1)": "N/A",
                    "Nearest distance (m)": round(float(transport_distance), 1) if pd.notna(transport_distance) else "Not available",
                }
            )

        detail_df = pd.DataFrame(detail_rows)
        st.markdown("### Office metric breakdown")
        st.dataframe(detail_df, use_container_width=True)

    c1, c2 = st.columns([1, 3])
    with c1:
        if st.button("Open location drilldown"):
            navigate_to(st.session_state, route="drilldown", standalone_page_path="pages/4_Location_Drilldown.py")
    with c2:
        st.caption("Use Location Drilldown for office-specific map and subcategory detail.")

with comparison_tab:
    st.subheader("Comparison")

    office_ids = office_scores_df["officeID"].astype(str).tolist()
    office_name_map = dict(zip(office_scores_df["officeID"].astype(str), office_scores_df["office_name"].astype(str)))

    c_a, c_b = st.columns(2)
    with c_a:
        office_a = st.selectbox(
            "Office A",
            options=office_ids,
            format_func=lambda oid: office_name_map.get(str(oid), str(oid)),
            key="comparison_office_a",
        )
    with c_b:
        office_b_default_index = 1 if len(office_ids) > 1 else 0
        office_b = st.selectbox(
            "Office B",
            options=office_ids,
            index=office_b_default_index,
            format_func=lambda oid: office_name_map.get(str(oid), str(oid)),
            key="comparison_office_b",
        )

    render_comparison_panel(
        office_scores_df=office_scores_df,
        office_a_id=str(office_a),
        office_b_id=str(office_b),
        selected_categories=amenity_categories,
    )
