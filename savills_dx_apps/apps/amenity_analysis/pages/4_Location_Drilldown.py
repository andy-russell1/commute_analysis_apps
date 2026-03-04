from __future__ import annotations

import pandas as pd
import streamlit as st

from apps.amenity_analysis.common import (
    HAS_PYDECK,
    PRIMARY_RADIUS_KEY,
    RESULTS_BY_RADIUS_KEY,
    SELECTED_METRICS_KEY,
    TRANSPORT_METRIC,
    WEIGHTS_NORM_KEY,
    build_amenity_points,
    build_location_drilldown_map,
    build_office_scores,
    get_amenity_bucket_names,
    init_amenity_state,
    navigate_to,
    render_density_map,
    safe_set_page_config,
)
from dx_core.scoring.amenity_index import bucket_slug, count_column, nearest_distance_column


safe_set_page_config(page_title="Amenity Analysis - Location Drilldown", page_icon="🧭", layout="wide")
init_amenity_state(st.session_state)

st.title("Location Drilldown")

results_by_radius = st.session_state[RESULTS_BY_RADIUS_KEY]
if not results_by_radius:
    st.info("Run analysis from Controls page first.")
    if st.button("Open Controls"):
        navigate_to(st.session_state, route="controls", standalone_page_path="pages/2_Controls.py")
    st.stop()

radii = sorted(results_by_radius.keys())
selected_radius = int(st.session_state.get(PRIMARY_RADIUS_KEY, radii[0]))
if selected_radius not in radii:
    selected_radius = radii[0]

summary_df = results_by_radius[selected_radius]["summary_df"]
poi_df = results_by_radius[selected_radius]["poi_df"]
if summary_df.empty:
    st.warning("No office data available.")
    st.stop()

if "office_name" not in summary_df.columns:
    summary_df = summary_df.copy()
    summary_df["office_name"] = summary_df["address"].astype(str).str.split(",").str[0].str.strip()

office_lookup = (
    summary_df[["officeID", "office_name"]]
    .drop_duplicates()
    .assign(office_name=lambda d: d["office_name"].fillna(d["officeID"]))
)
office_ids = office_lookup["officeID"].astype(str).tolist()
name_by_id = dict(zip(office_lookup["officeID"].astype(str), office_lookup["office_name"].astype(str)))

control_col_1, control_col_2 = st.columns(2)
with control_col_1:
    selected_office = st.selectbox(
        "Office",
        options=office_ids,
        format_func=lambda office_id: name_by_id.get(str(office_id), str(office_id)),
    )
with control_col_2:
    density_mode = st.radio("Density type", options=["Hex", "Heatmap"], horizontal=True)

selected_row_match = summary_df[summary_df["officeID"].astype(str) == selected_office]
if selected_row_match.empty:
    st.warning("Selected office not found in results.")
    st.stop()
selected_office_row = selected_row_match.iloc[0]

selected_metrics = st.session_state.get(SELECTED_METRICS_KEY, [])
selected_categories = [metric for metric in selected_metrics if metric in get_amenity_bucket_names()]
weights_norm = st.session_state.get(WEIGHTS_NORM_KEY, {})

office_pois = poi_df[poi_df["officeID"].astype(str) == selected_office].copy() if not poi_df.empty else poi_df

selected_summary_df = selected_office_row.to_frame().T.copy()
office_scores_df = build_office_scores(summary_df=selected_summary_df, selected_metrics=selected_metrics)
amenity_points_df = build_amenity_points(
    poi_df=office_pois,
    office_scores_df=office_scores_df,
    selected_categories=selected_categories,
    weights_norm=weights_norm,
)

office_name = str(selected_office_row.get("office_name", selected_office))
st.caption(f"Showing {office_name} at {selected_radius}m radius")

if HAS_PYDECK:
    deck = render_density_map(
        office_scores_df=office_scores_df,
        amenity_points_df=amenity_points_df,
        radius_m=int(selected_radius),
        selected_categories=selected_categories,
        density_type=density_mode,
        ring_radii=[int(selected_radius)],
    )
    if deck is not None:
        st.pydeck_chart(deck, use_container_width=True)
    else:
        fig = build_location_drilldown_map(
            office_row=selected_office_row,
            poi_df=office_pois,
            radii_m=[int(selected_radius)],
            title=f"{office_name} drilldown ({selected_radius}m)",
        )
        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})
else:
    fig = build_location_drilldown_map(
        office_row=selected_office_row,
        poi_df=office_pois,
        radii_m=[int(selected_radius)],
        title=f"{office_name} drilldown ({selected_radius}m)",
    )
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

st.markdown("### Full breakdown")
metric_rows: list[dict[str, object]] = []
for metric in selected_metrics:
    if metric == TRANSPORT_METRIC:
        transport_distance = pd.to_numeric(
            selected_office_row.get("nearest_public_transport_stop_distance_m"), errors="coerce"
        )
        transport_subscore = pd.to_numeric(
            selected_office_row.get(f"normalised_{bucket_slug(TRANSPORT_METRIC)}"), errors="coerce"
        )
        metric_rows.append(
            {
                "Metric": metric,
                "Subcategory": "Nearest stop distance",
                "Count within radius": "N/A",
                "Nearest distance (m)": round(float(transport_distance), 1) if pd.notna(transport_distance) else None,
                "Subscore (0-1)": round(float(transport_subscore), 3) if pd.notna(transport_subscore) else None,
            }
        )
        continue

    slug = bucket_slug(metric)
    count_val = pd.to_numeric(selected_office_row.get(count_column(metric)), errors="coerce")
    nearest_val = pd.to_numeric(selected_office_row.get(nearest_distance_column(metric)), errors="coerce")
    subscore_val = pd.to_numeric(selected_office_row.get(f"normalised_{slug}"), errors="coerce")
    metric_rows.append(
        {
            "Metric": metric,
            "Subcategory": "All",
            "Count within radius": int(count_val) if pd.notna(count_val) else 0,
            "Nearest distance (m)": round(float(nearest_val), 1) if pd.notna(nearest_val) else None,
            "Subscore (0-1)": round(float(subscore_val), 3) if pd.notna(subscore_val) else None,
        }
    )

metric_breakdown_df = pd.DataFrame(metric_rows)
if not metric_breakdown_df.empty:
    st.dataframe(metric_breakdown_df.fillna("Not available"), use_container_width=True)
else:
    st.info("No metric breakdown available for this office.")

st.markdown("### Subcategory breakdown")
if office_pois.empty:
    st.info("No amenity POIs available for this office at the selected radius.")
else:
    subcategory_df = (
        office_pois.groupby(["bucket", "tag_key", "tag_value"], dropna=False, as_index=False)
        .agg(
            count=("osm_id", "count"),
            nearest_distance_m=("distance_m", "min"),
            avg_distance_m=("distance_m", "mean"),
        )
        .rename(
            columns={
                "bucket": "Metric",
                "tag_key": "Tag key",
                "tag_value": "Tag value",
                "count": "POI count",
                "nearest_distance_m": "Nearest distance (m)",
                "avg_distance_m": "Average distance (m)",
            }
        )
        .sort_values(["Metric", "POI count", "Nearest distance (m)"], ascending=[True, False, True])
    )
    subcategory_df["Tag key"] = subcategory_df["Tag key"].fillna("Unknown")
    subcategory_df["Tag value"] = subcategory_df["Tag value"].fillna("Unknown")
    subcategory_df["Nearest distance (m)"] = pd.to_numeric(
        subcategory_df["Nearest distance (m)"], errors="coerce"
    ).round(1)
    subcategory_df["Average distance (m)"] = pd.to_numeric(
        subcategory_df["Average distance (m)"], errors="coerce"
    ).round(1)
    st.dataframe(subcategory_df, use_container_width=True)

if st.button("Back to Overview"):
    navigate_to(st.session_state, route="overview", standalone_page_path="pages/3_Overview.py")
