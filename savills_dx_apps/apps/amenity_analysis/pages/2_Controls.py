from __future__ import annotations

import pandas as pd
import streamlit as st

from apps.amenity_analysis.common import (
    ANALYSIS_MESSAGES_KEY,
    CACHE_STATS_KEY,
    CONTROLS_VIEW_KEY,
    LAST_RUN_CONFIG_KEY,
    PRIMARY_RADIUS_KEY,
    RESULTS_BY_RADIUS_KEY,
    RESULTS_POI_KEY,
    RESULTS_SUMMARY_KEY,
    SELECTED_METRICS_KEY,
    SELECTED_RADII_KEY,
    SITES_DF_KEY,
    WEIGHTS_NORM_KEY,
    WEIGHTS_RAW_KEY,
    get_metric_options,
    init_amenity_state,
    navigate_to,
    run_multi_radius_analysis,
    safe_set_page_config,
)
from shared.data.naptan_loader import load_naptan_stops
from shared.ui.page_header import render_page_header
from shared.scoring.amenity_index import normalise_weights


def _render_scoring_view() -> None:
    st.subheader("Scoring system")
    st.write("This is how the KPI is calculated in plain English.")
    st.markdown("1. For each selected metric, each office gets a raw value.")
    st.markdown("2. Raw values are converted to a 0 to 1 score so metrics are comparable.")
    st.markdown("3. The metric scores are combined using your weights.")
    st.markdown("4. The final KPI is shown on a 0 to 100 scale.")

    st.markdown("### How each metric is interpreted")
    st.markdown("- Amenity metrics (Lunch & coffee, Green, Fitness): higher count within radius is better.")
    st.markdown("- Public transport: shorter distance to nearest stop is better.")
    st.markdown("- If all offices are identical on a metric, each office gets a neutral 0.5 score for that metric.")

    st.markdown("### Weights")
    st.markdown("- Weights are always normalised to 100%.")
    st.markdown("- If you select one metric only, it automatically gets 100%.")
    st.markdown("- If input weights do not sum to 100, the app rescales them proportionally.")

    st.markdown("### Worked example")
    example_df = pd.DataFrame(
        {
            "Office": ["A", "B"],
            "Lunch count (400m)": [4, 10],
            "Lunch score (0-1)": [0.0, 1.0],
            "Transport distance (m)": [450, 250],
            "Transport score (0-1)": [0.0, 1.0],
            "Weight Lunch": [0.6, 0.6],
            "Weight Transport": [0.4, 0.4],
            "Final KPI": [0.0 * 0.6 + 0.0 * 0.4, 1.0 * 0.6 + 1.0 * 0.4],
        }
    )
    example_df["Final KPI"] = (example_df["Final KPI"] * 100).round(1)
    st.dataframe(example_df, use_container_width=True)


safe_set_page_config(page_title="Amenity Analysis - Controls", page_icon="🎛️", layout="wide")
init_amenity_state(st.session_state)

render_page_header("Controls")

sites_df = st.session_state[SITES_DF_KEY]
if sites_df.empty:
    st.warning("Load valid sites first on Setup page.")
    if st.button("Open Setup"):
        navigate_to(st.session_state, route="setup", standalone_page_path="pages/1_Setup.py")
    st.stop()

view_options = ["Analysis controls", "Scoring system"]
requested_view = st.session_state.get(CONTROLS_VIEW_KEY, "Analysis controls")
if requested_view not in view_options:
    requested_view = "Analysis controls"

selected_view = st.radio(
    "View",
    options=view_options,
    index=view_options.index(requested_view),
    horizontal=True,
)
st.session_state[CONTROLS_VIEW_KEY] = selected_view

if selected_view == "Scoring system":
    _render_scoring_view()
    if st.button("Back to analysis controls"):
        st.session_state[CONTROLS_VIEW_KEY] = "Analysis controls"
        st.rerun()
    st.stop()

metric_options = get_metric_options()
previous_metrics = st.session_state.get(SELECTED_METRICS_KEY, metric_options)
selected_metrics: list[str] = []

st.markdown("### Metrics to include")
for metric in metric_options:
    if st.checkbox(metric, value=metric in previous_metrics, key=f"metric_{metric}"):
        selected_metrics.append(metric)

if "Public transport" in selected_metrics:
    _, naptan_message = load_naptan_stops()
    if naptan_message:
        st.warning(
            "Public transport metric is selected, but NaPTAN data is unavailable. "
            "Transport scores will show as not available until a NaPTAN file is added."
        )

st.markdown("### Radius")
configured_radii = st.session_state.get(SELECTED_RADII_KEY, [1000])
selected_radius = int(configured_radii[0]) if configured_radii else 1000
st.info(f"Using radius from Setup: {selected_radius}m")
if st.button("Change radius in Setup"):
    navigate_to(st.session_state, route="setup", standalone_page_path="pages/1_Setup.py")

st.markdown("### Weights")
st.caption("Applied weights are always normalised to 100%.")

raw_weights: dict[str, float] = {}
prior_raw = st.session_state.get(WEIGHTS_RAW_KEY, {})

if not selected_metrics:
    st.info("Select at least one metric.")
elif len(selected_metrics) == 1:
    only_metric = selected_metrics[0]
    raw_weights[only_metric] = 100.0
    st.dataframe(
        [{"metric": only_metric, "input_weight": 100.0, "applied_weight_%": 100.0}],
        use_container_width=True,
    )
else:
    default = max(int(round(100 / len(selected_metrics))), 1)
    for metric in selected_metrics:
        raw_weights[metric] = float(
            st.slider(
                f"{metric} weight",
                min_value=0,
                max_value=100,
                value=int(round(prior_raw.get(metric, default))),
                step=1,
                key=f"weight_slider_{metric}",
            )
        )

    applied = normalise_weights(raw_weights, selected_metrics)
    st.dataframe(
        [
            {
                "metric": metric,
                "input_weight": raw_weights.get(metric, 0.0),
                "applied_weight_%": round(applied.get(metric, 0.0) * 100.0, 1),
            }
            for metric in selected_metrics
        ],
        use_container_width=True,
    )
    st.caption(
        f"Input total: {sum(raw_weights.values()):.1f}. Applied total: {sum(applied.values()) * 100.0:.1f}%"
    )

if st.button("Run analysis", type="primary"):
    if not selected_metrics:
        st.error("Select at least one metric.")
    else:
        with st.spinner("Running analysis..."):
            run_result = run_multi_radius_analysis(
                sites_df=sites_df,
                selected_metrics=selected_metrics,
                selected_radii_m=[int(selected_radius)],
                raw_weights=raw_weights,
            )

        results_by_radius = {
            radius: {
                "summary_df": radius_result.summary_df,
                "poi_df": radius_result.poi_df,
                "messages": radius_result.messages,
                "cache_stats": radius_result.cache_stats,
            }
            for radius, radius_result in run_result.by_radius.items()
        }

        primary_radius = min(results_by_radius.keys()) if results_by_radius else int(selected_radius)
        primary_result = results_by_radius.get(primary_radius, {})

        st.session_state[SELECTED_METRICS_KEY] = selected_metrics
        st.session_state[SELECTED_RADII_KEY] = [int(selected_radius)]
        st.session_state[PRIMARY_RADIUS_KEY] = int(primary_radius)
        st.session_state[WEIGHTS_RAW_KEY] = {metric: raw_weights.get(metric, 0.0) for metric in selected_metrics}
        st.session_state[WEIGHTS_NORM_KEY] = run_result.weights_normalised
        st.session_state[RESULTS_BY_RADIUS_KEY] = results_by_radius
        st.session_state[RESULTS_SUMMARY_KEY] = primary_result.get("summary_df", st.session_state[RESULTS_SUMMARY_KEY])
        st.session_state[RESULTS_POI_KEY] = primary_result.get("poi_df", st.session_state[RESULTS_POI_KEY])
        st.session_state[ANALYSIS_MESSAGES_KEY] = run_result.messages
        st.session_state[CACHE_STATS_KEY] = run_result.cache_stats
        st.session_state[LAST_RUN_CONFIG_KEY] = {
            "metrics": selected_metrics,
            "radii_m": [int(selected_radius)],
            "primary_radius_m": int(primary_radius),
        }
        st.session_state[CONTROLS_VIEW_KEY] = "Analysis controls"

        st.success("Analysis complete.")
        navigate_to(st.session_state, route="overview", standalone_page_path="pages/3_Overview.py")
