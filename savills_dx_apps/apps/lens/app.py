from __future__ import annotations

import streamlit as st

from apps.lens.common import render_nav_link, safe_set_page_config
from apps.lens.core import model
from apps.lens.core.constants import MODE_ADVANCED

safe_set_page_config(page_title="LENS Location Scoring", page_icon="L", layout="wide")

model.render_page_header(
    "LENS Location Evaluation",
    caption="Decision-first location scoring for client-ready recommendations.",
)

context = model.render_sidebar()
model.ensure_context_ready(context, upload_message="Upload a workbook from the sidebar to start.")
model.ensure_data_validation(context, prefix="Data validation failed. Resolve these issues before scoring:")

bundle = model.get_results_bundle(context)
if bundle is None:
    st.stop()
if "weight_validation" in bundle and not bundle["weight_validation"].is_valid:
    st.warning("Weights currently fail validation. Go to Weights and Scoring to continue.")
    render_nav_link(
        "Open Weights and Scoring",
        route="weights",
        standalone_page_path="pages/1_Weights_and_Scoring.py",
        key="lens_home_fix_weights_1",
    )
    st.stop()
if "direction_validation" in bundle and not bundle["direction_validation"].is_valid:
    st.warning("Direction overrides currently fail validation. Go to Weights and Scoring to continue.")
    render_nav_link(
        "Open Weights and Scoring",
        route="weights",
        standalone_page_path="pages/1_Weights_and_Scoring.py",
        key="lens_home_fix_weights_2",
    )
    st.stop()

ranks = model.build_city_ranks(bundle["overall_scores"])
top3 = ranks.head(3).copy()

col1, col2, col3 = st.columns(3)
col1.metric("Recommended Cities", int(top3.shape[0]))
col2.metric("Cities Evaluated", int(ranks.shape[0]))
col3.metric("Mode", context["mode"])

st.subheader("Top Recommendations")
if context["mode"] == MODE_ADVANCED:
    top_table = top3[["overall_rank", "city", "overall_index", "overall_score", "overall_tier"]].rename(
        columns={
            "overall_rank": "rank",
            "city": "city",
            "overall_index": "overall_index",
            "overall_score": "audit_score",
            "overall_tier": "tier",
        }
    )
    st.dataframe(model.format_table_for_display(top_table, decimals=1), use_container_width=True, hide_index=True)
else:
    top_table = top3[["overall_rank", "city", "overall_index", "distance_to_leader", "overall_tier"]].rename(
        columns={
            "overall_rank": "rank",
            "city": "city",
            "overall_index": "overall_index",
            "distance_to_leader": "distance_to_leader",
            "overall_tier": "tier",
        }
    )
    st.dataframe(model.format_table_for_display(top_table, decimals=1), use_container_width=True, hide_index=True)

best_city = str(top3.iloc[0]["city"])
best_drill = model.build_city_drilldown(bundle, best_city, top_n=3)
strongest_macro = best_drill["summary"]["strongest_macro"]
tradeoff_macro = best_drill["summary"]["tradeoff_macro"]

st.subheader("Quick Narrative")
st.write(
    f"**{best_city}** currently ranks #1 overall. Its strongest performance is in **{strongest_macro}**, "
    f"while **{tradeoff_macro}** is the main tradeoff area to monitor."
)

st.subheader("Next Steps")
st.caption("These are clickable links. Select one to open that page.")
render_nav_link(
    "Open Results Dashboard",
    route="results",
    standalone_page_path="pages/2_Results_Dashboard.py",
    key="lens_home_nav_results",
)
render_nav_link(
    "Open Weights and Scoring",
    route="weights",
    standalone_page_path="pages/1_Weights_and_Scoring.py",
    key="lens_home_nav_weights",
)
render_nav_link(
    "Open Export",
    route="export",
    standalone_page_path="pages/4_Export.py",
    key="lens_home_nav_export",
)
render_nav_link(
    "Open Methodology and Glossary",
    route="methodology",
    standalone_page_path="pages/5_Methodology_and_Glossary.py",
    key="lens_home_nav_methodology",
)

if context["mode"] == "Advanced":
    parsed = context["parsed"]
    with st.expander("Technical Preview (Advanced)", expanded=False):
        criteria_preview = parsed["criteria"][
            ["macro", "major", "micro", "macro_weight_template", "major_weight_template", "minor_weight_template"]
        ].copy()
        st.caption("Criteria Structure Preview")
        st.dataframe(model.format_table_for_display(criteria_preview, decimals=3), use_container_width=True)
        st.caption("Raw Data Preview")
        raw_preview = parsed["raw_data"][["macro", "major", "micro", "source", *parsed["city_columns"]]].head(10)
        st.dataframe(model.format_table_for_display(raw_preview, decimals=3), use_container_width=True)

if context["data_validation"].warnings:
    st.info(f"Workbook loaded with {len(context['data_validation'].warnings)} warning(s).")
    if context["mode"] == "Advanced":
        with st.expander("Data Quality Details", expanded=False):
            for idx, warning in enumerate(context["data_validation"].warnings, start=1):
                st.write(f"{idx}. {warning}")
