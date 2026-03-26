from __future__ import annotations

import streamlit as st

from apps.lens.common import safe_set_page_config
from apps.lens.core import model, visuals


def render_page() -> None:
    safe_set_page_config(page_title="Results Dashboard", layout="wide")

    model.render_page_header("Results Dashboard")
    model.render_dashboard_chrome()
    context = model.render_sidebar()
    model.ensure_context_ready(context)
    model.ensure_data_validation(context, prefix="Input data has blocking errors.")

    bundle = model.ensure_results_bundle(model.get_results_bundle(context))
    macro_order = ["Talent", "Operating Environment", "Risk", "Cost"]
    macro_options = sorted(bundle["macro_scores"]["macro"].dropna().unique().tolist())

    st.subheader("Overall Weighted Scoring")
    overall_fig = visuals.overall_stacked_bar(
        macro_scores=bundle["macro_scores"],
        overall_scores=bundle["overall_scores"],
        macro_order=macro_order,
    )
    st.plotly_chart(overall_fig, use_container_width=True)

    st.subheader("Macro Drilldown")
    macro_focus_col, _ = st.columns([1, 2.4])
    with macro_focus_col:
        selected_macro = st.selectbox(
            "Macro focus",
            options=macro_options,
            index=0,
            key="lens_results_macro_focus",
        )
    drilldown_fig = visuals.macro_drilldown_bar(
        bundle["major_scores"],
        bundle["macro_scores"],
        selected_macro=selected_macro,
    )
    st.plotly_chart(drilldown_fig, use_container_width=True)

    st.subheader("Cost per Capability")
    if not context["capability_macros"]:
        st.warning("Select at least one macro for capability in the sidebar.")
    else:
        size_options = [
            ("Population", "population"),
            ("Overall Index (0-100)", "overall_index"),
        ]
        size_label = st.selectbox(
            "Bubble size",
            options=[label for label, _ in size_options],
            index=0,
            key="lens_results_bubble_size",
        )
        selected_size_col = dict(size_options).get(size_label, "population")
        bubble_fig = visuals.capability_cost_bubble(
            bundle["capability_cost"],
            size_col=selected_size_col,
            size_label=size_label,
        )
        st.plotly_chart(bubble_fig, use_container_width=True)
        st.caption("Cost per Capability stays fixed on the indexed 0-100 axes.")
