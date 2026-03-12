from __future__ import annotations

import streamlit as st

from apps.lens.common import safe_set_page_config

from apps.lens.core import model, visuals
from apps.lens.core.constants import MODE_ADVANCED

safe_set_page_config(page_title="Data Matrix", layout="wide")

model.render_page_header("Data Matrix")
context = model.render_sidebar()
model.ensure_context_ready(context)
model.ensure_data_validation(context, prefix="Input data has blocking errors.")
bundle = model.ensure_results_bundle(model.get_results_bundle(context))

parsed = context["parsed"]
micro_scores = bundle["micro_scores"].copy()
mode = context["mode"]

macro_options = sorted(micro_scores["macro"].dropna().unique().tolist())
selected_macros = st.multiselect("Filter macro", options=macro_options, default=macro_options)

major_base = micro_scores[micro_scores["macro"].isin(selected_macros)]
major_options = sorted(major_base["major"].dropna().unique().tolist())
selected_majors = st.multiselect("Filter major", options=major_options, default=major_options)

filtered = micro_scores[
    micro_scores["macro"].isin(selected_macros) & micro_scores["major"].isin(selected_majors)
].copy()
filtered["micro_display"] = [
    model.get_micro_display_name(macro, major, micro)
    for macro, major, micro in zip(filtered["macro"], filtered["major"], filtered["micro"], strict=False)
]
filtered["micro_label"] = filtered["macro"] + " | " + filtered["major"] + " | " + filtered["micro_display"]

if filtered.empty:
    st.warning("No rows match current filters.")
    st.stop()

view_labels = ["Computed Ranks", "Scores", "Raw (units vary)"]
has_rank_data = filtered["rank"].notna().any()
st.session_state["lens_matrix_view"] = model.resolve_matrix_view_preference(
    st.session_state.get("lens_matrix_view"),
    has_rank_data=has_rank_data,
)

st.session_state["lens_matrix_view"] = st.radio(
    "Default open tab",
    options=view_labels,
    index=view_labels.index(st.session_state["lens_matrix_view"]),
    horizontal=True,
)

ordered_views = [st.session_state["lens_matrix_view"]] + [
    label for label in view_labels if label != st.session_state["lens_matrix_view"]
]
tabs = st.tabs(ordered_views)

local_decimals = 3


def _render_view(view_label: str) -> None:
    metric_col = {
        "Computed Ranks": "rank",
        "Scores": "score",
        "Raw (units vary)": "raw_value",
    }[view_label]

    matrix = filtered.pivot(index="micro_label", columns="city", values=metric_col).sort_index()
    st.plotly_chart(visuals.data_matrix_heatmap(matrix, title=f"{view_label} Heatmap"), use_container_width=True)

    if metric_col == "raw_value":
        st.dataframe(model.format_table_for_display(matrix), use_container_width=True)
    else:
        st.dataframe(model.format_table_for_display(matrix, decimals=local_decimals), use_container_width=True)

    if mode == MODE_ADVANCED and not parsed["rank_reference"].empty:
        with st.expander("Reference rank rows from input (advanced)", expanded=False):
            rank_reference = parsed["rank_reference"].copy()
            rank_reference = rank_reference[
                rank_reference["macro"].isin(selected_macros) & rank_reference["major"].isin(selected_majors)
            ]
            st.dataframe(model.format_table_for_display(rank_reference, decimals=local_decimals), use_container_width=True, hide_index=True)


for label, tab in zip(ordered_views, tabs, strict=False):
    with tab:
        _render_view(label)

