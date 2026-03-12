from __future__ import annotations

import streamlit as st

from apps.lens.common import safe_set_page_config

from apps.lens.core import model, visuals
from apps.lens.core.constants import MODE_ADVANCED

safe_set_page_config(page_title="Results Dashboard", layout="wide")

model.render_page_header("Results Dashboard")
context = model.render_sidebar()
model.ensure_context_ready(context)
model.ensure_data_validation(context, prefix="Input data has blocking errors.")

bundle = model.ensure_results_bundle(model.get_results_bundle(context))
macro_order = ["Talent", "Operating Environment", "Risk", "Cost"]

st.subheader("Overall Weighted Scoring")
overall_fig = visuals.overall_stacked_bar(
    macro_scores=bundle["macro_scores"],
    overall_scores=bundle["overall_scores"],
    macro_order=macro_order,
)
st.plotly_chart(overall_fig, use_container_width=True)

st.subheader("Macro Drilldown")
macro_options = sorted(bundle["macro_scores"]["macro"].dropna().unique().tolist())
selected_macro = st.selectbox("Select macro", options=macro_options, index=0)
drilldown_fig = visuals.macro_drilldown_bar(bundle["major_scores"], selected_macro=selected_macro)
st.plotly_chart(drilldown_fig, use_container_width=True)

st.subheader("Cost vs Capability")
if not context["capability_macros"]:
    st.warning("Select at least one macro for capability in the sidebar.")
else:
    bubble_fig = visuals.capability_cost_bubble(bundle["capability_cost"])
    st.plotly_chart(bubble_fig, use_container_width=True)

st.subheader("City Drilldown")
ranks = model.build_city_ranks(bundle["overall_scores"])
default_city = str(ranks.iloc[0]["city"])
selected_city = st.selectbox(
    "Choose city",
    options=ranks["city"].tolist(),
    index=ranks["city"].tolist().index(default_city),
)

top_n = 5 if context["mode"] != MODE_ADVANCED else 10
drill = model.build_city_drilldown(bundle, selected_city, top_n=top_n)
summary = drill["summary"]

metric_col1, metric_col2, metric_col3 = st.columns(3)
if context["mode"] == MODE_ADVANCED:
    metric_col1.metric("Overall Index (0-100)", f"{summary['overall_index']:.1f}")
    metric_col2.metric("Audit Score", f"{summary['overall_score']:.3f}")
    metric_col3.metric("Overall Rank", f"{summary['overall_rank']} of {len(ranks)}")
else:
    metric_col1.metric("Overall Index (0-100)", f"{summary['overall_index']:.1f}")
    metric_col2.metric("Overall Rank", f"{summary['overall_rank']} of {len(ranks)}")
    metric_col3.metric("Distance to Leader", f"{summary['distance_to_leader']:.1f}")

st.caption(
    f"Tradeoff focus for {selected_city}: {summary['tradeoff_macro']}. Tier: {summary['overall_tier']}. "
    "Strengths/weaknesses compare this city versus cross-city average contribution by criterion."
)

st.plotly_chart(visuals.macro_contribution_bar(drill["macro_summary"]), use_container_width=True)

strengths = drill["strengths"].copy()
weaknesses = drill["weaknesses"].copy()

left, right = st.columns(2)
with left:
    st.markdown("**Top Strengths**")
    if context["mode"] == MODE_ADVANCED:
        table = strengths[
            ["macro", "major", "micro_display", "direction", "effective_micro_weight", "score", "delta"]
        ].rename(columns={"micro_display": "micro"})
        st.dataframe(
            model.format_table_for_display(table, decimals=3),
            use_container_width=True,
            hide_index=True,
        )
    else:
        for _, row in strengths.head(3).iterrows():
            direction = "Higher is better" if row["direction"] == "higher" else "Lower is better"
            st.write(
                f"- {row['micro_display']} ({row['macro']} > {row['major']}): "
                f"weight {row['effective_micro_weight']:.3f}, score {row['score']:.3f}, {direction}"
            )

with right:
    st.markdown("**Top Weaknesses**")
    if context["mode"] == MODE_ADVANCED:
        table = weaknesses[
            ["macro", "major", "micro_display", "direction", "effective_micro_weight", "score", "delta"]
        ].rename(columns={"micro_display": "micro"})
        st.dataframe(
            model.format_table_for_display(table, decimals=3),
            use_container_width=True,
            hide_index=True,
        )
    else:
        for _, row in weaknesses.head(3).iterrows():
            direction = "Higher is better" if row["direction"] == "higher" else "Lower is better"
            st.write(
                f"- {row['micro_display']} ({row['macro']} > {row['major']}): "
                f"weight {row['effective_micro_weight']:.3f}, score {row['score']:.3f}, {direction}"
            )

if context["mode"] == MODE_ADVANCED:
    st.subheader("Hierarchy Breakdown")
    st.dataframe(model.format_table_for_display(drill["compact_breakdown"], decimals=3), use_container_width=True, hide_index=True)

    for macro in sorted(drill["hierarchy_breakdown"]["macro"].dropna().unique().tolist()):
        with st.expander(f"{macro}", expanded=False):
            macro_majors = drill["major_summary"][drill["major_summary"]["macro"] == macro].copy().sort_values("major")
            for _, major_row in macro_majors.iterrows():
                major_name = str(major_row["major"])
                st.markdown(f"**{major_name}** - score {major_row['major_score']:.3f}, weight {major_row['major_weight']:.3f}")
                micro_rows = drill["micro_details"][
                    (drill["micro_details"]["macro"] == macro) & (drill["micro_details"]["major"] == major_name)
                ].copy()
                micro_rows = micro_rows.sort_values("contribution", ascending=False)
                st.dataframe(
                    model.format_table_for_display(
                        micro_rows[["micro_display", "direction", "score", "effective_micro_weight", "contribution"]].rename(
                            columns={"micro_display": "micro"}
                        ),
                        decimals=3,
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

    with st.expander("Show full driver table (advanced)", expanded=False):
        st.dataframe(
            model.format_table_for_display(
                drill["micro_details"][
                    ["macro", "major", "micro_display", "direction", "score", "effective_micro_weight", "contribution", "delta"]
                ]
                .rename(columns={"micro_display": "micro"})
                .sort_values("contribution", ascending=False),
                decimals=3,
            ),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Show audit score table (advanced)", expanded=False):
        st.dataframe(
            model.format_table_for_display(bundle["city_scores"].sort_values("overall_score", ascending=False), decimals=1),
            use_container_width=True,
            hide_index=True,
        )

