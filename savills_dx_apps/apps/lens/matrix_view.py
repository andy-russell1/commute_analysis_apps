from __future__ import annotations

import streamlit as st

from apps.lens.common import safe_set_page_config
from apps.lens.core import model, visuals


def render_page() -> None:
    safe_set_page_config(page_title="Data Matrix", layout="wide")

    model.render_page_header("Data Matrix")
    context = model.render_sidebar()
    model.ensure_context_ready(context)
    model.ensure_data_validation(context, prefix="Input data has blocking errors.")
    bundle = model.ensure_results_bundle(model.get_results_bundle(context))

    parsed = context["parsed"]
    micro_scores = bundle["micro_scores"].copy()

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

    view_labels = ["Computed Ranks", "Score Index (0-100)", "Raw (units vary)"]
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

    def render_view(view_label: str) -> None:
        metric_col = {
            "Computed Ranks": "rank",
            "Score Index (0-100)": "score_index",
            "Raw (units vary)": "raw_value",
        }[view_label]

        matrix = filtered.pivot(index="micro_label", columns="city", values=metric_col).sort_index()
        reverse_scale = metric_col == "rank"
        st.plotly_chart(
            visuals.data_matrix_heatmap(
                matrix,
                title=f"{view_label} Heatmap",
                value_label=view_label,
                reverse_scale=reverse_scale,
            ),
            use_container_width=True,
        )

        matrix_preview = model.build_matrix_preview_table(filtered, metric_col, value_label="Metric")
        city_display_columns = [model.to_proper_case_label(city) for city in sorted(filtered["city"].dropna().unique().tolist())]
        if metric_col == "raw_value":
            reverse_by_metric = (
                filtered[["micro_label", "direction"]]
                .drop_duplicates("micro_label")
                .set_index("micro_label")["direction"]
                .map(lambda value: str(value).lower() == "lower")
            )
            reverse_rows = matrix_preview["Metric"].map(reverse_by_metric).fillna(False)
            st.caption("Raw values are shown in source units. Shading is row-aware so stronger positions stand out by criterion.")
            st.dataframe(
                model.style_relative_value_table(
                    model.format_table_for_display(matrix_preview),
                    value_columns=city_display_columns,
                    reverse_rows=reverse_rows,
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption(
                "Computed ranks and scores are generated in Python from the uploaded raw values and current weighting settings."
            )
            st.dataframe(
                model.style_relative_value_table(
                    model.format_table_for_display(matrix_preview, decimals=local_decimals),
                    value_columns=city_display_columns,
                    decimals=local_decimals,
                    reverse_rows={idx: metric_col == "rank" for idx in matrix_preview.index},
                ),
                use_container_width=True,
                hide_index=True,
            )

        if not parsed["rank_reference"].empty:
            with st.expander("Uploaded reference rank rows (reference input only)", expanded=False):
                rank_reference = parsed["rank_reference"].copy()
                rank_reference = rank_reference[
                    rank_reference["macro"].isin(selected_macros) & rank_reference["major"].isin(selected_majors)
                ]
                if rank_reference.empty:
                    st.caption("No uploaded reference rank rows match the current filters.")
                else:
                    st.caption("These uploaded rank rows are retained for audit only and are not used as the live scoring source.")
                    reference_preview = model.build_reference_rank_audit_table(rank_reference)
                    st.dataframe(
                        model.style_relative_value_table(
                            model.format_table_for_display(reference_preview, decimals=0),
                            value_columns=[model.to_proper_case_label(city) for city in parsed["city_columns"]],
                            decimals=0,
                            reverse_rows={idx: True for idx in reference_preview.index},
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

    for label, tab in zip(ordered_views, tabs, strict=False):
        with tab:
            render_view(label)
