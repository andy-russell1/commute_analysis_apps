from __future__ import annotations

import streamlit as st

from apps.lens.common import render_nav_link, safe_set_page_config
from apps.lens.core import model
from shared.ui.kpi import render_kpi_strip


def render_page() -> None:
    safe_set_page_config(page_title="LENS Location Scoring", page_icon="L", layout="wide")

    model.render_page_header(
        "LENS Location Evaluation",
        caption="Decision-first location scoring for client-ready recommendations.",
    )
    model.render_dashboard_chrome()

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
    top_table = model.build_home_recommendations_table(bundle, top_n=3)

    render_kpi_strip(
        [
            ("Recommended Cities", int(top3.shape[0])),
            ("Cities Evaluated", int(ranks.shape[0])),
            ("Mode", context["mode"]),
        ],
        columns=3,
    )

    st.subheader("Top Recommendations")
    st.dataframe(model.format_table_for_display(top_table, decimals=1), use_container_width=True, hide_index=True)

    best_city = str(top3.iloc[0]["city"])
    best_drill = model.build_city_drilldown(bundle, best_city, top_n=3)
    strongest_macro = best_drill["summary"]["strongest_macro"]
    tradeoff_macro = best_drill["summary"]["tradeoff_macro"]

    model.render_context_sentence(
        "Top-line answer",
        f"{best_city} currently leads the portfolio, with {strongest_macro} as the strongest macro and {tradeoff_macro} as the main trade-off area.",
    )

    st.subheader("Quick Narrative")
    st.write(
        f"**{best_city}** currently ranks #1 overall. Its strongest performance is in **{strongest_macro}**, "
        f"while **{tradeoff_macro}** is the main trade-off area to monitor."
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
        "Open Benchmarking",
        route="benchmarking",
        standalone_page_path="pages/3_Benchmarking.py",
        key="lens_home_nav_benchmarking",
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
        standalone_page_path="pages/5_Export.py",
        key="lens_home_nav_export",
    )
    render_nav_link(
        "Open Methodology and Glossary",
        route="methodology",
        standalone_page_path="pages/6_Methodology_and_Glossary.py",
        key="lens_home_nav_methodology",
    )

    parsed = context["parsed"]
    with st.expander("Technical Preview", expanded=False):
        st.caption("A concise audit view of the uploaded structure, live raw inputs, and any uploaded reference ranks.")

        criteria_preview = parsed["criteria"][
            ["macro", "major", "micro", "macro_weight_template", "major_weight_template", "minor_weight_template"]
        ].copy()
        st.caption("Criteria structure")
        st.dataframe(model.format_table_for_display(criteria_preview, decimals=3), use_container_width=True, hide_index=True)

        st.caption("Raw data preview")
        raw_preview = parsed["raw_data"][["macro", "major", "micro", "source", *parsed["city_columns"]]].head(10).copy()
        raw_preview_display = model.format_table_for_display(raw_preview, decimals=3)
        raw_preview_reverse = raw_preview["source"].fillna("").str.contains("lower is better", case=False, regex=False)
        st.dataframe(
            model.style_relative_value_table(
                raw_preview_display,
                value_columns=[model.to_proper_case_label(city) for city in parsed["city_columns"]],
                decimals=3,
                reverse_rows=raw_preview_reverse,
            ),
            use_container_width=True,
            hide_index=True,
        )

        if not parsed["rank_reference"].empty:
            st.caption("Uploaded reference rank rows")
            reference_preview = model.build_reference_rank_audit_table(parsed["rank_reference"])
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

    if context["data_validation"].warnings:
        st.info(f"Workbook loaded with {len(context['data_validation'].warnings)} warning(s).")
        if context["mode"] == "Advanced":
            with st.expander("Data Quality Details", expanded=False):
                for idx, warning in enumerate(context["data_validation"].warnings, start=1):
                    st.write(f"{idx}. {warning}")
