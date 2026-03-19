from __future__ import annotations

import pandas as pd
import streamlit as st

from apps.lens.common import safe_set_page_config
from apps.lens.core import model, visuals
from apps.lens.core.constants import MODE_ADVANCED
from shared.ui.kpi import render_kpi_strip


def _format_delta(value: float) -> str:
    if pd.isna(value):
        return "-"
    return f"{value:+.1f}"


def _render_flat_callout(title: str, body: str) -> None:
    st.markdown(
        "<div class='lens-surface' style='margin:0.15rem 0 0.9rem 0;padding:0.35rem 0.9rem 0.35rem 0.95rem;"
        "border-left:4px solid #FFDF00;'>"
        f"<div style='font-size:0.98rem;font-weight:600;color:inherit;line-height:1.15;margin-bottom:0.28rem;'>{title}</div>"
        f"<div style='font-size:1.02rem;color:inherit;line-height:1.35;'>{body}</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def _render_flat_insight(title: str, items: list[str], empty_message: str, caption: str | None = None) -> None:
    bullet_items = items or [empty_message]
    caption_html = (
        f"<div style='font-size:0.85rem;color:inherit;opacity:0.72;line-height:1.3;margin-bottom:0.3rem;'>{caption}</div>"
        if caption
        else ""
    )
    bullets = "".join(
        f"<li style='margin:0.18rem 0;'>{item}</li>"
        for item in bullet_items
    )
    st.markdown(
        "<div class='lens-surface' style='margin:0 0 0.8rem 0;padding:0.4rem 0.9rem 0.45rem 0.95rem;"
        "border-left:4px solid #FFDF00;'>"
        f"<div style='font-size:0.98rem;font-weight:600;color:inherit;line-height:1.15;margin-bottom:0.28rem;'>{title}</div>"
        f"{caption_html}"
        f"<ul style='margin:0 0 0 1rem;padding:0;font-size:0.98rem;color:inherit;line-height:1.35;'>{bullets}</ul>"
        "</div>",
        unsafe_allow_html=True,
    )


def _render_benchmark_profile_key(profile_df: pd.DataFrame) -> None:
    legend_items = visuals.profile_legend_items(profile_df)
    series_items = legend_items.get("series", [])
    group_items = legend_items.get("groups", [])
    if not series_items and not group_items:
        return

    def _series_swatch(item: dict[str, str]) -> str:
        colour = str(item.get("colour", "#D5DDE5"))
        style = str(item.get("style", "primary"))
        line_style = "dashed" if style == "secondary" else "solid"
        fill = f"linear-gradient(180deg, rgba(0,0,0,0) 0%, {colour}33 100%)"
        return (
            "<span style='display:inline-flex;align-items:center;gap:0.5rem;'>"
            "<span style='position:relative;display:inline-block;width:2rem;height:0.95rem;'>"
            f"<span style='position:absolute;inset:0;border-radius:999px;background:{fill};'></span>"
            f"<span style='position:absolute;left:0;right:0;top:50%;transform:translateY(-50%);"
            f"border-top:3px {line_style} {colour};'></span>"
            f"<span style='position:absolute;left:50%;top:50%;width:0.5rem;height:0.5rem;border-radius:50%;"
            f"transform:translate(-50%, -50%);background:{colour};border:1px solid #262A43;'></span>"
            "</span>"
            f"<span>{item.get('label', '')}</span>"
            "</span>"
        )

    def _group_swatch(item: dict[str, str]) -> str:
        colour = str(item.get("colour", "#D5DDE5"))
        return (
            "<span style='display:inline-flex;align-items:center;gap:0.45rem;'>"
            f"<span style='display:inline-block;width:0.95rem;height:0.95rem;border-radius:0.22rem;"
            f"background:{colour}2E;border:1px solid {colour}66;'></span>"
            f"<span>{item.get('label', '')}</span>"
            "</span>"
        )

    series_html = "".join(_series_swatch(item) for item in series_items)
    groups_html = "".join(_group_swatch(item) for item in group_items)
    st.markdown(
        "<div style='margin:-0.55rem 0 0.15rem 0;padding:0 0.15rem 0 0;'>"
        "<div style='font-size:0.82rem;font-weight:600;letter-spacing:0.02em;opacity:0.82;margin-bottom:0.35rem;'>"
        "Benchmark key</div>"
        f"<div style='display:flex;flex-wrap:wrap;gap:0.9rem 1.3rem;font-size:0.9rem;line-height:1.25;'>{series_html}</div>"
        + (
            "<div style='font-size:0.82rem;font-weight:600;letter-spacing:0.02em;opacity:0.72;"
            "margin:0.55rem 0 0.35rem 0;'>Macro underlays</div>"
            f"<div style='display:flex;flex-wrap:wrap;gap:0.7rem 1.05rem;font-size:0.86rem;line-height:1.2;'>{groups_html}</div>"
            if groups_html
            else ""
        )
        + "</div>",
        unsafe_allow_html=True,
    )


def _render_city_strength_tables(drill: dict, scoring_basis: dict, scoring_method: str, mode: str) -> None:
    strengths = drill["strengths"].copy()
    weaknesses = drill["weaknesses"].copy()

    left, right = st.columns(2)
    with left:
        st.subheader("Top Strengths")
        if mode == MODE_ADVANCED:
            table = strengths[
                ["macro", "major", "micro_display", scoring_basis["column"], "delta"]
            ].rename(columns={"micro_display": "micro", scoring_basis["column"]: scoring_basis["label"]})
            st.dataframe(model.format_table_for_display(table, decimals=3), use_container_width=True, hide_index=True)
        else:
            for _, row in strengths.head(3).iterrows():
                st.write(
                    f"- {row['micro_display']} ({row['macro']} > {row['major']}): "
                    f"{model.format_driver_basis_value(row, scoring_method)}"
                )
    with right:
        st.subheader("Top Weaknesses")
        if mode == MODE_ADVANCED:
            table = weaknesses[
                ["macro", "major", "micro_display", scoring_basis["column"], "delta"]
            ].rename(columns={"micro_display": "micro", scoring_basis["column"]: scoring_basis["label"]})
            st.dataframe(model.format_table_for_display(table, decimals=3), use_container_width=True, hide_index=True)
        else:
            for _, row in weaknesses.head(3).iterrows():
                st.write(
                    f"- {row['micro_display']} ({row['macro']} > {row['major']}): "
                    f"{model.format_driver_basis_value(row, scoring_method)}"
                )


def _render_city_detail_support(drill: dict, bundle: dict, ranks: pd.DataFrame) -> None:
    with st.expander("Selected city hierarchy breakdown", expanded=False):
        st.dataframe(
            model.format_table_for_display(drill["compact_breakdown"], decimals=3),
            use_container_width=True,
            hide_index=True,
        )

        for macro in sorted(drill["hierarchy_breakdown"]["macro"].dropna().unique().tolist()):
            with st.expander(f"{macro}", expanded=False):
                macro_majors = drill["major_summary"][drill["major_summary"]["macro"] == macro].copy().sort_values("major")
                for _, major_row in macro_majors.iterrows():
                    major_name = str(major_row["major"])
                    st.markdown(
                        f"**{major_name}** - index {major_row.get('major_index', major_row['major_score']):.1f}, "
                        f"weight {major_row['major_weight']:.3f}"
                    )
                    micro_rows = drill["micro_details"][
                        (drill["micro_details"]["macro"] == macro) & (drill["micro_details"]["major"] == major_name)
                    ].copy()
                    micro_rows = micro_rows.sort_values("contribution", ascending=False)
                    st.dataframe(
                        model.format_table_for_display(
                            micro_rows[
                                ["micro_display", "direction", "score_index", "effective_micro_weight", "contribution"]
                            ].rename(columns={"micro_display": "micro"}),
                            decimals=3,
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

    with st.expander("Selected city support tables", expanded=False):
        st.dataframe(
            model.format_table_for_display(
                drill["micro_details"][
                    [
                        "macro",
                        "major",
                        "micro_display",
                        "direction",
                        "score_index",
                        "effective_micro_weight",
                        "contribution",
                        "delta",
                    ]
                ]
                .rename(columns={"micro_display": "micro"})
                .sort_values("contribution", ascending=False),
                decimals=3,
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.dataframe(
            model.format_table_for_display(
                model.build_home_recommendations_table(bundle, top_n=len(ranks)).sort_values("rank"),
                decimals=1,
            ),
            use_container_width=True,
            hide_index=True,
        )


def render_page() -> None:
    safe_set_page_config(page_title="Benchmarking", layout="wide")

    model.render_page_header(
        "Benchmarking",
        caption="Side-by-side benchmark comparisons on the current LENS scoring outputs.",
    )
    model.render_dashboard_chrome()
    context = model.render_sidebar()
    model.ensure_context_ready(context)
    model.ensure_data_validation(context, prefix="Input data has blocking errors.")

    bundle = model.ensure_results_bundle(model.get_results_bundle(context))
    scoring_basis = model.get_active_scoring_basis(context["scoring_method"])
    ranks = model.build_city_ranks(bundle["overall_scores"])
    city_options = ranks["city"].tolist()
    default_city = str(ranks.iloc[0]["city"])

    comparison_options = {
        "Benchmark city": "city",
        "Portfolio average": "average",
        "Best performer": "best",
    }
    default_comparison_label = "Benchmark city"
    active_comparison_label = st.session_state.get("lens_benchmark_comparison_mode", default_comparison_label)
    active_comparison_mode = comparison_options.get(active_comparison_label, comparison_options[default_comparison_label])
    controls = st.columns(4 if active_comparison_mode == "city" else 3)
    with controls[0]:
        comparison_label = st.selectbox(
            "Comparison",
            options=list(comparison_options.keys()),
            index=0,
            key="lens_benchmark_comparison_mode",
        )
    comparison_mode = comparison_options[comparison_label]
    with controls[1]:
        selected_city = st.selectbox(
            "Selected city",
            options=city_options,
            index=city_options.index(default_city),
            key="lens_benchmark_selected_city",
        )
    benchmark_options = [city for city in city_options if city != selected_city]
    if not benchmark_options:
        benchmark_options = city_options
    if comparison_mode == "city":
        with controls[2]:
            benchmark_city = st.selectbox(
                "Benchmark city",
                options=benchmark_options,
                index=0,
                key="lens_benchmark_city",
            )
        detail_column = controls[3]
    else:
        benchmark_city = None
        detail_column = controls[2]
    with detail_column:
        detail_level = st.selectbox(
            "Benchmark detail",
            options=["Macro", "Major", "Micro"],
            index=0,
            key="lens_benchmark_detail_level",
        )
    benchmark_city_value = benchmark_city if comparison_mode == "city" else None

    benchmark = model.build_benchmark_profile(
        bundle,
        selected_city,
        comparison_mode=comparison_mode,
        benchmark_city=benchmark_city_value,
        level=detail_level,
        scoring_method=context["scoring_method"],
    )
    profile_df = benchmark["profile"]
    benchmark_label = benchmark["benchmark_label"]
    top_n = 5 if context["mode"] != MODE_ADVANCED else 10
    drill = model.build_city_drilldown(bundle, selected_city, top_n=top_n)
    city_summary = drill["summary"]
    overview = model.build_benchmark_overview(
        bundle,
        selected_city,
        comparison_mode=comparison_mode,
        benchmark_city=benchmark_city_value,
    )
    summary = model.build_benchmark_profile_summary(profile_df, selected_city, benchmark_label, detail_level, top_n=4)

    st.subheader("Selected City Snapshot")
    render_kpi_strip(
        [
            ("Selected index", f"{city_summary['overall_index']:.1f}"),
            ("Capability index", f"{city_summary['capability_score']:.1f}"),
            ("Cost index", f"{city_summary['cost_score']:.1f}"),
            ("Rank", f"{city_summary['overall_rank']} of {len(ranks)}"),
        ]
    )

    st.markdown(
        "<h2 style='margin:1.1rem 0 0.15rem 0;font-size:2rem;font-weight:600;line-height:1.15;'>"
        "Benchmark Comparison</h2>",
        unsafe_allow_html=True,
    )

    if profile_df.empty:
        st.info("No benchmark data is available for the selected comparison.")
        return

    chart_col, summary_col = st.columns([1.9, 1.1])
    with chart_col:
        st.subheader("Benchmark Profile")
        st.plotly_chart(visuals.city_profile_radar(profile_df), use_container_width=True)
        _render_benchmark_profile_key(profile_df)
    with summary_col:
        _render_flat_insight(
            "Top areas ahead",
            summary["strengths"],
            "No material areas ahead on this benchmark selection.",
        )
        _render_flat_insight(
            "Top areas behind",
            summary["weaknesses"],
            "No material areas behind on this benchmark selection.",
        )
        _render_flat_insight(
            "Benchmark Summary",
            [
                f"{selected_city} is being benchmarked against {benchmark_label}.",
                f"Charts default to indexed 0-100 outputs.",
                f"Supporting tables follow the active {scoring_basis['narrative_label']} basis where available.",
            ],
            empty_message="No benchmark summary is available.",
        )

    comparison_table = summary["comparison_table"].copy()
    if not comparison_table.empty:
        st.subheader("Largest Benchmark Gaps")
        st.plotly_chart(visuals.benchmark_delta_bar(comparison_table), use_container_width=True)

        st.subheader("Supporting Comparison Table")
        display_table = comparison_table.copy()
        display_table["comparison_group"] = display_table["group_label"].where(
            display_table["group_label"] != display_table["item_label"],
            "",
        )
        basis_label = summary["basis_label"]
        table_columns = [
            "item_label",
            "comparison_group",
            "city_score",
            "benchmark_score",
            "delta_to_benchmark",
        ]
        rename_map = {
            "item_label": "Measure",
            "comparison_group": "Group",
            "city_score": f"{selected_city} Index",
            "benchmark_score": f"{benchmark_label} Index",
            "delta_to_benchmark": "Index Gap",
        }
        if comparison_table["city_basis"].notna().any() or comparison_table["benchmark_basis"].notna().any():
            table_columns.extend(["city_basis", "benchmark_basis", "delta_to_benchmark_basis"])
            rename_map.update(
                {
                    "city_basis": f"{selected_city} {basis_label}",
                    "benchmark_basis": f"{benchmark_label} {basis_label}",
                    "delta_to_benchmark_basis": f"{basis_label} Gap",
                }
            )
        st.dataframe(
            model.format_table_for_display(
                display_table[table_columns].rename(columns=rename_map),
                decimals=1,
                criterion_id_mode="label",
            ),
            use_container_width=True,
            hide_index=True,
        )

    if context["mode"] == MODE_ADVANCED:
        _render_city_detail_support(drill, bundle, ranks)

    _render_city_strength_tables(drill, scoring_basis, context["scoring_method"], context["mode"])
