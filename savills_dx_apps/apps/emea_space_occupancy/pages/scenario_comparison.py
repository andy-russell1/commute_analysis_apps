from __future__ import annotations

import pandas as pd
import streamlit as st

from apps.emea_space_occupancy import engine, explainability, ui, visuals
from apps.emea_space_occupancy.common import build_live_bundle, build_scenario_library, prepare_page


def render_page() -> None:
    context = prepare_page(
        "Scenario Comparison",
        "Side-by-side comparison of baseline, seed, and saved scenarios using common portfolio metrics.",
    )
    if not context.get("ready"):
        ui.render_callout("Workbook not ready", "Load a valid workbook before comparing scenarios.", tone="warning")
        return

    live_bundle = build_live_bundle(context)
    baseline_bundle = engine.compute_live_scenario(
        context["clean_sheets"],
        engine.build_working_assumptions(context["clean_sheets"], "Base 2026"),
        context["filters"],
        workbook_name=context["workbook_name"],
        workbook_hash=context["workbook_hash"],
    )
    library = build_scenario_library(context, live_bundle)
    baseline_entry = {
        "snapshot_key": "live::baseline",
        "scenario_name": baseline_bundle["scenario_name"],
        "origin": "Live Baseline",
        "summary": baseline_bundle["summary"],
        "calculated_outputs": baseline_bundle["outputs"],
        "assumptions_used": baseline_bundle["assumptions"],
        "assumption_count": int(len(baseline_bundle["assumptions"].index)),
        "output_site_count": int(len(baseline_bundle["outputs"].index)),
        "timestamp": context.get("last_run"),
        "calculation_timestamp": context.get("last_run"),
        "workbook_name": context["workbook_name"],
        "workbook_hash": context["workbook_hash"],
        "filters": context["filters"],
        "basis_scenario_name": baseline_bundle["scenario_name"],
        "basis_origin": "Baseline",
        "notes": "",
    }
    if baseline_entry["scenario_name"] != live_bundle["scenario_name"]:
        library = [baseline_entry, *library]

    labels = [entry["display_label"] if "display_label" in entry else f"{entry['scenario_name']} [{entry['origin']}]" for entry in library]
    default_labels = labels[: min(3, len(labels))]
    auto_recommendation = engine.auto_recommended_scenario(library)

    ui.render_hero_panel(
        eyebrow="Scenario Review",
        title="Compare options side by side before choosing a preferred route",
        body=(
            "This page is designed for live comparison: keep the shortlist tight, understand the score and risk trade-offs, "
            "and isolate where actions change between scenarios."
        ),
        badges=[
            ("Available scenarios", str(len(labels))),
            ("Default selection", str(len(default_labels))),
            ("Auto recommendation", auto_recommendation["scenario_name"] if auto_recommendation else "Not available"),
            ("Filters", "Persisting from sidebar"),
        ],
        tone="accent",
    )

    with st.container(border=True):
        explainability.render_provenance_panel(
            context=context,
            scenario_name="Selected comparison set",
            scenario_origin="Mixed scenario comparison",
            basis_scenario_name=context["active_scenario"],
            basis_origin="Seed Scenario",
            calculation_timestamp=context.get("last_run"),
            assumption_count=len(context["working_assumptions"].index),
            notes=context.get("working_notes", ""),
        )
        explainability.render_scenario_formula_cards(expanded=False)
        explainability.render_score_component_glossary(expanded=False)

    with st.container(border=True):
        ui.render_section_heading(
            "Scenario selection",
            "Choose two or more scenarios to compare. Keeping this list focused makes the scorecards easier to read in a client session.",
            eyebrow="Control bar",
        )
        selected_labels = st.multiselect("Choose 2 or more scenarios", options=labels, default=default_labels)
    selected_entries = [entry for entry, label in zip(library, labels, strict=False) if label in selected_labels]
    if len(selected_entries) < 2:
        ui.render_empty_state(
            "Select at least two scenarios",
            "Add one more scenario to activate the scorecards, action-shift view, and recommended comparison narrative.",
        )
        return

    auto_selected = engine.auto_recommended_scenario(selected_entries)

    with st.container(border=True):
        ui.render_section_heading(
            "Selected scenario provenance",
            "Shows whether each line in the comparison is baseline, seed, live, or saved, alongside the workbook and timestamp context.",
            eyebrow="Comparison audit trail",
        )
        st.dataframe(explainability.scenario_provenance_table(selected_entries), use_container_width=True, hide_index=True)

    summary_table = engine.comparison_summary_table(selected_entries)
    best_score = summary_table.sort_values("Scenario Score", ascending=False).head(1)
    lowest_risk = summary_table.sort_values(["High Risk Sites", "Scenario Score"], ascending=[True, False]).head(1)
    smallest_gap = summary_table.assign(_abs_gap=summary_table["Seat Gap"].abs()).sort_values(["_abs_gap", "Scenario Score"]).head(1)

    visuals.render_metric_row(
        [
            (
                "Best score",
                best_score.iloc[0]["Scenario"] if not best_score.empty else "None",
                f"{best_score.iloc[0]['Scenario Score']:.1f}" if not best_score.empty else None,
            ),
            (
                "Lowest high-risk count",
                lowest_risk.iloc[0]["Scenario"] if not lowest_risk.empty else "None",
                f"{int(lowest_risk.iloc[0]['High Risk Sites'])} site(s)" if not lowest_risk.empty else None,
            ),
            (
                "Tightest seat gap",
                smallest_gap.iloc[0]["Scenario"] if not smallest_gap.empty else "None",
                visuals.format_number(smallest_gap.iloc[0]["Seat Gap"]) if not smallest_gap.empty else None,
            ),
        ],
        columns=3,
    )

    if auto_selected is not None:
        with st.container(border=True):
            ui.render_section_heading(
                "Recommended comparison narrative",
                "Summarises which option currently leads the selected shortlist, what is driving the lead, and what could still change the answer.",
                eyebrow="What this means",
            )
            explainability.render_driver_panel(
                title="Key delta drivers for the lead option",
                summary=auto_selected.get("summary", {}),
                baseline_summary=baseline_entry.get("summary", {}),
                outputs=auto_selected.get("calculated_outputs", pd.DataFrame()),
                baseline_label=baseline_entry["scenario_name"],
            )
            explainability.render_recommendation_panel(
                preferred_entry=auto_selected,
                baseline_entry=baseline_entry,
                comparison_entries=selected_entries,
                title="Why this option leads the shortlist",
            )

    scorecard_col, chart_col = st.columns([1.2, 1.0])
    with scorecard_col:
        with st.container(border=True):
            ui.render_section_heading(
                "Scenario scorecards",
                "A concise portfolio-level comparison of demand, capacity, area, standards, and risk.",
                eyebrow="Executive table",
            )
            st.dataframe(summary_table, use_container_width=True, hide_index=True)
    with chart_col:
        with st.container(border=True):
            ui.render_section_heading(
                "Score comparison",
                "Use this visual to frame the shortlist before exploring site-level action changes.",
                eyebrow="Portfolio score",
            )
            fig = visuals.bar_chart(summary_table, x="Scenario", y="Scenario Score", color="Origin", title="Scenario score comparison")
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            else:
                ui.render_empty_state("No score comparison available", "The selected scenarios do not yet produce a comparison chart.")

    site_table = engine.comparison_site_table(selected_entries)
    if not site_table.empty:
        changed_sites = site_table.groupby("site_name")["action_flag"].nunique().reset_index(name="Action Variants")
        changed_sites = changed_sites[changed_sites["Action Variants"] > 1].sort_values("Action Variants", ascending=False)
        change_tab, full_tab = st.tabs(["Action shifts", "Full comparison"])
        with change_tab:
            with st.container(border=True):
                ui.render_section_heading(
                    "Sites with action shifts",
                    "These are the sites where the recommendation changes between selected scenarios, making them the best places to focus the discussion.",
                    eyebrow="What changes",
                )
                if not changed_sites.empty:
                    ui.render_callout(
                        "Primary comparison question",
                        (
                            f"{len(changed_sites)} site(s) change action across the selected scenarios. "
                            "Use these sites to explain the practical difference between options."
                        ),
                        tone="accent",
                    )
                    st.dataframe(changed_sites.rename(columns={"site_name": "Site"}), use_container_width=True, hide_index=True)
                else:
                    ui.render_empty_state(
                        "No action shifts detected",
                        "The selected scenarios lead to the same action recommendation at every site in scope.",
                    )
        with full_tab:
            with st.container(border=True):
                ui.render_section_heading(
                    "Full site-by-site comparison",
                    "Detailed view across all selected scenarios so you can inspect seat gaps, risks, and action flags line by line.",
                    eyebrow="Detailed comparison",
                )
                display = site_table.rename(
                    columns={
                        "site_name": "Site",
                        "region": "Region",
                        "scenario_name": "Scenario",
                        "forecast_headcount": "Forecast headcount",
                        "required_seats": "Required seats",
                        "existing_seats": "Existing seats",
                        "seat_gap": "Seat gap",
                        "required_area_sqm": "Required area",
                        "area_gap_sqm": "Area gap",
                        "action_flag": "Action",
                        "action_reason": "Action reason",
                        "risk_rating": "Risk",
                        "key_risk": "Key risk",
                        "why_this_changed": "Why this changed",
                        "standards_compliance_score": "Standards",
                        "capacity_fit_score": "Capacity fit",
                        "utilisation_fit_score": "Utilisation fit",
                        "implementation_simplicity_score": "Implementation simplicity",
                        "consolidation_efficiency_score": "Consolidation efficiency",
                        "scenario_score": "Scenario score",
                        "origin": "Origin",
                    }
                )
                st.dataframe(
                    display[
                        [
                            "Site",
                            "Region",
                            "Scenario",
                            "Origin",
                            "Forecast headcount",
                            "Required seats",
                            "Existing seats",
                            "Seat gap",
                            "Required area",
                            "Area gap",
                            "Action",
                            "Action reason",
                            "Risk",
                            "Key risk",
                            "Why this changed",
                            "Capacity fit",
                            "Utilisation fit",
                            "Standards",
                            "Implementation simplicity",
                            "Consolidation efficiency",
                            "Scenario score",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
    else:
        ui.render_empty_state(
            "No site comparison available",
            "The selected scenarios do not currently return any comparable site-level outputs.",
        )

    if auto_selected is not None:
        ui.render_callout(
            "Recommended scenario highlight",
            (
                f"{auto_selected['scenario_name']} [{auto_selected['origin']}] currently leads the selected shortlist. "
                "Use the provenance and rationale blocks above to explain why."
            ),
            tone="success",
        )
