from __future__ import annotations

from datetime import datetime, timezone

import streamlit as st

from apps.emea_space_occupancy import engine, explainability, state, ui, visuals
from apps.emea_space_occupancy.common import build_live_bundle, prepare_page


def render_page() -> None:
    context = prepare_page(
        "Scenario Builder",
        "Live scenario modelling workspace with transparent assumptions, recalculation, and snapshot capture.",
    )
    if not context.get("ready"):
        ui.render_callout("Workbook not ready", "Load a valid workbook before running live scenario calculations.", tone="warning")
        return

    last_run = context.get("last_run")
    last_run_label = str(last_run).replace("T", " ").replace("+00:00", " UTC") if last_run else "Working copy not yet refreshed"

    ui.render_hero_panel(
        eyebrow="Live Modelling",
        title="Build, test, and save a scenario while keeping the rationale visible",
        body=(
            "This page is optimised for a live client session: name the working scenario, recalculate quickly, "
            "review the change against the baseline, and save the result when it is ready to compare."
        ),
        badges=[
            ("Working scenario", context["working_scenario_name"]),
            ("Seed scenario", context["active_scenario"]),
            ("Last calculation", last_run_label),
            ("Saved snapshots", str(len(context["saved_snapshots"]))),
        ],
        tone="accent",
    )

    with st.container(border=True):
        explainability.render_provenance_panel(
            context=context,
            scenario_name=context["working_scenario_name"],
            scenario_origin="Live Scenario",
            basis_scenario_name=context["active_scenario"],
            basis_origin="Seed Scenario",
            calculation_timestamp=context.get("last_run"),
            assumption_count=len(context["working_assumptions"].index),
            notes=context.get("working_notes", ""),
        )
        explainability.render_assumption_summary_strip(
            context["clean_sheets"],
            context["working_assumptions"],
            basis_scenario_name=context["active_scenario"],
        )
        explainability.render_scenario_formula_cards(expanded=False)
        explainability.render_score_component_glossary(expanded=False)

    control_col, refresh_col, save_col = st.columns([1.6, 0.7, 0.7])
    with control_col:
        scenario_name = st.text_input("Scenario name", value=context["working_scenario_name"], key="eso_builder_scenario_name")
    with refresh_col:
        st.write("")
        if st.button("Refresh live calculation", key="eso_refresh_live", use_container_width=True):
            st.session_state[state.LAST_RUN_KEY] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            st.rerun()
    with save_col:
        st.write("")
        save_snapshot_clicked = st.button("Save scenario snapshot", key="eso_save_snapshot", use_container_width=True)

    if scenario_name != context["working_scenario_name"]:
        updated = context["working_assumptions"].copy()
        updated["scenario_name"] = scenario_name
        state.set_working_assumptions(st.session_state, updated)
        st.session_state[state.WORKING_SCENARIO_NAME_KEY] = scenario_name
        st.rerun()

    live_bundle = build_live_bundle(context)
    baseline_assumptions = engine.build_working_assumptions(context["clean_sheets"], "Base 2026")
    baseline_bundle = engine.compute_live_scenario(
        context["clean_sheets"],
        baseline_assumptions,
        context["filters"],
        workbook_name=context["workbook_name"],
        workbook_hash=context["workbook_hash"],
    )
    summary = live_bundle.get("summary", {})
    base_summary = baseline_bundle.get("summary", {})
    outputs = live_bundle["outputs"]

    visuals.render_metric_row(
        [
            (
                "Forecast Headcount",
                visuals.format_number(summary.get("forecast_headcount", 0)),
                f"{summary.get('forecast_headcount', 0) - base_summary.get('forecast_headcount', 0):+,.0f}",
            ),
            (
                "Required Seats",
                visuals.format_number(summary.get("required_seats", 0)),
                f"{summary.get('required_seats', 0) - base_summary.get('required_seats', 0):+,.0f}",
            ),
            (
                "Seat Gap",
                visuals.format_number(summary.get("seat_gap", 0)),
                f"{summary.get('seat_gap', 0) - base_summary.get('seat_gap', 0):+,.0f}",
            ),
            (
                "Required Area",
                visuals.format_number(summary.get("required_area_sqm", 0), "sqm"),
                f"{summary.get('required_area_sqm', 0) - base_summary.get('required_area_sqm', 0):+,.0f}",
            ),
            (
                "Scenario Score",
                visuals.format_number(summary.get("scenario_score", 0), "score"),
                f"{summary.get('scenario_score', 0) - base_summary.get('scenario_score', 0):.1f}",
            ),
        ],
        columns=5,
    )

    with st.container(border=True):
        ui.render_section_heading(
            "What changed in this live scenario",
            "A concise view of the main deltas, why they moved, and where challenge should focus.",
            eyebrow="Delta drivers",
        )
        explainability.render_driver_panel(
            title="Active drivers versus baseline",
            summary=summary,
            baseline_summary=base_summary,
            outputs=outputs,
            baseline_label="Base 2026",
        )

    with st.container(border=True):
        ui.render_section_heading(
            "Scenario score components",
            "Keeps the weighted model transparent by showing the component scores that roll up into the overall scenario result.",
            eyebrow="Scoring transparency",
        )
        visuals.render_metric_row(
            [
                ("Capacity fit", visuals.format_number(summary.get("capacity_fit_score", 0), "score"), None),
                ("Utilisation fit", visuals.format_number(summary.get("utilisation_fit_score", 0), "score"), None),
                ("Standards alignment", visuals.format_number(summary.get("standards_compliance_score", 0), "score"), None),
                (
                    "Implementation simplicity",
                    visuals.format_number(summary.get("implementation_simplicity_score", 0), "score"),
                    None,
                ),
                (
                    "Consolidation efficiency",
                    visuals.format_number(summary.get("consolidation_efficiency_score", 0), "score"),
                    None,
                ),
            ],
            columns=5,
        )

    if live_bundle.get("warnings"):
        with st.container(border=True):
            ui.render_section_heading(
                "Modelling warnings",
                "These are not blockers, but they should be explained before using the result as a recommendation.",
                eyebrow="Validation for the room",
            )
            for warning in live_bundle["warnings"]:
                ui.render_callout("Working scenario warning", warning, tone="warning")

    priority_site = None
    if not outputs.empty:
        ranked = outputs.assign(_risk_rank=outputs["risk_rating"].map(ui.risk_rank)).sort_values(
            ["_risk_rank", "seat_gap", "scenario_score"],
            ascending=[True, True, False],
        )
        if not ranked.empty:
            priority_site = ranked.iloc[0]

    if priority_site is not None:
        ui.render_callout(
            "Lead recommendation from this run",
            (
                f"Start the review with {priority_site['site_name']} in {priority_site['region']}: it currently needs "
                f"{priority_site['action_flag']} treatment, carries {priority_site['risk_rating']} risk, and shows a "
                f"seat gap of {visuals.format_number(priority_site['seat_gap'])}."
            ),
            tone="accent",
        )
    with st.container(border=True):
        ui.render_section_heading(
            "Lead action rationale",
            "Explains the top site recommendation in plain English so the room can test the output without reverse-engineering the model.",
            eyebrow="Why this flag appeared",
        )
        explainability.render_lead_site_rationale(priority_site, title="Priority site explanation")

    action_col, risk_col = st.columns(2)
    with action_col:
        with st.container(border=True):
            ui.render_section_heading(
                "Action distribution",
                "Shows how the live scenario is allocating sites across the delivery response types.",
                eyebrow="Portfolio action mix",
            )
            action_summary = live_bundle["outputs"].groupby("action_flag", dropna=False).size().reset_index(name="Sites")
            fig = visuals.bar_chart(action_summary, x="action_flag", y="Sites", title="Action flag distribution")
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            else:
                ui.render_empty_state("No action mix available", "There are no scenario outputs for the current filters.")
    with risk_col:
        with st.container(border=True):
            ui.render_section_heading(
                "Risk distribution",
                "Use this before stepping into site-level detail so the room understands the overall exposure.",
                eyebrow="Risk profile",
            )
            risk_summary = live_bundle["outputs"].groupby("risk_rating", dropna=False).size().reset_index(name="Sites")
            fig = visuals.bar_chart(risk_summary, x="risk_rating", y="Sites", title="Risk rating distribution")
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            else:
                ui.render_empty_state("No risk mix available", "There are no scenario outputs for the current filters.")

    shortlist_tab, full_tab = st.tabs(["Priority shortlist", "Full site impact table"])
    with shortlist_tab:
        with st.container(border=True):
            ui.render_section_heading(
                "Priority shortlist",
                "The first sites to discuss live because they combine risk, required action, and planning delta.",
                eyebrow="Client demo shortcut",
            )
            if not live_bundle["outputs"].empty:
                shortlist = (
                    live_bundle["outputs"]
                    .assign(_risk_rank=live_bundle["outputs"]["risk_rating"].map(ui.risk_rank))
                    .sort_values(["_risk_rank", "seat_gap", "scenario_score"], ascending=[True, True, False])
                    .head(12)
                    .rename(
                        columns={
                            "site_name": "Site",
                            "region": "Region",
                            "peak_attendance": "Peak attendance",
                            "required_seats": "Required seats",
                            "existing_seats": "Existing seats",
                            "seat_gap": "Seat gap",
                            "required_area_sqm": "Required area",
                            "area_gap_sqm": "Area gap",
                            "action_flag": "Action",
                            "risk_rating": "Risk",
                            "action_reason": "Action reason",
                            "why_this_changed": "Why this changed",
                            "standards_compliance_score": "Standards",
                            "scenario_score": "Scenario score",
                        }
                    )
                )
                st.dataframe(
                    shortlist[
                        [
                            "Site",
                            "Region",
                            "Peak attendance",
                            "Required seats",
                            "Existing seats",
                            "Seat gap",
                            "Required area",
                            "Area gap",
                            "Action",
                            "Risk",
                            "Standards",
                            "Scenario score",
                            "Action reason",
                            "Why this changed",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                ui.render_empty_state("No site impacts available", "Refresh the model or widen the filters to generate a shortlist.")
    with full_tab:
        with st.container(border=True):
            ui.render_section_heading(
                "Full site impact table",
                "A sortable client-facing view of the live scenario outcome for every site in scope.",
                eyebrow="Detailed review",
            )
            if not live_bundle["outputs"].empty:
                impact_table = (
                    live_bundle["outputs"]
                    .assign(_risk_rank=live_bundle["outputs"]["risk_rating"].map(ui.risk_rank))
                    .sort_values(["_risk_rank", "seat_gap"], ascending=[True, True])
                    .rename(
                        columns={
                            "site_name": "Site",
                            "region": "Region",
                            "forecast_headcount": "Forecast headcount",
                            "peak_attendance": "Peak attendance",
                            "required_seats": "Required seats",
                            "existing_seats": "Existing seats",
                            "seat_gap": "Seat gap",
                            "required_area_sqm": "Required area",
                            "area_gap_sqm": "Area gap",
                            "action_flag": "Action",
                            "risk_rating": "Risk",
                            "action_reason": "Action reason",
                            "why_this_changed": "Why this changed",
                            "standards_compliance_score": "Standards",
                            "scenario_score": "Scenario score",
                        }
                    )
                )
                st.dataframe(
                    impact_table[
                        [
                            "Site",
                            "Region",
                            "Forecast headcount",
                            "Peak attendance",
                            "Required seats",
                            "Existing seats",
                            "Seat gap",
                            "Required area",
                            "Area gap",
                            "Action",
                            "Risk",
                            "Standards",
                            "Scenario score",
                            "Action reason",
                            "Why this changed",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                ui.render_empty_state("No site impacts available", "Refresh the model or widen the filters to view site-level outputs.")

    if save_snapshot_clicked:
        snapshot = engine.build_snapshot(
            live_bundle,
            active_filters=context["filters"],
            origin="Saved Scenario Snapshot",
            scenario_name=scenario_name,
            basis_scenario_name=context["active_scenario"],
            basis_origin="Seed Scenario",
            notes=context.get("working_notes", ""),
            calculation_timestamp=context.get("last_run"),
        )
        state.add_saved_snapshot(st.session_state, snapshot)
        st.success(f"Saved snapshot '{scenario_name}'.")
