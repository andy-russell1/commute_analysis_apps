from __future__ import annotations

import streamlit as st
import pandas as pd

from shared.runtime.downloads import df_to_csv_bytes

from apps.emea_space_occupancy import engine, explainability, ui, visuals
from apps.emea_space_occupancy.common import build_live_bundle, prepare_page


def render_page() -> None:
    context = prepare_page(
        "Space Planning Outputs",
        "Matrix-style planning outputs that bridge strategy modelling into building and floor implementation views.",
    )
    if not context.get("ready"):
        ui.render_callout("Workbook not ready", "Load a valid workbook before reviewing space planning outputs.", tone="warning")
        return

    live_bundle = build_live_bundle(context)
    baseline_bundle = engine.compute_live_scenario(
        context["clean_sheets"],
        engine.build_working_assumptions(context["clean_sheets"], "Base 2026"),
        context["filters"],
        workbook_name=context["workbook_name"],
        workbook_hash=context["workbook_hash"],
    )
    plans = engine.build_space_plan(context["clean_sheets"], live_bundle, context["filters"])
    building_plan = plans["building_plan"]
    floor_plan = plans["floor_plan"]

    ui.render_hero_panel(
        eyebrow="Planning Outputs",
        title="Move from strategy into building and floor implementation views",
        body=(
            "This page turns the live scenario into implementation-ready planning tables so you can move from portfolio "
            "strategy into where the intervention would actually land."
        ),
        badges=[
            ("Scenario", live_bundle.get("scenario_name", context["active_scenario"])),
            ("Buildings in plan", visuals.format_number(building_plan["building_id"].nunique() if not building_plan.empty else 0)),
            ("Floors in plan", visuals.format_number(floor_plan["floor_id"].nunique() if not floor_plan.empty else 0)),
        ],
    )

    with st.container(border=True):
        explainability.render_provenance_panel(
            context=context,
            scenario_name=live_bundle.get("scenario_name", context["active_scenario"]),
            scenario_origin="Live Scenario",
            basis_scenario_name=context["active_scenario"],
            basis_origin="Seed Scenario",
            calculation_timestamp=context.get("last_run"),
            assumption_count=len(live_bundle.get("assumptions", pd.DataFrame()).index),
            notes=context.get("working_notes", ""),
        )
        explainability.render_planning_formula_cards(expanded=False)

    visuals.render_metric_row(
        [
            ("Required Seats", visuals.format_number(live_bundle.get("summary", {}).get("required_seats", 0)), None),
            ("Existing Seats", visuals.format_number(live_bundle.get("summary", {}).get("existing_seats", 0)), None),
            ("Required Area", visuals.format_number(live_bundle.get("summary", {}).get("required_area_sqm", 0), "sqm"), None),
            ("Area Gap", visuals.format_number(live_bundle.get("summary", {}).get("area_gap_sqm", 0), "sqm"), None),
        ],
        columns=4,
    )

    with st.container(border=True):
        ui.render_section_heading(
            "Why the planning outputs look this way",
            "Summarises the live scenario delta and the planning signals that matter most before moving into building and floor detail.",
            eyebrow="Planning rationale",
        )
        explainability.render_driver_panel(
            title="Planning drivers versus baseline",
            summary=live_bundle.get("summary", {}),
            baseline_summary=baseline_bundle.get("summary", {}),
            outputs=live_bundle.get("outputs", building_plan),
            baseline_label="Base 2026",
        )
        explainability.render_lead_site_rationale(
            explainability.priority_site(live_bundle.get("outputs", pd.DataFrame())),
            title="Lead site rationale behind the planning view",
        )

    ui.render_callout(
        "Implementation read-out",
        "Use the building view for executive discussion and the floor view when the conversation turns to practical stacking or intervention planning.",
        tone="info",
    )

    building_tab, floor_tab = st.tabs(["Building planning matrix", "Floor planning summary"])

    with building_tab:
        with st.container(border=True):
            ui.render_section_heading(
                "Building planning matrix",
                "The best view for explaining how the scenario translates into building-level targets and delivery flags.",
                eyebrow="Executive planning view",
            )
            if not building_plan.empty:
                display = building_plan[
                    [
                        "site_name",
                        "building_name",
                        "seat_capacity_total",
                        "target_seats",
                        "target_area_sqm",
                        "seat_gap_target",
                        "implementation_flag",
                        "allocation_basis",
                        "planning_confidence",
                        "action_flag",
                        "risk_rating",
                        "action_reason",
                    ]
                ].rename(
                    columns={
                        "site_name": "Site",
                        "building_name": "Building",
                        "seat_capacity_total": "Existing seats",
                        "target_seats": "Target seats",
                        "target_area_sqm": "Target area",
                        "seat_gap_target": "Seat gap",
                        "implementation_flag": "Implementation flag",
                        "allocation_basis": "Allocation basis",
                        "planning_confidence": "Planning confidence",
                        "action_flag": "Action",
                        "risk_rating": "Risk",
                        "action_reason": "Action reason",
                    }
                ).sort_values("Target seats", ascending=False)
                st.dataframe(display, use_container_width=True, hide_index=True)
                fig = visuals.bar_chart(
                    building_plan.sort_values("target_seats", ascending=False).head(15),
                    x="building_name",
                    y="target_seats",
                    title="Target seats by building",
                )
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                ui.render_empty_state("No building planning output", "The current filters do not return any building-level plan rows.")

    with floor_tab:
        with st.container(border=True):
            ui.render_section_heading(
                "Floor planning summary",
                "A practical view for implementation teams once the strategic direction has been agreed.",
                eyebrow="Delivery detail",
            )
            if not floor_plan.empty:
                display = floor_plan[
                    [
                        "site_name",
                        "building_name",
                        "floor_name",
                        "floor_sequence",
                        "seat_capacity",
                        "target_seats",
                        "target_area_sqm",
                        "seat_gap_target",
                        "intervention_flag",
                        "allocation_basis",
                        "planning_confidence",
                        "why_this_changed",
                    ]
                ].rename(
                    columns={
                        "site_name": "Site",
                        "building_name": "Building",
                        "floor_name": "Floor",
                        "floor_sequence": "Floor seq",
                        "seat_capacity": "Existing seats",
                        "target_seats": "Target seats",
                        "target_area_sqm": "Target area",
                        "seat_gap_target": "Seat gap",
                        "intervention_flag": "Intervention flag",
                        "allocation_basis": "Allocation basis",
                        "planning_confidence": "Planning confidence",
                        "why_this_changed": "Why this changed",
                    }
                ).sort_values(["Site", "Floor seq"])
                st.dataframe(display, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download planning summary (CSV)",
                    data=df_to_csv_bytes(
                        floor_plan[
                            [
                                "site_name",
                                "building_name",
                                "floor_name",
                                "seat_capacity",
                                "target_seats",
                                "target_area_sqm",
                                "seat_gap_target",
                                "intervention_flag",
                                "allocation_basis",
                                "planning_confidence",
                                "why_this_changed",
                            ]
                        ]
                    ),
                    file_name="emea_space_planning_summary.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            else:
                ui.render_empty_state("No floor planning output", "The current filters do not return any floor-level plan rows.")
