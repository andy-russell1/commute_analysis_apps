from __future__ import annotations

import pandas as pd
import streamlit as st

from shared.runtime.downloads import df_to_csv_bytes

from apps.emea_space_occupancy import engine, explainability, exports, ui, visuals
from apps.emea_space_occupancy.common import build_live_bundle, build_scenario_library, prepare_page


def render_page() -> None:
    context = prepare_page(
        "Exports & Audit",
        "Download structured outputs and inspect provenance, active filters, assumptions, and snapshot lineage.",
    )
    if not context.get("ready"):
        ui.render_callout("Workbook not ready", "Load a valid workbook before exporting outputs or reviewing provenance.", tone="warning")
        return

    baseline = engine.build_portfolio_baseline(context["clean_sheets"], context["filters"])
    live_bundle = build_live_bundle(context)
    library = build_scenario_library(context, live_bundle)
    comparison_summary = engine.comparison_summary_table(library)
    comparison_site = engine.comparison_site_table(library)
    decision_pack_summary = pd.DataFrame(
        [
            {
                "workbook_name": context["workbook_name"],
                "workbook_hash": context["workbook_hash"],
                "active_scenario": context["active_scenario"],
                "working_live_scenario": live_bundle.get("scenario_name"),
                "preferred_scenario_key": context["preferred_scenario_key"],
                "active_filter_summary": exports._filter_summary(context["filters"]),
                "filters": str(context["filters"]),
                "last_calculation_timestamp": context.get("last_run"),
                "working_notes_captured": "Yes" if str(context.get("working_notes", "")).strip() else "No",
                "working_assumption_count": len(context["working_assumptions"].index),
                "saved_snapshots": len(context["saved_snapshots"]),
            }
        ]
    )
    snapshots_for_export = [
        *context["seed_snapshots"],
        *context["saved_snapshots"],
        engine.build_snapshot(
            live_bundle,
            active_filters=context["filters"],
            origin="Live Scenario",
            scenario_name=live_bundle["scenario_name"],
            basis_scenario_name=context["active_scenario"],
            basis_origin="Seed Scenario",
            notes=context.get("working_notes", ""),
            calculation_timestamp=context.get("last_run"),
        ),
    ]

    ui.render_hero_panel(
        eyebrow="Exports and Audit",
        title="Package the outputs and keep the session provenance visible",
        body=(
            "This page groups the most useful downloads for a client handover while retaining the workbook hash, "
            "active filters, and snapshot lineage needed for auditability."
        ),
        badges=[
            ("Saved snapshots", str(len(context["saved_snapshots"]))),
            ("Scenario library", str(len(library))),
            ("Validation", context["validation"].get("status", "Unknown")),
            ("Workbook", context["workbook_name"]),
        ],
    )

    visuals_items = [
        ("Downloads available", "6", None),
        ("Saved snapshots", str(len(context["saved_snapshots"])), None),
        ("Seed scenarios", str(len(context["seed_snapshots"])), None),
        ("Validation issues", str(len(context["validation"].get("issues", pd.DataFrame()))), None),
    ]
    visuals.render_metric_row(visuals_items, columns=4)

    with st.container(border=True):
        explainability.render_provenance_panel(
            context=context,
            scenario_name=live_bundle.get("scenario_name", context["active_scenario"]),
            scenario_origin="Audit view across baseline, seed, live, and saved outputs",
            basis_scenario_name=context["active_scenario"],
            basis_origin="Seed Scenario",
            calculation_timestamp=context.get("last_run"),
            assumption_count=len(context["working_assumptions"].index),
            notes=context.get("working_notes", ""),
        )
        ui.render_bullet_panel(
            "How provenance is captured",
            [
                "Every exported scenario now carries workbook name and hash, calculation timestamp, scenario origin, seed / basis scenario, active filters, and assumption counts.",
                "Saved snapshots preserve their own save timestamp while keeping seed-vs-live lineage separate from the current working view.",
                "The executive Excel package includes baseline, comparison, decision-pack, assumptions, outputs, and snapshot metadata in one handover file.",
            ],
            empty_message="Provenance guidance is not available.",
            tone="info",
        )

    ui.render_callout(
        "Export guidance",
        "Use the executive Excel package for a single handover artifact, and use the CSV downloads when you need to inspect or reuse individual datasets.",
        tone="info",
    )

    download_col, audit_export_col = st.columns(2)
    with download_col:
        with st.container(border=True):
            ui.render_section_heading(
                "Core downloads",
                "The most common exports for baseline, comparison, and snapshot review.",
                eyebrow="Client handover",
            )
            st.download_button(
                "Download baseline sites (CSV)",
                data=df_to_csv_bytes(baseline["site_table"]),
                file_name="emea_space_baseline_sites.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.download_button(
                "Download comparison summary (CSV)",
                data=df_to_csv_bytes(comparison_summary),
                file_name="emea_space_scenario_comparison.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.download_button(
                "Download executive export package (Excel)",
                data=exports.build_excel_package(
                    validation_result=context["validation"],
                    baseline_site_table=baseline["site_table"],
                    comparison_summary=comparison_summary,
                    comparison_site=comparison_site,
                    decision_pack_summary=decision_pack_summary,
                    snapshots=snapshots_for_export,
                ),
                file_name="emea_space_occupancy_export_package.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
    with audit_export_col:
        with st.container(border=True):
            ui.render_section_heading(
                "Detailed audit exports",
                "Use these when you need raw assumptions, validation detail, or snapshot-level outputs.",
                eyebrow="Supporting files",
            )
            st.download_button(
                "Download snapshot outputs (CSV)",
                data=df_to_csv_bytes(exports.scenario_outputs_export_table(snapshots_for_export)),
                file_name="emea_space_snapshot_outputs.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.download_button(
                "Download assumptions used (CSV)",
                data=df_to_csv_bytes(exports.assumptions_export_table(snapshots_for_export)),
                file_name="emea_space_assumptions.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.download_button(
                "Download validation summary (CSV)",
                data=df_to_csv_bytes(context["validation"].get("issues", pd.DataFrame())),
                file_name="emea_space_validation_summary.csv",
                mime="text/csv",
                use_container_width=True,
            )

    audit_rows = [
        {"Item": "Loaded workbook", "Value": context["workbook_name"]},
        {"Item": "Workbook hash", "Value": context["workbook_hash"]},
        {"Item": "Active scenario", "Value": context["active_scenario"]},
        {"Item": "Working live scenario", "Value": live_bundle.get("scenario_name")},
        {"Item": "Last calculation timestamp", "Value": context.get("last_run") or "Live working view"},
        {"Item": "Active filters", "Value": exports._filter_summary(context["filters"])},
        {"Item": "Working assumption rows", "Value": str(len(context["working_assumptions"].index))},
        {"Item": "Working notes captured", "Value": "Yes" if str(context.get("working_notes", "")).strip() else "No"},
        {"Item": "Saved scenario names", "Value": ", ".join(snapshot["scenario_name"] for snapshot in context["saved_snapshots"])},
        {"Item": "Seed scenarios available", "Value": ", ".join(snapshot["scenario_name"] for snapshot in context["seed_snapshots"])},
        {"Item": "Current live scenario origin", "Value": "Live Scenario"},
    ]
    overview_tab, snapshot_tab = st.tabs(["Audit overview", "Snapshot lineage"])
    with overview_tab:
        with st.container(border=True):
            ui.render_section_heading(
                "Audit and provenance",
                "Current session metadata captured alongside every exportable output.",
                eyebrow="Session metadata",
            )
            st.dataframe(audit_rows, use_container_width=True, hide_index=True)
            st.dataframe(
                explainability.scenario_provenance_table(snapshots_for_export),
                use_container_width=True,
                hide_index=True,
            )
    with snapshot_tab:
        with st.container(border=True):
            ui.render_section_heading(
                "Snapshot metadata",
                "Lineage across seed scenarios, saved snapshots, and the current live scenario.",
                eyebrow="Snapshot lineage",
            )
            st.dataframe(exports.snapshot_metadata_table(snapshots_for_export), use_container_width=True, hide_index=True)
