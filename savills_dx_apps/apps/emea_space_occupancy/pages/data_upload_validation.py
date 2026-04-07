from __future__ import annotations

import streamlit as st

from apps.emea_space_occupancy import exports, state, ui, visuals
from apps.emea_space_occupancy.common import prepare_page


def render_page() -> None:
    context = prepare_page(
        "Data Upload & Validation",
        "Load the bundled demo workbook or replace it with a client workbook in the same multi-sheet structure.",
    )
    validation_result = context.get("validation", {})
    available_core = validation_result.get("available_core_sheets", [])

    ui.render_hero_panel(
        eyebrow="Workbook Readiness",
        title="Confirm the source workbook before presenting portfolio outputs",
        body=(
            "This page is the control point for workbook quality. Replace the demo workbook here, review the validation "
            "status, and only then move into the baseline or scenario pages."
        ),
        badges=[
            ("Workbook", context.get("workbook_name", "Bundled demo workbook")),
            ("Validation", validation_result.get("status", "Unknown")),
            ("Quality score", str(validation_result.get("quality_score", 0))),
            ("Core sheets ready", f"{len(available_core)}/7"),
        ],
    )

    upload_col, button_col = st.columns([1.6, 0.6])
    with upload_col:
        with st.container(border=True):
            ui.render_section_heading(
                "Replace workbook",
                "Upload a client workbook that follows the same multi-sheet structure as the bundled demo.",
                eyebrow="Data source",
            )
            uploaded = st.file_uploader("Upload replacement workbook", type=["xlsx"], key="eso_page_upload")
            if uploaded is not None:
                state.set_workbook_override(
                    st.session_state,
                    file_bytes=uploaded.getvalue(),
                    workbook_name=uploaded.name,
                )
                st.rerun()
    with button_col:
        with st.container(border=True):
            ui.render_section_heading(
                "Reset",
                "Return to the bundled demo workbook if you want to reset the session quickly.",
                eyebrow="Fallback",
            )
            if st.button("Reset to bundled demo", key="eso_page_reset_demo", use_container_width=True):
                state.reset_to_demo_workbook(st.session_state)
                st.rerun()

    visuals.render_metric_row(
        [
            ("Quality Score", visuals.format_number(validation_result.get("quality_score", 0), "score"), None),
            ("Critical Issues", visuals.format_number(validation_result.get("critical_count", 0)), None),
            ("Warnings", visuals.format_number(validation_result.get("warning_count", 0)), None),
            ("Validation Status", validation_result.get("status", "Unknown"), None),
        ],
        columns=4,
    )

    if validation_result.get("status") == "Blocked":
        ui.render_callout(
            "Validation blocked",
            "Core modelling sheets are not yet validated. Resolve the critical issues below before relying on live outputs.",
            tone="critical",
        )
    elif validation_result.get("warning_count", 0) > 0:
        ui.render_callout(
            "Validation ready with warnings",
            "The workbook is usable, but the warning log should be reviewed before presenting outputs externally.",
            tone="warning",
        )
    else:
        ui.render_callout("Workbook validated successfully", "The workbook is ready to drive the baseline and scenario pages.", tone="success")

    summary_col, relationship_col = st.columns([1.15, 0.85])
    with summary_col:
        with st.container(border=True):
            ui.render_section_heading(
                "Per-sheet validation summary",
                "Quickly shows row counts, issue counts, and readiness by worksheet.",
                eyebrow="Sheet-level review",
            )
            sheet_summary = validation_result.get("sheet_summary")
            if sheet_summary is not None and not getattr(sheet_summary, "empty", True):
                st.dataframe(sheet_summary, use_container_width=True, hide_index=True)
            else:
                ui.render_empty_state("No sheet summary available", "Upload or restore a workbook to generate the validation summary.")

    with relationship_col:
        with st.container(border=True):
            ui.render_section_heading(
                "Join integrity",
                "Confirms whether key relationships between sheets still hold.",
                eyebrow="Relationship checks",
            )
            relationship_summary = validation_result.get("relationship_summary")
            if relationship_summary is not None and not relationship_summary.empty:
                st.dataframe(relationship_summary, use_container_width=True, hide_index=True)
            else:
                ui.render_empty_state("No relationship issues", "No relationship issues were recorded for the current workbook.")

    with st.container(border=True):
        ui.render_section_heading(
            "Issue log",
            "Use this detail when the workbook is blocked or when you need to explain a warning to a client team.",
            eyebrow="Detailed diagnostics",
        )
        issues = validation_result.get("issues")
        if issues is not None and not issues.empty:
            st.dataframe(issues, use_container_width=True, hide_index=True)
            issue_records = validation_result.get("issue_records", {})
            selectable_issue_ids = [issue_id for issue_id in issues["issue_id"].tolist() if issue_id in issue_records]
            if selectable_issue_ids:
                selected_issue = st.selectbox("Inspect failed records", options=selectable_issue_ids)
                st.dataframe(issue_records[selected_issue], use_container_width=True, hide_index=True)
        else:
            ui.render_empty_state("No validation issues recorded", "The current workbook does not have any logged validation issues.")

    with st.container(border=True):
        ui.render_section_heading(
            "Validation export",
            "Download the issue log when you need to share a remediation pack outside the live app session.",
            eyebrow="Audit output",
        )
        st.download_button(
            "Download validation log (Excel)",
            data=exports.validation_log_bytes(validation_result),
            file_name="emea_space_occupancy_validation_log.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
