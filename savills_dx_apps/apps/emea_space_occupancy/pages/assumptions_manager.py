from __future__ import annotations

import pandas as pd
import streamlit as st

from apps.emea_space_occupancy import engine, state, ui, visuals
from apps.emea_space_occupancy.common import prepare_page


def _upsert_global_row(assumptions: pd.DataFrame, *, category: str, name: str, value: float, unit: str) -> pd.DataFrame:
    out = assumptions.copy()
    mask = (
        out["parameter_name"].map(engine._normalise_parameter_name) == engine._normalise_parameter_name(name)
    ) & (out["scope_level"].astype(str).str.lower() == "global")
    if mask.any():
        out.loc[mask, "value"] = value
        out.loc[mask, "unit"] = unit
    else:
        template = out.iloc[0].to_dict() if not out.empty else {}
        template.update(
            {
                "parameter_category": category,
                "parameter_name": name,
                "scope_level": "Global",
                "scope_value": "All",
                "value": value,
                "unit": unit,
                "driver_note": "Session override",
                "owner": "Session",
                "version_status": "Working",
            }
        )
        out = pd.concat([out, pd.DataFrame([template])], ignore_index=True)
    return out


def render_page() -> None:
    context = prepare_page(
        "Assumptions Manager",
        "Session-backed scenario assumptions with editable planning controls and auditable parameter rows.",
    )
    if not context.get("ready"):
        ui.render_callout("Workbook not ready", "Load a valid workbook before editing scenario assumptions.", tone="warning")
        return

    assumptions = context["working_assumptions"].copy()
    working_name = context["working_scenario_name"]
    working_notes = context["working_notes"]
    horizon_value = engine._base_horizon_months(assumptions)
    growth = float(engine.resolve_parameter_value(assumptions, "Headcount Growth Default", {}, default=0.035) or 0.035)
    avg_uplift = float(engine.resolve_parameter_value(assumptions, "Average Attendance Uplift", {}, default=0.0) or 0.0)
    peak_buffer = float(engine.resolve_parameter_value(assumptions, "Peak Attendance Buffer", {}, default=0.08) or 0.08)
    desk_share = float(engine.resolve_parameter_value(assumptions, "Desk Sharing Ratio Target", {}, default=1.2) or 1.2)
    sqm_target = float(engine.resolve_parameter_value(assumptions, "sqm per Person Target", {}, default=9.0) or 9.0)
    meeting_ratio = float(engine.resolve_parameter_value(assumptions, "Meeting Seats per 100 Staff", {}, default=14.0) or 14.0)
    collaboration = float(engine.resolve_parameter_value(assumptions, "Collaboration Area Target", {}, default=0.18) or 0.18)

    ui.render_hero_panel(
        eyebrow="Scenario Controls",
        title="Shape the live planning logic without leaving the session",
        body=(
            "Use the quick controls for live client workshops. Keep the detailed table for advanced overrides, "
            "auditable edits, or scenario-specific exceptions."
        ),
        badges=[
            ("Working scenario", working_name),
            ("Active seed", context["active_scenario"]),
            ("Saved drafts", str(len(context["draft_names"]))),
            ("Planning horizon", f"{horizon_value} months"),
        ],
        tone="accent",
    )

    visuals.render_metric_row(
        [
            ("Headcount growth", f"{growth:.1%}", None),
            ("Attendance uplift", f"{avg_uplift:+.1%}", None),
            ("Peak buffer", f"{peak_buffer:.1%}", None),
            ("Desk sharing", f"{desk_share:.2f}x", None),
            ("sqm per person", f"{sqm_target:.1f}", None),
            ("Meeting seats / 100", f"{meeting_ratio:.0f}", None),
        ],
        columns=3,
    )

    ui.render_callout(
        "Recommended usage",
        "For most client sessions, adjust the quick controls, refresh the scenario, and only then edit the detailed parameter rows if a location or business unit needs a targeted exception.",
        tone="info",
    )

    ui.render_bullet_panel(
        "Modelling rules in play",
        [
            "Forecasts interpolate from the current position to 12 and 24 months, with a damped extension beyond 24 months.",
            "Average attendance honours uplift assumptions but will not fall below any anchor-day attendance floor.",
            "Peak attendance uses both the live peak buffer and an observed/site-type minimum peak gap so crowding stays credible.",
            "Planning standards resolve in this order: scoped scenario override, workstyle-weighted default, site-type fallback, then portfolio fallback.",
        ],
        empty_message="Modelling rules are not available.",
        tone="info",
    )

    col_1, col_2 = st.columns([1.3, 1.0])
    with col_1:
        with st.container(border=True):
            ui.render_section_heading(
                "Quick controls",
                "Adjust the assumptions that change demand, attendance, and planning standards most visibly in a client conversation.",
                eyebrow="Primary workflow",
            )
            with st.form("eso_assumption_controls"):
                scenario_name = st.text_input("Working scenario name", value=working_name)
                horizon = st.selectbox(
                    "Planning horizon (months)",
                    options=[12, 18, 24, 36],
                    index=[12, 18, 24, 36].index(horizon_value) if horizon_value in [12, 18, 24, 36] else 0,
                )
                st.markdown("**Demand and attendance**")
                growth_value = st.number_input("Headcount growth / decline", value=growth, step=0.01, format="%.3f")
                avg_value = st.number_input("Average attendance uplift", value=avg_uplift, step=0.01, format="%.3f")
                peak_value = st.number_input("Peak attendance buffer", value=peak_buffer, step=0.01, format="%.3f")
                st.markdown("**Planning standards**")
                desk_value = st.number_input("Desk sharing ratio", value=desk_share, step=0.01, format="%.3f")
                sqm_value = st.number_input("sqm per person target", value=sqm_target, step=0.1, format="%.2f")
                meeting_value = st.number_input("Meeting seats per 100 staff", value=meeting_ratio, step=1.0, format="%.1f")
                collab_value = st.number_input("Support / collaboration allowance", value=collaboration, step=0.01, format="%.3f")
                notes_value = st.text_area("Notes / rationale", value=working_notes, height=110)
                submitted = st.form_submit_button("Apply working controls", use_container_width=True)

        if submitted:
            updated = assumptions.copy()
            updated["scenario_name"] = scenario_name
            updated["planning_horizon_months"] = horizon
            updated = _upsert_global_row(updated, category="Demand", name="Headcount Growth Default", value=growth_value, unit="pct")
            updated = _upsert_global_row(updated, category="Attendance", name="Average Attendance Uplift", value=avg_value, unit="pct_pts")
            updated = _upsert_global_row(updated, category="Attendance", name="Peak Attendance Buffer", value=peak_value, unit="pct")
            updated = _upsert_global_row(updated, category="Planning", name="Desk Sharing Ratio Target", value=desk_value, unit="ratio")
            updated = _upsert_global_row(updated, category="Planning", name="sqm per Person Target", value=sqm_value, unit="sqm")
            updated = _upsert_global_row(updated, category="Planning", name="Meeting Seats per 100 Staff", value=meeting_value, unit="seats")
            updated = _upsert_global_row(updated, category="Planning", name="Collaboration Area Target", value=collab_value, unit="pct")
            state.set_working_assumptions(st.session_state, updated)
            st.session_state[state.WORKING_SCENARIO_NAME_KEY] = scenario_name
            st.session_state[state.WORKING_NOTES_KEY] = notes_value
            st.rerun()

    with col_2:
        with st.container(border=True):
            ui.render_section_heading(
                "Scenario loading and drafts",
                "Swap between workbook seeds, saved session drafts, and your current working copy without leaving the page.",
                eyebrow="Session management",
            )
            load_seed = st.selectbox("Load from seed scenario", options=context["scenario_names"], key="eso_load_seed")
            if st.button("Load seed assumptions", key="eso_load_seed_btn", use_container_width=True):
                state.set_active_scenario(st.session_state, load_seed)
                st.rerun()

            if context["draft_names"]:
                load_draft = st.selectbox("Load saved draft", options=context["draft_names"], key="eso_load_draft")
                if st.button("Load saved draft", key="eso_load_draft_btn", use_container_width=True):
                    state.load_assumption_draft(st.session_state, load_draft)
                    st.rerun()
            else:
                ui.render_empty_state(
                    "No saved drafts yet",
                    "Save the current working copy as a draft if you want to preserve alternative options during the session.",
                )

            draft_name = st.text_input("Save current assumptions as draft", value=working_name, key="eso_save_draft_name")
            if st.button("Save draft", key="eso_save_draft_btn", use_container_width=True):
                state.save_assumption_draft(st.session_state, draft_name)
                st.success(f"Saved draft '{draft_name}'.")
            if st.button("Reset to selected seed", key="eso_reset_to_seed_btn", use_container_width=True):
                state.set_active_scenario(st.session_state, context["active_scenario"])
                st.rerun()

            ui.render_bullet_panel(
                "Current working basis",
                [
                    f"Scenario name: {working_name}",
                    f"Planning horizon: {horizon_value} months",
                    f"Notes captured: {'Yes' if working_notes.strip() else 'No'}",
                ],
                empty_message="No working basis is available yet.",
            )

    with st.container(border=True):
        ui.render_section_heading(
            "Detailed parameter table",
            "Use this editor for auditable row-level changes, scope-specific overrides, or advanced scenario tuning.",
            eyebrow="Advanced controls",
        )
        edited = st.data_editor(
            assumptions.sort_values(["parameter_category", "parameter_name", "scope_level", "scope_value"]),
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",
            key="eso_assumptions_editor",
        )
        if st.button("Save table edits", key="eso_save_table_edits", use_container_width=True):
            state.set_working_assumptions(st.session_state, pd.DataFrame(edited))
            st.rerun()
