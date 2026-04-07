from __future__ import annotations

import pandas as pd
import streamlit as st

from apps.emea_space_occupancy import engine, explainability, state, ui, visuals
from apps.emea_space_occupancy.common import build_live_bundle, build_scenario_library, prepare_page
from apps.emea_space_occupancy.config import PREFERRED_SCENARIO_AUTO_KEY


def render_page() -> None:
    context = prepare_page(
        "Decision Pack",
        "Governance-ready recommendation view with headline impacts, risks, assumptions, and a manually pin-able preferred scenario.",
    )
    if not context.get("ready"):
        ui.render_callout("Workbook not ready", "Load a valid workbook before creating a decision-ready recommendation.", tone="warning")
        return

    live_bundle = build_live_bundle(context)
    baseline_bundle = engine.compute_live_scenario(
        context["clean_sheets"],
        engine.build_working_assumptions(context["clean_sheets"], "Base 2026"),
        context["filters"],
        workbook_name=context["workbook_name"],
        workbook_hash=context["workbook_hash"],
    )
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
        "display_label": f"{baseline_bundle['scenario_name']} [Live Baseline]",
    }
    library = build_scenario_library(context, live_bundle)
    if baseline_bundle["scenario_name"] != live_bundle["scenario_name"]:
        library.insert(0, baseline_entry)

    options = {PREFERRED_SCENARIO_AUTO_KEY: "Auto recommended"}
    options.update({entry["snapshot_key"]: entry["display_label"] for entry in library})
    current_pref = context["preferred_scenario_key"]
    auto_entry, preferred_entry = engine.resolve_preferred_scenario(library, current_pref)

    ui.render_hero_panel(
        eyebrow="Decision Pack",
        title="Turn the scenario library into a governance-ready recommendation",
        body=(
            "Pin a preferred scenario only when you want to override the model's automatic recommendation. "
            "The page then summarises the choice, the rationale, and the sites that need attention."
        ),
        badges=[
            ("Selection mode", "Manual pin" if current_pref != PREFERRED_SCENARIO_AUTO_KEY else "Auto recommended"),
            ("Preferred scenario", preferred_entry["scenario_name"] if preferred_entry else "Not available"),
            ("Auto recommendation", auto_entry["scenario_name"] if auto_entry else "Not available"),
            ("Scenarios reviewed", str(len(library))),
        ],
        tone="accent",
    )

    with st.container(border=True):
        ui.render_section_heading(
            "Preferred scenario control",
            "Leave this on auto for the model's recommended answer, or pin a scenario if the room wants to override the default.",
            eyebrow="Governance setting",
        )
        selected_key = st.selectbox(
            "Preferred scenario pin",
            options=list(options.keys()),
            format_func=lambda key: options[key],
            index=list(options.keys()).index(current_pref) if current_pref in options else 0,
        )
    if selected_key != context["preferred_scenario_key"]:
        state.set_preferred_scenario(st.session_state, selected_key)
        st.rerun()

    auto_entry, preferred_entry = engine.resolve_preferred_scenario(library, selected_key)
    preferred_summary = preferred_entry.get("summary", {}) if preferred_entry else {}
    baseline_summary = baseline_bundle.get("summary", {})
    preferred_assumptions = preferred_entry.get("assumptions_used", pd.DataFrame()) if preferred_entry else pd.DataFrame()

    with st.container(border=True):
        explainability.render_provenance_panel(
            context=context,
            scenario_name=preferred_entry["scenario_name"] if preferred_entry else "No preferred scenario",
            scenario_origin=preferred_entry.get("origin", "Unavailable") if preferred_entry else "Unavailable",
            basis_scenario_name=preferred_entry.get("basis_scenario_name", context["active_scenario"]) if preferred_entry else context["active_scenario"],
            basis_origin=preferred_entry.get("basis_origin", "Seed Scenario") if preferred_entry else "Seed Scenario",
            calculation_timestamp=preferred_entry.get("calculation_timestamp", context.get("last_run")) if preferred_entry else context.get("last_run"),
            assumption_count=preferred_entry.get("assumption_count", int(len(preferred_assumptions.index))) if preferred_entry else None,
            notes=preferred_entry.get("notes", context.get("working_notes", "")) if preferred_entry else context.get("working_notes", ""),
        )
        if preferred_entry is not None and not preferred_assumptions.empty:
            explainability.render_assumption_summary_strip(
                context["clean_sheets"],
                preferred_assumptions,
                basis_scenario_name=preferred_entry.get("basis_scenario_name", context["active_scenario"]),
            )
        explainability.render_score_component_glossary(expanded=False)
        st.dataframe(
            explainability.scenario_provenance_table(
                [entry for entry in [preferred_entry, auto_entry, baseline_entry] if entry is not None]
            ).drop_duplicates(),
            use_container_width=True,
            hide_index=True,
        )

    visuals.render_metric_row(
        [
            ("Preferred scenario", preferred_entry["scenario_name"] if preferred_entry else "None", None),
            ("Auto recommendation", auto_entry["scenario_name"] if auto_entry else "None", None),
            (
                "Scenario score",
                visuals.format_number(preferred_summary.get("scenario_score", 0), "score"),
                f"{preferred_summary.get('scenario_score', 0) - baseline_summary.get('scenario_score', 0):+.1f}",
            ),
            (
                "Seat gap",
                visuals.format_number(preferred_summary.get("seat_gap", 0)),
                f"{preferred_summary.get('seat_gap', 0) - baseline_summary.get('seat_gap', 0):+,.0f}",
            ),
            (
                "High-risk sites",
                visuals.format_number(preferred_summary.get("high_risk_sites", 0)),
                f"{preferred_summary.get('high_risk_sites', 0) - baseline_summary.get('high_risk_sites', 0):+,.0f}",
            ),
        ],
        columns=5,
    )

    if preferred_entry is not None:
        recommendation_body = engine.decision_pack_narrative(preferred_entry, baseline_entry=baseline_bundle)
        ui.render_callout(
            "Executive recommendation",
            recommendation_body,
            tone="success" if selected_key == PREFERRED_SCENARIO_AUTO_KEY else "accent",
        )
    else:
        ui.render_empty_state(
            "No preferred scenario available",
            "The decision pack needs at least one scenario in the library to produce a recommendation.",
        )

    if preferred_entry is not None:
        with st.container(border=True):
            ui.render_section_heading(
                "Decision support narrative",
                "States what changed, why it matters, the decision implied by the model, and what could still change the answer.",
                eyebrow="What this means",
            )
            explainability.render_driver_panel(
                title="Key deltas versus baseline",
                summary=preferred_summary,
                baseline_summary=baseline_summary,
                outputs=preferred_entry.get("calculated_outputs", pd.DataFrame()),
                baseline_label=baseline_entry["scenario_name"],
            )
            explainability.render_recommendation_panel(
                preferred_entry=preferred_entry,
                baseline_entry=baseline_entry,
                comparison_entries=library,
                title="Recommendation rationale block",
            )

    rationale_col, governance_col = st.columns([1.2, 0.9])
    with rationale_col:
        with st.container(border=True):
            ui.render_section_heading(
                "Why this scenario",
                "Use these talking points to explain the recommendation before moving into the action shortlist.",
                eyebrow="Decision rationale",
            )
            if preferred_entry is not None:
                ui.render_bullet_panel(
                    "Recommendation summary",
                    [
                        f"Scenario score: {visuals.format_number(preferred_summary.get('scenario_score', 0), 'score')}.",
                        f"Capacity fit: {visuals.format_number(preferred_summary.get('capacity_fit_score', 0), 'score')}.",
                        f"Utilisation fit: {visuals.format_number(preferred_summary.get('utilisation_fit_score', 0), 'score')}.",
                        f"Standards compliance: {visuals.format_number(preferred_summary.get('standards_compliance_score', 0), 'score')}.",
                        (
                            f"Implementation simplicity: "
                            f"{visuals.format_number(preferred_summary.get('implementation_simplicity_score', 0), 'score')}."
                        ),
                        (
                            f"Consolidation efficiency: "
                            f"{visuals.format_number(preferred_summary.get('consolidation_efficiency_score', 0), 'score')}."
                        ),
                        f"High-risk sites: {visuals.format_number(preferred_summary.get('high_risk_sites', 0))}.",
                        (
                            f"Portfolio action mix: {visuals.format_number(preferred_summary.get('expand_sites', 0))} expand, "
                            f"{visuals.format_number(preferred_summary.get('consolidate_sites', 0))} consolidate, "
                            f"{visuals.format_number(preferred_summary.get('maintain_sites', 0))} maintain."
                        ),
                    ],
                    empty_message="There is no recommendation summary available yet.",
                )
            else:
                ui.render_empty_state("No rationale available", "There is no preferred scenario to explain yet.")
    with governance_col:
        with st.container(border=True):
            ui.render_section_heading(
                "Governance notes",
                "Clarifies whether the pack reflects the model default or a manual override from the session.",
                eyebrow="Control status",
            )
            if auto_entry is not None and preferred_entry is not None:
                ui.render_bullet_panel(
                    "Selection state",
                    [
                        f"Auto recommended scenario: {auto_entry['scenario_name']}.",
                        (
                            f"Preferred scenario currently {'matches' if auto_entry['snapshot_key'] == preferred_entry['snapshot_key'] else 'overrides'} "
                            "the model default."
                        ),
                        f"Current workbook: {context['workbook_name']}.",
                    ],
                    empty_message="Selection state is not available.",
                    tone="info",
                )

    assumptions_used = preferred_assumptions
    with st.expander("Assumptions summary", expanded=False):
        if not assumptions_used.empty:
            st.dataframe(
                assumptions_used[["parameter_category", "parameter_name", "scope_level", "scope_value", "value", "unit", "driver_note"]]
                .sort_values(["parameter_category", "parameter_name"]),
                use_container_width=True,
                hide_index=True,
            )
        else:
            ui.render_empty_state("No assumptions summary available", "There are no stored assumptions for the preferred scenario.")

    if preferred_entry is not None:
        outputs = engine.comparison_site_table([preferred_entry])
        shortlist = outputs.reindex(
            columns=[
                "site_name",
                "region",
                "required_seats",
                "seat_gap",
                "required_area_sqm",
                "action_flag",
                "action_reason",
                "risk_rating",
                "key_risk",
                "why_this_changed",
            ]
        )
        shortlist = (
            shortlist.assign(_risk_rank=shortlist["risk_rating"].map(ui.risk_rank))
            .sort_values(["_risk_rank", "seat_gap"], ascending=[True, True])
            .rename(
                columns={
                    "site_name": "Site",
                    "region": "Region",
                    "required_seats": "Required seats",
                    "seat_gap": "Seat gap",
                    "required_area_sqm": "Required area",
                    "action_flag": "Action",
                    "action_reason": "Action reason",
                    "risk_rating": "Risk",
                    "key_risk": "Key risk",
                    "why_this_changed": "Why this changed",
                }
            )
        )
        top_col, risk_col = st.columns(2)
        with top_col:
            with st.container(border=True):
                ui.render_section_heading(
                    "Recommended actions by site",
                    "The best shortlist to use when moving from strategy into practical intervention decisions.",
                    eyebrow="Action shortlist",
                )
                st.dataframe(shortlist.head(12), use_container_width=True, hide_index=True)
        with risk_col:
            with st.container(border=True):
                ui.render_section_heading(
                    "Risk mix",
                    "Provides a quick visual on how much exposure remains in the preferred scenario.",
                    eyebrow="Portfolio exposure",
                )
                risk_summary = outputs.groupby("risk_rating", dropna=False).size().reset_index(name="Sites")
                fig = visuals.bar_chart(risk_summary, x="risk_rating", y="Sites", title="Risk mix")
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    ui.render_empty_state("No risk mix available", "There are no comparable site outputs for the preferred scenario.")
