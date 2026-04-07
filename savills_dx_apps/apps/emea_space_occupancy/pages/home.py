from __future__ import annotations

import pandas as pd
import streamlit as st

from apps.emea_space_occupancy import engine, ui, visuals
from apps.emea_space_occupancy.common import build_live_bundle, build_scenario_library, prepare_page


def _home_headline(live_summary: dict[str, float]) -> str:
    seat_gap = float(live_summary.get("seat_gap", 0) or 0)
    expand_sites = int(live_summary.get("expand_sites", 0) or 0)
    consolidate_sites = int(live_summary.get("consolidate_sites", 0) or 0)
    if seat_gap > 0:
        return (
            f"The current working scenario retains {visuals.format_number(seat_gap)} seats of portfolio headroom. "
            f"The immediate discussion should focus on the {expand_sites} site(s) still flagged for expansion or re-stack action."
        )
    return (
        f"The current working scenario is short by {visuals.format_number(abs(seat_gap))} seats. "
        f"That makes the {consolidate_sites} consolidation opportunities and the expansion list the key decisions to resolve."
    )


def render_page() -> None:
    context = prepare_page("Home", "Executive landing page for EMEA portfolio performance and scenario context.")
    if not context.get("ready"):
        ui.render_callout(
            "Workbook not ready",
            "Use Data Upload & Validation to inspect the source workbook before relying on portfolio outputs.",
            tone="warning",
        )
        return

    baseline = engine.build_portfolio_baseline(context["clean_sheets"], context["filters"])
    live_bundle = build_live_bundle(context)
    library = build_scenario_library(context, live_bundle)
    summary = baseline["portfolio_summary"]
    live_summary = live_bundle.get("summary", {})
    site_table = baseline["site_table"]
    outputs = live_bundle.get("outputs", pd.DataFrame()).copy()

    ui.render_hero_panel(
        eyebrow="Executive Overview",
        title="Portfolio planning in one live client-ready flow",
        body=(
            "Use this landing page to orient the conversation quickly: confirm the current workbook quality, "
            "understand the portfolio footprint, and surface where the live scenario needs action."
        ),
        badges=[
            ("Validation", context["validation"].get("status", "Unknown")),
            ("Active scenario", live_bundle.get("scenario_name", context["active_scenario"])),
            ("Scope", f"{visuals.format_number(summary.get('total_sites', 0))} sites"),
            ("Decision flow", f"Page {context['current_page_index']} of {context['total_pages']}"),
        ],
        tone="accent",
    )

    visuals.render_metric_row(
        [
            ("Total sites", visuals.format_number(summary.get("total_sites", 0)), None),
            ("Total buildings", visuals.format_number(summary.get("total_buildings", 0)), None),
            ("Seat capacity", visuals.format_number(summary.get("total_seat_capacity", 0)), None),
            ("Current headcount", visuals.format_number(summary.get("current_headcount", 0)), None),
            ("Desk utilisation", visuals.format_number(summary.get("average_desk_utilisation_pct", 0), "pct"), None),
            ("Risk sites", visuals.format_number(summary.get("portfolio_risk_sites_count", 0)), None),
        ],
        columns=3,
    )

    priority_site = pd.DataFrame()
    if not outputs.empty:
        priority_site = (
            outputs.assign(_risk_rank=outputs["risk_rating"].map(ui.risk_rank))
            .sort_values(["_risk_rank", "seat_gap", "scenario_score"], ascending=[True, True, False])
            .head(1)
        )

    intro_col, scenario_col = st.columns([1.35, 1.0])
    with intro_col:
        with st.container(border=True):
            ui.render_section_heading(
                "What matters now",
                "A short narrative you can use to open a client conversation before drilling into detail.",
                eyebrow="Read-out",
            )
            ui.render_callout("Headline planning signal", _home_headline(live_summary), tone="info")
            bullets = [
                (
                    f"Highest immediate attention: {priority_site.iloc[0]['site_name']} in "
                    f"{priority_site.iloc[0]['region']} ({priority_site.iloc[0]['action_flag']}, "
                    f"{priority_site.iloc[0]['risk_rating']} risk)."
                )
                if not priority_site.empty
                else "No site-level pressure point is currently available for the active filter scope.",
                (
                    f"Standards alignment sits at "
                    f"{visuals.format_number(live_summary.get('standards_compliance_score', 0), 'score')}."
                ),
                (
                    f"{visuals.format_number(live_summary.get('high_risk_sites', 0))} site(s) remain high risk and "
                    f"{visuals.format_number(live_summary.get('expand_sites', 0))} site(s) currently need expansion."
                ),
            ]
            ui.render_bullet_panel(
                "Suggested talk track",
                bullets,
                empty_message="The live model is ready, but there are no specific recommendations for the active scope.",
            )
    with scenario_col:
        with st.container(border=True):
            ui.render_section_heading(
                "Scenario library snapshot",
                "Seed and saved scenarios available for comparison from the current workbook and session.",
                eyebrow="Comparison setup",
            )
            if library:
                library_table = pd.DataFrame(
                    [
                        {
                            "Scenario": item["scenario_name"],
                            "Origin": item["origin"],
                            "Score": round(float(item.get("summary", {}).get("scenario_score", 0) or 0), 1),
                            "Seat gap": visuals.format_number(item.get("summary", {}).get("seat_gap", 0)),
                        }
                        for item in library[:5]
                    ]
                )
                st.dataframe(library_table, use_container_width=True, hide_index=True)
            else:
                ui.render_empty_state(
                    "No scenarios available",
                    "Load a workbook with scenario seeds or save a working snapshot to build out the comparison library.",
                )

    chart_col, mix_col = st.columns([1.2, 1.0])
    with chart_col:
        with st.container(border=True):
            ui.render_section_heading(
                "Regional capacity profile",
                "A quick sense-check of where the estate is concentrated before moving into detailed recommendations.",
                eyebrow="Portfolio shape",
            )
            if not site_table.empty:
                region_summary = site_table.groupby("region", dropna=False).agg(
                    current_headcount=("current_headcount", "sum"),
                    existing_seats=("existing_seats", "sum"),
                ).reset_index()
                fig = visuals.bar_chart(region_summary, x="region", y="existing_seats", title="Seat capacity by region")
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                ui.render_empty_state("No regional view available", "The current filters remove all portfolio rows.")
    with mix_col:
        with st.container(border=True):
            ui.render_section_heading(
                "Site type mix",
                "Helps frame whether the current discussion is driven by hubs, headquarters, or smaller offices.",
                eyebrow="Portfolio mix",
            )
            if not site_table.empty:
                site_type_summary = site_table.groupby("site_type", dropna=False).agg(Sites=("site_id", "nunique")).reset_index()
                fig = visuals.donut_chart(site_type_summary, names="site_type", values="Sites", title="Portfolio mix by site type")
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                ui.render_empty_state("No site mix available", "There are no sites in the current filter selection.")

    insight_col_1, insight_col_2, insight_col_3 = st.columns(3)
    with insight_col_1:
        ui.render_stat_card(
            "Forecast demand",
            visuals.format_number(live_summary.get("forecast_headcount", 0)),
            (
                f"Modelled against {visuals.format_number(live_summary.get('existing_seats', 0))} existing seats "
                "for the current scenario scope."
            ),
            tone="default",
        )
    with insight_col_2:
        ui.render_stat_card(
            "Pressure points",
            visuals.format_number(live_summary.get("expand_sites", 0)),
            (
                f"Sites currently triggering expansion, with "
                f"{visuals.format_number(live_summary.get('high_risk_sites', 0))} also marked high risk."
            ),
            tone="warning",
        )
    with insight_col_3:
        ui.render_stat_card(
            "Scenario score",
            visuals.format_number(live_summary.get("scenario_score", 0), "score"),
            (
                f"Backed by a standards alignment score of "
                f"{visuals.format_number(live_summary.get('standards_compliance_score', 0), 'score')}."
            ),
            tone="success",
        )

    with st.container(border=True):
        ui.render_section_heading(
            "Priority sites",
            "These are the first sites to bring into the live narrative because they combine risk, action need, and portfolio impact.",
            eyebrow="Recommended next click",
        )
        if not outputs.empty:
            exceptions = (
                outputs.assign(_risk_rank=outputs["risk_rating"].map(ui.risk_rank))
                .sort_values(["_risk_rank", "seat_gap", "scenario_score"], ascending=[True, True, False])
                .head(12)
                .rename(
                    columns={
                        "site_name": "Site",
                        "region": "Region",
                        "forecast_headcount": "Forecast headcount",
                        "required_seats": "Required seats",
                        "existing_seats": "Existing seats",
                        "seat_gap": "Seat gap",
                        "action_flag": "Recommended action",
                        "risk_rating": "Risk",
                        "scenario_score": "Scenario score",
                    }
                )
            )
            if not priority_site.empty:
                ui.render_callout(
                    "Lead site for review",
                    (
                        f"{priority_site.iloc[0]['site_name']} is the clearest near-term intervention site, "
                        f"currently flagged as {priority_site.iloc[0]['action_flag']} with "
                        f"{priority_site.iloc[0]['risk_rating']} risk."
                    ),
                    tone="accent",
                )
            st.dataframe(
                exceptions[
                    [
                        "Site",
                        "Region",
                        "Forecast headcount",
                        "Required seats",
                        "Existing seats",
                        "Seat gap",
                        "Recommended action",
                        "Risk",
                        "Scenario score",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            ui.render_empty_state(
                "No exception rows available",
                "The current filter combination does not return any live scenario rows to review.",
            )
