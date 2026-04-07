from __future__ import annotations

import streamlit as st

from apps.emea_space_occupancy import engine, ui, visuals
from apps.emea_space_occupancy.common import prepare_page


def render_page() -> None:
    context = prepare_page(
        "Portfolio Baseline",
        "Current-state portfolio footprint, demand-capacity position, and baseline exceptions.",
    )
    if not context.get("ready"):
        ui.render_callout("Workbook not ready", "Load a valid workbook before reviewing the portfolio baseline.", tone="warning")
        return

    baseline = engine.build_portfolio_baseline(context["clean_sheets"], context["filters"])
    site_table = baseline["site_table"]
    summary = baseline["portfolio_summary"]

    ui.render_hero_panel(
        eyebrow="Current-State Baseline",
        title="Understand the estate before changing the scenario",
        body=(
            "This page frames the existing portfolio footprint, space balance, and baseline exception ranking so the room "
            "has a clear starting point before reviewing interventions."
        ),
        badges=[
            ("Sites", visuals.format_number(summary.get("total_sites", 0))),
            ("Buildings", visuals.format_number(summary.get("total_buildings", 0))),
            ("Validation", context["validation"].get("status", "Unknown")),
        ],
    )

    visuals.render_metric_row(
        [
            ("Current Headcount", visuals.format_number(summary.get("current_headcount", 0)), None),
            ("Current Seats", visuals.format_number(summary.get("total_seat_capacity", 0)), None),
            (
                "Seat Surplus / Deficit",
                visuals.format_number(site_table.get("seat_gap_baseline", []).sum() if not site_table.empty else 0),
                None,
            ),
            ("Usable Area", visuals.format_number(summary.get("usable_area_sqm", 0), "sqm"), None),
            ("Annual Property Cost", visuals.format_number(summary.get("annual_property_cost_eur", 0), "eur"), None),
        ],
        columns=5,
    )

    lease_expiries = int(site_table.get("lease_expiry_within_24m", []).sum()) if not site_table.empty else 0
    ui.render_callout(
        "Baseline narrative",
        (
            f"The current scope includes {visuals.format_number(lease_expiries)} site(s) with lease expiry inside 24 months. "
            "Use these alongside the seat-gap ranking to prioritise which locations need immediate review."
        ),
        tone="info",
    )

    grain = st.radio("Summary grain", options=["Site", "Building", "Floor"], index=0, horizontal=True)
    display_table = {
        "Site": site_table,
        "Building": baseline["building_table"],
        "Floor": baseline["floor_table"],
    }[grain]
    table_col, chart_col = st.columns([1.25, 1.0])

    with chart_col:
        with st.container(border=True):
            ui.render_section_heading(
                "Capacity view",
                "A quick baseline visual before moving into the table or exception list.",
                eyebrow="Portfolio shape",
            )
            if not site_table.empty:
                top_gap = site_table[["site_name", "current_headcount", "existing_seats"]].copy()
                top_gap = top_gap.sort_values("current_headcount", ascending=False).head(12)
                fig = visuals.bar_chart(top_gap, x="site_name", y="existing_seats", title="Seats by site")
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                ui.render_empty_state("No capacity view available", "The active filters do not return any baseline sites.")

    with table_col:
        with st.container(border=True):
            ui.render_section_heading(
                f"{grain}-level summary",
                "Pre-sorted current-state output for the chosen grain.",
                eyebrow="Baseline table",
            )
            if not display_table.empty:
                if grain == "Site":
                    table_view = display_table[
                        [
                            "site_name",
                            "region",
                            "current_headcount",
                            "existing_seats",
                            "seat_gap_baseline",
                            "annual_property_cost_eur",
                            "avg_desk_utilisation_pct",
                            "earliest_lease_end_date",
                        ]
                    ].rename(
                        columns={
                            "site_name": "Site",
                            "region": "Region",
                            "current_headcount": "Current headcount",
                            "existing_seats": "Existing seats",
                            "seat_gap_baseline": "Seat gap",
                            "annual_property_cost_eur": "Annual property cost",
                            "avg_desk_utilisation_pct": "Desk utilisation",
                            "earliest_lease_end_date": "Lease end",
                        }
                    ).sort_values("Seat gap", ascending=True)
                elif grain == "Building":
                    table_view = display_table[
                        [
                            "site_name",
                            "building_name",
                            "current_headcount_allocated",
                            "seat_capacity_total",
                            "seat_gap_baseline",
                            "annual_property_cost_eur",
                            "lease_end_date",
                        ]
                    ].rename(
                        columns={
                            "site_name": "Site",
                            "building_name": "Building",
                            "current_headcount_allocated": "Allocated headcount",
                            "seat_capacity_total": "Seat capacity",
                            "seat_gap_baseline": "Seat gap",
                            "annual_property_cost_eur": "Annual property cost",
                            "lease_end_date": "Lease end",
                        }
                    ).sort_values("Seat gap", ascending=True)
                else:
                    table_view = display_table[
                        [
                            "site_name",
                            "building_name",
                            "floor_name",
                            "current_headcount_allocated",
                            "seat_capacity",
                            "seat_gap_baseline",
                        ]
                    ].rename(
                        columns={
                            "site_name": "Site",
                            "building_name": "Building",
                            "floor_name": "Floor",
                            "current_headcount_allocated": "Allocated headcount",
                            "seat_capacity": "Seat capacity",
                            "seat_gap_baseline": "Seat gap",
                        }
                    ).sort_values("Seat gap", ascending=True)
                st.dataframe(table_view, use_container_width=True, hide_index=True)
            else:
                ui.render_empty_state("No baseline rows available", "There are no rows at the selected grain for the current filters.")

    mix_col, exception_col = st.columns([1.0, 1.25])
    with mix_col:
        with st.container(border=True):
            ui.render_section_heading(
                "Space mix",
                "Highlights how the current estate is allocated across workspace types.",
                eyebrow="Spatial profile",
            )
            if not baseline["space_mix"].empty:
                mix_fig = visuals.donut_chart(baseline["space_mix"], names="space_type", values="area_sqm", title="Area by space type")
                if mix_fig is not None:
                    st.plotly_chart(mix_fig, use_container_width=True)
            else:
                ui.render_empty_state("No space mix available", "Space inventory is not available for the current filters.")

    with exception_col:
        with st.container(border=True):
            ui.render_section_heading(
                "Exception ranking",
                "Use this list to identify where the baseline position already looks weak before scenario assumptions are applied.",
                eyebrow="Priority list",
            )
            if not site_table.empty:
                exceptions = site_table[
                    [
                        "site_name",
                        "region",
                        "current_headcount",
                        "existing_seats",
                        "seat_gap_baseline",
                        "annual_property_cost_eur",
                        "earliest_lease_end_date",
                        "avg_desk_utilisation_pct",
                    ]
                ].rename(
                    columns={
                        "site_name": "Site",
                        "region": "Region",
                        "current_headcount": "Current headcount",
                        "existing_seats": "Existing seats",
                        "seat_gap_baseline": "Seat gap",
                        "annual_property_cost_eur": "Annual property cost",
                        "earliest_lease_end_date": "Lease end",
                        "avg_desk_utilisation_pct": "Desk utilisation",
                    }
                ).sort_values(["Seat gap", "Annual property cost"], ascending=[True, False])
                st.dataframe(exceptions.head(15), use_container_width=True, hide_index=True)
            else:
                ui.render_empty_state("No baseline exceptions", "There are no exception rows in the current filter scope.")
